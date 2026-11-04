"""Choose the conformal alpha without touching the test period.

The calibration split is cut in time order into two halves. The conformal gate is
fitted on the earlier half and scored on the later half, for each candidate alpha.
The cheapest alpha is then used with the gate refitted on the whole calibration
split, and the test period is scored once. Reviewers are not simulated here: the
cost of an escalated case is the review price plus the reviewer's error rate times
the cost of the wrong decision, so the numbers have no sampling noise.

Small calibration halves hold few fraud cases, so the chosen alpha is itself noisy.
Treat it as a starting value, not a finding. ASCII only.
"""
import numpy as np

from .backtest import best_threshold
from .conformal import ClassConditionalConformal


def expected_system_cost(actions, y, amt, review_cost, false_block_cost, reviewer_error):
    """Expected cost of a set of approve, block and escalate actions."""
    a, y, amt = np.asarray(actions), np.asarray(y), np.asarray(amt, float)
    esc = a == "escalate"
    cost = float(amt[(a == "approve") & (y == 1)].sum())
    cost += false_block_cost * float(((a == "block") & (y == 0)).sum())
    cost += review_cost * float(esc.sum())
    wrong = float(amt[esc & (y == 1)].sum()) + false_block_cost * float((esc & (y == 0)).sum())
    return cost + reviewer_error * wrong


def choose_alpha(p_fit, y_fit, p_eval, y_eval, amt_eval, review_cost,
                 false_block_cost, reviewer_error, alphas):
    """Return the cheapest alpha on the evaluation half and the cost of each alpha.

    Only calibration arrays are accepted, so the test period cannot leak in.
    Ties go to the smaller alpha, which is the stricter guarantee.
    """
    costs = {}
    for a in sorted(alphas):
        gate = ClassConditionalConformal(a).fit(p_fit, y_fit)
        costs[a] = expected_system_cost(gate.actions(p_eval), y_eval, amt_eval,
                                        review_cost, false_block_cost, reviewer_error)
    best = min(costs, key=lambda a: (costs[a], a))
    return best, costs


def run_sweep(pipe, cfg):
    frame, y, p, m = pipe["frame"], pipe["y"], pipe["p"], pipe["masks"]
    amt = frame["amt"].to_numpy(float)
    fb = cfg["costs"]["false_block"]
    err = cfg["reviewer"]["error_rate"]
    alphas = cfg["sweep"]["alphas"]

    cal = np.flatnonzero(m["calibration"])
    half = len(cal) // 2
    fit_i, eval_i = cal[:half], cal[half:]
    te = np.flatnonzero(m["test"])

    thr = best_threshold(pipe, cfg)
    model_actions = np.where(p[te] >= thr, "block", "approve")
    model_cost = expected_system_cost(model_actions, y[te], amt[te], 0.0, fb, err)
    default_gate = ClassConditionalConformal(cfg["alpha"]).fit(p[cal], y[cal])
    default_actions = default_gate.actions(p[te])

    rows = []
    for price in cfg["sweep"]["review_prices"]:
        best, costs = choose_alpha(p[fit_i], y[fit_i], p[eval_i], y[eval_i], amt[eval_i],
                                   price, fb, err, alphas)
        gate = ClassConditionalConformal(best).fit(p[cal], y[cal])
        rep = gate.report(p[te], y[te])
        acts = gate.actions(p[te])
        rows.append({
            "review_price": price, "chosen_alpha": best,
            "escalation_rate": rep["escalation_rate"],
            "coverage_fraud": rep["coverage_fraud"],
            "test_cost_chosen": expected_system_cost(acts, y[te], amt[te], price, fb, err),
            "test_cost_default_alpha": expected_system_cost(
                default_actions, y[te], amt[te], price, fb, err),
            "test_cost_model_only": model_cost,
            "eval_half_costs": {str(k): v for k, v in costs.items()},
        })
    winners = [r["review_price"] for r in rows if r["test_cost_chosen"] <= model_cost]
    return {
        "test_rows": int(len(te)), "test_frauds": int(y[te].sum()),
        "calibration_fit_rows": int(len(fit_i)), "calibration_eval_rows": int(len(eval_i)),
        "calibration_eval_frauds": int(y[eval_i].sum()),
        "reviewer_error": err, "default_alpha": cfg["alpha"],
        "model_threshold": thr, "rows": rows,
        "highest_price_where_conformal_wins": max(winners) if winners else None,
    }
