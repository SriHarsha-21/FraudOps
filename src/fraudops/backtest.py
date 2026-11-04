"""Backtest on the held-out period.

Tier A (full test period, no agents): rules only, model only, model plus conformal.
Tier B (agents, on escalated cases): does the investigation add anything beyond the
model, and how much human work could it remove? Escalated cases are sampled up to a
cap because agent runs are slower. Reviewers are simulated: the true decision is
returned with a configurable error rate. That is an assumption, not measured.
ASCII only.
"""
import numpy as np
from scipy.stats import chi2

from .agents.graph import build_graph, resume_case, start_case
from .agents.store import DecisionStore
from .agents.tools import CaseContext


def decision_cost(decision, y, amt, false_block_cost):
    if decision == "approve" and y == 1:
        return float(amt)
    if decision == "block" and y == 0:
        return float(false_block_cost)
    return 0.0


def summarize(actions, y, amt, reviews, review_cost, false_block_cost):
    """actions must be approve or block for every case (reviews already resolved)."""
    a, y, amt = np.asarray(actions), np.asarray(y), np.asarray(amt, float)
    blocked = a == "block"
    tp = int((blocked & (y == 1)).sum())
    fp = int((blocked & (y == 0)).sum())
    missed = float(amt[(a == "approve") & (y == 1)].sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / int((y == 1).sum()) if (y == 1).any() else None,
        "missed_fraud_amount": missed, "false_blocks": fp, "reviews": int(reviews),
        "cost": missed + false_block_cost * fp + review_cost * reviews,
    }


def mcnemar(correct_a, correct_b):
    """McNemar test with Edwards continuity correction on per-case correctness."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "a_only": n10, "b_only": n01}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"statistic": float(stat), "p_value": float(chi2.sf(stat, 1)),
            "a_only": n10, "b_only": n01}


def _oracle(y, rng, error_rate):
    truth = "block" if y == 1 else "approve"
    if rng.random() < error_rate:
        return "approve" if truth == "block" else "block"
    return truth


def rules_actions(pipe):
    """Block if any of three features exceeds its 99th percentile in training data."""
    X, m = pipe["X"], pipe["masks"]
    train = X[m["train"]]
    cols = ["count_1h", "dist_km", "amt_over_median"]
    limits = {c: np.nanquantile(train[c], 0.99) for c in cols}
    test = X[m["test"]]
    hit = np.zeros(len(test), bool)
    for c in cols:
        hit |= (test[c].to_numpy() > limits[c]) & ~np.isnan(test[c].to_numpy())
    return np.where(hit, "block", "approve"), limits


def best_threshold(pipe, cfg):
    """Cost-minimizing probability threshold chosen on the calibration split."""
    p, y, m = pipe["p"], pipe["y"], pipe["masks"]
    amt = pipe["frame"]["amt"].to_numpy(float)
    pc, yc, ac = p[m["calibration"]], y[m["calibration"]], amt[m["calibration"]]
    best_t, best_cost = 0.5, np.inf
    for t in np.unique(np.quantile(pc, np.linspace(0.5, 0.9999, 200))):
        act = np.where(pc >= t, "block", "approve")
        cost = (ac[(act == "approve") & (yc == 1)].sum()
                + cfg["costs"]["false_block"] * ((act == "block") & (yc == 0)).sum())
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t


def agent_eval(pipe, cfg, cap, seed):
    ctx = CaseContext(pipe, cfg)
    store = DecisionStore()
    app = build_graph(ctx, store)
    rng = np.random.default_rng(seed)
    frame, y, p = pipe["frame"], pipe["y"], pipe["p"]
    amt = frame["amt"].to_numpy(float)
    tids = frame["txn_id"].to_numpy()
    test_rows = np.flatnonzero(pipe["masks"]["test"])
    esc = [i for i in test_rows if ctx.gate_actions[i] == "escalate"]
    if not esc:
        return {"escalated_in_test": 0}
    sample = rng.choice(esc, size=min(cap, len(esc)), replace=False)
    thr = best_threshold(pipe, cfg)
    err = cfg["reviewer"]["error_rate"]
    rc, fb = cfg["costs"]["review"], cfg["costs"]["false_block"]
    agent_auto = agent_right = model_right = 0
    cost_human = cost_agent = cost_model = 0.0
    for i in sample:
        res = start_case(app, tids[i], cfg)
        rec = res["payload"]["recommendation"]
        human = _oracle(y[i], rng, err)
        resume_case(app, tids[i], {"decision": human, "reviewer": "oracle"}, cfg)
        model_dec = "block" if p[i] >= thr else "approve"
        truth = "block" if y[i] == 1 else "approve"
        cost_human += rc + decision_cost(human, y[i], amt[i], fb)
        cost_model += decision_cost(model_dec, y[i], amt[i], fb)
        if rec in ("approve", "block"):
            agent_auto += 1
            agent_right += int(rec == truth)
            model_right += int(model_dec == truth)
            cost_agent += decision_cost(rec, y[i], amt[i], fb)
        else:
            cost_agent += rc + decision_cost(human, y[i], amt[i], fb)
    n = len(sample)
    return {
        "escalated_in_test": len(esc), "escalated_evaluated": int(n),
        "agent_resolved_share": agent_auto / n,
        "agent_accuracy_on_resolved": agent_right / agent_auto if agent_auto else None,
        "model_accuracy_on_same_cases": model_right / agent_auto if agent_auto else None,
        "total_cost": {"all_to_human": cost_human, "agent_resolves_confident": cost_agent,
                       "model_threshold_only": cost_model},
        "records_stored": store.count(), "chain_ok": store.verify_chain(),
    }


def run_backtest(pipe, cfg, seed=0):
    p, y, m = pipe["p"], pipe["y"], pipe["masks"]
    te = m["test"]
    frame = pipe["frame"]
    amt = frame["amt"].to_numpy(float)[te]
    yt, pt = y[te], p[te]
    rc, fb = cfg["costs"]["review"], cfg["costs"]["false_block"]
    rng = np.random.default_rng(seed)
    truth = np.where(yt == 1, "block", "approve")

    rules, limits = rules_actions(pipe)
    thr = best_threshold(pipe, cfg)
    model = np.where(pt >= thr, "block", "approve")
    gate = pipe["conformal"].actions(pt)
    resolved = gate.copy()
    esc = gate == "escalate"
    for j in np.flatnonzero(esc):
        resolved[j] = _oracle(yt[j], rng, cfg["reviewer"]["error_rate"])

    systems = {
        "1_rules_only": summarize(rules, yt, amt, 0, rc, fb),
        "2_model_only": summarize(model, yt, amt, 0, rc, fb),
        "3_model_plus_conformal_human": summarize(resolved, yt, amt, int(esc.sum()), rc, fb),
    }
    right = {"1": rules == truth, "2": model == truth, "3": resolved == truth}
    tests = {"1_vs_2": mcnemar(right["1"], right["2"]),
             "2_vs_3": mcnemar(right["2"], right["3"])}
    return {
        "test_rows": int(te.sum()), "test_frauds": int(yt.sum()),
        "model_threshold": thr, "rule_limits": {k: float(v) for k, v in limits.items()},
        "systems": systems, "mcnemar": tests,
        "agents": agent_eval(pipe, cfg, cfg["backtest"]["agent_cap"], seed),
    }
