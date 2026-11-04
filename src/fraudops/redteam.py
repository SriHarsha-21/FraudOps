"""Red team: value-preserving mutations of fraud, and the blue team's response.

The attacker rewrites fraud cases in the test period so that each case looks more
like normal use while the total value stolen stays the same, in whole cents.
Detection is then measured on the same trained system, with features rebuilt from
the mutated timeline. The blue response retrains on a training period in which part
of the fraud has been mutated, and a challenger replaces the champion only if it
holds up on the clean test period.

Every row keeps the period it came from in part_orig, so splits do not move when
mutated rows change timestamps. Mutations here are a rule-based attacker written by
the author, so robustness is measured against this attacker only. ASCII only.
"""
import numpy as np
import pandas as pd

from .backtest import best_threshold
from .conformal import ClassConditionalConformal
from .detector import fit_temperature, predict_logit, sigmoid, train_detector
from .features import build_features
from .sweep import expected_system_cost

ATTACKS = ["amount_shaving", "splitting", "timing_spread", "category_swap", "local_looking"]
ALL_ATTACKS = ATTACKS + ["combined"]


def with_parts(pipe):
    """Copy of the pipeline result whose frame records each row's original period."""
    m = pipe["masks"]
    part = np.where(m["train"], "train",
                    np.where(m["calibration"], "calibration",
                             np.where(m["test"], "test", "unused")))
    frame = pipe["frame"].copy()
    frame["part_orig"] = part
    out = dict(pipe)
    out["frame"] = frame
    return out


def make_context(pipe, cfg):
    """What the attacker knows: each card's usual amount and the safest category."""
    fr = pipe["frame"]
    train = fr[fr["part_orig"] == "train"]
    rates = pipe["category_rates"]
    safe = sorted(rates, key=lambda c: (rates[c], c))[0]
    return {"median": train.groupby("cc_num")["amt"].median(),
            "global_median": float(train["amt"].median()),
            "safe_category": safe, "cfg": cfg["red_team"]}


def _cents(x):
    return np.round(np.asarray(x, float) * 100).astype(np.int64)


def _explode(fr, counts):
    rep = fr.loc[np.repeat(fr.index.to_numpy(), counts)].copy()
    j = np.concatenate([np.arange(c) for c in counts]) if len(counts) else np.array([], int)
    return rep, j


def _shift(rep, j, gap_minutes):
    rep["ts"] = rep["ts"] + pd.to_timedelta(j * gap_minutes, unit="m")
    return rep


def amount_shaving(fr, ctx, rng):
    """Cap each piece at a multiple of the card's usual amount and move the rest later."""
    rt = ctx["cfg"]
    med = fr["cc_num"].map(ctx["median"]).fillna(ctx["global_median"]).to_numpy(float)
    cap_c = np.maximum(100, np.floor(rt["shave_k"] * med * 100)).astype(np.int64)
    a_c = _cents(fr["amt"])
    counts = np.maximum(1, -(-a_c // cap_c))
    rep, j = _explode(fr, counts)
    cap_r, a_r, n_r = (np.repeat(v, counts) for v in (cap_c, a_c, counts))
    cents = np.where(j < n_r - 1, cap_r, a_r - (n_r - 1) * cap_r)
    rep["amt"] = cents / 100.0
    return _shift(rep, j, rt["shave_gap_min"])


def splitting(fr, ctx, rng):
    """Replace one purchase with equal pieces spaced beyond the one-hour velocity window."""
    rt = ctx["cfg"]
    a_c = _cents(fr["amt"])
    counts = np.minimum(rt["split_n"], np.maximum(1, a_c)).astype(np.int64)
    base = a_c // counts
    rep, j = _explode(fr, counts)
    base_r, a_r, n_r = (np.repeat(v, counts) for v in (base, a_c, counts))
    cents = np.where(j < n_r - 1, base_r, a_r - (n_r - 1) * base_r)
    rep["amt"] = cents / 100.0
    return _shift(rep, j, rt["split_gap_min"])


def timing_spread(fr, ctx, rng):
    """Push later fraud on the same card forward so bursts are at least a few hours apart."""
    gap = np.timedelta64(int(ctx["cfg"]["spread_gap_hours"] * 3600) * 10 ** 9, "ns")
    out = fr.sort_values(["cc_num", "ts"], kind="mergesort").copy()
    ts = out["ts"].to_numpy().astype("datetime64[ns]")
    cards = out["cc_num"].to_numpy()
    for i in range(1, len(out)):
        if cards[i] == cards[i - 1] and ts[i] < ts[i - 1] + gap:
            ts[i] = ts[i - 1] + gap
    out["ts"] = ts.astype(fr["ts"].to_numpy().dtype)
    return out


def category_swap(fr, ctx, rng):
    out = fr.copy()
    out["category"] = ctx["safe_category"]
    return out


def local_looking(fr, ctx, rng):
    """Move the merchant to within a couple of kilometres of the customer."""
    s = ctx["cfg"]["local_sigma_deg"]
    out = fr.copy()
    out["merch_lat"] = out["lat"] + rng.normal(0, s, len(out))
    out["merch_long"] = out["long"] + rng.normal(0, s, len(out))
    return out


def combined(fr, ctx, rng):
    """The three cheapest evasions together: safe category, local merchant, split."""
    return splitting(local_looking(category_swap(fr, ctx, rng), ctx, rng), ctx, rng)


_APPLY = {"amount_shaving": amount_shaving, "splitting": splitting,
          "timing_spread": timing_spread, "category_swap": category_swap,
          "local_looking": local_looking, "combined": combined}


def mutate_cases(frame, assignment, ctx, seed):
    """Rewrite the rows named in assignment (a Series of attack names indexed by row).

    Raises AssertionError if an attack changes the total value taken, in cents.
    """
    rng = np.random.default_rng(seed)
    base = frame.copy()
    base["case_id"] = -1
    base.loc[assignment.index, "case_id"] = assignment.index.to_numpy()
    parts = [base.drop(index=assignment.index)]
    before = int(_cents(base.loc[assignment.index, "amt"]).sum())
    after = 0
    for attack in sorted(set(assignment)):
        idx = assignment.index[assignment.to_numpy() == attack]
        rows = _APPLY[attack](base.loc[idx], ctx, rng)
        after += int(_cents(rows["amt"]).sum())
        parts.append(rows)
    if before != after:
        raise AssertionError("attack changed the total value taken: %d vs %d cents"
                             % (before, after))
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("ts", kind="mergesort").reset_index(drop=True)
    if "txn_id" in out.columns:
        out["txn_id"] = ["R%08d" % i for i in range(len(out))]
    return out


def score_world(pipe, fr):
    """Probabilities from the fixed trained system on a (possibly mutated) frame."""
    X = build_features(fr, pipe["categories"])
    return sigmoid(predict_logit(pipe["detector"], X) / pipe["temperature"])


def evaluate_world(fr, p, gate, thr, cfg):
    """Test-period results for model only and model plus conformal gate."""
    te = (fr["part_orig"] == "test").to_numpy()
    y = fr["is_fraud"].to_numpy()[te]
    amt = fr["amt"].to_numpy(float)[te]
    pt = np.asarray(p)[te]
    err = cfg["reviewer"]["error_rate"]
    rc, fb = cfg["costs"]["review"], cfg["costs"]["false_block"]
    fraud = y == 1
    total = float(amt[fraud].sum())

    def system(act):
        act = np.asarray(act)
        stopped = np.where(act == "block", 1.0, np.where(act == "escalate", 1.0 - err, 0.0))
        return {
            "value_recall": float((amt[fraud] * stopped[fraud]).sum() / total) if total else None,
            "row_recall": float(stopped[fraud].mean()) if fraud.any() else None,
            "silent_miss_rate": float((act[fraud] == "approve").mean()) if fraud.any() else None,
            "escalation_rate": float((act == "escalate").mean()),
            "cost": expected_system_cost(act, y, amt, rc, fb, err),
        }

    rep = gate.report(pt, y)
    return {
        "fraud_rows": int(fraud.sum()), "fraud_value": total,
        "model_only": system(np.where(pt >= thr, "block", "approve")),
        "model_plus_gate": system(gate.actions(pt)),
        "gate": {"coverage_fraud": rep["coverage_fraud"],
                 "coverage_legit": rep["coverage_legit"]},
    }


def attacked(pipe, attack, ctx, cfg, seed, thr):
    fr = pipe["frame"]
    target = fr.index[(fr["part_orig"] == "test") & (fr["is_fraud"] == 1)]
    fm = mutate_cases(fr, pd.Series(attack, index=target, dtype=object), ctx, seed)
    return evaluate_world(fm, score_world(pipe, fm), pipe["conformal"], thr, cfg)


def build_challenger(pipe, ctx, cfg, seed, attacks=None):
    """Retrain on a training period where some fraud has been mutated.

    attacks lists the attack types the blue team gets to see. The default is all five.
    """
    attacks = list(ATTACKS if attacks is None else attacks)
    fr = pipe["frame"]
    rt = cfg["red_team"]
    train_max = fr.loc[fr["part_orig"] == "train", "ts"].max()
    cutoff = train_max - pd.Timedelta(hours=rt["train_margin_hours"])
    elig = fr.index[(fr["part_orig"] == "train") & (fr["is_fraud"] == 1) & (fr["ts"] <= cutoff)]
    rng = np.random.default_rng(seed + 1)
    pick = rng.integers(0, len(attacks) + 1, size=len(elig))
    chosen = pick < len(attacks)
    assign = pd.Series([attacks[k] for k in pick[chosen]], index=elig[chosen], dtype=object)
    fr2 = mutate_cases(fr, assign, ctx, seed + 2)

    cats = pipe["categories"]
    X = build_features(fr2, cats)
    y = fr2["is_fraud"].to_numpy()
    part = fr2["part_orig"].to_numpy()
    masks = {k: part == k for k in ("train", "calibration", "test")}
    det = train_detector(X[masks["train"]], y[masks["train"]], cfg["detector"], cfg["seed"])
    logit = predict_logit(det, X)
    temp = fit_temperature(logit[masks["calibration"]], y[masks["calibration"]])
    p = sigmoid(logit / temp)
    gate = ClassConditionalConformal(cfg["alpha"]).fit(
        p[masks["calibration"]], y[masks["calibration"]])
    return {"frame": fr2, "p": p, "y": y, "masks": masks, "detector": det,
            "temperature": temp, "conformal": gate, "categories": cats,
            "mutated_training_cases": int(len(assign)),
            "training_attacks": sorted(set(assign))}


def promotion_decision(champion_clean, challenger_clean, cfg):
    """Promote only if the challenger is no worse on clean data and keeps coverage."""
    rt = cfg["red_team"]
    c0 = champion_clean["model_plus_gate"]["cost"]
    c1 = challenger_clean["model_plus_gate"]["cost"]
    cov = challenger_clean["gate"]["coverage_fraud"]
    cost_ok = c1 <= c0 * (1.0 + rt["promote_cost_tol"])
    cov_ok = cov is not None and cov >= 1.0 - cfg["alpha"] - rt["promote_coverage_margin"]
    return {"promote": bool(cost_ok and cov_ok), "cost_ok": bool(cost_ok),
            "coverage_ok": bool(cov_ok), "champion_clean_cost": c0,
            "challenger_clean_cost": c1, "challenger_fraud_coverage": cov}


def run_red_team(pipe, cfg, seed=0):
    champ = with_parts(pipe)
    ctx = make_context(champ, cfg)
    thr_c = best_threshold(champ, cfg)
    champion = {"clean": evaluate_world(champ["frame"], champ["p"], champ["conformal"],
                                        thr_c, cfg)}
    for i, a in enumerate(ALL_ATTACKS):
        champion[a] = attacked(champ, a, ctx, cfg, seed + 10 + i, thr_c)

    chall = build_challenger(champ, ctx, cfg, seed)
    thr_h = best_threshold(chall, cfg)
    challenger = {"clean": evaluate_world(chall["frame"], chall["p"], chall["conformal"],
                                          thr_h, cfg)}
    for i, a in enumerate(ALL_ATTACKS):
        challenger[a] = attacked(chall, a, ctx, cfg, seed + 10 + i, thr_h)
    return {
        "reviewer_error": cfg["reviewer"]["error_rate"], "alpha": cfg["alpha"],
        "safe_category": ctx["safe_category"],
        "mutated_training_cases": chall["mutated_training_cases"],
        "champion": champion, "challenger": challenger,
        "promotion": promotion_decision(champion["clean"], challenger["clean"], cfg),
    }


def _recovered_share(clean_recall, champion_recall, challenger_recall, floor=0.02):
    """Share of the champion's recall loss that the challenger wins back.

    None when the attack barely hurt the champion, because the ratio is then noise.
    """
    loss = clean_recall - champion_recall
    if loss < floor:
        return None
    return float((challenger_recall - champion_recall) / loss)


def run_generalization(pipe, cfg, seed=0):
    """Does the blue retrain survive attacks it has not seen?

    Two tests. Shifted parameters: the standard challenger faces every attack with
    different settings (tighter cap, more pieces, longer gaps). Held-out attack: a
    challenger is retrained without one attack type and then faces that attack.
    """
    rt = cfg["red_team"]
    champ = with_parts(pipe)
    ctx = make_context(champ, cfg)
    thr_c = best_threshold(champ, cfg)
    clean = evaluate_world(champ["frame"], champ["p"], champ["conformal"], thr_c, cfg)
    shifted_ctx = dict(ctx, cfg={**rt, **rt["shifted"]})
    full = build_challenger(champ, ctx, cfg, seed)
    thr_f = best_threshold(full, cfg)

    def row(test, attack, champ_res, chall_res):
        cm, hm = champ_res["model_only"], chall_res["model_only"]
        return {
            "test": test, "attack": attack, "champion": champ_res, "challenger": chall_res,
            "model_recall_recovered_share": _recovered_share(
                clean["model_only"]["value_recall"], cm["value_recall"], hm["value_recall"]),
        }

    rows = []
    for i, a in enumerate(ATTACKS):
        rows.append(row("shifted_parameters", a,
                        attacked(champ, a, shifted_ctx, cfg, seed + 50 + i, thr_c),
                        attacked(full, a, shifted_ctx, cfg, seed + 50 + i, thr_f)))
    for i, a in enumerate(rt["holdout_attacks"]):
        ch = build_challenger(champ, ctx, cfg, seed, attacks=[x for x in ATTACKS if x != a])
        assert a not in ch["training_attacks"]
        rows.append(row("held_out_attack", a,
                        attacked(champ, a, ctx, cfg, seed + 80 + i, thr_c),
                        attacked(ch, a, ctx, cfg, seed + 80 + i, best_threshold(ch, cfg))))
    return {"champion_clean_model_recall": clean["model_only"]["value_recall"],
            "shifted_settings": rt["shifted"], "rows": rows}
