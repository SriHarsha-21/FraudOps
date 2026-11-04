"""Evaluate the drift monitor: validity, power against a fraud surge, and the real test stream.

Usage:
    python scripts/run_monitor.py
    python scripts/run_monitor.py --train fraudTrain.csv --max-cards 200
ASCII only.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraudops.agents.tools import CaseContext  # noqa: E402
from fraudops.monitor import ScoreMonitor  # noqa: E402
from fraudops.pipeline import fit_pipeline, load_frames  # noqa: E402


def halves(p_cal, rng):
    idx = rng.permutation(len(p_cal))
    h = len(idx) // 2
    return idx[:h], idx[h:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--max-cards", type=int, default=0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "config.json"))
    ap.add_argument("--out", default="results/monitor.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    train_df, test_df = load_frames(args.train, args.test, args.max_cards, cfg["seed"])
    pipe = fit_pipeline(train_df, test_df, cfg)
    ctx = CaseContext(pipe, cfg)
    masks = pipe["masks"]
    p_cal = ctx.p[masks["calibration"]]
    y_cal = ctx.y[masks["calibration"]]
    thr = cfg["monitor"]["threshold"]
    out = {"threshold": thr, "false_alarm_bound": min(1.0, 2.0 / thr), "trials": args.trials}

    alarms = 0
    for t in range(args.trials):
        rng = np.random.default_rng(1000 + t)
        a, b = halves(p_cal, rng)
        mon = ScoreMonitor(p_cal[a], thr, seed=t)
        mon.run(p_cal[b])
        alarms += mon.alarm_at is not None
    out["control_alarm_rate"] = alarms / args.trials

    surge = {}
    for q in (0.02, 0.05, 0.10):
        delays, caught = [], 0
        for t in range(args.trials):
            rng = np.random.default_rng(5000 + t)
            a, b = halves(p_cal, rng)
            stream, ystream = p_cal[b].copy(), y_cal[b]
            pool = stream[ystream == 1]
            if len(pool) < 3:
                continue
            start = len(stream) // 2
            swap = rng.random(len(stream) - start) < q
            stream[start:][swap] = rng.choice(pool, size=int(swap.sum()))
            mon = ScoreMonitor(p_cal[a], thr, seed=t)
            mon.run(stream)
            if mon.alarm_at is not None and mon.alarm_at >= start:
                caught += 1
                delays.append(mon.alarm_at - start)
        surge["q=%.2f" % q] = {
            "detected": caught / args.trials,
            "median_delay_cases": float(np.median(delays)) if delays else None,
            "stream_cases": int(len(p_cal) // 2)}
    out["surge"] = surge

    rows = np.flatnonzero(masks["test"])
    real = ScoreMonitor(p_cal, thr, seed=cfg["monitor"]["seed"])
    real.run(ctx.p[rows])
    rep = real.report()
    rep["auto_approvals_after_alarm"] = 0 if rep["alarm_at"] is None else int(
        sum(ctx.gate_actions[i] == "approve" for i in rows[rep["alarm_at"]:]))
    out["real_test_stream"] = rep

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
