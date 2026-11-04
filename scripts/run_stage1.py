"""Stage 1: clean, build features, split by time, train, calibrate, conformal report.

Usage:
    python scripts/run_stage1.py                      # synthetic fixture dry run
    python scripts/run_stage1.py --train fraudTrain.csv --test fraudTest.csv --max-cards 200
ASCII only.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraudops.data import clean, make_fixture  # noqa: E402
from fraudops.detector import ece, expected_cost, rank_metrics  # noqa: E402
from fraudops.pipeline import fit_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", help="path to fraudTrain.csv (omit for the fixture)")
    ap.add_argument("--test", help="path to fraudTest.csv (optional)")
    ap.add_argument("--max-cards", type=int, default=0, help="keep this many cards")
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "config.json"))
    ap.add_argument("--out", default="results/stage1_metrics.json")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)

    if args.train:
        raw_train = pd.read_csv(args.train)
        raw_test = pd.read_csv(args.test) if args.test else None
    else:
        raw_train, raw_test = make_fixture(seed=cfg["seed"]), None

    train_df = clean(raw_train)
    test_df = clean(raw_test) if raw_test is not None else None
    if args.max_cards:
        cards = np.random.default_rng(cfg["seed"]).permutation(
            train_df["cc_num"].unique())[:args.max_cards]
        train_df = train_df[train_df["cc_num"].isin(cards)].reset_index(drop=True)
        if test_df is not None:
            test_df = test_df[test_df["cc_num"].isin(cards)].reset_index(drop=True)

    pipe = fit_pipeline(train_df, test_df, cfg)
    frame, y, p, m = pipe["frame"], pipe["y"], pipe["p"], pipe["masks"]
    te = m["test"]
    conf = pipe["conformal"]
    actions = conf.actions(p[te])
    result = {
        "rows": {k: int(v.sum()) for k, v in m.items()},
        "fraud_rate": {k: float(y[v].mean()) for k, v in m.items()},
        "temperature": pipe["temperature"],
        "ece_test": ece(p[te], y[te]),
        "ranking": rank_metrics(p[te], y[te]),
        "conformal": conf.report(p[te], y[te]),
        "expected_cost": expected_cost(actions, y[te], frame.loc[te, "amt"],
                                       cfg["costs"]["review"], cfg["costs"]["false_block"]),
        "date_range": [str(frame["ts"].min()), str(frame["ts"].max())],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
