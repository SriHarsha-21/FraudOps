"""Stage 3: backtest the systems and write results/backtest.json plus a markdown table.

Usage:
    python scripts/run_backtest.py                       # synthetic fixture dry run
    python scripts/run_backtest.py --train fraudTrain.csv --test fraudTest.csv --max-cards 200
ASCII only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraudops.backtest import run_backtest  # noqa: E402
from fraudops.pipeline import fit_pipeline, load_frames  # noqa: E402


def table(result):
    rows = ["| System | Precision | Recall | Missed fraud amount | False blocks | Reviews | Cost |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in result["systems"].items():
        fmt = lambda v: "n/a" if v is None else "%.3f" % v
        rows.append("| %s | %s | %s | %.0f | %d | %d | %.0f |" % (
            name, fmt(s["precision"]), fmt(s["recall"]), s["missed_fraud_amount"],
            s["false_blocks"], s["reviews"], s["cost"]))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--max-cards", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "config.json"))
    ap.add_argument("--out", default="results/backtest.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    train_df, test_df = load_frames(args.train, args.test, args.max_cards, cfg["seed"])
    pipe = fit_pipeline(train_df, test_df, cfg)
    result = run_backtest(pipe, cfg, seed=cfg["seed"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    md = table(result)
    with open(args.out.replace(".json", ".md"), "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(json.dumps({k: result[k] for k in ("mcnemar", "agents")}, indent=2))


if __name__ == "__main__":
    main()
