"""Stage 3b: choose alpha on the calibration split and show cost across review prices.

Usage:
    python scripts/run_sweep.py                       # synthetic fixture dry run
    python scripts/run_sweep.py --train fraudTrain.csv --max-cards 200
ASCII only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraudops.pipeline import fit_pipeline, load_frames  # noqa: E402
from fraudops.sweep import run_sweep  # noqa: E402


def table(result):
    rows = ["| Review price | Chosen alpha | Escalation rate | Fraud coverage | "
            "Cost (chosen) | Cost (alpha 0.1) | Cost (model only) |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["rows"]:
        cov = "n/a" if r["coverage_fraud"] is None else "%.3f" % r["coverage_fraud"]
        rows.append("| %.2f | %.2f | %.3f | %s | %.0f | %.0f | %.0f |" % (
            r["review_price"], r["chosen_alpha"], r["escalation_rate"], cov,
            r["test_cost_chosen"], r["test_cost_default_alpha"], r["test_cost_model_only"]))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--max-cards", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "config.json"))
    ap.add_argument("--out", default="results/sweep.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    train_df, test_df = load_frames(args.train, args.test, args.max_cards, cfg["seed"])
    pipe = fit_pipeline(train_df, test_df, cfg)
    result = run_sweep(pipe, cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    md = table(result)
    with open(args.out.replace(".json", ".md"), "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(json.dumps({k: result[k] for k in (
        "test_rows", "test_frauds", "calibration_eval_rows", "calibration_eval_frauds",
        "reviewer_error", "highest_price_where_conformal_wins")}, indent=2))


if __name__ == "__main__":
    main()
