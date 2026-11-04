"""Stage 4b: does the blue retrain survive attacks it has not seen?

Usage:
    python scripts/run_generalization.py                       # synthetic fixture dry run
    python scripts/run_generalization.py --train fraudTrain.csv --max-cards 200
ASCII only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraudops.pipeline import fit_pipeline, load_frames  # noqa: E402
from fraudops.redteam import run_generalization  # noqa: E402


def _f(v):
    return "n/a" if v is None else "%.3f" % v


def table(result):
    rows = ["| Test | Attack | Fraud rows | Champion model recall | Champion gate recall | "
            "Challenger model recall | Challenger gate recall | Challenger gate silent miss | "
            "Recall loss recovered |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["rows"]:
        c, h = r["champion"], r["challenger"]
        rows.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            r["test"], r["attack"], c["fraud_rows"], _f(c["model_only"]["value_recall"]),
            _f(c["model_plus_gate"]["value_recall"]), _f(h["model_only"]["value_recall"]),
            _f(h["model_plus_gate"]["value_recall"]),
            _f(h["model_plus_gate"]["silent_miss_rate"]),
            _f(r["model_recall_recovered_share"])))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--max-cards", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "config.json"))
    ap.add_argument("--out", default="results/generalization.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    train_df, test_df = load_frames(args.train, args.test, args.max_cards, cfg["seed"])
    pipe = fit_pipeline(train_df, test_df, cfg)
    result = run_generalization(pipe, cfg, seed=cfg["seed"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    md = table(result)
    with open(args.out.replace(".json", ".md"), "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(json.dumps({"champion_clean_model_recall": result["champion_clean_model_recall"],
                      "shifted_settings": result["shifted_settings"]}, indent=2))


if __name__ == "__main__":
    main()
