"""Stage 4: red team attacks, robustness drop, blue retrain and promotion decision.

Usage:
    python scripts/run_redteam.py                       # synthetic fixture dry run
    python scripts/run_redteam.py --train fraudTrain.csv --max-cards 200
ASCII only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraudops.pipeline import fit_pipeline, load_frames  # noqa: E402
from fraudops.redteam import ALL_ATTACKS, run_red_team  # noqa: E402


def _f(v):
    return "n/a" if v is None else "%.3f" % v


def table(block):
    rows = ["| Attack | Fraud rows | Model value recall | Gate value recall | "
            "Model silent miss | Gate silent miss | Gate fraud coverage | "
            "Model cost | Gate cost |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in ["clean"] + ALL_ATTACKS:
        r = block[name]
        m, g = r["model_only"], r["model_plus_gate"]
        rows.append("| %s | %d | %s | %s | %s | %s | %s | %.0f | %.0f |" % (
            name, r["fraud_rows"], _f(m["value_recall"]), _f(g["value_recall"]),
            _f(m["silent_miss_rate"]), _f(g["silent_miss_rate"]),
            _f(r["gate"]["coverage_fraud"]), m["cost"], g["cost"]))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--max-cards", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "config.json"))
    ap.add_argument("--out", default="results/redteam.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    train_df, test_df = load_frames(args.train, args.test, args.max_cards, cfg["seed"])
    pipe = fit_pipeline(train_df, test_df, cfg)
    result = run_red_team(pipe, cfg, seed=cfg["seed"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    md = "Champion (trained on clean data)\n\n%s\n\nChallenger (retrained with mutated fraud)\n\n%s" % (
        table(result["champion"]), table(result["challenger"]))
    with open(args.out.replace(".json", ".md"), "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(json.dumps({k: result[k] for k in (
        "safe_category", "mutated_training_cases", "reviewer_error", "promotion")}, indent=2))


if __name__ == "__main__":
    main()
