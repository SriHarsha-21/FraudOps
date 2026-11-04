"""Compare the rule planner with a local LLM planner on the same escalated cases.

Needs a running Ollama server. Usage:
    python scripts/run_llm_compare.py --n 20
    python scripts/run_llm_compare.py --train fraudTrain.csv --max-cards 200 --n 30
ASCII only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraudops.agents.graph import build_graph, make_checkpointer, start_case  # noqa: E402
from fraudops.agents.llm import make_ollama_planner, ollama_chat  # noqa: E402
from fraudops.agents.planner import rule_planner  # noqa: E402
from fraudops.agents.store import DecisionStore  # noqa: E402
from fraudops.agents.tools import CaseContext  # noqa: E402
from fraudops.pipeline import fit_pipeline, load_frames  # noqa: E402
import numpy as np  # noqa: E402


def run(ctx, cfg, planner, ids):
    app = build_graph(ctx, DecisionStore(":memory:"), make_checkpointer(), planner=planner)
    out = {}
    for tid in ids:
        payload = start_case(app, tid, cfg)["payload"]
        out[tid] = ([e["tool"] for e in payload["evidence"]], payload["recommendation"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--max-cards", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model")
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "config.json"))
    ap.add_argument("--out", default="results/llm_compare.json")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    train_df, test_df = load_frames(args.train, args.test, args.max_cards, cfg["seed"])
    pipe = fit_pipeline(train_df, test_df, cfg)
    ctx = CaseContext(pipe, cfg)
    rows = np.flatnonzero(pipe["masks"]["test"])
    ids = [ctx.frame["txn_id"].iloc[i] for i in rows if ctx.gate_actions[i] == "escalate"]
    ids = ids[:args.n]
    llm = cfg["llm"]
    name = args.model or llm["model"]
    planner = make_ollama_planner(lambda p: ollama_chat(p, name, llm["host"], llm["timeout_s"]))
    base = run(ctx, cfg, rule_planner, ids)
    test = run(ctx, cfg, planner, ids)
    same_tools = sum(set(base[t][0]) == set(test[t][0]) for t in ids)
    same_order = sum(base[t][0] == test[t][0] for t in ids)
    same_rec = sum(base[t][1] == test[t][1] for t in ids)
    n = max(len(ids), 1)
    result = {"model": name, "cases": len(ids), "llm_calls": planner.stats["calls"],
              "fallbacks_to_rules": planner.stats["fallbacks"],
              "same_tool_set": same_tools / n, "same_tool_order": same_order / n,
              "same_recommendation": same_rec / n,
              "mean_tools_rule": sum(len(base[t][0]) for t in ids) / n,
              "mean_tools_llm": sum(len(test[t][0]) for t in ids) / n}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
