"""LangGraph workflow: score -> conformal gate -> bounded investigation -> human review.

Design rules enforced in code, not in prompts:
  * The investigation loop is capped by agent.max_steps.
  * A planner can only name an unused tool or say done. Anything else is rejected.
  * Only a human or the conformal gate can set a verdict. No LLM output does.
  * Escalated cases pause at a LangGraph interrupt until a reviewer decides.
ASCII only.
"""
import hashlib
import json
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .planner import rule_planner, validate_choice
from .tools import make_tools

GRAPH_VERSION = "1"


class State(TypedDict, total=False):
    txn_id: str
    p_fraud: float
    gate_action: str
    evidence: list
    steps: int
    next_tool: str
    invalid_planner_outputs: int
    tool_error: bool
    recommendation: str
    reasons: list
    verdict: str
    decided_by: str
    reviewer: str
    trace: list


def config_hash(cfg):
    body = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def make_checkpointer(path=":memory:"):
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def build_graph(ctx, store, checkpointer=None, planner=rule_planner):
    cfg = ctx.cfg
    agent_cfg = cfg["agent"]
    tools = make_tools(ctx)
    cfg_hash = config_hash(cfg)

    def score(state):
        i = ctx.row(state["txn_id"])
        p = float(ctx.p[i])
        action = str(ctx.gate_actions[i])
        reasons = []
        if ctx.trust_level == "manual-only" and action != "escalate":
            action = "escalate"
            reasons.append("trust level manual-only sends every case to a human")
        elif ctx.trust_level == "cautious" and action == "approve":
            action = "escalate"
            reasons.append("trust level cautious disables automatic approval")
        if action == "escalate" and not reasons:
            reasons.append("conformal set held both labels or neither")
        event = {"event": "score", "p_fraud": round(p, 4), "gate_action": action}
        return {"p_fraud": p, "gate_action": action, "evidence": [], "steps": 0,
                "invalid_planner_outputs": 0, "tool_error": False,
                "reasons": reasons, "trace": [event]}

    def route_gate(state):
        return "plan" if state["gate_action"] == "escalate" else "finalize"

    def plan(state):
        raw, bad = "", False
        if state["steps"] >= agent_cfg["max_steps"]:
            nxt, why = "done", "step cap reached"
        else:
            raw = planner(state)
            nxt, bad = validate_choice(raw, state)
            why = "planner choice rejected" if bad else "planner choice accepted"
        event = {"event": "plan", "raw": str(raw)[:40], "chosen": nxt, "note": why}
        return {"next_tool": nxt,
                "invalid_planner_outputs": state["invalid_planner_outputs"] + int(bad),
                "trace": state["trace"] + [event]}

    def route_plan(state):
        return "recommend" if state["next_tool"] == "done" else "act"

    def act(state):
        name = state["next_tool"]
        try:
            result = tools[name](state["txn_id"])
            evidence = state["evidence"] + [result]
            event = {"event": "act", "tool": name, "signal": round(result["signal"], 3),
                     "note": result["note"]}
            return {"evidence": evidence, "steps": state["steps"] + 1,
                    "trace": state["trace"] + [event]}
        except Exception as exc:  # a tool failure always escalates
            event = {"event": "act", "tool": name, "error": type(exc).__name__}
            return {"steps": state["steps"] + 1, "tool_error": True,
                    "reasons": state["reasons"] + ["tool error in " + name],
                    "trace": state["trace"] + [event]}

    def recommend(state):
        sigs = [e["signal"] for e in state["evidence"]]
        ev = (0.6 * max(sigs) + 0.4 * sum(sigs) / len(sigs)) if sigs else 0.0
        combined = 0.5 * state["p_fraud"] + 0.5 * ev
        if combined >= agent_cfg["block_at"]:
            rec = "block"
        elif combined <= agent_cfg["approve_at"]:
            rec = "approve"
        else:
            rec = "review"
        reasons = list(state["reasons"])
        model_says_fraud = state["p_fraud"] >= 0.5
        if (rec == "approve" and model_says_fraud) or (rec == "block" and not model_says_fraud):
            reasons.append("evidence contradicts the model verdict")
        if state["tool_error"]:
            rec = "review"
        if state["invalid_planner_outputs"]:
            reasons.append("planner output was rejected")
        event = {"event": "recommend", "combined_score": round(combined, 4),
                 "recommendation": rec}
        return {"recommendation": rec, "reasons": reasons,
                "trace": state["trace"] + [event]}

    def human_review(state):
        answer = interrupt({
            "txn_id": state["txn_id"], "p_fraud": state["p_fraud"],
            "gate_action": state["gate_action"], "evidence": state["evidence"],
            "recommendation": state["recommendation"], "reasons": state["reasons"]})
        decision = answer.get("decision") if isinstance(answer, dict) else None
        if decision not in ("approve", "block"):
            raise ValueError("decision must be 'approve' or 'block'")
        event = {"event": "human", "decision": decision}
        return {"verdict": decision, "decided_by": "human",
                "reviewer": str(answer.get("reviewer", "unknown"))[:40],
                "trace": state["trace"] + [event]}

    def finalize(state):
        verdict = state.get("verdict") or state["gate_action"]
        decided_by = state.get("decided_by", "auto")
        record = {
            "case_id": state["txn_id"], "txn_id": state["txn_id"],
            "versions": {"config_hash": cfg_hash, "graph": GRAPH_VERSION,
                         "planner": getattr(planner, "__name__", "planner"),
                         "alpha": cfg["alpha"], "trust_level": ctx.trust_level},
            "p_fraud": state["p_fraud"], "gate_action": state["gate_action"],
            "evidence": state["evidence"], "recommendation": state.get("recommendation"),
            "reasons": state["reasons"], "trace": state["trace"],
            "verdict": verdict, "decided_by": decided_by,
            "reviewer": state.get("reviewer"),
        }
        if not store.append(state["txn_id"], record):
            existing = store.get(state["txn_id"])
            return {"verdict": existing["verdict"], "decided_by": existing["decided_by"]}
        return {"verdict": verdict, "decided_by": decided_by}

    g = StateGraph(State)
    for name, fn in [("score", score), ("plan", plan), ("act", act),
                     ("recommend", recommend), ("human_review", human_review),
                     ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "score")
    g.add_conditional_edges("score", route_gate, ["plan", "finalize"])
    g.add_conditional_edges("plan", route_plan, ["act", "recommend"])
    g.add_edge("act", "plan")
    g.add_edge("recommend", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or make_checkpointer())


def _config(cfg, thread_id):
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": cfg["agent"]["recursion_limit"]}


def start_case(app, txn_id, cfg):
    """Run a case. Returns {'status': 'paused', 'payload': ...} or {'status': 'done', ...}."""
    out = app.invoke({"txn_id": txn_id}, _config(cfg, txn_id))
    return _result(out)


def resume_case(app, txn_id, answer, cfg):
    """Resume a paused case with {'decision': 'approve'|'block', 'reviewer': name}."""
    out = app.invoke(Command(resume=answer), _config(cfg, txn_id))
    return _result(out)


def _result(out):
    if "__interrupt__" in out:
        return {"status": "paused", "payload": out["__interrupt__"][0].value}
    return {"status": "done", "verdict": out["verdict"], "decided_by": out["decided_by"],
            "state": out}
