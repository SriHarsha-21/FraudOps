"""Verifier: checks that the agent's suggestion follows from its own evidence.

It runs between recommend and human review. It can only add flags for the reviewer to see.
It can never approve, block or remove a case from review, and a verifier error becomes a
flag (fail closed). The optional model second opinion is also flag-only. ASCII only.
"""
import re

_UNSAFE = re.compile(r"[^A-Za-z0-9 .,%:;()_/-]")


def rule_verifier(state):
    """Return a list of short flags. An empty list means the suggestion is supported."""
    ev = state.get("evidence", [])
    sigs = [e["signal"] for e in ev]
    rec = state.get("recommendation")
    p = state.get("p_fraud", 0.0)
    flags = []
    if not ev:
        flags.append("no evidence was gathered")
    if rec == "block" and sigs and max(sigs) < 0.3:
        flags.append("block suggested but every signal is weak")
    if rec == "approve" and sigs and max(sigs) >= 0.8:
        flags.append("approve suggested despite a strong signal")
    if rec == "approve" and p >= 0.5:
        flags.append("approve suggested against a high model score")
    if rec == "block" and p < 0.2 and sigs and max(sigs) < 0.6:
        flags.append("block suggested against a low model score")
    if len(sigs) >= 2 and max(sigs) >= 0.8 and min(sigs) <= 0.1:
        flags.append("tool signals disagree strongly")
    if state.get("tool_error"):
        flags.append("a tool failed during the investigation")
    return flags


def make_llm_verifier(chat, base=rule_verifier):
    """Rule checks plus a model second opinion. The model can only add a flag."""
    stats = {"calls": 0, "failures": 0, "disagreements": 0}

    def llm_verifier(state):
        flags = list(base(state))
        ev = state.get("evidence", [])
        facts = "; ".join("%s %.2f" % (e["tool"], e["signal"]) for e in ev) or "none"
        prompt = (
            "A fraud agent suggests: %s. Model fraud probability %.2f. Tool signals from 0 to 1: %s. "
            "Does the evidence support the suggestion? Reply with exactly one word: "
            "SUPPORTED or UNSUPPORTED."
            % (state.get("recommendation"), state.get("p_fraud", 0.0), facts))
        stats["calls"] += 1
        try:
            answer = _UNSAFE.sub("", str(chat(prompt))).lower()
            if "unsupported" in answer:
                stats["disagreements"] += 1
                flags.append("second opinion: evidence does not support the suggestion")
        except Exception:
            stats["failures"] += 1
        return flags

    llm_verifier.stats = stats
    return llm_verifier
