"""Planners choose the next tool. Their output is validated by code.

A planner can only name a tool or say done. It can never set a verdict.
ASCII only.
"""
import os

from .tools import TOOL_NAMES


def rule_planner(state):
    """Default planner. Needs no API key. Stops early when evidence is conclusive."""
    used = {e["tool"] for e in state.get("evidence", [])}
    sigs = [e["signal"] for e in state.get("evidence", [])]
    if sigs and max(sigs) >= 0.9:
        return "done"
    for name in TOOL_NAMES:
        if name not in used:
            return name
    return "done"


def llm_planner(state):
    """Optional planner that asks Claude for the next tool. Needs ANTHROPIC_API_KEY.

    Only tool names and numbers are sent. Raw text fields are never included.
    Not exercised by the tests because it needs network access and a key.
    """
    import anthropic

    client = anthropic.Anthropic()
    prompt = (
        "You help investigate a card transaction. Tools: " + ", ".join(TOOL_NAMES)
        + ". Evidence so far: "
        + str([(e["tool"], round(e["signal"], 2)) for e in state.get("evidence", [])])
        + ". Reply with ONLY the next tool name, or the word done."
    )
    msg = client.messages.create(
        model=os.getenv("AGENT_MODEL", "claude-sonnet-5-5"), max_tokens=20,
        temperature=0, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text.strip().lower()


def validate_choice(choice, state):
    """Return (tool_or_done, was_invalid). Unknown or repeated tools become done."""
    used = {e["tool"] for e in state.get("evidence", [])}
    if choice == "done":
        return "done", False
    if choice in TOOL_NAMES and choice not in used:
        return choice, False
    return "done", True
