"""Local LLM helpers (Ollama): a tool planner and a case explainer.

Safety rules enforced in code:
  * Prompts carry only tool names, numbers and short notes written by our own tools.
    Raw merchant text, names and card numbers are never sent.
  * The planner can only name a tool or say done. The graph validates every answer.
  * The explainer writes text for a reviewer. It never sets or changes a verdict.
  * If the model is down or answers badly, the system falls back to the rule planner
    or a fixed template. The workflow never depends on the model being up.
ASCII only.
"""
import json
import re
import urllib.request

from .planner import rule_planner
from .tools import TOOL_NAMES

_CHOICE = re.compile(r"\b(" + "|".join(TOOL_NAMES + ["done"]) + r")\b")
_UNSAFE = re.compile(r"[^A-Za-z0-9 .,%:;()_/-]")


def ollama_chat(prompt, model, host="http://localhost:11434", timeout=60):
    """One non-streaming chat call to a local Ollama server. Temperature 0."""
    body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["message"]["content"])


def _first_choice(text):
    """The first tool name or the word done in the reply. Anything else is returned as is."""
    low = str(text).lower()
    hit = _CHOICE.search(low)
    return hit.group(1) if hit else low.strip()[:40]


def make_ollama_planner(chat, fallback=rule_planner):
    """Return a planner that asks a model for the next tool. chat(prompt) returns text."""
    stats = {"calls": 0, "fallbacks": 0}

    def ollama_planner(state):
        evidence = state.get("evidence", [])
        used = [e["tool"] for e in evidence]
        left = [t for t in TOOL_NAMES if t not in used]
        prompt = (
            "You investigate a card transaction that a fraud model could not settle. "
            "Model fraud probability: %.2f. Tools already used with their signal from 0 to 1: %s. "
            "Tools still available: %s. Pick the single most useful next tool, or reply done "
            "if the evidence is already conclusive. Reply with ONLY one tool name or the word done."
            % (state.get("p_fraud", 0.0),
               ", ".join("%s=%.2f" % (e["tool"], e["signal"]) for e in evidence) or "none",
               ", ".join(left) or "none"))
        stats["calls"] += 1
        try:
            return _first_choice(chat(prompt))
        except Exception:
            stats["fallbacks"] += 1
            return fallback(state)

    ollama_planner.stats = stats
    return ollama_planner


def _clean(text, limit):
    return _UNSAFE.sub("", str(text).replace("\n", " ")).strip()[:limit]


def template_explanation(payload):
    """Fixed-format summary built from numbers and our own tool notes. No model involved."""
    ev = sorted(payload.get("evidence", []), key=lambda e: -e["signal"])
    parts = ["Model fraud probability %.2f and the gate escalated the case."
             % payload["p_fraud"]]
    if ev:
        top = ev[0]
        parts.append("Strongest evidence: %s at %.2f (%s)."
                     % (top["tool"], top["signal"], _clean(top["note"], 90)))
    parts.append("The agent suggests: %s." % payload["recommendation"])
    return " ".join(parts)


def make_explainer(chat=None, max_chars=400):
    """Return explain(payload) -> {"text", "source"}. With chat=None it only uses the template."""

    def explain(payload):
        base = template_explanation(payload)
        if chat is None:
            return {"text": base, "source": "template"}
        facts = "; ".join("%s %.2f: %s" % (e["tool"], e["signal"], _clean(e["note"], 90))
                          for e in payload.get("evidence", []))
        prompt = (
            "Write two short sentences for a fraud analyst. Use only these facts and do not "
            "give a verdict. Model fraud probability %.2f. Evidence: %s. Agent suggestion: %s."
            % (payload["p_fraud"], facts or "none", payload["recommendation"]))
        try:
            text = _clean(chat(prompt), max_chars)
            if len(text) < 20:
                raise ValueError("answer too short")
            return {"text": text, "source": "llm"}
        except Exception:
            return {"text": base, "source": "template"}

    return explain
