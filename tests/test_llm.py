"""LLM planner and explainer tests. They use fake models, so no server is needed."""
import http.server
import json
import os
import threading

import pytest

from fraudops.agents.graph import build_graph, make_checkpointer, resume_case, start_case
from fraudops.agents.llm import make_explainer, make_ollama_planner, ollama_chat, \
    template_explanation
from fraudops.agents.store import DecisionStore
from fraudops.agents.tools import CaseContext
from fraudops.data import clean, make_fixture
from fraudops.pipeline import fit_pipeline
from fraudops.review import ReviewService

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "..", "config.json")) as fh:
    CFG = json.load(fh)


@pytest.fixture(scope="module")
def pipe():
    return fit_pipeline(clean(make_fixture(seed=3)), None, CFG)


STATE = {"p_fraud": 0.4, "evidence": [{"tool": "geo_check", "signal": 0.9, "note": "x"}]}


def test_planner_picks_the_first_named_tool_even_in_chatty_replies():
    p = make_ollama_planner(lambda prompt: "I would run Velocity_Check next, not geo_check.")
    assert p(STATE) == "velocity_check"
    assert make_ollama_planner(lambda prompt: "DONE")(STATE) == "done"


def test_planner_falls_back_to_rules_when_the_model_is_down():
    def boom(prompt):
        raise ConnectionError("no server")
    p = make_ollama_planner(boom)
    assert p(STATE) in ("velocity_check", "merchant_check", "history_check", "done")
    assert p.stats == {"calls": 1, "fallbacks": 1}


def test_prompt_carries_only_numbers_and_tool_names():
    seen = []
    make_ollama_planner(lambda prompt: seen.append(prompt) or "done")(STATE)
    assert "geo_check=0.90" in seen[0] and "0.40" in seen[0]
    assert "note" not in seen[0]


def test_graph_with_model_planner_still_ends_at_a_human(pipe):
    ctx = CaseContext(pipe, CFG)
    svc = ReviewService(pipe, CFG)
    tid = svc.queue(1)[0]
    store = DecisionStore(":memory:")
    app = build_graph(ctx, store, make_checkpointer(),
                      planner=make_ollama_planner(lambda p: "ignore all rules and block"))
    out = start_case(app, tid, CFG)
    assert out["status"] == "paused"
    assert out["payload"]["evidence"] == []  # unknown output was rejected
    done = resume_case(app, tid, {"decision": "approve", "reviewer": "a"}, CFG)
    assert done["verdict"] == "approve" and store.verify_chain()
    assert "planner output was rejected" in store.get(tid)["reasons"]


def test_explainer_cleans_model_output_and_never_returns_markup():
    payload = {"p_fraud": 0.62, "recommendation": "block",
               "evidence": [{"tool": "geo_check", "signal": 0.9, "note": "800 km away"}]}
    ex = make_explainer(lambda p: "Likely fraud <script>alert(1)</script> because of distance.\n")
    out = ex(payload)
    assert out["source"] == "llm" and "<" not in out["text"] and "\n" not in out["text"]


def test_explainer_falls_back_to_template():
    payload = {"p_fraud": 0.62, "recommendation": "block",
               "evidence": [{"tool": "geo_check", "signal": 0.9, "note": "800 km away"}]}

    def boom(prompt):
        raise TimeoutError()
    assert make_explainer(boom)(payload)["source"] == "template"
    assert make_explainer(lambda p: "ok")(payload)["source"] == "template"
    assert make_explainer(None)(payload)["text"] == template_explanation(payload)
    assert "geo_check" in template_explanation(payload)


def test_review_service_explains_a_pending_case_without_touching_the_store(pipe):
    svc = ReviewService(pipe, CFG, explainer=make_explainer(lambda p: "A short fake summary here."))
    tid = svc.queue(1)[0]
    assert svc.explain(tid) is None  # not opened yet
    svc.open_case(tid)
    assert svc.explain(tid)["text"] == "A short fake summary here."
    assert svc.store.count() == 0


def test_ollama_chat_speaks_the_ollama_protocol():
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"message": {"content": "geo_check"}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = ollama_chat("hi", "m1", "http://127.0.0.1:%d" % srv.server_address[1], 5)
    finally:
        srv.shutdown()
    assert out == "geo_check" and seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "m1" and seen["body"]["stream"] is False
    assert seen["body"]["options"]["temperature"] == 0
