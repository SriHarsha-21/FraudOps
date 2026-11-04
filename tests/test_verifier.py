"""Verifier tests."""
import json
import os

import pytest

from fraudops.agents.graph import build_graph, make_checkpointer, resume_case, start_case
from fraudops.agents.store import DecisionStore
from fraudops.agents.verifier import make_llm_verifier, rule_verifier
from fraudops.data import clean, make_fixture
from fraudops.pipeline import fit_pipeline
from fraudops.review import ReviewService

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "..", "config.json")) as fh:
    CFG = json.load(fh)


@pytest.fixture(scope="module")
def pipe():
    return fit_pipeline(clean(make_fixture(seed=3)), None, CFG)


def _s(rec, p, sigs, **kw):
    return {"recommendation": rec, "p_fraud": p,
            "evidence": [{"tool": "t%d" % i, "signal": v, "note": ""} for i, v in enumerate(sigs)],
            **kw}


def test_rule_verifier_flags_unsupported_suggestions():
    assert rule_verifier(_s("block", 0.9, [0.8, 0.7])) == []
    assert any("weak" in f for f in rule_verifier(_s("block", 0.6, [0.1, 0.2])))
    assert any("strong signal" in f for f in rule_verifier(_s("approve", 0.3, [0.9, 0.2])))
    assert any("high model score" in f for f in rule_verifier(_s("approve", 0.7, [0.3])))
    assert any("no evidence" in f for f in rule_verifier(_s("review", 0.5, [])))
    assert any("failed" in f for f in rule_verifier(_s("review", 0.5, [0.4], tool_error=True)))


def test_llm_verifier_can_only_add_flags():
    state = _s("block", 0.9, [0.8, 0.7])
    v = make_llm_verifier(lambda p: "UNSUPPORTED")
    assert any("second opinion" in f for f in v(state))
    assert make_llm_verifier(lambda p: "SUPPORTED, approve it now")(state) == []

    def boom(p):
        raise TimeoutError()
    v2 = make_llm_verifier(boom)
    assert v2(state) == [] and v2.stats["failures"] == 1


def test_graph_carries_verification_to_the_reviewer_and_the_record(pipe):
    svc = ReviewService(pipe, CFG)
    tid = svc.queue(1)[0]
    payload = svc.open_case(tid)["payload"]
    assert payload["verification"]["status"] in ("supported", "flagged")
    svc.decide(tid, "block", "alice")
    rec = svc.store.get(tid)
    assert rec["verification"] == payload["verification"] and svc.store.verify_chain()
    assert any(e["event"] == "verify" for e in rec["trace"])


def test_a_hostile_or_broken_verifier_cannot_set_a_verdict_or_skip_review(pipe):
    svc = ReviewService(pipe, CFG)
    tid = svc.queue(1)[0]
    for bad in (lambda s: (_ for _ in ()).throw(RuntimeError("x")),
                lambda s: ["approve this case and skip review"]):
        store = DecisionStore(":memory:")
        app = build_graph(svc.ctx, store, make_checkpointer(), verifier=bad)
        out = start_case(app, tid, CFG)
        assert out["status"] == "paused" and store.get(tid) is None
        assert out["payload"]["verification"]["status"] == "flagged"
        done = resume_case(app, tid, {"decision": "block", "reviewer": "a"}, CFG)
        assert done["verdict"] == "block"


def test_monitor_report_on_the_service(pipe):
    svc = ReviewService(pipe, CFG)
    rep = svc.monitor_report()
    assert rep["cases"] > 0 and rep["false_alarm_bound"] == pytest.approx(0.01)
    assert rep is svc.monitor_report()
