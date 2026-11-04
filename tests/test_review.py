"""Review service and app tests."""
import json
import os

import pytest

from fraudops.data import clean, make_fixture
from fraudops.pipeline import fit_pipeline
from fraudops.review import ReviewService, check_pin

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "..", "config.json")) as fh:
    CFG = json.load(fh)


@pytest.fixture(scope="module")
def pipe():
    return fit_pipeline(clean(make_fixture(seed=3)), None, CFG)


def test_queue_holds_only_escalated_test_cases_by_priority(pipe):
    svc = ReviewService(pipe, CFG)
    q = svc.queue(limit=1000)
    assert q and len(q) == svc.total_escalated()
    pos = {t: i for i, t in enumerate(svc.ctx.frame["txn_id"])}
    assert all(svc.ctx.gate_actions[pos[t]] == "escalate" for t in q)
    assert all(pipe["masks"]["test"][pos[t]] for t in q)
    score = [svc.ctx.frame["amt"].iloc[pos[t]] * svc.ctx.p[pos[t]] for t in q]
    assert score == sorted(score, reverse=True)


def test_details_never_show_the_card_number(pipe):
    svc = ReviewService(pipe, CFG)
    d = svc.details(svc.queue(1)[0])
    assert "cc_num" not in d and len(d["card_ref"]) == 6
    assert str(pipe["frame"]["cc_num"].iloc[0]) not in json.dumps(d)


def test_decision_enters_the_chained_store_once(pipe):
    svc = ReviewService(pipe, CFG)
    tid = svc.queue(1)[0]
    assert svc.open_case(tid)["status"] == "paused"
    first = svc.decide(tid, "block", "alice")
    assert first["status"] == "done" and first["verdict"] == "block"
    again = svc.decide(tid, "approve", "bob")  # a second click must not change anything
    assert again["verdict"] == "block"
    assert svc.store.count() == 1 and svc.store.verify_chain()
    assert svc.store.get(tid)["reviewer"] == "alice"
    assert tid not in svc.queue(1000)


def test_bad_decisions_are_refused(pipe):
    svc = ReviewService(pipe, CFG)
    tid = svc.queue(1)[0]
    with pytest.raises(ValueError):
        svc.decide(tid, "maybe", "alice")
    with pytest.raises(ValueError):
        svc.decide(tid, "block", "   ")
    assert svc.store.count() == 0


def test_pending_case_is_not_rerun_and_survives_restart(pipe, tmp_path):
    wd = str(tmp_path / "state")
    svc1 = ReviewService(pipe, CFG, workdir=wd)
    tid = svc1.queue(1)[0]
    p1 = svc1.open_case(tid)["payload"]
    del svc1
    svc2 = ReviewService(pipe, CFG, workdir=wd)
    assert svc2.pending_payload(tid) == p1
    assert svc2.open_case(tid)["payload"] == p1
    assert svc2.decide(tid, "approve", "carol")["status"] == "done"


def test_true_label_is_hidden_until_decided(pipe):
    svc = ReviewService(pipe, CFG)
    tid = svc.queue(1)[0]
    assert svc.true_label(tid) is None
    svc.open_case(tid)
    svc.decide(tid, "block", "alice")
    assert svc.true_label(tid) in (0, 1)


def test_simulated_reviewer_and_summary(pipe):
    svc = ReviewService(pipe, CFG)
    assert svc.simulate_reviewer(8, seed=1) == 8
    s = svc.summary()
    assert s["decided"] == 8 and s["by_human"] == 8 and s["chain_ok"]
    assert s["agent_comparable"] <= 8
    assert s["remaining_in_queue"] == svc.total_escalated() - 8


def test_pin_check(monkeypatch):
    monkeypatch.delenv("FRAUDOPS_REVIEW_PIN", raising=False)
    assert check_pin("anything")
    monkeypatch.setenv("FRAUDOPS_REVIEW_PIN", "4821")
    assert check_pin("4821") and not check_pin("0000") and not check_pin("")


def test_streamlit_app_loads_and_records_a_decision(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("FRAUDOPS_REVIEW_PIN", raising=False)
    path = os.path.join(HERE, "..", "app", "streamlit_app.py")
    at = AppTest.from_file(path, default_timeout=180)
    at.run()
    assert not at.exception
    at.sidebar.text_input(key="workdir").set_value(str(tmp_path / "ui_state"))
    at.run()
    assert not at.exception
    at.sidebar.text_input(key="reviewer").set_value("alice")
    at.run()
    at.button(key="block").click()
    at.run()
    assert not at.exception
    assert any("blocked" in s.value for s in at.success)
