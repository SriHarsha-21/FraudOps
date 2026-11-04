"""Backtest tests: metric arithmetic, McNemar, and an end-to-end dry run."""
import json
import os

import numpy as np
import pytest

from fraudops.backtest import decision_cost, mcnemar, run_backtest, summarize
from fraudops.data import clean, make_fixture
from fraudops.pipeline import fit_pipeline

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "..", "config.json")) as fh:
    CFG = json.load(fh)


def test_summarize_arithmetic():
    actions = np.array(["block", "block", "approve", "approve"])
    y = np.array([1, 0, 1, 0])
    amt = np.array([100.0, 50.0, 300.0, 20.0])
    s = summarize(actions, y, amt, reviews=2, review_cost=5.0, false_block_cost=25.0)
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert s["missed_fraud_amount"] == 300.0 and s["false_blocks"] == 1
    assert s["cost"] == 300.0 + 25.0 + 10.0


def test_decision_cost_cases():
    assert decision_cost("approve", 1, 80.0, 25.0) == 80.0
    assert decision_cost("block", 0, 80.0, 25.0) == 25.0
    assert decision_cost("block", 1, 80.0, 25.0) == 0.0
    assert decision_cost("approve", 0, 80.0, 25.0) == 0.0


def test_mcnemar_detects_a_clear_difference_and_ties():
    a = np.array([True] * 50 + [False] * 5 + [True] * 45)
    b = np.array([True] * 50 + [True] * 5 + [True] * 45)
    r = mcnemar(a, b)
    assert r["b_only"] == 5 and r["a_only"] == 0
    assert mcnemar(a, a)["p_value"] == 1.0
    wide = mcnemar(np.array([True] * 60 + [False] * 40), np.array([True] * 100))
    assert wide["p_value"] < 0.001


@pytest.fixture(scope="module")
def result():
    pipe = fit_pipeline(clean(make_fixture(seed=5)), None, CFG)
    return run_backtest(pipe, CFG, seed=1)


def test_backtest_runs_and_is_reproducible(result):
    pipe = fit_pipeline(clean(make_fixture(seed=5)), None, CFG)
    again = run_backtest(pipe, CFG, seed=1)
    assert json.dumps(result, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_backtest_has_all_systems_and_a_valid_chain(result):
    assert set(result["systems"]) == {
        "1_rules_only", "2_model_only", "3_model_plus_conformal_human"}
    agents = result["agents"]
    if agents.get("escalated_in_test"):
        assert agents["chain_ok"] is True
        assert agents["records_stored"] == agents["escalated_evaluated"]
        assert 0.0 <= agents["agent_resolved_share"] <= 1.0
