"""Sweep tests: cost arithmetic, alpha choice uses only calibration data, determinism."""
import json
import os

import numpy as np
import pytest

from fraudops.data import clean, make_fixture
from fraudops.pipeline import fit_pipeline
from fraudops.sweep import choose_alpha, expected_system_cost, run_sweep

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "..", "config.json")) as fh:
    CFG = json.load(fh)


def test_expected_cost_arithmetic():
    actions = np.array(["approve", "block", "escalate", "escalate", "approve"])
    y = np.array([1, 0, 1, 0, 0])
    amt = np.array([100.0, 40.0, 60.0, 30.0, 10.0])
    # approve a fraud: 100. block a legit: 25. two reviews at 2 each: 4.
    # reviewer errs 10 percent: wrong on the fraud costs 60, on the legit costs 25.
    got = expected_system_cost(actions, y, amt, 2.0, 25.0, 0.1)
    assert got == pytest.approx(100.0 + 25.0 + 4.0 + 0.1 * (60.0 + 25.0))


def test_no_escalation_means_reviewer_error_is_irrelevant():
    actions = np.array(["approve", "block"])
    y = np.array([1, 0])
    amt = np.array([50.0, 10.0])
    a = expected_system_cost(actions, y, amt, 5.0, 25.0, 0.0)
    b = expected_system_cost(actions, y, amt, 5.0, 25.0, 0.5)
    assert a == b == 75.0


def _calibration_arrays(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(600) < 0.08).astype(int)
    p = np.clip(np.where(y == 1, rng.beta(5, 2, 600), rng.beta(1, 12, 600)), 0, 1)
    amt = rng.uniform(5, 300, 600)
    return p[:300], y[:300], p[300:], y[300:], amt[300:]


def test_choose_alpha_is_deterministic_and_in_the_grid():
    pf, yf, pe, ye, ae = _calibration_arrays(0)
    grid = [0.02, 0.05, 0.1, 0.2]
    a1, c1 = choose_alpha(pf, yf, pe, ye, ae, 1.0, 25.0, 0.05, grid)
    a2, c2 = choose_alpha(pf, yf, pe, ye, ae, 1.0, 25.0, 0.05, grid)
    assert a1 == a2 and a1 in grid and c1 == c2


def test_free_reviews_prefer_a_stricter_alpha_than_costly_reviews():
    pf, yf, pe, ye, ae = _calibration_arrays(1)
    grid = [0.02, 0.05, 0.1, 0.2]
    cheap, _ = choose_alpha(pf, yf, pe, ye, ae, 0.0, 25.0, 0.0, grid)
    dear, _ = choose_alpha(pf, yf, pe, ye, ae, 500.0, 25.0, 0.0, grid)
    assert cheap <= dear


@pytest.fixture(scope="module")
def sweep():
    pipe = fit_pipeline(clean(make_fixture(seed=5)), None, CFG)
    return pipe, run_sweep(pipe, CFG)


def test_sweep_output_shape(sweep):
    _, res = sweep
    assert len(res["rows"]) == len(CFG["sweep"]["review_prices"])
    for r in res["rows"]:
        assert r["chosen_alpha"] in CFG["sweep"]["alphas"]
        assert r["test_cost_chosen"] >= 0.0
        assert 0.0 <= r["escalation_rate"] <= 1.0


def test_test_labels_do_not_change_the_chosen_alpha(sweep):
    pipe, base = sweep
    flipped = dict(pipe)
    y2 = pipe["y"].copy()
    te = pipe["masks"]["test"]
    y2[te] = 1 - y2[te]
    flipped["y"] = y2
    again = run_sweep(flipped, CFG)
    assert [r["chosen_alpha"] for r in base["rows"]] == \
        [r["chosen_alpha"] for r in again["rows"]]
