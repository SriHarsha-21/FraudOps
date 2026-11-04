"""Drift monitor tests."""
import numpy as np
import pytest

from fraudops.monitor import ScoreMonitor, TrustGuard


class Ctx:
    trust_level = "normal"


def _cal(n=500, seed=0):
    return np.random.default_rng(seed).beta(2, 8, n)


def test_pvalues_are_uniform_for_exchangeable_scores():
    cal = _cal(2000)
    mon = ScoreMonitor(cal, seed=1)
    p = mon.pvalues(np.random.default_rng(2).beta(2, 8, 20000))
    assert abs(p.mean() - 0.5) < 0.02
    assert abs((p < 0.1).mean() - 0.1) < 0.01


def test_false_alarm_rate_stays_under_the_bound():
    alarms = 0
    trials = 200
    for t in range(trials):
        rng = np.random.default_rng(t)
        mon = ScoreMonitor(rng.beta(2, 8, 300), threshold=200, seed=t)
        mon.run(rng.beta(2, 8, 1500))
        alarms += mon.alarm_at is not None
    assert alarms / trials <= 0.03  # bound is 1 percent, allow sampling noise


def test_a_shift_to_higher_scores_raises_an_up_alarm():
    rng = np.random.default_rng(0)
    mon = ScoreMonitor(rng.beta(2, 8, 500), threshold=200, seed=0)
    mon.run(rng.beta(2, 8, 300))
    assert mon.alarm_at is None
    mon.run(rng.beta(4, 5, 400))
    assert mon.alarm_at is not None and mon.alarm_at >= 300 and mon.side == "up"


def test_a_shift_to_lower_scores_raises_a_down_alarm():
    rng = np.random.default_rng(1)
    mon = ScoreMonitor(rng.beta(4, 5, 500), threshold=200, seed=0)
    mon.run(rng.beta(1, 12, 800))
    assert mon.side == "down"


def test_batch_and_one_at_a_time_agree():
    rng = np.random.default_rng(3)
    cal, stream = rng.beta(2, 8, 400), rng.beta(3, 6, 300)
    a = ScoreMonitor(cal, seed=5)
    a.run(stream)
    b = ScoreMonitor(cal, seed=5)
    for s in stream:
        b.update(s)
    assert a.alarm_at == b.alarm_at
    assert a.report()["peak_log10_e"] == pytest.approx(b.report()["peak_log10_e"], rel=1e-6)


def test_guard_tightens_on_alarm_and_only_a_named_human_relaxes():
    rng = np.random.default_rng(0)
    ctx = Ctx()
    guard = TrustGuard(ctx, ScoreMonitor(rng.beta(2, 8, 500), threshold=50, seed=0))
    assert guard.observe(rng.beta(2, 8, 100)) == "normal"
    assert guard.observe(rng.beta(5, 4, 300)) == "cautious"
    with pytest.raises(ValueError):
        guard.relax("  ")
    assert ctx.trust_level == "cautious"
    guard.relax("alice")
    assert ctx.trust_level == "normal" and guard.monitor.alarm_at is None
    assert [e["event"] for e in guard.log] == ["tightened", "relaxed"]


def test_too_few_calibration_scores_are_refused():
    with pytest.raises(ValueError):
        ScoreMonitor(np.arange(10.0))
