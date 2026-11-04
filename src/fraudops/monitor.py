"""Drift and surge monitor: a conformal test martingale on the stream of model scores.

Idea. Each incoming score is turned into a conformal p-value against the calibration
scores plus every earlier stream score. If the stream still looks like the calibration data, those p-values are uniform and
the martingale stays near 1. A fraud wave or a shift in attacker behaviour makes small (or
large) p-values too common and the martingale grows. By Ville's inequality the chance that
either martingale ever reaches the threshold T under no shift is at most 1/T, so the two
tails together give a false alarm bound of 2/T at any stopping time. Real streams drift in
time, so the guarantee is for exchangeable data only. It is an alarm, not a proof of attack.

TrustGuard acts on an alarm. It can only tighten the gate (normal to cautious). Relaxing it
again needs a named human. ASCII only.
"""
import bisect

import numpy as np

EPS = np.linspace(0.05, 0.95, 19)
_LOG_EPS = np.log(EPS)
_EPS_M1 = EPS - 1.0


class ScoreMonitor:
    """The reference set starts as the calibration scores and grows with every score seen.

    Growing the reference is what makes the p-values exactly uniform under exchangeability.
    A fixed calibration set is not: its small sampling error repeats on every case and a long
    stream would eventually reveal it as a false alarm.
    """

    def __init__(self, calib_scores, threshold=200.0, seed=0):
        cal = np.sort(np.asarray(calib_scores, float))
        if len(cal) < 50:
            raise ValueError("need at least 50 calibration scores")
        if threshold <= 1:
            raise ValueError("threshold must exceed 1")
        self._cal0 = cal
        self.threshold = float(threshold)
        self.log_t = float(np.log(threshold))
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def false_alarm_bound(self):
        return min(1.0, 2.0 / self.threshold)

    def reset(self):
        self.ref = self._cal0.tolist()
        self.n_seen = 0
        self.up = np.zeros(len(EPS))
        self.down = np.zeros(len(EPS))
        self.alarm_at = None
        self.side = None
        self.peak_log = 0.0

    def pvalues(self, scores):
        """Smoothed p-values against the current reference. Small means a high score."""
        s = np.asarray(scores, float)
        ref = np.asarray(self.ref)
        n = len(ref)
        right = np.searchsorted(ref, s, side="right")
        left = np.searchsorted(ref, s, side="left")
        u = self.rng.random(len(s))
        p = ((n - right) + u * ((right - left) + 1)) / (n + 1)
        return np.clip(p, 1e-12, 1 - 1e-12)

    @staticmethod
    def _lse(v):
        m = v.max()
        return m + np.log(np.exp(v - m).sum())

    def run(self, scores):
        """Feed scores in time order, one at a time. Returns the log martingales per case."""
        s = np.asarray(scores, float)
        m = len(s)
        mu, md = np.empty(m), np.empty(m)
        u = self.rng.random(m)
        k = np.log(len(EPS))
        ref = self.ref
        up, down = self.up, self.down
        for j in range(m):
            x = float(s[j])
            n = len(ref)
            right = bisect.bisect_right(ref, x)
            left = bisect.bisect_left(ref, x)
            p = ((n - right) + u[j] * ((right - left) + 1)) / (n + 1)
            p = min(max(p, 1e-12), 1 - 1e-12)
            up = up + _LOG_EPS + _EPS_M1 * np.log(p)
            down = down + _LOG_EPS + _EPS_M1 * np.log(1.0 - p)
            bisect.insort(ref, x)
            mu[j] = self._lse(up) - k
            md[j] = self._lse(down) - k
        self.up, self.down = up, down
        if m:
            worst = np.maximum(mu, md)
            hit = np.flatnonzero(worst >= self.log_t)
            if hit.size and self.alarm_at is None:
                j = int(hit[0])
                self.alarm_at = self.n_seen + j
                self.side = "up" if mu[j] >= md[j] else "down"
            self.peak_log = max(self.peak_log, float(worst.max()))
        self.n_seen += m
        return {"log_up": mu, "log_down": md}

    def update(self, score):
        self.run([score])
        return self.alarm_at is not None

    def report(self):
        return {"cases": self.n_seen, "threshold": self.threshold,
                "false_alarm_bound": self.false_alarm_bound,
                "alarm_at": self.alarm_at, "side": self.side,
                "peak_log10_e": self.peak_log / np.log(10.0)}


class TrustGuard:
    """Acts on the monitor. It can tighten the gate. Only a named human can relax it."""

    def __init__(self, ctx, monitor):
        self.ctx, self.monitor = ctx, monitor
        self.log = []

    def observe(self, scores):
        self.monitor.run(scores)
        if self.monitor.alarm_at is not None and self.ctx.trust_level == "normal":
            self.ctx.trust_level = "cautious"
            self.log.append({"event": "tightened", "at": self.monitor.alarm_at,
                             "side": self.monitor.side})
        return self.ctx.trust_level

    def relax(self, reviewer):
        if not str(reviewer).strip():
            raise ValueError("a reviewer name is required to relax the gate")
        self.ctx.trust_level = "normal"
        self.monitor.reset()
        self.log.append({"event": "relaxed", "by": str(reviewer).strip()[:40]})
