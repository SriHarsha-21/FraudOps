"""Class-conditional (Mondrian) split conformal prediction for binary labels.

Each class gets its own threshold, so coverage holds for the rare fraud class too.
Guarantee: for each class the true label is in the set with probability at least
1 - alpha, if calibration and test data are exchangeable. ASCII only.
"""
import math

import numpy as np


class ClassConditionalConformal:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = {0: math.inf, 1: math.inf}

    def fit(self, p_fraud, y):
        p_fraud, y = np.asarray(p_fraud, float), np.asarray(y)
        prob = {0: 1.0 - p_fraud, 1: p_fraud}
        for k in (0, 1):
            scores = 1.0 - prob[k][y == k]
            n = len(scores)
            level = math.ceil((n + 1) * (1 - self.alpha)) / n if n else 2.0
            self.q[k] = math.inf if level > 1 else float(
                np.quantile(scores, level, method="higher"))
        return self

    def sets(self, p_fraud):
        """Boolean array of shape (n, 2): column 0 = legitimate, column 1 = fraud."""
        p_fraud = np.asarray(p_fraud, float)
        legit = (1.0 - (1.0 - p_fraud)) <= self.q[0]
        fraud = (1.0 - p_fraud) <= self.q[1]
        return np.column_stack([legit, fraud])

    def actions(self, p_fraud):
        """approve if only legit, block if only fraud, escalate otherwise."""
        s = self.sets(p_fraud)
        out = np.full(len(s), "escalate", dtype=object)
        out[s[:, 0] & ~s[:, 1]] = "approve"
        out[~s[:, 0] & s[:, 1]] = "block"
        return out

    def report(self, p_fraud, y):
        y = np.asarray(y)
        s = self.sets(p_fraud)
        act = self.actions(p_fraud)
        covered = s[np.arange(len(y)), y]
        auto = act != "escalate"
        correct = ((act == "approve") & (y == 0)) | ((act == "block") & (y == 1))
        return {
            "alpha": self.alpha,
            "coverage_legit": float(covered[y == 0].mean()) if (y == 0).any() else None,
            "coverage_fraud": float(covered[y == 1].mean()) if (y == 1).any() else None,
            "escalation_rate": float((act == "escalate").mean()),
            "auto_accuracy": float(correct[auto].mean()) if auto.any() else None,
            "thresholds": {"legit": self.q[0], "fraud": self.q[1]},
        }
