"""Gradient-boosted detector, temperature scaling and basic metrics.

ASCII only.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import CATEGORICAL, FEATURES

_EPS = 1e-6


def train_detector(X, y, cfg, seed):
    """Fit the detector. Returns a dict holding the model and the prior correction."""
    weight = float(cfg["fraud_weight"])
    sample_weight = np.where(np.asarray(y) == 1, weight, 1.0)
    mask = [c in CATEGORICAL for c in FEATURES]
    clf = HistGradientBoostingClassifier(
        learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"],
        max_leaf_nodes=cfg["max_leaf_nodes"], min_samples_leaf=cfg["min_samples_leaf"],
        l2_regularization=cfg["l2"], categorical_features=mask,
        early_stopping=False, random_state=seed)
    clf.fit(X[FEATURES], y, sample_weight=sample_weight)
    return {"model": clf, "log_weight": float(np.log(weight))}


def predict_logit(detector, X):
    """Log-odds of fraud, corrected for any fraud class weight used in training."""
    p = detector["model"].predict_proba(X[FEATURES])[:, 1]
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p)) - detector["log_weight"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits, y):
    """Find T that minimizes log loss of sigmoid(logit / T). Ranking is unchanged."""
    y = np.asarray(y, float)

    def nll(t):
        p = np.clip(sigmoid(logits / t), _EPS, 1 - _EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def ece(p, y, bins=10):
    """Expected calibration error for the fraud probability."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def rank_metrics(p, y):
    y = np.asarray(y)
    if y.min() == y.max():
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p))}


def expected_cost(actions, y, amt, review_cost, false_block_cost):
    """Missed fraud costs its amount. Reviews and false blocks cost fixed amounts."""
    actions, y, amt = np.asarray(actions), np.asarray(y), np.asarray(amt, float)
    missed = amt[(actions == "approve") & (y == 1)].sum()
    reviews = review_cost * (actions == "escalate").sum()
    false_blocks = false_block_cost * ((actions == "block") & (y == 0)).sum()
    return float(missed + reviews + false_blocks)
