"""Evidence tools with an explicit contract.

Each tool takes a transaction id and returns
    {"tool": str, "signal": float in [0, 1], "note": str}
Tools read precomputed features only. They have no network, shell or write access.
ASCII only.
"""
import math

import numpy as np

TOOL_NAMES = ["velocity_check", "geo_check", "merchant_check", "history_check"]


class CaseContext:
    """Read-only view of the scored data used by tools and the graph."""

    def __init__(self, pipe, cfg):
        self.frame = pipe["frame"]
        self.X = pipe["X"]
        self.p = pipe["p"]
        self.y = pipe["y"]
        self.conformal = pipe["conformal"]
        self.rates = pipe["category_rates"]
        self.cfg = cfg
        self.pos = {t: i for i, t in enumerate(self.frame["txn_id"])}
        self.max_rate = max(self.rates.values()) if self.rates else 1.0
        self.trust_level = "normal"
        self.gate_actions = self.conformal.actions(self.p)

    def row(self, txn_id):
        return self.pos[txn_id]


def _clip01(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def make_tools(ctx):
    scale = ctx.cfg["tools"]

    def velocity_check(txn_id):
        i = ctx.row(txn_id)
        n = float(ctx.X["count_1h"].iloc[i])
        return {"tool": "velocity_check",
                "signal": _clip01((n - 1) / scale["velocity_full_at"]),
                "note": "%d earlier transactions on this card in the last hour" % n}

    def geo_check(txn_id):
        i = ctx.row(txn_id)
        d = float(ctx.X["dist_km"].iloc[i])
        return {"tool": "geo_check", "signal": _clip01(d / scale["geo_full_at_km"]),
                "note": "merchant is %.0f km from the customer" % d}

    def merchant_check(txn_id):
        i = ctx.row(txn_id)
        cat = ctx.frame["category"].iloc[i]
        rate = ctx.rates.get(cat, 0.0)
        return {"tool": "merchant_check", "signal": _clip01(rate / ctx.max_rate),
                "note": "category %s has a training fraud rate of %.2f%%" % (cat, 100 * rate)}

    def history_check(txn_id):
        i = ctx.row(txn_id)
        r = float(ctx.X["amt_over_median"].iloc[i])
        if np.isnan(r) or r <= 0:
            return {"tool": "history_check", "signal": 0.3,
                    "note": "no earlier history for this card"}
        sig = _clip01(math.log(max(r, 1e-9)) / math.log(scale["history_full_at_ratio"]))
        return {"tool": "history_check", "signal": sig,
                "note": "amount is %.1f times the card's usual median" % r}

    return {"velocity_check": velocity_check, "geo_check": geo_check,
            "merchant_check": merchant_check, "history_check": history_check}
