"""Feature builders. Every feature for a row uses only data strictly before it.

ASCII only.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "amt", "log_amt", "age", "hour", "dow", "dist_km", "city_pop", "cat_code",
    "count_1h", "count_24h", "sum_24h", "sum_7d", "secs_since_last",
    "amt_over_median",
]
CATEGORICAL = ["cat_code"]


def fit_categories(df):
    """Category list from the training frame. Unknown categories map past the end."""
    return sorted(df["category"].unique().tolist())


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _card_history(df):
    """Per-card rolling counts, sums, gap and median of strictly earlier rows."""
    n = len(df)
    d = pd.DataFrame({
        "card": df["cc_num"].to_numpy(),
        "ts": df["ts"].to_numpy(),
        "amt": df["amt"].to_numpy(float),
        "pos": np.arange(n),
    })
    d = d.sort_values(["card", "ts", "pos"], kind="mergesort").reset_index(drop=True)
    gap = d.groupby("card")["ts"].diff().dt.total_seconds()
    med = d.groupby("card")["amt"].transform(
        lambda s: s.expanding().median().shift(1))
    ind = d.set_index("ts")
    g = ind.groupby("card")["amt"]
    res = {}

    def put(name, arr, fill=None):
        full = np.full(n, np.nan)
        full[d["pos"].to_numpy()] = np.asarray(arr, float)
        if fill is not None:
            full = np.nan_to_num(full, nan=fill)
        res[name] = full

    put("count_1h", g.rolling("1h", closed="left").count().to_numpy(), fill=0.0)
    put("count_24h", g.rolling("24h", closed="left").count().to_numpy(), fill=0.0)
    put("sum_24h", g.rolling("24h", closed="left").sum().to_numpy(), fill=0.0)
    put("sum_7d", g.rolling("7D", closed="left").sum().to_numpy(), fill=0.0)
    put("secs_since_last", gap.to_numpy())
    ratio = d["amt"].to_numpy() / med.to_numpy()
    ratio[~np.isfinite(ratio)] = np.nan
    put("amt_over_median", ratio)
    return res


def build_features(df, categories):
    """Return a feature frame aligned to df. Gender and state are not features."""
    out = pd.DataFrame(index=df.index)
    amt = df["amt"].to_numpy(float)
    out["amt"] = amt
    out["log_amt"] = np.log1p(amt)
    out["age"] = (df["ts"] - df["dob"]).dt.days / 365.25
    out["hour"] = df["ts"].dt.hour
    out["dow"] = df["ts"].dt.dayofweek
    out["dist_km"] = haversine_km(df["lat"].to_numpy(float), df["long"].to_numpy(float),
                                  df["merch_lat"].to_numpy(float),
                                  df["merch_long"].to_numpy(float))
    out["city_pop"] = df["city_pop"].to_numpy(float)
    cat_map = {c: i for i, c in enumerate(categories)}
    out["cat_code"] = df["category"].map(cat_map).fillna(len(categories)).astype(int)
    for name, arr in _card_history(df).items():
        out[name] = arr
    return out[FEATURES]
