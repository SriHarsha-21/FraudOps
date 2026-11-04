"""Schema checks, cleaning, synthetic test fixture and time-ordered splits.

Written from scratch for this project. ASCII only.
"""
import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "state", "lat", "long", "city_pop", "job", "dob", "merch_lat", "merch_long",
    "is_fraud",
]
# Identity-like or unused columns. Dropped so no agent or prompt ever sees them.
DROP_COLUMNS = ["first", "last", "street", "trans_num", "city", "zip", "Unnamed: 0"]
TEXT_COLUMNS = ["merchant", "category", "job"]

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9 _&'-]")
_INJECTION_WORDS = re.compile(
    r"(ignore|disregard|override|system prompt|instruction|approve|assistant|"
    r"you are|forget)", re.IGNORECASE)


def validate_schema(df):
    """Raise ValueError if a required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))


def sanitize_text(value, max_len=64):
    """Strip markup and control characters and truncate.

    Character filtering alone does not stop natural-language injection. The real
    control is that raw text fields are never placed in an LLM prompt. This
    function is the second layer.
    """
    return _UNSAFE_CHARS.sub("", str(value))[:max_len].strip()


def looks_like_injection(value):
    """Flag text that resembles an instruction aimed at a model."""
    return bool(_INJECTION_WORDS.search(str(value)))


def _parse_dates(series, fmt):
    """Parse with a pinned format, falling back to inference if the format differs."""
    try:
        return pd.to_datetime(series, format=fmt)
    except (ValueError, TypeError):
        return pd.to_datetime(series)


def clean(df):
    """Validate, parse, sanitize and sort a raw transaction frame."""
    validate_schema(df)
    out = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()
    out["ts"] = _parse_dates(out["trans_date_trans_time"], "%Y-%m-%d %H:%M:%S")
    out = out.drop(columns=["trans_date_trans_time"])
    out["dob"] = _parse_dates(out["dob"], "%Y-%m-%d")
    out["cc_num"] = out["cc_num"].astype(str)
    out["merchant"] = out["merchant"].astype(str).str.replace(r"^fraud_", "", regex=True)
    flag = pd.Series(False, index=out.index)
    for col in TEXT_COLUMNS:
        flag = flag | out[col].map(looks_like_injection)
        out[col] = out[col].map(sanitize_text)
    out["text_flag"] = flag
    out["is_fraud"] = out["is_fraud"].astype(int)
    out = out.sort_values("ts", kind="mergesort").reset_index(drop=True)
    return out


def time_split(df, train_frac=0.7, calib_frac=0.15):
    """Split a time-sorted frame into train, calibration and test by time."""
    n = len(df)
    ts = df["ts"].sort_values().reset_index(drop=True)
    t1 = ts.iloc[int(n * train_frac)]
    t2 = ts.iloc[min(n - 1, int(n * (train_frac + calib_frac)))]
    train = df[df["ts"] <= t1]
    calib = df[(df["ts"] > t1) & (df["ts"] <= t2)]
    test = df[df["ts"] > t2]
    return train, calib, test


def make_fixture(n_cards=120, per_card=80, days=90, fraud_rate=0.03, seed=0):
    """Small synthetic frame with the same columns as the Sparkov-style data.

    Used only for tests and dry runs. Fraud is made detectable on purpose:
    large amounts, night hours and far-away merchants.
    """
    rng = np.random.default_rng(seed)
    cats = ["grocery_pos", "gas_transport", "food_dining", "shopping_net",
            "misc_net", "entertainment", "home"]
    start = pd.Timestamp("2019-01-01")
    rows = []
    for c in range(n_cards):
        home_lat = rng.uniform(30, 47)
        home_long = rng.uniform(-120, -75)
        dob = pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.integers(0, 15000)))
        gender = str(rng.choice(["F", "M"]))
        state = str(rng.choice(["TX", "NY", "CA", "FL"]))
        base = rng.uniform(20, 80)
        for _ in range(int(rng.poisson(per_card)) + 10):
            ts = start + pd.Timedelta(seconds=float(rng.uniform(0, days * 86400)))
            fraud = rng.random() < fraud_rate
            # About 40 percent of fraud is "stealth": it looks close to normal use.
            stealth = fraud and rng.random() < 0.4
            if stealth:
                amt = float(base * rng.uniform(1.5, 3.0))
                mlat = home_lat + rng.normal(0, 0.5)
                mlong = home_long + rng.normal(0, 0.5)
                cat = str(rng.choice(cats))
            elif fraud:
                amt = float(rng.uniform(300, 1500))
                ts = ts.normalize() + pd.Timedelta(hours=int(rng.integers(0, 4)),
                                                   minutes=int(rng.integers(0, 60)))
                mlat = home_lat + rng.normal(0, 3)
                mlong = home_long + rng.normal(0, 3)
                cat = str(rng.choice(["shopping_net", "misc_net"]))
            else:
                amt = float(abs(rng.normal(base, base * 0.4)) + 1)
                mlat = home_lat + rng.normal(0, 0.3)
                mlong = home_long + rng.normal(0, 0.3)
                cat = str(rng.choice(cats))
            rows.append({
                "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "cc_num": 4000000000000000 + c,
                "merchant": "fraud_Merchant " + str(int(rng.integers(0, 50))),
                "category": cat, "amt": amt, "first": "A", "last": "B",
                "gender": gender, "street": "1 Main St", "city": "Town",
                "state": state, "zip": "12345", "lat": home_lat, "long": home_long,
                "city_pop": int(rng.integers(500, 500000)), "job": "Engineer",
                "dob": dob.strftime("%Y-%m-%d"), "trans_num": "t%d" % len(rows),
                "unix_time": int(ts.timestamp()), "merch_lat": mlat,
                "merch_long": mlong, "is_fraud": int(fraud),
            })
    return pd.DataFrame(rows)
