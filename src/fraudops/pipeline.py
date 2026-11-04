"""Fit the stage 1 pipeline end to end and return everything later stages need.

ASCII only.
"""
import numpy as np
import pandas as pd

from .conformal import ClassConditionalConformal
from .data import time_split
from .detector import fit_temperature, predict_logit, sigmoid, train_detector
from .features import build_features, fit_categories


def load_frames(train_path, test_path, max_cards, seed):
    """Load and clean raw files, or build the synthetic fixture if no path is given."""
    from .data import clean, make_fixture

    if train_path:
        raw_train = pd.read_csv(train_path)
        raw_test = pd.read_csv(test_path) if test_path else None
    else:
        raw_train, raw_test = make_fixture(seed=seed), None
    train_df = clean(raw_train)
    test_df = clean(raw_test) if raw_test is not None else None
    if max_cards:
        cards = np.random.default_rng(seed).permutation(
            train_df["cc_num"].unique())[:max_cards]
        train_df = train_df[train_df["cc_num"].isin(cards)].reset_index(drop=True)
        if test_df is not None:
            test_df = test_df[test_df["cc_num"].isin(cards)].reset_index(drop=True)
    return train_df, test_df


def fit_pipeline(train_df, test_df, cfg):
    """Train, calibrate and score. train_df and test_df are cleaned frames.

    If test_df is None the train frame is split by time into train, calibration
    and test. Otherwise train_df is split 80/20 by time into train and calibration
    and test_df is the held-out period. Features use the full timeline so test rows
    keep their card history.
    """
    seed = cfg["seed"]
    if test_df is not None:
        both = pd.concat([train_df.assign(part="train"), test_df.assign(part="test")],
                         ignore_index=True).sort_values("ts", kind="mergesort")
        frame = both.reset_index(drop=True)
        cats = fit_categories(train_df)
        is_test = (frame["part"] == "test").to_numpy()
        fit_part = frame[~is_test]
        cut = fit_part["ts"].sort_values().iloc[int(len(fit_part) * 0.8)]
        tr_m = (~is_test) & (frame["ts"] <= cut).to_numpy()
        ca_m = (~is_test) & (frame["ts"] > cut).to_numpy()
        te_m = is_test
    else:
        frame = train_df.reset_index(drop=True)
        cats = fit_categories(frame)
        tr, ca, te = time_split(frame, cfg["split"]["train_frac"],
                                cfg["split"]["calib_frac"])
        tr_m = frame.index.isin(tr.index)
        ca_m = frame.index.isin(ca.index)
        te_m = frame.index.isin(te.index)

    X = build_features(frame, cats)
    y = frame["is_fraud"].to_numpy()
    assert frame.loc[tr_m, "ts"].max() < frame.loc[ca_m, "ts"].min(), "splits overlap"
    assert frame.loc[ca_m, "ts"].max() < frame.loc[te_m, "ts"].min(), "splits overlap"

    det = train_detector(X[tr_m], y[tr_m], cfg["detector"], seed)
    logit = predict_logit(det, X)
    temp = fit_temperature(logit[ca_m], y[ca_m])
    p = sigmoid(logit / temp)
    conf = ClassConditionalConformal(cfg["alpha"]).fit(p[ca_m], y[ca_m])

    frame = frame.copy()
    frame["txn_id"] = ["T%07d" % i for i in range(len(frame))]
    rates = frame[tr_m].groupby("category")["is_fraud"].mean().to_dict()
    return {
        "frame": frame, "X": X, "y": y, "p": p, "conformal": conf, "detector": det,
        "temperature": temp, "categories": cats, "category_rates": rates,
        "masks": {"train": tr_m, "calibration": ca_m, "test": te_m},
    }
