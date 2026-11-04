"""Red team tests: value is preserved, only test fraud changes, and rules hold."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from fraudops.data import clean, make_fixture
from fraudops.pipeline import fit_pipeline
from fraudops.redteam import (ALL_ATTACKS, ATTACKS, make_context, mutate_cases,
                              promotion_decision, run_red_team, with_parts)

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "..", "config.json")) as fh:
    CFG = json.load(fh)


@pytest.fixture(scope="module")
def world():
    pipe = fit_pipeline(clean(make_fixture(seed=7)), None, CFG)
    champ = with_parts(pipe)
    return champ, make_context(champ, CFG)


def _targets(champ):
    fr = champ["frame"]
    return fr.index[(fr["part_orig"] == "test") & (fr["is_fraud"] == 1)]


def _cents(s):
    return int(np.round(np.asarray(s, float) * 100).sum())


@pytest.mark.parametrize("attack", ALL_ATTACKS)
def test_every_attack_preserves_total_value_in_cents(world, attack):
    champ, ctx = world
    fr, tgt = champ["frame"], _targets(champ)
    out = mutate_cases(fr, pd.Series(attack, index=tgt, dtype=object), ctx, seed=1)
    before = _cents(fr.loc[tgt, "amt"])
    after = _cents(out.loc[out["case_id"] >= 0, "amt"])
    assert before == after


@pytest.mark.parametrize("attack", ALL_ATTACKS)
def test_only_test_fraud_rows_change(world, attack):
    champ, ctx = world
    fr, tgt = champ["frame"], _targets(champ)
    out = mutate_cases(fr, pd.Series(attack, index=tgt, dtype=object), ctx, seed=2)
    key = ["cc_num", "ts", "amt", "category", "is_fraud", "part_orig"]
    untouched_before = fr.drop(index=tgt)[key].sort_values(key).reset_index(drop=True)
    untouched_after = out[out["case_id"] < 0][key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(untouched_before, untouched_after)
    assert (out.loc[out["case_id"] >= 0, "part_orig"] == "test").all()
    assert (out.loc[out["case_id"] >= 0, "is_fraud"] == 1).all()


def test_splitting_makes_equal_pieces_beyond_the_velocity_window(world):
    champ, ctx = world
    fr, tgt = champ["frame"], _targets(champ)
    out = mutate_cases(fr, pd.Series("splitting", index=tgt, dtype=object), ctx, seed=3)
    pieces = out[out["case_id"] >= 0].sort_values(["case_id", "ts"])
    assert len(pieces) == len(tgt) * CFG["red_team"]["split_n"]
    gaps = pieces.groupby("case_id")["ts"].diff().dropna().dt.total_seconds() / 60
    assert (gaps >= CFG["red_team"]["split_gap_min"] - 1e-9).all()
    assert (pieces["amt"] > 0).all()


def test_shaving_keeps_each_piece_under_the_cap(world):
    champ, ctx = world
    fr, tgt = champ["frame"], _targets(champ)
    out = mutate_cases(fr, pd.Series("amount_shaving", index=tgt, dtype=object), ctx, seed=4)
    pieces = out[out["case_id"] >= 0]
    med = pieces["cc_num"].map(ctx["median"]).fillna(ctx["global_median"])
    cap = np.maximum(1.0, np.floor(CFG["red_team"]["shave_k"] * med * 100) / 100)
    assert (pieces["amt"] <= cap + 1e-9).all()


def test_timing_spread_separates_fraud_on_the_same_card(world):
    champ, ctx = world
    fr, tgt = champ["frame"], _targets(champ)
    out = mutate_cases(fr, pd.Series("timing_spread", index=tgt, dtype=object), ctx, seed=5)
    pieces = out[out["case_id"] >= 0].sort_values(["cc_num", "ts"])
    gaps = pieces.groupby("cc_num")["ts"].diff().dropna().dt.total_seconds() / 3600
    assert (gaps >= CFG["red_team"]["spread_gap_hours"] - 1e-9).all()


def test_category_swap_and_local_looking_change_only_their_field(world):
    champ, ctx = world
    fr, tgt = champ["frame"], _targets(champ)
    sw = mutate_cases(fr, pd.Series("category_swap", index=tgt, dtype=object), ctx, seed=6)
    assert (sw.loc[sw["case_id"] >= 0, "category"] == ctx["safe_category"]).all()
    lk = mutate_cases(fr, pd.Series("local_looking", index=tgt, dtype=object), ctx, seed=6)
    moved = lk[lk["case_id"] >= 0]
    assert (np.abs(moved["merch_lat"] - moved["lat"]) < 0.2).all()


def test_mutation_is_deterministic_for_a_seed(world):
    champ, ctx = world
    fr, tgt = champ["frame"], _targets(champ)
    a = pd.Series("local_looking", index=tgt, dtype=object)
    x = mutate_cases(fr, a, ctx, seed=9)
    y = mutate_cases(fr, a, ctx, seed=9)
    pd.testing.assert_frame_equal(x, y)


def test_promotion_requires_cost_and_coverage():
    ok = {"model_plus_gate": {"cost": 100.0}, "gate": {"coverage_fraud": 0.9}}
    cheaper = {"model_plus_gate": {"cost": 90.0}, "gate": {"coverage_fraud": 0.9}}
    dearer = {"model_plus_gate": {"cost": 120.0}, "gate": {"coverage_fraud": 0.9}}
    low_cov = {"model_plus_gate": {"cost": 90.0}, "gate": {"coverage_fraud": 0.5}}
    assert promotion_decision(ok, cheaper, CFG)["promote"] is True
    assert promotion_decision(ok, dearer, CFG)["promote"] is False
    assert promotion_decision(ok, low_cov, CFG)["promote"] is False


def test_run_red_team_output_shape():
    pipe = fit_pipeline(clean(make_fixture(seed=8)), None, CFG)
    res = run_red_team(pipe, CFG, seed=1)
    for side in ("champion", "challenger"):
        assert set(res[side]) == {"clean"} | set(ALL_ATTACKS)
        for r in res[side].values():
            assert r["fraud_rows"] > 0
            assert 0.0 <= r["model_only"]["value_recall"] <= 1.0
            assert r["model_plus_gate"]["silent_miss_rate"] <= 1.0
    assert isinstance(res["promotion"]["promote"], bool)
    assert len(ATTACKS) == 5
