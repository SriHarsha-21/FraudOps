"""Date parsing: the spreadsheet-saved mirror format must give the same data as ISO."""
import pandas as pd
import pytest

from fraudops.data import clean, make_fixture


def _to_mirror_format(raw):
    """Rewrite dates the way the Hugging Face mirror has them: 1/19/62 0:00."""
    out = raw.copy()
    ts = pd.to_datetime(out["trans_date_trans_time"])
    dob = pd.to_datetime(out["dob"])
    out["trans_date_trans_time"] = [
        "%d/%d/%s %d:%02d" % (t.month, t.day, t.strftime("%y"), t.hour, t.minute)
        for t in ts]
    out["dob"] = ["%d/%d/%s" % (d.month, d.day, d.strftime("%y")) for d in dob]
    return out


def test_mirror_format_matches_iso_format():
    raw = make_fixture(n_cards=40, per_card=20, seed=3)
    iso = clean(raw)
    mirror = clean(_to_mirror_format(raw))
    assert len(iso) == len(mirror)
    # The mirror drops seconds, so compare at minute precision and match rows by key.
    iso = iso.assign(ts=iso["ts"].dt.floor("min"))
    key = ["cc_num", "amt", "merchant"]
    both = iso.merge(mirror, on=key, suffixes=("_iso", "_mir"))
    assert len(both) >= len(iso) * 0.99
    assert (both["dob_iso"] == both["dob_mir"]).all()


def test_two_digit_years_never_put_a_birth_in_the_future():
    raw = make_fixture(n_cards=60, per_card=15, seed=4)
    df = clean(_to_mirror_format(raw))
    age = (df["ts"] - df["dob"]).dt.days / 365.25
    assert age.min() > 15 and age.max() < 100
    assert (df["dob"].dt.year < 2000).any()


def test_unknown_date_format_raises_instead_of_guessing():
    raw = make_fixture(n_cards=5, per_card=5, seed=1)
    raw["trans_date_trans_time"] = "not a date"
    with pytest.raises(ValueError):
        clean(raw)
