"""Business-rule tests. Everything runs on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptproxy import POWERTRAINS, allocate_region, confusion_from_reference, country_tilts, relabel
from ptproxy.allocate import iron_rule_gap
from ptproxy.backtest import METHODS, run
from ptproxy.labels import relabel_frame
from ptproxy.synthetic import make_world


@pytest.fixture(scope="module")
def world():
    return make_world(seed=11)


def _year(df, y):
    return df[df["year"] == y]


@pytest.fixture(scope="module")
def tilts(world):
    y = world.last_actual
    return country_tilts(world.observed, world.region_mix, [y - 2, y - 1, y])


# --- allocation ---------------------------------------------------------------


def test_iron_rule_holds_exactly(world, tilts):
    y = 2030
    a = allocate_region(_year(world.country_volumes, y), _year(world.region_pt, y), tilts)
    assert iron_rule_gap(a, _year(world.region_pt, y)) < 1e-3


def test_country_totals_preserved(world, tilts):
    y = 2028
    cv = _year(world.country_volumes, y)
    a = allocate_region(cv, _year(world.region_pt, y), tilts)
    got = a.groupby(["country", "model"])["volume"].sum()
    want = cv.set_index(["country", "model"])["volume"]
    np.testing.assert_allclose(got.reindex(want.index), want, rtol=1e-6)


def test_no_negative_volumes(world, tilts):
    a = allocate_region(_year(world.country_volumes, 2030), _year(world.region_pt, 2030), tilts)
    assert (a["volume"] >= 0).all()


def test_mono_powertrain_model_is_exact(world, tilts):
    a = allocate_region(_year(world.country_volumes, 2027), _year(world.region_pt, 2027), tilts)
    corvo = a[a["model"] == "Corvo"]  # BEV-only
    assert corvo.loc[corvo["powertrain"] != "BEV", "volume"].abs().max() < 1e-6


def test_naive_split_equals_regional_mix(world):
    y = 2027
    a = allocate_region(_year(world.country_volumes, y), _year(world.region_pt, y), None)
    sh = a.assign(share=a["volume"] / a.groupby(["country", "model"])["volume"].transform("sum"))
    reg = _year(world.region_mix, y)
    m = sh.merge(reg, on=["model", "year", "powertrain"], suffixes=("", "_r"))
    np.testing.assert_allclose(m["share"], m["share_r"], atol=1e-9)


def test_inconsistent_releases_are_rejected(world):
    y = 2027
    cv = _year(world.country_volumes, y).copy()
    cv["volume"] *= 1.05  # country source 5 % above the region: different releases
    with pytest.raises(ValueError, match="differs from region"):
        allocate_region(cv, _year(world.region_pt, y), None)


def test_missing_regional_forecast_is_reported(world):
    cv = _year(world.country_volumes, 2027)
    with pytest.raises(KeyError, match="No regional powertrain forecast"):
        allocate_region(cv, _year(world.region_pt, 2026), None)


def test_tilts_are_zero_when_country_matches_region():
    reg = pd.DataFrame(
        {"model": "X", "year": 2025, "powertrain": list(POWERTRAINS), "share": [0.4, 0.1, 0.2, 0.2, 0.1]}
    )
    obs = reg.assign(country="A", volume=reg["share"] * 1e5).drop(columns="share")
    t = country_tilts(obs, reg, [2025])
    assert t["tilt"].abs().max() < 1e-3


def test_deterministic(world, tilts):
    y = 2029
    a1 = allocate_region(_year(world.country_volumes, y), _year(world.region_pt, y), tilts)
    a2 = allocate_region(_year(world.country_volumes, y), _year(world.region_pt, y), tilts)
    pd.testing.assert_frame_equal(a1, a2)


# --- label correction ---------------------------------------------------------


def test_identity_confusion_changes_nothing():
    pts = list(POWERTRAINS)
    ref = pd.DataFrame({"true_pt": pts, "observed_pt": pts, "volume": 1.0})
    obs = pd.Series([10.0, 5, 30, 20, 35], index=pts)
    out = relabel(obs, pd.Series(0.2, index=pts), confusion_from_reference(ref), em_iter=5)
    np.testing.assert_allclose(out.reindex(pts), obs)


def test_relabel_conserves_volume(world):
    fixed = relabel_frame(world.observed, world.region_mix, world.reference)
    a = fixed.groupby(["country", "model", "year"])["volume"].sum()
    b = world.observed.groupby(["country", "model", "year"])["volume"].sum()
    np.testing.assert_allclose(a.reindex(b.index), b, rtol=1e-9)


def test_vectorised_matches_reference_implementation(world):
    fixed = relabel_frame(world.observed, world.region_mix, world.reference, em_iter=3)
    c, m, y = "Iberica", "Delta", 2024

    def pick(df):
        sel = (df["model"] == m) & (df["year"] == y)
        if "country" in df:
            sel &= df["country"] == c
        return df[sel].set_index("powertrain")

    prior = pick(world.region_mix)["share"]
    one = relabel(pick(world.observed)["volume"], prior, confusion_from_reference(world.reference, c), em_iter=3)
    got = pick(fixed)["volume"]
    np.testing.assert_allclose(got.reindex(one.index), one, rtol=1e-9)


# --- end to end ---------------------------------------------------------------


def test_backtest_ranks_methods(world):
    bt = run(world, origins=(2024,), horizons=(1,))
    err = bt.set_index("method")["error_pp"]
    assert err[METHODS[2]] < err[METHODS[1]] < err[METHODS[0]]
