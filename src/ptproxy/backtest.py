"""Rolling-origin backtest: how well does each method predict country mixes?

From each forecast origin T, tilts are estimated on actuals T-2..T only, then
used to allocate the regional forecast for T+1..T+3. Errors are measured on
the country-level powertrain mix (all models), in percentage points, weighted
by country volume.

The regional forecast is taken as known so that the test isolates what this
module controls: the split between countries.
"""

from __future__ import annotations

import pandas as pd

from ptproxy.allocate import allocate_region, country_tilts
from ptproxy.labels import relabel_frame
from ptproxy.synthetic import World

METHODS = ("Naive (regional mix)", "Proxy, raw labels", "Proxy, corrected labels")


def country_mix(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["country", "year", "powertrain"], as_index=False)["volume"].sum()
    g["share"] = g["volume"] / g.groupby(["country", "year"])["volume"].transform("sum")
    return g


def mix_error_pp(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Volume-weighted mean absolute error of country mixes, in pp."""
    p, t = country_mix(pred), country_mix(truth)
    m = t.merge(p, on=["country", "year", "powertrain"], suffixes=("_t", "_p"))
    w = m.groupby(["country", "year"])["volume_t"].transform("sum")
    return float(((m["share_t"] - m["share_p"]).abs() * w).sum() / w.sum() * 100)


def run(world: World, origins: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025), horizons=(1, 2, 3)) -> pd.DataFrame:
    """Return ``origin, horizon, method, error_pp``."""
    fixed = relabel_frame(world.observed, world.region_mix, world.reference)
    out = []
    for T in origins:
        fit_years = [T - 2, T - 1, T]
        tilts = {
            METHODS[0]: None,
            METHODS[1]: country_tilts(world.observed, world.region_mix, fit_years),
            METHODS[2]: country_tilts(fixed, world.region_mix, fit_years),
        }
        for h in horizons:
            y = T + h
            cv = world.country_volumes[world.country_volumes["year"] == y]
            rp = world.region_pt[world.region_pt["year"] == y]
            truth = world.truth[world.truth["year"] == y]
            for name, tl in tilts.items():
                pred = allocate_region(cv, rp, tl)
                out.append((T, h, name, mix_error_pp(pred, truth)))
    return pd.DataFrame(out, columns=["origin", "horizon", "method", "error_pp"])
