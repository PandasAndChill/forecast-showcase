"""Allocate a regional powertrain forecast down to its member countries.

The problem
-----------
A vendor forecasts the powertrain mix (BEV, PHEV, HEV, MHEV, ICE) for a
*region* as a whole, per model and year. A second source forecasts the
*volume* of each model in each member country, but with no powertrain split.
We need the country × model × powertrain table.

The rule that must never break ("iron rule")
--------------------------------------------
For every model, year and powertrain, the countries must add up exactly to the
regional forecast. The proxy redistributes the mix between countries; it never
invents or loses a single regional unit.

Method
------
1. Seed each country with the regional mix, tilted by the country's own
   observed preference (:func:`country_tilts`).
2. Rake the seed with iterative proportional fitting (IPF / RAS) so that
   rows match country volumes and columns match regional powertrain volumes.
   Mono-powertrain models are therefore exact by construction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ptproxy._const import POWERTRAINS

log = logging.getLogger(__name__)

EPS = 1e-9


def country_tilts(
    observed: pd.DataFrame,
    region_mix: pd.DataFrame,
    years: list[int],
    shrink: float = 2_000.0,
) -> pd.DataFrame:
    """Estimate each country's persistent powertrain preference as a log-ratio.

    The tilt for country *c*, model *k*, powertrain *p* is
    ``log(share_c / share_region)`` pooled over ``years``. Model-level tilts are
    shrunk toward the country's all-model tilt with weight ``n / (n + shrink)``,
    where *n* is the country-model volume — so thin evidence falls back to the
    country's general behaviour instead of producing noise.

    Args:
        observed: columns ``country, model, year, powertrain, volume`` (actuals).
        region_mix: columns ``model, year, powertrain, share`` (shares sum to 1).
        years: actual years to pool.
        shrink: pseudo-volume of the shrinkage prior.

    Returns:
        DataFrame ``country, model, powertrain, tilt``.
    """
    obs = observed[observed["year"].isin(years)]
    reg = region_mix[region_mix["year"].isin(years)]

    # Expected volume if each country followed the regional mix exactly.
    cm_tot = obs.groupby(["country", "model", "year"], as_index=False)["volume"].sum()
    expected = cm_tot.merge(reg, on=["model", "year"])
    expected["expected"] = expected["volume"] * expected["share"]

    keys = ["country", "model", "powertrain"]
    o = obs.groupby(keys)["volume"].sum()
    e = expected.groupby(keys)["expected"].sum()
    both = pd.concat([o.rename("obs"), e], axis=1).fillna(0.0).reset_index()
    both = both[both["expected"] > 0]

    # Country-level tilt (all models pooled) acts as the shrinkage target.
    ctry = both.groupby(["country", "powertrain"])[["obs", "expected"]].sum()
    ctry["tilt_c"] = np.log((ctry["obs"] + 1) / (ctry["expected"] + 1))
    both = both.merge(ctry["tilt_c"].reset_index(), on=["country", "powertrain"])

    n = both.groupby(["country", "model"])["obs"].transform("sum")
    w = n / (n + shrink)
    raw = np.log((both["obs"] + 1) / (both["expected"] + 1))
    both["tilt"] = w * raw + (1 - w) * both["tilt_c"]
    return both[keys + ["tilt"]]


def _ipf(seed: np.ndarray, rows: np.ndarray, cols: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Rake ``seed`` (countries × powertrains) to row and column targets."""
    x = seed.copy()
    for _ in range(max_iter):
        x *= (rows / np.maximum(x.sum(axis=1), EPS))[:, None]
        x *= cols / np.maximum(x.sum(axis=0), EPS)
        if np.abs(x.sum(axis=1) - rows).max() < tol:
            break
    return x


def allocate_region(
    country_volumes: pd.DataFrame,
    region_powertrain: pd.DataFrame,
    tilts: pd.DataFrame | None = None,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> pd.DataFrame:
    """Split country model volumes by powertrain, consistent with the region.

    Args:
        country_volumes: ``country, model, year, volume`` — no powertrain split.
        region_powertrain: ``model, year, powertrain, volume`` — regional forecast.
        tilts: output of :func:`country_tilts`; ``None`` gives the naive split
            (every country inherits the regional mix).

    Returns:
        ``country, model, year, powertrain, volume``. For each model × year ×
        powertrain the country volumes sum to the regional volume.

    Raises:
        ValueError: if country volumes and the regional total disagree by more
            than 0.5 % for any model-year (inputs from different releases).
    """
    pts = list(POWERTRAINS)
    reg = (
        region_powertrain.pivot_table(index=["model", "year"], columns="powertrain", values="volume", aggfunc="sum")
        .reindex(columns=pts)
        .fillna(0.0)
    )
    tilt_map = (
        {}
        if tilts is None
        else {
            (c, m): g.set_index("powertrain")["tilt"].reindex(pts).fillna(0.0).to_numpy()
            for (c, m), g in tilts.groupby(["country", "model"])
        }
    )
    ctry_fallback = (
        {}
        if tilts is None
        else (tilts.groupby(["country", "powertrain"])["tilt"].mean().unstack().reindex(columns=pts).fillna(0.0))
    )

    out: list[pd.DataFrame] = []
    for (model, year), g in country_volumes.groupby(["model", "year"]):
        if (model, year) not in reg.index:
            raise KeyError(f"No regional powertrain forecast for {model} {year}")
        cols = reg.loc[(model, year)].to_numpy(dtype=float)
        rows = g["volume"].to_numpy(dtype=float)
        gap = rows.sum() / max(cols.sum(), EPS) - 1
        if abs(gap) > 0.005:
            raise ValueError(f"{model} {year}: country total differs from region by {gap:+.2%}")
        rows = rows * cols.sum() / max(rows.sum(), EPS)  # absorb rounding only

        countries = g["country"].to_list()
        tilt = np.zeros((len(countries), len(pts)))
        for i, c in enumerate(countries):
            if (c, model) in tilt_map:
                tilt[i] = tilt_map[(c, model)]
            elif len(ctry_fallback) and c in ctry_fallback.index:
                tilt[i] = ctry_fallback.loc[c].to_numpy()
        share = cols / max(cols.sum(), EPS)
        seed = rows[:, None] * share[None, :] * np.exp(tilt)
        x = _ipf(seed, rows, cols, tol, max_iter) if tilts is not None else rows[:, None] * share[None, :]

        df = pd.DataFrame(x, columns=pts)
        df.insert(0, "country", countries)
        out.append(
            df.melt(id_vars="country", var_name="powertrain", value_name="volume").assign(model=model, year=year)
        )

    res = pd.concat(out, ignore_index=True)[["country", "model", "year", "powertrain", "volume"]]
    log.info("allocated %d rows", len(res))
    return res


def iron_rule_gap(allocated: pd.DataFrame, region_powertrain: pd.DataFrame) -> float:
    """Largest absolute gap (units) between summed countries and the region."""
    s = allocated.groupby(["model", "year", "powertrain"])["volume"].sum()
    r = region_powertrain.set_index(["model", "year", "powertrain"])["volume"]
    return float((s - r.reindex(s.index).fillna(0.0)).abs().max())
