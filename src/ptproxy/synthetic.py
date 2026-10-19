"""Synthetic European "Others" region — no real data anywhere in this repo.

Eight fictional countries, fourteen fictional models, 2019–2030. Each country
has a persistent powertrain preference (Nordia buys BEVs, Hellenia hybrids, …),
so the regional average is a poor description of any single country — exactly
the situation the proxy is built for.

The generator returns the *truth* (which a real analyst never sees) alongside
what the analyst *does* see: noisy-label registrations up to the last actual
year, a reference sample for label correction, and the regional forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ptproxy._const import POWERTRAINS

COUNTRIES: dict[str, dict[str, float]] = {
    # name: size (k units / yr) and powertrain tilts (log-odds vs. average)
    "Nordia": {"size": 140, "BEV": 2.6, "PHEV": 0.6, "HEV": -0.4, "MHEV": -0.6, "ICE": -1.6},
    "Alpinia": {"size": 260, "BEV": 0.5, "PHEV": 0.7, "HEV": -0.1, "MHEV": 0.5, "ICE": -0.3},
    "Batavia": {"size": 330, "BEV": 1.1, "PHEV": 0.4, "HEV": 0.2, "MHEV": -0.3, "ICE": -0.6},
    "Iberica": {"size": 520, "BEV": -0.6, "PHEV": 0.1, "HEV": 0.7, "MHEV": 0.2, "ICE": 0.2},
    "Danubia": {"size": 210, "BEV": -0.7, "PHEV": -0.3, "HEV": 0.3, "MHEV": 0.4, "ICE": 0.6},
    "Hellenia": {"size": 120, "BEV": -1.0, "PHEV": -0.2, "HEV": 1.1, "MHEV": 0.1, "ICE": 0.3},
    "Carpathia": {"size": 150, "BEV": -1.2, "PHEV": -0.6, "HEV": 0.6, "MHEV": 0.0, "ICE": 1.0},
    "Baltica": {"size": 60, "BEV": -0.3, "PHEV": -0.4, "HEV": 0.9, "MHEV": -0.2, "ICE": 0.4},
}

# model: (segment, available powertrains)
MODELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Arca": ("A", ("BEV", "MHEV", "ICE")),
    "Brisa": ("B", ("HEV", "MHEV", "ICE")),
    "Corvo": ("B", ("BEV",)),
    "Delta": ("B-SUV", ("BEV", "PHEV", "HEV", "MHEV", "ICE")),
    "Ember": ("B-SUV", ("HEV", "ICE")),
    "Fjord": ("C", ("BEV", "PHEV", "MHEV", "ICE")),
    "Gale": ("C", ("HEV", "MHEV")),
    "Helix": ("C-SUV", ("BEV", "PHEV", "HEV", "MHEV", "ICE")),
    "Iris": ("C-SUV", ("BEV",)),
    "Jura": ("C-SUV", ("PHEV", "HEV", "MHEV")),
    "Kestrel": ("D", ("BEV", "PHEV", "MHEV")),
    "Lumen": ("D-SUV", ("BEV", "PHEV", "HEV")),
    "Mistral": ("D-SUV", ("BEV",)),
    "Nova": ("E", ("PHEV", "MHEV", "ICE")),
}

TREND = {"BEV": 0.30, "PHEV": 0.10, "HEV": 0.08, "MHEV": 0.06, "ICE": -0.28}

# Share of mild hybrids each country's registration feed files as full hybrids.
MHEV_AS_HEV = {
    "Nordia": 0.05,
    "Alpinia": 0.10,
    "Batavia": 0.15,
    "Iberica": 0.55,
    "Danubia": 0.40,
    "Hellenia": 0.60,
    "Carpathia": 0.30,
    "Baltica": 0.20,
}
# Countries the cleaner reference source covers.
REFERENCE_COVERAGE = ("Alpinia", "Iberica", "Danubia", "Carpathia")


def label_noise(mhev_as_hev: float) -> pd.DataFrame:
    """P(filed label | true powertrain) for one country's registration feed."""
    m = mhev_as_hev
    return pd.DataFrame(
        [
            [0.99, 0.01, 0.00, 0.00, 0.00],
            [0.00, 0.96, 0.03, 0.00, 0.01],
            [0.00, 0.00, 0.95, 0.03, 0.02],
            [0.00, 0.00, m, 0.92 - m, 0.08],
            [0.00, 0.00, 0.01, 0.02, 0.97],
        ],
        index=list(POWERTRAINS),
        columns=list(POWERTRAINS),
    )


@dataclass
class World:
    truth: pd.DataFrame  # country, model, year, powertrain, volume
    observed: pd.DataFrame  # same, labels noisy, actual years only
    reference: pd.DataFrame  # country, true_pt, observed_pt, volume
    region_pt: pd.DataFrame  # model, year, powertrain, volume
    region_mix: pd.DataFrame  # model, year, powertrain, share
    country_volumes: pd.DataFrame  # country, model, year, volume (no split)
    segments: dict[str, str]
    last_actual: int


def make_world(seed: int = 7, years: range = range(2019, 2031), last_actual: int = 2025) -> World:
    """Generate one reproducible synthetic world."""
    rng = np.random.default_rng(seed)
    pts = list(POWERTRAINS)
    yrs = np.array(list(years))
    model_pop = rng.dirichlet(np.ones(len(MODELS)) * 3)
    base = {m: rng.normal(0, 0.6, len(pts)) for m in MODELS}

    rows = []
    for c, spec in COUNTRIES.items():
        ctilt = np.array([spec[p] for p in pts])
        for k, (m, (_, avail)) in enumerate(MODELS.items()):
            mask = np.array([p in avail for p in pts])
            mtilt = rng.normal(0, 0.25, len(pts))  # model-specific country taste
            pop = model_pop[k] * rng.lognormal(0, 0.35)
            for t in yrs:
                logit = base[m] + np.array([TREND[p] for p in pts]) * (t - 2019) + ctilt + mtilt
                logit += rng.normal(0, 0.08, len(pts))
                w = np.where(mask, np.exp(logit), 0.0)
                vol = spec["size"] * 1_000 * pop * (1 + 0.015 * (t - 2019))
                for p, v in zip(pts, vol * w / w.sum(), strict=True):
                    rows.append((c, m, int(t), p, v))
    truth = pd.DataFrame(rows, columns=["country", "model", "year", "powertrain", "volume"])

    # What registration feeds report: expected misfiling, actual years only.
    act = truth[truth["year"] <= last_actual]
    wide = act.pivot_table(index=["country", "model", "year"], columns="powertrain", values="volume").reindex(
        columns=pts
    )
    parts = []
    for c, g in wide.fillna(0.0).groupby(level="country"):
        o = g @ label_noise(MHEV_AS_HEV[c]).to_numpy()
        o.columns = pts
        parts.append(o)
    observed = (
        pd.concat(parts)
        .reset_index()
        .melt(id_vars=["country", "model", "year"], var_name="powertrain", value_name="volume")
    )

    # Reference sample: ~5 % of vehicles in covered countries, seen in both sources.
    ref_rows = []
    for c in REFERENCE_COVERAGE:
        tot_true = act[act["country"] == c].groupby("powertrain")["volume"].sum()
        noise = label_noise(MHEV_AS_HEV[c])
        for p in pts:
            n = int(tot_true.get(p, 0) * 0.05)
            if n:
                draws = rng.multinomial(n, noise.loc[p].to_numpy())
                ref_rows += [(c, p, q, d) for q, d in zip(pts, draws, strict=True) if d]
    reference = pd.DataFrame(ref_rows, columns=["country", "true_pt", "observed_pt", "volume"])

    region_pt = truth.groupby(["model", "year", "powertrain"], as_index=False)["volume"].sum()
    region_mix = region_pt.assign(
        share=region_pt["volume"] / region_pt.groupby(["model", "year"])["volume"].transform("sum")
    )
    country_volumes = truth.groupby(["country", "model", "year"], as_index=False)["volume"].sum()
    return World(
        truth=truth,
        observed=observed,
        reference=reference,
        region_pt=region_pt,
        region_mix=region_mix[["model", "year", "powertrain", "share"]],
        country_volumes=country_volumes,
        segments={m: s for m, (s, _) in MODELS.items()},
        last_actual=last_actual,
    )
