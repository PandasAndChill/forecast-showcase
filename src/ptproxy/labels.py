"""Correct mislabelled powertrains in registration data.

Registration feeds often blur neighbouring technologies — a mild hybrid
(MHEV) filed as a full hybrid (HEV), or as plain ICE. Using those labels
as-is biases every country tilt built on them.

A smaller, cleaner reference source tells us how true powertrains map to
filed labels: the confusion matrix ``P(observed | true)``. It covers only some
countries, and misfiling habits differ by country, so:

* **covered countries** use their own confusion matrix;
* **uncovered countries** use the pooled matrix — a weaker but unbiased guess.

Bayes' rule then gives ``P(true | observed)``. The prior on the true mix starts
from the regional forecast and is re-estimated by EM on the country's own data,
so a country's genuine preferences are not pulled back toward the average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ptproxy._const import POWERTRAINS


def confusion_from_reference(reference: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Build ``P(observed | true)`` from a reference sample.

    Args:
        reference: ``[country,] true_pt, observed_pt, volume`` — the same
            vehicles seen in both sources.
        country: restrict to one country; ``None`` pools all of them.

    Returns:
        Square DataFrame, index = true powertrain, columns = observed label;
        each row sums to 1. Unseen rows default to the identity.
    """
    pts = list(POWERTRAINS)
    if country is not None:
        reference = reference[reference["country"] == country]
    m = reference.pivot_table(index="true_pt", columns="observed_pt", values="volume", aggfunc="sum")
    m = m.reindex(index=pts, columns=pts).fillna(0.0)
    for p in pts:
        if m.loc[p].sum() == 0:
            m.loc[p, p] = 1.0
    return m.div(m.sum(axis=1), axis=0)


def relabel(observed: pd.Series, prior: pd.Series, confusion: pd.DataFrame, em_iter: int = 0) -> pd.Series:
    """Redistribute observed volumes by label onto estimated true powertrains.

    Volume is conserved exactly: ``relabel(...).sum() == observed.sum()``.

    Args:
        observed: volume by observed label.
        prior: prior share by true powertrain (e.g. regional forecast mix).
        confusion: output of :func:`confusion_from_reference`.
        em_iter: EM refinements of the prior on this group's own volumes.
    """
    pts = list(POWERTRAINS)
    obs = observed.reindex(pts).fillna(0.0)
    pri = prior.reindex(pts).fillna(0.0)
    est = _bayes(obs, pri, confusion)
    for _ in range(em_iter):
        if est.sum() <= 0:
            break
        est = _bayes(obs, est / est.sum(), confusion)
    return est.rename("volume")


def _bayes(obs: pd.Series, pri: pd.Series, confusion: pd.DataFrame) -> pd.Series:
    pts = list(POWERTRAINS)
    joint = confusion.mul(pri, axis=0)  # P(obs | true) * P(true)
    col = joint.sum(axis=0)
    # Labels the prior cannot explain keep their own label.
    post = joint.div(col.where(col > 0, 1.0), axis=1)
    for p in pts:
        if col[p] == 0:
            post[p] = 0.0
            post.loc[p, p] = 1.0
    return post.mul(obs, axis=1).sum(axis=1)


def _bayes_batch(obs: np.ndarray, pri: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Vectorised :func:`_bayes` over groups: obs, pri are (n, P); conf is (P, P)."""
    joint = conf[None, :, :] * pri[:, :, None]  # (n, true, obs)
    col = joint.sum(axis=1, keepdims=True)  # (n, 1, obs)
    post = np.divide(joint, col, out=np.zeros_like(joint), where=col > 0)
    unexplained = col[:, 0, :] == 0  # labels the prior cannot explain
    est = np.einsum("nto,no->nt", post, obs)
    return est + obs * unexplained  # ...keep their own label


def relabel_frame(
    observed: pd.DataFrame, region_mix: pd.DataFrame, reference: pd.DataFrame, em_iter: int = 20
) -> pd.DataFrame:
    """Correct every country × model × year group (direct where covered, pooled elsewhere).

    Same maths as :func:`relabel`, vectorised per country.
    """
    pts = list(POWERTRAINS)
    pooled = confusion_from_reference(reference)
    own = {c: confusion_from_reference(reference, c) for c in reference["country"].unique()}
    keys = ["country", "model", "year"]
    obs = observed.pivot_table(index=keys, columns="powertrain", values="volume", aggfunc="sum")
    obs = obs.reindex(columns=pts).fillna(0.0)
    pri = region_mix.pivot_table(index=["model", "year"], columns="powertrain", values="share", aggfunc="sum")
    pri = pri.reindex(columns=pts).fillna(0.0)

    parts = []
    for c, o in obs.groupby(level="country"):
        conf = own.get(c, pooled).to_numpy()
        ov = o.to_numpy()
        p = pri.reindex(o.index.droplevel("country")).fillna(0.0).to_numpy()
        est = _bayes_batch(ov, p, conf)
        for _ in range(em_iter):
            tot = est.sum(axis=1, keepdims=True)
            est = _bayes_batch(ov, np.divide(est, tot, out=np.zeros_like(est), where=tot > 0), conf)
        parts.append(pd.DataFrame(est, index=o.index, columns=pts))
    wide = pd.concat(parts)
    wide.columns.name = "powertrain"
    return wide.stack().rename("volume").reset_index()[keys + ["powertrain", "volume"]]
