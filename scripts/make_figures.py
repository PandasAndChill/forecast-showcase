"""Regenerate every figure in assets/ from the synthetic world.

python scripts/make_figures.py          # writes assets/*.png + assets/backtest.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptproxy import POWERTRAINS  # noqa: E402
from ptproxy.allocate import allocate_region, country_tilts, iron_rule_gap  # noqa: E402
from ptproxy.backtest import METHODS, country_mix, run  # noqa: E402
from ptproxy.labels import relabel_frame  # noqa: E402
from ptproxy.synthetic import REFERENCE_COVERAGE, make_world  # noqa: E402

OUT = ROOT / "assets"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PT_COLOR = dict(zip(POWERTRAINS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], strict=True))
METHOD_COLOR = {METHODS[0]: "#a3a29d", METHODS[1]: "#eb6834", METHODS[2]: "#2a78d6"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def _save(fig, name: str, note: str = "Synthetic data — fictional countries and models.", y: float = -0.03) -> None:
    fig.text(0.01, y, note, fontsize=8, color=INK2)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _stacked_bars(ax, mix: pd.DataFrame, order: list[str]) -> None:
    """100 % horizontal stacked bars with direct % labels (relief for low-contrast hues)."""
    left = np.zeros(len(order))
    for p in POWERTRAINS:
        vals = mix.reindex(order)[p].fillna(0).to_numpy() * 100
        ax.barh(order, vals, left=left, color=PT_COLOR[p], edgecolor=SURFACE, linewidth=2, height=0.7, label=p)
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=INK)
        left += vals
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of registrations (%)")


def fig_problem(w) -> None:
    year = w.last_actual
    t = w.truth[w.truth["year"] == year]
    ctry = country_mix(t).pivot(index="country", columns="powertrain", values="share")
    reg = t.groupby("powertrain")["volume"].sum()
    ctry.loc["REGION AVERAGE"] = reg / reg.sum()
    order = ["REGION AVERAGE"] + ctry.drop("REGION AVERAGE").sort_values("BEV", ascending=False).index.to_list()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    _stacked_bars(ax, ctry, order)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_title(f"The regional average describes no country — powertrain mix, {year}")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    _save(fig, "01_problem_regional_average.png", y=-0.07)


def fig_forecast(w, fixed) -> None:
    fit = [w.last_actual - 2, w.last_actual - 1, w.last_actual]
    fut = list(range(w.last_actual + 1, 2031))
    cv = w.country_volumes[w.country_volumes["year"].isin(fut)]
    rp = w.region_pt[w.region_pt["year"].isin(fut)]
    proxy = country_mix(allocate_region(cv, rp, country_tilts(fixed, w.region_mix, fit)))
    naive = country_mix(allocate_region(cv, rp, None))
    truth = country_mix(w.truth)
    show = ["Nordia", "Batavia", "Iberica", "Carpathia"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, c in zip(axes, show, strict=True):

        def bev(df, c=c):
            s = df[(df["country"] == c) & (df["powertrain"] == "BEV")].set_index("year")["share"] * 100
            return s.sort_index()

        tr = bev(truth)
        ax.axvspan(w.last_actual + 0.5, 2030.5, color="#f1f0ec", zorder=0)
        ax.plot(tr.index, tr.values, color=INK, lw=2, label="Truth (hidden from the model)")
        ax.plot(
            bev(naive).index,
            bev(naive).values,
            color=METHOD_COLOR[METHODS[0]],
            lw=2,
            ls=(0, (2, 2)),
            label="Naive: regional mix",
        )
        ax.plot(
            bev(proxy).index,
            bev(proxy).values,
            color=METHOD_COLOR[METHODS[2]],
            lw=2,
            ls=(0, (5, 2)),
            label="Proxy forecast",
        )
        ax.set_title(c, fontsize=11)
        ax.set_xticks([2019, 2025, 2030])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("BEV share (%)")
    axes[0].text(w.last_actual + 0.8, 93, "forecast", fontsize=8, color=INK2)
    fig.suptitle(
        "Country BEV share: proxy keeps each country's character, naive collapses to the average",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(
        fig,
        "02_forecast_by_country.png",
        "Synthetic data. Forecast origin 2025; the regional forecast is taken as given.",
        y=-0.1,
    )


def fig_backtest(bt: pd.DataFrame) -> None:
    agg = bt.groupby(["horizon", "method"])["error_pp"].mean().unstack()[list(METHODS)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(agg))
    width = 0.26
    for i, m in enumerate(METHODS):
        bars = ax.bar(x + (i - 1) * width, agg[m], width - 0.03, color=METHOD_COLOR[m], label=m)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.08,
                f"{b.get_height():.1f}",
                ha="center",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x, [f"{h} year{'s' if h > 1 else ''} ahead" for h in agg.index])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mean absolute error, country mix (pp)")
    ax.set_ylim(0, agg.to_numpy().max() * 1.35)
    ax.set_title("Rolling-origin backtest: country-mix error from ~4 pp to ~0.7 pp")
    ax.legend(frameon=False, loc="upper right")
    _save(
        fig,
        "03_backtest_error.png",
        f"Synthetic data. Origins {bt.origin.min()}–{bt.origin.max()}, "
        "volume-weighted over 8 countries × 5 powertrains.",
    )


def fig_labels(w, fixed) -> None:
    y = w.last_actual

    def mhev(df):
        m = country_mix(df[df["year"] == y])
        return m[m["powertrain"] == "MHEV"].set_index("country")["share"] * 100

    d = pd.DataFrame({"Filed label": mhev(w.observed), "Corrected": mhev(fixed), "Truth": mhev(w.truth)})
    d = d.sort_values("Truth", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    yy = np.arange(len(d))
    ax.hlines(yy, d.min(axis=1), d.max(axis=1), color=GRID, lw=3, zorder=1)
    ax.scatter(
        d["Filed label"],
        yy,
        s=60,
        color=METHOD_COLOR[METHODS[1]],
        label="Filed label",
        zorder=3,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.scatter(
        d["Corrected"],
        yy,
        s=60,
        color=METHOD_COLOR[METHODS[2]],
        label="Corrected",
        zorder=3,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.scatter(d["Truth"], yy, s=110, facecolor="none", edgecolor=INK, linewidth=1.6, label="Truth", zorder=4)
    labels = [f"{c}  ●" if c in REFERENCE_COVERAGE else c for c in d.index]
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Mild-hybrid share of registrations, {y} (%)")
    ax.set_title("Label correction recovers mild hybrids misfiled as full hybrids")
    ax.legend(frameon=False, loc="lower right")
    _save(
        fig,
        "04_label_correction.png",
        "Synthetic data. ● = covered by the reference source (direct correction); others use the pooled matrix.",
    )


def fig_iron_rule(w, fixed) -> float:
    fit = [w.last_actual - 2, w.last_actual - 1, w.last_actual]
    fut = list(range(w.last_actual + 1, 2031))
    cv = w.country_volumes[w.country_volumes["year"].isin(fut)]
    rp = w.region_pt[w.region_pt["year"].isin(fut)]
    a = allocate_region(cv, rp, country_tilts(fixed, w.region_mix, fit))
    s = a.groupby(["model", "year", "powertrain"])["volume"].sum()
    r = rp.set_index(["model", "year", "powertrain"])["volume"].reindex(s.index)
    keep = r > 1
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(r[keep] / 1e3, s[keep] / 1e3, s=14, color=PT_COLOR["BEV"], alpha=0.7, edgecolor="none")
    lim = [r[keep].min() / 1e3 * 0.7, r.max() / 1e3 * 1.3]
    ax.plot(lim, lim, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Regional forecast (k units)")
    ax.set_ylabel("Sum of allocated countries (k units)")
    gap = iron_rule_gap(a, rp)
    ax.set_title("Iron rule: countries add up exactly")
    ax.text(
        0.03,
        0.95,
        f"{keep.sum()} model × powertrain × year cells\nlargest gap: {gap:.1e} units",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=INK,
    )
    _save(fig, "05_iron_rule.png")
    return gap


def main() -> None:
    OUT.mkdir(exist_ok=True)
    w = make_world()
    fixed = relabel_frame(w.observed, w.region_mix, w.reference)
    fig_problem(w)
    fig_forecast(w, fixed)
    fig_labels(w, fixed)
    fig_iron_rule(w, fixed)
    bt = run(w)
    bt.round(4).to_csv(OUT / "backtest.csv", index=False)
    fig_backtest(bt)
    print(bt.groupby(["method", "horizon"])["error_pp"].mean().unstack().round(2))


if __name__ == "__main__":
    main()
