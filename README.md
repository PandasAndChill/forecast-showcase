# Powertrain Mix Forecaster — country-level EV forecasts for markets nobody forecasts

[![CI](https://github.com/PandasAndChill/forecast-showcase/actions/workflows/ci.yml/badge.svg)](https://github.com/PandasAndChill/forecast-showcase/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Data](https://img.shields.io/badge/data-100%25%20synthetic-lightgrey)

> Public showcase of a forecasting tool I designed and built for the market-intelligence team of a
> European car maker. The production tool runs on licensed data that cannot leave the company; this
> repository reproduces **the method, the code structure and the validation** on a synthetic market,
> so every number and chart below can be regenerated with one command.

| | |
|---|---|
| **Problem** | Vendors forecast the BEV / PHEV / HEV / MHEV / ICE mix for the big markets and for the smaller ones only *as one block*. Planning needs each country. |
| **Solution** | Split the block forecast into countries using each country's own registration behaviour — while guaranteeing the countries add back up to the vendor forecast *exactly*. Clean mislabelled registrations first. |
| **Result** | On a rolling backtest, country-mix error falls from **~4 pp** (copy the regional average) to **~0.7 pp**, 1–3 years ahead. The tool now runs as a stage of the team's forecast build. |

---

## The product: Forecast Studio

The method lives inside **Forecast Studio**, a Mac app I built on top of the production engine
so the forecast can be explored, challenged and changed live in a meeting — not emailed as a
spreadsheet. Shown here on synthetic data.

![Forecast Studio guided tour](assets/app/studio_tour.gif)

| | |
|---|---|
| **Answer first** | Each view opens with one headline sentence generated from the data (“Battery-electric becomes Europe's largest powertrain in 2027 — 51% of new cars by 2030”), then the evidence. |
| **What-if, safely** | Set a target (“BEV 75% of German C-SUV by 2030”) and see the consequence for the segment, the country and Europe *before* applying. Totals always reconcile, locked powertrains never move, impossible edits are refused with the reason, and every change is logged and undoable. |
| **Explore** | Any cut of the forecast — country, segment, brand, down to the model generation code — from a bank of ready-made analyses or a chart builder. Every chart writes its own one-line insight. |
| **Report → PowerPoint** | Collect charts into a report and export a deck with **native, editable PowerPoint charts**, numbers recomputed from the forecast at export time. |
| **Ask (local AI)** | Plain-language questions answered with a chart by a model running **on the laptop** (Ollama / LM Studio). The model never sees the data: it can only call the analysis engine through three tools, so every number in its answer is computed, not generated. Without a model, common questions are answered by rules. |
| **Compare** | Scenario vs base — and release vs release: which markets a new vendor forecast upgraded or downgraded. |
| **Smaller markets** | The method on this page, run on the real data: coverage, each country's mix against the block, and the iron-rule check. |
| **Demo mode** | One key blurs every number on screen (KPIs, tables, tooltips, axes, numbers in sentences) while shapes stay visible — on by default for confidential data. |
| **Present** | A 12-step guided tour with talking points, for walking a stakeholder through it. |
| **⌘K** | One search box to jump to any view, run any analysis, or send a question to Ask — no menus to hunt through during a call. |

<table><tr>
<td><img src="assets/app/studio_explore.png" alt="Explore"><br><sub>Explore: chart bank and builder, down to model generation</sub></td>
<td><img src="assets/app/studio_ask.png" alt="Ask the forecast"><br><sub>Ask: local AI answers with computed charts</sub></td>
</tr><tr>
<td><img src="assets/app/studio_report.png" alt="Report"><br><sub>Report: one click to an editable PowerPoint</sub></td>
<td><img src="assets/app/studio_editor.png" alt="Scenario editor with impact preview"><br><sub>Editor: the impact of an edit before it is applied</sub></td>
</tr><tr>
<td><img src="assets/app/studio_demo_mode.png" alt="Demo mode"><br><sub>Demo mode: numbers hidden, shapes kept</sub></td>
<td><img src="assets/app/studio_compare.png" alt="Compare scenarios"><br><sub>Compare: what a scenario changes</sub></td>
</tr><tr>
<td><img src="assets/app/studio_palette.png" alt="Command palette"><br><sub>⌘K: jump, analyse or ask from one box</sub></td>
<td><img src="assets/app/studio_overview_dark.png" alt="Dark mode"><br><sub>Follows the system light / dark setting</sub></td>
</tr></table>

**Under the hood.** Each forecast release is packaged into a compact, documented **data store**
(~7 MB for 35 markets: market and model level, proxy detail, vendor reference mix, lineage
manifest) with build-time checks — model vs market totals, no negatives, and the iron rule
(largest gap on the real data: 0.001 pts). One analysis engine serves the app, the reports, the
PowerPoint export and the AI, so a number is computed one way wherever it appears.

**Built with** Python, pandas, FastAPI, python-pptx, a local LLM via the OpenAI-compatible API, plain HTML/CSS/JS and ECharts. No build step, it runs
offline, and it launches with one click from the Desktop. Every number comes from the forecast
engine's service API, never from the interface. The app's code lives in the private repository
with the engine; this repository reproduces the method.

## 1. The business problem

The regional block hides very different markets. Copying its average onto each country — the
default when no better data exists — is wrong for every one of them:

![Regional average versus country mixes](assets/01_problem_regional_average.png)

Two constraints shape the solution:

- **The vendor forecast is the reference.** Whatever the split, countries must sum back to it for
  every model, powertrain and year. Anything else breaks reconciliation with the rest of the
  forecast and loses the business's trust instantly.
- **Registration data lies a little.** Feeds misfile neighbouring technologies (mild hybrids filed
  as full hybrids is the classic), and the error rate differs by country.

## 2. Approach

```mermaid
flowchart LR
    A["Registrations<br/>(country × model × label)"] --> B["Label correction<br/>Bayes + EM, direct where<br/>reference data exists"]
    R["Reference sample<br/>(true vs filed label)"] --> B
    B --> C["Country tilts<br/>log-ratio vs region,<br/>shrunk for thin data"]
    V["Regional powertrain<br/>forecast (vendor)"] --> D
    S["Country model volumes<br/>(no powertrain split)"] --> D
    C --> D["Raking / IPF<br/>rows = country volumes<br/>cols = regional forecast"]
    D --> E["Country × model ×<br/>powertrain forecast"]
    E --> G{{"Gate: iron rule,<br/>totals, no negatives"}}
```

1. **Correct the labels** — a cleaner reference source gives `P(filed label | true powertrain)`.
   Covered countries use their own matrix, others the pooled one. Bayes' rule, with a prior
   re-estimated by EM on each country's own data, moves misfiled volume back. Volume is conserved.
2. **Measure each country's character** — the log-ratio of its mix to the regional mix, per model,
   shrunk toward the country's overall tilt when the model is thin in that market.
3. **Allocate with a hard constraint** — seed = regional mix × country tilt, then rake (iterative
   proportional fitting) until rows match country volumes and columns match the regional forecast.
   Single-powertrain models are exact by construction.

Details and design choices: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## 3. Results (synthetic market)

**Backtest.** From each origin 2021–2025, tilts are fitted on the three preceding years only, then
scored 1–3 years ahead against the hidden truth:

![Backtest error](assets/03_backtest_error.png)

The proxy keeps each country's trajectory where the naive split collapses everything to the average:

![BEV share by country](assets/02_forecast_by_country.png)

Label correction works best where the reference source covers the country (●). Elsewhere the
pooled matrix is a reasonable but imperfect guess — over-correcting clean feeds and
under-correcting the worst one — which is why widening reference coverage is the next step:

![Label correction](assets/04_label_correction.png)

And the constraint the business cares about most holds to floating-point precision:

<img src="assets/05_iron_rule.png" width="420" alt="Iron rule check">

## 4. From analysis to a production tool

The proxy is one stage of a larger forecasting pipeline I built and maintain. What runs in
production (described here, not included — it is tied to licensed data):

| Area | What it looks like |
|---|---|
| **Pipeline** | 11-stage reproducible build: ingestion adapters per source → validation → canonical long-form model → forecast → reconciliation → the proxy stage → Excel/report outputs with a revision report. One command to refresh on a new data release. |
| **Quality gates** | Every run checks the iron rule, country and regional totals, negative volumes and release consistency, and stops the build on failure. |
| **Testing** | ~900 automated tests on synthetic fixtures; a rolling-origin backtest re-run on each methodology change. |
| **Engineering hygiene** | Private GitHub repo with CI (lint, tests, data-leak guards), pre-commit and pre-push hooks that block proprietary files, configuration-driven paths, Architecture Decision Records for every method change. |
| **Interface** | Forecast Studio (above): FastAPI over the engine's service layer; the interface holds no business arithmetic, so it was swapped from an earlier Streamlit prototype without touching a calculation. |
| **Confidentiality by design** | Code and documentation are versioned; data, mappings and outputs never are. This public repo is a direct product of that separation. |

## 5. How it was built — iteratively

The method was not designed upfront; each version shipped, was measured, and the measurement drove
the next step.

| Iteration | What changed | Why |
|---|---|---|
| v0 | Regional mix copied to every country | Baseline: the default when no country detail exists. |
| v1 | Country tilts + raking with the iron rule | Countries differ a lot; the vendor total must hold. |
| v2 | Label correction (direct + Bayes) | Backtest residuals traced to misfiled mild hybrids. |
| v3 | Tilts persist beyond the vendor horizon | Long-range extension was snapping countries back to the average. |
| v4 | Integrated as a pipeline stage with gates and a refresh script | From analysis to something the team runs every release. |
| v5 | Forecast Studio: scenarios, previewed impact, demo mode, guided tour | The forecast gets used when people can question it live. |
| v6 | Data store per release, Explore, reports to PowerPoint, release comparison, local AI | From a forecast to an analysis product; questions answered in plain words, privately. |
| v7 *(in progress)* | Review loop with the market experts | Edge cases (tiny markets, new variants) belong to people who know them. |

This showcase followed the same loop. The first synthetic test showed label correction making
things *worse*: misfiling was identical in every country, so raking absorbed it, and the regional
prior pulled countries toward the average. I made misfiling country-specific (as it is in reality)
and switched to an EM-updated prior — the result above. See [docs/DELIVERY.md](docs/DELIVERY.md).

## 6. Run it yourself

```bash
git clone https://github.com/PandasAndChill/forecast-showcase && cd forecast-showcase
pip install -e ".[dev]"
python scripts/make_figures.py   # regenerates assets/ and assets/backtest.csv (~10 s)
pytest -q                        # business-rule tests
```

```text
src/ptproxy/
  allocate.py    country tilts + raking under the iron rule
  labels.py      confusion matrices, Bayes/EM relabelling (vectorised)
  synthetic.py   fictional 8-country region, 14 models, 2019–2030
  backtest.py    rolling-origin evaluation, three methods
scripts/make_figures.py
tests/test_proxy.py
docs/            methodology, delivery log
```

## 7. What is not here

No real data, vendor names, mappings, forecasts or production code. Countries (Nordia, Iberica, …)
and models are fictional; the synthetic market is built to reproduce the *structure* of the real
problem (strong country preferences, country-specific misfiling, partial reference coverage), not
its numbers.

## 8. Related projects

- **[VoxTube](https://github.com/PandasAndChill/voxtube-showcase)** — reads a product's public
  reception on YouTube, across markets and languages, and turns comments into insights a marketing
  team can use.
- **[PlaceSignal](https://github.com/PandasAndChill/placesignal-showcase)** — puts ratings and reviews
  on a map, finds what drives them, and shows the reviews behind every number.

---

**Petr Shovoldaev** — Lead Consumer & Market Analyst, Paris. Market intelligence × Python
automation: I build the tools my analyses run on.
