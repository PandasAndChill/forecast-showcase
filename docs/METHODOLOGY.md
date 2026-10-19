# Methodology

## Notation

| Symbol | Meaning |
|---|---|
| *c*, *k*, *p*, *t* | country, model, powertrain, year |
| *V<sub>c,k,t</sub>* | country model volume (no powertrain split) — input |
| *R<sub>k,p,t</sub>* | regional powertrain volume — vendor forecast, input |
| *r<sub>k,p,t</sub>* | regional mix, *R* / Σ<sub>p</sub>*R* |
| *X<sub>c,k,p,t</sub>* | output |

**Iron rule:** Σ<sub>c</sub> *X<sub>c,k,p,t</sub>* = *R<sub>k,p,t</sub>* for every *k, p, t*.
**Row rule:** Σ<sub>p</sub> *X<sub>c,k,p,t</sub>* = *V<sub>c,k,t</sub>*.

If country volumes and the regional total disagree by more than 0.5 % for a model-year, the inputs
come from different releases and the run stops — it does not silently rescale.

## Step 1 — Label correction

Registration feeds misfile neighbouring technologies, at country-specific rates.

1. From a reference sample (vehicles seen in both a clean source and the feed), estimate
   *C<sub>c</sub>*(o | p) = P(filed label o | true powertrain p) for each covered country, plus a
   pooled matrix for the rest.
2. For each country × model × year, invert with Bayes' rule:
   P(p | o) ∝ *C*(o | p) · π(p).
3. The prior π starts at the regional mix and is re-estimated by EM on the group's own corrected
   volumes (20 iterations). A regional prior alone pulls every country toward the average — the
   exact bias the proxy exists to remove.

Volume is conserved per group. Labels the prior cannot explain keep their filed label.

**Known limitation:** for uncovered countries the pooled matrix assumes average misfiling.
Countries with clean feeds get over-corrected, the worst offenders under-corrected
(see figure 4). Widening reference coverage is the main lever.

## Step 2 — Country tilts

τ<sub>c,k,p</sub> = log((O<sub>c,k,p</sub> + 1) / (E<sub>c,k,p</sub> + 1)), pooled over the last three
actual years, where *O* is corrected registrations and *E* the volume the country would have had
at the regional mix.

Model-level tilts are shrunk toward the country's all-model tilt:
τ̂ = w·τ<sub>model</sub> + (1 − w)·τ<sub>country</sub>, with w = n / (n + 2 000) and *n* the
country-model volume. A model sold in hundreds of units in a market does not get to define that
market's preferences.

Tilts are held constant over the horizon. In production the same tilt carries into the long-range
extension beyond the vendor horizon, so countries do not snap back to the regional mix in the
first extended year.

## Step 3 — Allocation by raking

Seed: *S<sub>c,p</sub>* = *V<sub>c</sub>* · *r<sub>p</sub>* · exp(τ̂<sub>c,p</sub>), per model-year.

Iterative proportional fitting alternately rescales rows to *V* and columns to *R* until the
largest row gap is below 10⁻⁶ units (typically < 20 iterations). Properties:

- both margins hold — the iron rule is exact, not approximate;
- the result is the matrix closest to the seed (minimum KL divergence) that satisfies both margins,
  so country tilts are preserved as far as the vendor forecast allows;
- powertrains a model does not offer stay at zero; single-powertrain models are exact;
- no negatives can appear.

The naive baseline is the seed with τ = 0: every country inherits the regional mix (this also
satisfies the iron rule, trivially).

## Evaluation

Rolling origin *T* ∈ {2021 … 2025}; tilts fitted on *T*−2 … *T*; forecasts scored at *T*+1 … *T*+3.
Metric: mean absolute error of the **country-level** mix (all models), in percentage points,
weighted by country volume. The regional forecast is taken as known, which isolates the part this
module controls — the split between countries.

| Method | 1 yr | 2 yr | 3 yr |
|---|---|---|---|
| Naive (regional mix) | 4.3 | 4.2 | 3.9 |
| Proxy, raw labels | 1.4 | 1.4 | 1.3 |
| Proxy, corrected labels | **0.7** | **0.7** | **0.7** |

Synthetic market; regenerate with `python scripts/make_figures.py` (raw rows in `assets/backtest.csv`).

## Why not …

- **…a regression on country covariates (income, charging points, incentives)?** Covariates forecast *levels*, but the vendor already forecasts the regional level; what is missing
  is the *split*, which the country's own revealed behaviour captures directly, with far fewer
  assumptions and inputs to maintain.
- **…scaling each country's last actual mix?** It ignores the vendor's view of how each model's mix
  evolves and breaks the iron rule.
- **…a full hierarchical Bayesian model?** Heavier to run and explain for a modest gain at this
  data size; shrinkage gives most of the benefit transparently. A candidate if more markets are added.
