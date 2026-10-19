# Delivery log — how the method evolved

Short iterations, each shipped and measured before the next. The production tool keeps one
Architecture Decision Record per methodology change; this is the public summary.

## v0 — Baseline
Regional mix copied to every country. Always consistent with the vendor, always wrong for any
single country. Kept as the benchmark every later version must beat.

## v1 — Country tilts + raking
- **Decision:** the vendor forecast is the hard constraint ("iron rule"); the proxy only moves mix
  *between* countries.
- **Why raking:** it satisfies both margins exactly and stays as close as possible to the seed — no
  hand-tuned post-adjustments.
- **Learning:** multi-powertrain models need the tilt; single-powertrain models are exact for free.

## v2 — Label correction
- **Trigger:** backtest residuals concentrated on hybrids in specific countries. Inspection showed
  mild hybrids filed as full hybrids in registration data.
- **Decision:** use a cleaner reference source where it exists (direct correction) and a Bayesian
  relabel elsewhere; exclude a market whose reference data did not match its feed.
- **Result:** the error in the affected cells fell by more than half.

## v3 — Persistence beyond the vendor horizon
- **Trigger:** in the long-range extension, countries jumped back to the regional mix in the first
  extended year — a visible artefact in charts.
- **Decision:** carry country tilts through the extension.

## v4 — Productionisation
- Pipeline stage after reconciliation, quality gate on every run, one-command refresh on a new data
  release, revision report against the previous run.
- Repository hardening: CI, data-leak guards, hooks, clean-up of exploratory scripts into a quarantine
  folder (with approval) rather than deletion.

## v5 — Expert review *(in progress)*
Small markets, new variants and edge cases reviewed with the market experts; overrides to flow
through a forecast editor rather than ad-hoc edits.

## This showcase
- **First run:** label correction *hurt* — 0.6 pp error with raw labels, 0.9 pp after correction. Cause: synthetic
  misfiling was identical in every country, which raking absorbs, and the regional prior pulled
  countries toward the average.
- **Fix:** country-specific misfiling (realistic), partial reference coverage, EM-updated prior.
- **Then:** vectorised the relabelling (60 s → 7 s for the full figure build) with a test pinning
  it to the readable reference implementation.
