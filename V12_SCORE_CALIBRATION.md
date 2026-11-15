# ZicZac Version 12 — League Score Calibration (Experimental)

**Released 15 November 2026.** The original v6/v7/v8 scoring engines, automatic
five-double selections, existing prediction archives and verified results have
not been modified. Version 12 is a separate challenger, NOT an accuracy claim.

## What changed

- **Deep Research:** Select *V12 Score Calibration · league evidence
  (experimental)*, choose an upcoming match, then Run Deep Research AI.
  The page displays its exact-score distribution and original v8 comparison.
- **Evidence gate:** V12 requires at least **80** genuinely completed,
  provider- and competition-matched league results from the previous **730
  days**, available before the prediction timestamp. It abstains if not met.
- **Conservative goal-score adjustment:** Calculate recent league-wide home
  and away means, create a Poisson reference grid, compare observed historical
  exact-score frequencies with that reference, shrink the correction with
  **120 pseudo-match equivalents**, cap score multipliers to 0.75–1.25, and
  apply a square-root adjustment to the original full v8 grid. Normalize to
  total probability 1. This does **not** force higher scores or remove 1–1.
- **Write-once prospective evidence:** The regular pre-match collector can
  save a distinct **v12** probability matrix alongside a newly saved **v8**
  prediction using identical verified evidence and input digest, without
  additional provider calls. V8 is never overwritten. There is no backfill
  after results or retroactive rewriting of predictions.
- **Performance:** v10 performance reporting also counts v12 frozen,
  pending and verified forecasts, proper score log loss, 1X2 Brier, exact
  hits and fair same-fixture v8 vs v12 hit comparisons.
- **Independent historical benchmark:**
  `python v12_calibration_benchmark.py work/verified_history.json --year 2026`
  uses local normalized completed match records, forecasts one hour before
  kickoffs and only uses earlier available history. Outputs
  `work/v12_calibration_holdout.json`. No network calls are made.

## Methodology caution

This is a **league-environment residual** correction against a simple
independent-Poisson league reference, **not** an already trained,
independently calibrated per-fixture v8 residual function. The league may
contain different team strengths; the adjustment may improve or worsen
probability quality. A more diverse modal-score distribution by itself is
not evidence of improved accuracy.

The first reported sample in
`data/v10_prospective/performance.json` had only two verified v8
fixtures and one verified five-double pair before this rollout. Those
historical results do not establish any V12 improvement.

## No automatic promotion

Do NOT make V12 the default model merely because it shows a different score,
lower 1–1 frequency, or an impressive retrospective best case. Require a
sufficient sample of **prospectively frozen same-evidence matches** (at least
200 as an initial reporting threshold, not a guarantee), lower held-out score
log loss, competitive 1X2 Brier, and a stable paired exact-hit comparison.
Analyse leagues separately and check uncertainty rather than choosing the
best result after seeing final scores.

No external xG, injury, odds, lineup, player strength, premium AI service or
proprietary model is claimed by this release. Existing external provider
research is not reweighted. No odds are fetched for V12.

## Files and validation

- `score_calibration_v12.py`: deterministic statistical challenger
- `research_ui.py`: on-demand experimental Deep Research option
- `v10_prospective.py`: separate immutable research forecast snapshots
- `v10_evaluate.py`, `v10_status_ui.py`, `capture_prematch.py`:
  verification and collector diagnostics
- `v12_calibration_benchmark.py`: chronological read-only replay
- `tests/test_score_calibration_v12.py`,
  `tests/test_v12_prospective.py`: no-leakage, probability integrity,
  original-preservation and paired-verification checks

Streamlit deployment of the latest GitHub commit must be checked separately.
Passed GitHub Actions tests validate code, not a live hosted session.
