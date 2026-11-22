# V26: Chronological score-tail calibration (research-only)

**Date:** 22 November 2026. **Live prediction model: unchanged.**

## Why V26 exists
V8 already includes Poisson/Dixon–Coles and related signals, while V25 compares
fixed sensitivity transforms on the *same* underlying goal estimates. None has
proved 70% five-double accuracy. The independently saved V8 audit reported
roughly 34.4% predicted 4+ total goals versus 28.8% observed on 59 verified
fixtures, which is only a tentative calibration concern.

This update tests a specific, *predeclared* adjustment instead of promising
a win: divide the complete exact-score grid into 0–1, 2–3, and 4+ total-goal
bins. On at least **30** earlier, independently verified, originally frozen
V8 fixtures, compare each bin's observed share with its archived forecasted
share. Shrink the residual toward no adjustment with an **80-game prior**,
clip each bin multiplier between 0.8 and 1.2, apply a fixed 0.65 correction
power and renormalize the complete grid. No outcome-dependent parameters
are optimized after checking the target game.

## Strict evidence and comparisons
1. For chronological historical testing, a match can influence a later
   prediction ONLY if its verified-final timestamp is at or before the
   target's *original forecast capture*, and its original V8 score matrix
   was saved before kickoff. Exact fixture/team identity is required.
2. The historical replay explicitly remains **retrospective model-development
   evidence**, even though each test uses prior results chronologically.
   It cannot establish a prospective V26 gain on its own.
3. A new V26 research packet is saved only when an **original V25 pair was
   frozen within the previous 20 minutes**; older V25 archives are never
   retrospectively assigned V26 probability grids.
4. Both V26 grids use exactly the same source V8 forecasts, V25 fixture pair
   and prior verified evidence cutoff. The training IDs, source digest,
   verification timestamps and fitted correction are saved write-once before
   either kickoff.
5. An independent result grader reproduces both V26 matrices from the saved
   original inputs and verifies the five optimal score combinations. Invalid,
   tampered or unmatched records are excluded. Pending games stay pending.
6. V26, V25 and V8 five-double hits are compared on the **same verified
   physical match pairs**, alongside exact-score log loss, full multiclass
   Brier score, and calibrated goal-tail metrics.

## Operations
- `python v26_temporal_score.py` writes new future shadow forecasts only
  in the narrow source-frozen window; it also updates the read-only study.
- `python v26_temporal_score.py --verify-only` recalculates only verified
  scores and the chronological study, without issuing new predictions.
- Pre-match GitHub scheduled collection runs the first command after the
  original V25 write-once freeze; independently verified result backfill
  uses the second command.
- Latest study JSON is `data/v26_temporal_calibration/status.json`.
  New write-once sets live in `data/v26_temporal_calibration/sets/`.

The V26 panel on the website labels historical and genuine prospective
performance separately and never claims calibrated probability is actual
accuracy. No new sports or LLM API calls, no changes to the production
model, no retrospective edits to V8/V25 picks.

## Research gate
Even a lower historical log loss does not justify a production promotion.
Future verified same-pair samples must show repeatable lower Brier/log
loss and a materially better five-double hit rate, with uncertainty bounds,
before any upgrade. The aspiration of 70% remains unproven.
