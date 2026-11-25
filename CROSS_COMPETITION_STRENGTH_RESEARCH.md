# Cross-competition strength research v1

Purpose: test whether domestic league-relative attack and defence information can improve a sparse UEFA score distribution without lowering V8's five-match requirement or treating unavailable European history as fabricated data.

## Fixed research hypothesis

- API-Football team identifiers only; current-season participation in exactly one supported domestic league is required.
- Existing history cache only, no extra requests. Source observation must be known, not stale, and at or before the decision cutoff. Existing history policy excludes unavailable results and conflicting duplicates.
- At least 50 domestic league results, eight team results, and three relevant venue results per team. Both teams must qualify.
- Six regularized opponent-adjustment iterations, 180-day half-life and prior of eight. A bounded 0.85–1.15 goal tilt modifies the existing European control grid without changing its support.
- Domestic relative strength is not absolute strength across countries. The method does not estimate league quality or UEFA coefficients, nor does it claim an accuracy gain.

## Next validation gate

The proposed scheduled integration uses the existing pre-match cache, with no extra provider calls. It preserves content-addressed source packets and implementation-bound write-once controls/challengers before kickoff. The read-only grader reproduces both score grids and checks independently verified regulation results; it reports matched exact-score hits and log loss. Outcome Brier, paired uncertainty and prospective five-double trials remain follow-up work. Actual source coverage and live scheduled execution have not yet been verified. Report missing/ambiguous contexts and coverage. Do not use earlier completed fixtures as prospective results, change existing V37/V41/V42 archives, or automatically promote. A five-double improvement requires a separate prospective matched-pair trial.

Run: `python -m pytest -q tests/test_cross_competition_strength.py`. No real provider calls or paid AI requests are needed for these synthetic tests.
