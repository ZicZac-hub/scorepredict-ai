# Cross-competition strength research v1

Purpose: test whether domestic league-relative attack and defence information can improve a sparse UEFA score distribution without lowering V8's five-match requirement or treating unavailable European history as fabricated data.

## Fixed research hypothesis

- API-Football team identifiers only; current-season participation in exactly one supported domestic league is required.
- Existing history cache only, no extra requests. Source observation must be known, not stale, and at or before the decision cutoff. Existing history policy excludes unavailable results and conflicting duplicates.
- At least 50 domestic league results, eight team results, and three relevant venue results per team. Both teams must qualify.
- Six regularized opponent-adjustment iterations, 180-day half-life and prior of eight. A bounded 0.85–1.15 goal tilt modifies the existing European control grid without changing its support.
- Domestic relative strength is not absolute strength across countries. The method does not estimate league quality or UEFA coefficients, nor does it claim an accuracy gain.

## Next validation gate

This is an offline prototype and synthetic-test suite, not a scheduled forecast collector. Before prospective use, acquire authentic source-stamped domestic and European history at the same cutoff, register source and implementation hashes, and freeze both control and challenger on new eligible fixtures before kickoff. Then evaluate independently verified regulation results with identical fixtures, proper exact-score log loss, outcome Brier, and paired uncertainty. Report missing/ambiguous contexts and coverage. Do not use earlier completed fixtures as prospective results, change existing V37/V41/V42 archives, or automatically promote. A five-double improvement requires a separate prospective matched-pair trial.

Run: `python -m pytest -q tests/test_cross_competition_strength.py`. No real provider calls or paid AI requests are needed for these synthetic tests.
