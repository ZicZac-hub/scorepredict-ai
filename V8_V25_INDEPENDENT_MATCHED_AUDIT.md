# V8 versus V25: independently matched prospective audit

**22 November 2026 (Australia/Sydney). Research only; not a model promotion.**

## One common denominator

Run: `python v8_v25_matched_audit.py`

Every counted opportunity must be an **original, saved V25 shadow trial** under
`data/pro_analyst_shadow/sets/YYYY-MM-DD.json`. Both methods are graded on
exactly those two fixtures and exactly the same independently verified final
scores. This is **not** a comparison of V25's chosen fixture pairs against
V8's different original automatically chosen daily pairs.

The auditor checks:

- An unaltered V25 frozen research decision and five distinct saved selections
  per method, made at least 45 minutes before both kickoffs.
- The **exact archived V8/V8-near forecast IDs** for both legs, including digest,
  model, team, fixture, provider, and kickoff identities; the original grids
  must be valid probability distributions with sufficient historical coverage.
- The original V8 baseline picks must actually be the highest-five joint
  outcomes on those two original grids; saved model-mass annotations must match.
- Both fixtures fall on the declared Sydney date, contain four distinct teams,
  and have source evidence captured before selection and kickoff.
- Results must have verified regulation-time scores with matching provider,
  fixture, home and away teams, and kickoff. Conflicting or missing results
  remain **pending** and never count as misses.
- The report computes matched five-double hit rates, Wilson intervals, paired
  discordances and an exact small-sample McNemar comparison.
- V8 per-leg probability diagnostics include top-one score calibration,
  full-multiclass score Brier and actual-score log loss.

## Critical calibration limitation

V25's saved robust selections are formed by a **lower envelope across four
sensitivity models using V8-based goal rates**. That envelope is neither a
proper normalized score matrix nor an independently calibrated distribution.

Consequently, **V25 full exact-score probability calibration is reported as
unavailable**. The report shows the probability *V8* assigned to the V25
selections strictly as a common-reference diagnostic; it is not a V25 forecast
probability, a valid estimate of real win probability, or a proxy for V25's
log loss. Do not present it as evidence that V25 is calibrated.

Future V25 revisions should freeze a separately versioned, normalized
distribution per fixture **before kickoff**, with provenance and source
timestamps, to permit truly symmetric V8/V25 score-distribution Brier and
log-loss comparisons. Never construct a historical probability after
settlement and relabel it prospective.

## Automation and tests

`tests/test_v8_v25_matched_audit.py` rejects post-match selections,
tampered source digests, conflicting result sources, altered V8 baseline choices
and incorrect identities. The `prediction-tests.yml` workflow runs the
real archived comparison with a positive source-coverage gate; the hourly
verified-results backfill updates `data/accuracy_diagnosis/v8_v25_matched.json`
only when evidence changes.

**No API calls, no rewriting archived picks, no production-prediction model
change.** The 70% hit-rate target remains unproven. A handful of completed
pairs cannot establish that V25 outperforms V8 or vice versa. The saved JSON
capture timestamps are validated but should not be mistaken for an
independent cryptographic notarization of archival timing.
