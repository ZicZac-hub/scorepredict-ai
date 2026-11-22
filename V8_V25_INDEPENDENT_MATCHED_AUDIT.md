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

## Prospective probability comparison

Each new write-once V25 set saves a separate score matrix per fixture under
`v25_probability`. Version `v25-equal-sensitivity-mixture-v1` equally weights
the Poisson, negative-binomial and bivariate Poisson sensitivity matrices on
the original V8 score grid. The components share V8-derived goal rates, and the
weights are fixed before results. Each matrix records the freeze timestamp and
inherits the fixture, source forecast ID, input digest and capture timestamp from
the enclosing leg. The auditor checks the evidence timestamp against the source
and requires the same cutoff as the saved selections.

The matched probability report compares top-one Brier, exact-score multiclass
Brier and exact-score log loss on identical verified legs. It also publishes
**paired V25-minus-V8 metric differences** (negative is better) and V8/V25
five-double hit rates restricted to exactly the same verified, probability-frozen
pairs. Missing older grids produce null differences, **never reconstructed
scores or artificial zero losses**. The matched report lists distinct fixture
identities so overlapping legs remain visible; these are descriptive metrics,
not a statistical superiority claim. The original V8-only
diagnostics still cover the wider archive. Pairs without two valid V25 matrices
can still count toward five-double performance once verified. Historical V25
matrices are never reconstructed. V8 mass on V25's robust five remains a
reference diagnostic, and the lower envelope still determines the five picks.

## Five-double policy comparison

New packets also save the five highest joint probabilities from the normalized
V25 mixture as a third research policy. All three policies use the same
fixtures, selection time and eventual regulation scores. The auditor validates
the saved mixture choices against their frozen matrices and reports wins on
the common cohort. Prior sets without mixture choices still contribute to the
original V8-versus-robust comparison. The mixture's estimated coverage assumes
independent matches. Its future hit rate is measured through the same
verified-results workflow.

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
