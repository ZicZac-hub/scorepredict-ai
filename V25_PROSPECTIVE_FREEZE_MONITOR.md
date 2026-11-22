# V25 prospective probability-freeze integrity monitor

**22 November 2026. Research only; production predictions remain V8-backed.**

## Why this exists

The historical original V25 shadow sets for 19–23 November contain five saved
score combinations but **do not contain their own frozen V25 per-fixture
probability grids**. This means their observed set wins and losses can be counted
on their original same-pair fixtures, but their probability calibration cannot.
Recreating V25 probability grids after learning match results would be hindsight.

New V25 write-once predictions created after the fixed rollout boundary
**2026-11-22 05:17:27 UTC** must carry:
- both independently reproducible normalized V25 per-match score matrices,
- exact source forecast identity, original evidence timestamps and input digest,
- five mixture-ranked joint outcomes frozen at the exact original selection time,
- archived V8 top-five and robust V25 five picks on the **same fixture pair**.

No old file is rewritten or enriched.

## Automated checks

The existing independent audit, `v8_v25_matched_audit.py`, checks the source
originals, original probabilities, frozen V25 normalization and reproducibility,
mixture top-five and independently matched regulation-time scores.

New `v25_prospective_monitor.py` then summarizes:
- legacy selection-only sets (not backfilled);
- total new prospective sets, valid frozen grids and mixture choices;
- pending new pairs (NOT defeats), verified new pairs and evaluation coverage;
- source archive exclusions and missing or invalid freeze evidence;
- an **initial** review threshold of 100 verified matched pairs, not an
  accuracy or statistical-significance guarantee.

If an after-rollout pair lacks both valid V25 probability matrices or the
frozen mixture policy, the monitor fails. Missing verified final scores remain
pending; a historical missing grid remains unavailable.

## Collection and publishing

- Every scheduled pre-match capture recomputes the independent V8–V25 audit and
  monitor **after** the V25 freeze and **before** publishing research archives.
- The permanent-data commit preflight refuses to publish an invalid newly
  frozen V25 packet, even when the workflow is recovering from another failure.
- The existing hourly verified-final backfill recomputes both audit and
  monitor before committing verified results.
- GitHub PR regression runs revalidate all original frozen source identities
  and checks the monitor's prospective integrity gate.
- The website research panel displays the number of frozen new V25 sets,
  completed pairs, pending pairs, and legacy non-backfilled selections.

No additional football-data API requests or LLM tokens are used. The
independent audit is read-only and makes no retrospective V25 forecasts.

## What counts as progress

The next genuinely saved V25 set must pass all source checks before it can
enter the probability-eligible cohort. After both games finish and their
regulation scores are independently verified, compare V25 versus V8 exact-score
log loss and multinomial Brier on the **same two fixtures**. Continue monitoring
hit rates of the five exact-score combinations, calibration, fixture overlap and
source staleness. A few matched outcomes cannot justify model promotion or a
70% five-double win-rate claim.

Review `data/accuracy_diagnosis/v25_freeze_monitor.json` for the latest saved
quality report and `data/accuracy_diagnosis/v8_v25_matched.json` for its
independently reproduced source-linked underlying evaluations.
