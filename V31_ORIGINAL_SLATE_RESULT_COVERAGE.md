# V31 — Complete original-slate verified-result collection

**22 November 2026. Research-only. No change to live score predictions.**

## Actual evidence motivating V31

- V28 froze genuine full original candidate slates for **22 and 23 November 2026 Sydney**, containing 3 and 210 eligible four-team match pairs.
- At 08:51 UTC on 22 November, V30 still had **0 verified candidate-pair rows**. All 24 frozen original physical fixtures were scheduled to kick off in the future. This is **not** evidence of failure, merely results not yet due.
- The first V28 original fixture starts **11:15 UTC on 22 November**; the result collector's conservative three-hour buffer first makes it eligible at **14:15 UTC**.
- The historical `backfill_verified_results.py` gathered due match IDs from the *latest* V8 forecast pointer. Later updates to that pointer could omit older fixtures that nevertheless remain in a frozen V28 research slate.

## V31 engineering change

The hourly result backfill now independently enrolls the ORIGINAL physical
fixture IDs from **every valid frozen V28 slate**, before considering the
latest-ledger records.

1. Run the same immutable-source, exact candidate-catalog, timestamp and
   five-selection replay implemented by V28's `verify_decision`.
2. Collect unique fixture IDs from every independently validated original
   daily candidate slate. Conflicting identities between archives are excluded.
3. Never issue a result request before kickoff plus **three hours**. A new
   official provider result is valid only if its provider, fixture, home and
   away IDs, original kickoff timestamp, regulation-time score basis and
   finished status match the archived original.
4. Prioritize due enrolled research fixtures ahead of ordinary latest-ledger
   fixtures. All requests share the existing **12 batches × 20 fixture IDs**
   per hourly run. V31 adds **no extra request budget** and never calls an LLM.
5. Preserve any existing recorded results and the entire original score
   forecast archive. Repeated runs are idempotent.
6. If the newest forecast index is unavailable, still retain the original
   verified-slate enrollment so result collection can proceed.
7. Explicitly record provider IDs due, not yet due, already saved, absent
   from the latest pointer, prioritized, deferred by request cap and rejected
   for source/provenance problems.

The existing V29 and V30 independent auditors regrade only AFTER a legitimate
verified provider final is recorded. Pending fixtures are not losses, and
multiple overlapping candidate pairs are not independent experiments.

## Source and grading integrity

`v31_slate_settlement.enrolled_due` does not reconstruct models or backfill
past score predictions. It reads only the pre-kickoff frozen V28 source files
and their original checksum-protected candidate catalogs.

`backfill_verified_results.py` remains responsible for independently
validating provider score and fixture identity, then updating only
`data/prematch/verified_results.json` and its collection status. No score
forecast, source evidence or historical ticket is modified.

## Status and monitoring

The dashboard reads
`data/diagnostics/result_backfill.json` under
`original_slate_settlement` to report:

- original fixtures already in the verified-result file
- fixtures still before the conservative verification time
- due original fixtures without saved results
- IDs selected/deferred by the existing provider-call cap
- validation failures and the next original settlement due time

The verified-results workflow runs hourly at minute 23 UTC. The pre-match
evidence collector has its separate 15-minute cadence; the two workflows
do not increase paid data requests for original slate enrollment.

**Accuracy remains unproven.** A success claim depends on independently
verified future regulation-time results, enough valid full days and
statistically meaningful comparisons. A 70% five-double hit rate has
not been established, and no live model is promoted.
