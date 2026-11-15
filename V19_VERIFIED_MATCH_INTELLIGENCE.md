# ZicZac V19 — Verified Match Intelligence (market-backed shadow)

**15 November 2026.** V19 is a real implementation of an evidence-gated
forecast challenger, not a claim that more AI models improve exact scores.

## Implemented

1. **Genuine archived 90-minute market model evidence:** Read previously
   captured immutable V9 bookmaker-informed or V10 multi-bookmaker market
   model records and verify the complete exact-score matrix, the 1X2
   de-vigged market vector, bookmaker identity, market update timestamp,
   odds age (at most 12 hours), source input digest, matching historical
   as-of date, two team IDs, provider, fixture ID, scheduled kickoff and
   original archived score distribution. A V10 consensus needs at least
   three distinct named bookmaker sources.
2. **BOTH original matches required:** Read each already-frozen V8 five
   exact-score double and locate **two** matching, authentic market
   forecast records of the *same model type*. Prefer V10 on both matches
   when available, otherwise V9 on both. Do not combine one real quote
   with a fabricated result for the other leg.
3. **Full-distribution five exact-score selection:** When eligible,
   compute the five highest products of the two market-informed exact
   score distributions, rather than arbitrary score diversity or repeated
   0–1 / 1–0 / 1–1 outputs. The top-five cells from each leg suffice
   mathematically to identify the top five joint cells.
4. **Separate write-once original-pair shadow:** Save the V19 experimental
   choice only before BOTH kickoffs, with a 45-minute safety minimum and
   no more than 36-hour advance window. Store both actual market-source
   forecast IDs and timestamps, both original source IDs, the full five
   choices and its modelled coverage. Original V8 picks never change.
5. **Independent full-time verification:** Only count exact-score double
   wins when *both* matches finish and are independently verified against
   the original provider, team IDs, kickoff and 90-minute final scores.
   Verify source integrity again before settlement and compare the V19
   and original V8 picks on exactly the same matches. Pending matches
   are not losses.
6. **Readable missing-evidence status:** V19 publishes a no-cost audit and
   readiness report in the existing scheduled pipeline and displays
   qualified pairs, recorded market snapshots, actual/pending hit rates,
   and missing xG/lineup/AI evidence alongside V17/V18 in Performance
   and Five Doubles.

## Important limitations and operator actions

- **This code does not create data the subscribed APIs do not provide.**
  There is no verified licensed shot-quality xG/xGA feed, trusted
  numerical starting-XI and injury adjustment model, or archived raw
  HTTP bookmaker response available from the current records.
- **The market forecast ledger contains the selected original provider
  metadata and actual model probabilities, NOT the independently
  replayable full raw bookmaker HTTP packet.** V19 explicitly marks
  this provenance limitation and never calls it a fully validated raw
  source.
- The scheduled workflow performs **zero additional paid requests**.
  It reuses the existing API-Football odds capture and never changes
  model-provider subscription, GitHub secrets or per-run query budgets.
- The six configured LLM provider names still must not be represented
  as actual independent successful AI votes. V19's numerical scores
  come from real archived statistical and market probabilities.
- There will often be **no V19 forecast** when the two originally frozen
  matches do not both have correctly timestamped market evidence.
  Abstention is safer than presenting fabricated predictions.
- This market V19 challenger may perform **worse** than V8; it has not
  demonstrated a real independent win-rate improvement. Test many
  future immutable pairs before any promotion.
- Exact-score doubles remain fundamentally harder than high-confidence
  broader match outcomes; 70% five-exact-score set accuracy is not
  established.

## Validation and integration

Code: `v19_match_intelligence.py`
UI: `v19_match_intelligence_ui.py`
Tests: `python -m pytest -q tests/test_v19_match_intelligence.py`
Workflow: `.github/workflows/prematch-evidence.yml`
Output: `data/v19_intelligence/doubles/YYYY-MM-DD.json`,
`data/v19_intelligence/performance.json` and
`data/v19_intelligence/readiness.json`.

Next external dependencies, requiring real provider consent, availability
and possible billing: licensed pre-match xG by fixture/team and timestamp;
source-verified starting-XI/injury snapshots with historically trained
impact coefficients; raw time-stamped bookmaker quote archiving; and
a genuine two-fixture multi-provider AI research capture. None of these
is falsely asserted as available in V19.
