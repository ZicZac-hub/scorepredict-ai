# V27 — Fully prospective, audited smart match-pair selection

**Created 22 November 2026. Research-only; production predictions unchanged.**

## What V27 changes

V25 selects one same-Sydney-day match pair by the highest robustness
lower-envelope score, calculated from four mathematical sensitivities of
the original V8 goal forecasts. V26 tests a separate correction to goal
probabilities using earlier independently verified results (if at least
30 prior qualifying fixtures exist). Neither is established as more
accurate than the original model.

**V27 changes pair selection**, not score-model probabilities. It evaluates
the COMPLETE eligible pair slate available at the original V25 daily decision
time and ranks it using a fixed, prespecified criterion:

`rank = (0.60 * original_V8_optimal_five_mass +
         0.40 * V25_robust_five_mass_evaluated_under_original_V8)
        * bounded_original_source_freshness_multiplier`

- Original source age 72 hours or less gets multiplier 1.
- Older frozen sources are downweighted linearly to a minimum multiplier
  of 0.75 at 14 days. Older than 14 days is ineligible.
- Requires 45+ original historical fixtures per modeled match, full
  normalized source-saved score grids, four distinct teams, genuine
  pre-match evidence cutoff and both matches on the same Sydney day.
- Research-only abstention if fewer than two different valid pairs,
  model-estimated optimal five-score mass less than 5.5%, or more than
  2,500 pairs (never silently discard candidates).
- The score is an uncalibrated *ranking heuristic*, NOT a true winning
  probability or a guaranteed improvement.

## Freeze / read-only grading protocol

1. Only process an original V25 set selected within the last 20 minutes.
   NEVER add a V27 selection retrospectively to an old V25 day.
2. Rank all eligible matches **as of V25's original decision time**. Store
   every available original forecast receipt and complete four-team pair
   catalog, source digests and a deterministic catalog checksum.
3. Save the decision or abstention **once**, before kickoff. If the chosen
   pair cannot remain at least 45 minutes from kickoff at actual V27 issue
   time, record a permanent abstention instead.
4. Recompute EVERY original candidate from the exact archived V8 source
   identity and score grid when assessing a saved decision. Tampered
   selection, incomplete catalogs and changed originals do not count.
5. Grade exact five-score double selections using two independently verified
   regulation-time scores. A result is pending until both matches finish
   and their identities match.
6. Compare V27's robust five against its own V8-optimal five **on the same
   chosen V27 match pair**; separately compare V27's chosen pair with
   V25's originally frozen choice **on the same day**. They may be different
   matches: this is a **same-day selection policy comparison**, not
   probability calibration on identical fixtures.
7. V26 calibration is tracked separately on the original V25 pair when
   there are enough previous verified results. Do not combine the V26
   denominator with V27's different selected fixtures.

## Runs / outputs

- `python v27_pair_selection.py` freezes genuinely new V27 choices and
  rechecks results (no extra provider or LLM requests).
- `python v27_pair_selection.py --verify-only` can only regrade existing
  decisions, without issuing new predictions.
- Freeze files: `data/v27_pair_selection/decisions/YYYY-MM-DD.json`
- Verified research status: `data/v27_pair_selection/status.json`
- Pre-match capture and hourly result verification workflows run the
  appropriate modes automatically.

A live prediction model promotion is blocked. Any claim of 70% observed
five-double accuracy requires sufficiently many independent future
pre-match frozen, same-cohort verified results and uncertainty estimates.
Historical or model-estimated coverage is NOT actual hit rate.
