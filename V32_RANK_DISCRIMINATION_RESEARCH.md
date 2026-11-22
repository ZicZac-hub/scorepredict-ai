# V32 — Is V30's original pair ranking actually useful?

Research-only. Scores and original forecasts stay immutable.

## Evidence as of 22 November 2026, 09:09 UTC

Original V28 slates: Sydney 22 November (3 candidate pairs, 3 fixtures) and
23 November (210 candidate pairs, 21 fixtures). All 213 pair rows are pending
independently checked regulation-time final scores; **zero complete days**
at the last stored V30 check. This is not an observed failure.

V31 prioritizes the 24 original fixture IDs using already-configured
API-Football results, only after kickoff + three hours. Neither V30 nor V32
uses AI to guess final results. The score verification must match provider,
fixture ID, team IDs, kickoff timestamp, finished status and score basis.

## New falsifiable ranking comparison

After V30 independently re-verifies the original write-once frozen candidate
catalog, V32 checks *only days where every original fixture has an
independently verified final*. For each complete Sydney day, it compares:

- The pre-kickoff **quality** and **robustness** rank separately.
- Top 1, 5, or 10 candidate pairs against disjoint bottom 1, 5, or 10
  candidate pairs **from the same day**, using identical numbers of
  5-combination score sets in the two arms (5, 25, or 50 tickets).
- Hit-pair fractions and within-day rank concordance, always marked as
  *correlated*, exploratory descriptive measures.
- One top-minus-bottom difference **per Sydney day**, unweighted across
  days. No rank significance test before 30 complete days.

For a small full slate, both arms automatically shrink to
floor(candidate_count / 2), so comparison sets never overlap and ticket
budgets remain equal. These top/bottom comparisons are retrospective
analyses of **original pre-frozen candidates**, *not new historical
selections, purchased tickets or a prospective 5-double policy hit rate*.

The V32 report is rebuilt after ordinary pre-match capture and independently
verified-results backfill, with zero extra sports API or AI model requests.

## Run

python v32_rank_discrimination.py

Output: data/v32_rank_discrimination/status.json

The initial review target is **30 fully verified daily slates**; even that
is only an initial study threshold and not proof of a 70% win rate. Changing
the live ranking policy requires subsequent prospective out-of-sample
confirmation and a separate production decision. Do not multiply the
number of candidate pairs to claim independent samples.
