# V16 — Verified paired-AI evidence and honest five-double uncertainty

Date: 15 November 2026. **Research-only upgrade**, not an automatic
replacement for the production V8 score selector. Existing immutable match
forecasts, five-double choices and historical results are never edited.

## Problem being fixed

V15 had an evidence-attribution gap: saved AI research on another V8
forecast horizon for the *same match* could be treated as relevant to the
original frozen five-double pair. Also the read-only AI audit previously
counted just fixture IDs and superficially valid score lists, rather than
the original source forecast, data digest, 1X2 response validity and capture
timing. Configured provider keys must never count as successful model votes.

## What V16 does

- **Source identity:** Accept an AI panel for a frozen double leg only when
  fixture provider and ID, both team IDs, kickoff, exact frozen source
  forecast ID and input digest all match. The source forecast must exist,
  predate the vote and predate kickoff.
- **Real model responses:** Validate provider names, integer score ranges,
  finite probability components and near-unit 1X2 totals. Invalid, absent,
  future or unrelated votes do not increase the provider count.
- **Predeclared chronology:** If partial and complete panels exist for one
  match, use the earliest eligible *complete* pre-kickoff panel; do not
  select a later panel because it happened to predict a winning result.
- **Joint completeness:** Require at least two genuine responses on
  each of the two matches. Report shared provider names and reasons for
  declining to produce an AI-assisted double. Do not imply all configured
  models have answered both fixtures.
- **Continuous no-cost audit:** Existing prematch GitHub workflow produces
  public \`data/v15_ai_double/source_audit.json\` after saved evidence is
  collected; it makes zero paid AI or football API requests. The Deep
  Research panel exposes per-leg counts and missing-evidence reasons.
- **Outcome uncertainty:** Evaluated, genuinely frozen V15 five-double
  records receive a Wilson 95% win-rate interval plus exact same-pair
  original V8 results and the two directions of disagreement. This
  interval reflects limited observations; it is not a guaranteed future
  rate. No challenger is automatically promoted.
- **Source preservation:** Stored V15 shadow records continue to be
  write-once, and V8 archives, old research panels, and previous
  score predictions remain immutable.

## Tests

\`python -m pytest -q tests/test_research_double_v15.py tests/test_v15_source_audit.py\`

Regression cases cover a genuine complete pair, wrong source ID, wrong
input digest, malformed probabilities, future-only votes, duplicate
panels, earliest complete panel selection, original write-once safety,
pending versus verified results, and no automatic promotion.

## Critical limitations

This upgrade improves **data quality, auditable provider coverage and
confidence reporting**, not demonstrated exact-score predictive accuracy.
Without genuine future pairs that have verified results, no six-AI
five-double win rate exists. Six LLMs reading the same statistical packet
are correlated; none provides true shot-level xG, confirmed lineups or
bookmaker odds unless those timestamped feeds are separately connected.

The optional two-fixture paid AI collector stays limited by
\`LLM_RESEARCH_MAX_MATCHES\` and requires explicit opt-in to increase the
number of charged provider calls. V16 does **not** change provider secrets,
billing settings or existing prediction weights.
