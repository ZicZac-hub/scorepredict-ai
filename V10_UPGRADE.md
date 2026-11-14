# ZicZac v10 — Verified Hybrid Research Upgrade

This upgrade was implemented on 14 November 2026. It does **not**
claim world's-best exact-score accuracy. It preserves all original v6/v7
models, pre-match snapshots, historical results, and existing five-double
selections without rewriting outcomes.

## Shipped features

1. **Durable, independent pre-kickoff challenger ledger**
   - `v10_prospective.py` computes v8 using only the existing verified
     historical as-of packet. The complete original score-probability grid,
     model version, forecast timestamp, fixture IDs, source evidence digest,
     score options and component weights are immutable.
   - V9 is saved **only** when real timestamped /odds are available from
     API-Football. Missing odds do not create a fake v9 forecast.
   - The additional v10 challenger aggregates **at least three** qualifying
     bookmaker quotations by the median of overround-adjusted implied
     probabilities from the SAME actual provider response. The response is
     requested only once per fixture. Real market timestamps, book counts
     and vendor names are archived, but never the API secret.
   - All source market models remain **experimental and not validated**.
     They do not silently change the v8 daily default.

2. **Verified context without invented xG**
   - `verified_features_v10.py` derives both teams' latest historical
     match dates, venue/form history and rest days from the existing
     two-season as-of policy.
   - API-Football's fixture statistics do not supply a complete official
     licensed xG feed for every match. True xG/xGA, player-strength
     adjustments and independent external LLM scores are marked as
     **unavailable**. They are not fabricated or numerically applied.

3. **Sydney-calendar five-double ranking**
   - `v10_evaluate.py` considers all available research-ready fixtures,
     not merely the first 18. It checks the same Sydney calendar day,
     four different teams, genuine frozen model history and five unique
     exact-score combinations.
   - It maximises the sum of five mutually exclusive modelled joint-score
     probabilities. Under independent fixture outcomes this is the
     mathematically correct objective for "at least one of five hits".
   - An explicitly **untuned** 4% modelled-coverage gate allows abstention.
     This gate is an operating guardrail, not a proven historical threshold.
   - Write-once files in `data/v10_prospective/doubles/YYYY-MM-DD.json`
     preserve the selected matches and five score pairs before kickoff.
   - v9 and v10 alternate combinations are only frozen if both match
     forecasts already existed **before the exact same selection time**,
     using the same original pair. No retroactive revised models are counted.

4. **Observed performance only**
   - `v10_evaluate.py` matches provider ID, both team IDs and kickoff
     timestamp and requires independently verified full-time regulation
     scores **after kickoff**.
   - Exact-score hits, top-three coverage, score log loss, 1X2 Brier
     score, same-fixture paired challenger comparisons, double wins,
     pending matches and model-calibration status stay separate.
   - Pending games are neither wins nor losses. Historical simulated
     matches do not enter prospective accuracy.
   - `v10_status_ui.py` surfaces the verified records in Daily Predictions
     and Finished Matches, and displays "no verified games" when required.

5. **Reliable collector / safe Git history**
   - The existing `.github/workflows/prematch-evidence.yml` now runs v10
     scoring and commits new append-only records. The 15-minute cadence
     remains; additional previously captured yet still-upcoming fixtures
     are researched **only on heavy refreshes**, capped to 12 per run.
   - `reconcile_capture_conflicts.py` resolves concurrent generated
     report conflicts safely, merges final-score dictionaries only when
     their event/score identities agree, and preserves the **first
     published** valid date selection on concurrent frozen-pick clashes.
   - **Immutable forecast ID collisions with different contents cause a
     safety stop.** No force-pushing or rewriting original forecasts.
   - If the optional `DATABASE_URL` secret is configured, the original
     provider ledger continues using its database; the separate v10
     shadow ledger is currently committed as immutable Git-backed files.
     A managed dedicated v10 database is not yet configured.

## Model validation and limitations

Earlier *historical* v8 experiments had 77 exact hits from 632 matches
versus 70 for v6; this was **not** a statistically decisive improvement
and does not guarantee future results. The 59-date historical paired
five-double replay had 2 v8 wins and 1 baseline win, also insufficient.
New v9/v10 have no independently established prospective superiority
and remain challengers, not auto-promoted defaults.

At the first successful production v10 ledger check, 24 v8 and 2 v9
forecasts were saved, with five upcoming frozen five-double sets.
There were no verified completed v10 matches yet. These counts
will change; current truth is `data/v10_prospective/performance.json`.

**External resources still required for further accuracy work:**
A properly licensed, historical **true xG/xGA** data feed with genuine
pre-match timestamps; verified player-strength injury and confirmed
starting-XI adjustment data; enough independently graded future matches
to fit/calibrate ensemble weights; and optional supported AI-provider
credentials (if a separately evaluated multi-LLM research panel is wanted).
None is claimed to be connected by this release.

## Evidence and files

- Source: `v10_prospective.py`, `multi_market_v10.py`,
  `verified_features_v10.py`, `v10_evaluate.py`
- Publication safety: `reconcile_capture_conflicts.py`
- UI: `v10_status_ui.py`, `app.py`
- Future results: `data/v10_prospective/performance.json`
- Original comparison: `data/research_v8/extended_historical_validation.json`
- Tests: `tests/test_v10_prospective.py`,
  `tests/test_multi_market_v10.py`,
  `tests/test_capture_conflicts_v10.py`
- Automation: `.github/workflows/prematch-evidence.yml`

**Goal:** improve correctness of the complete score distribution and
match-pair selection using reproducible, as-of-safe data. Do not claim
"world's most accurate", "guaranteed five doubles", or a verified win
rate until independently observed samples actually support that claim.
