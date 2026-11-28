# V45 Data Intelligence and Verification Engine

Open **V45 Data** in Matchday Studio to see evidence quality for archived fixtures kicking off in the next 24 hours, the latest published collection status, and the history and source impact audits. The evidence score measures readiness at the displayed cutoff. The weights are fixed operational defaults, and the score carries no numerical weight in forecast probabilities.

| Evidence | Points | Qualification |
| --- | ---: | --- |
| Historical results | 35 | Exact pre-match history receipt matches the forecast's original input digest and satisfies team and league history requirements. |
| Historical shot xG | 25 | The existing V21 source packet qualifies on fixture identity, chronology, sample size and venue coverage. Sportmonks has a separate availability and entitlement report. |
| Bookmaker markets | 20 | Time-qualified 1X2 and goal-total quotes match the fixture. Source packets may be at most three hours old at the check, and market quotes at most 12 hours old. |
| Confirmed starting XIs | 20 | Sportmonks confirms both XIs and API-Football independently reports the same 11 players per side on the matched fixture. |

## Collection and provider setup

- The existing 15-minute **Capture verified pre-match evidence** workflow continues to freeze original forecasts and exact history receipts and verify regulation results. V45 can check an existing API-Football forecast between 60 and 5 minutes before kickoff, subject to published lineups, workflow timing and the request budget.
- To enable lineup requests, set repository variable `V45_LINEUPS_ENABLED=true`. Use the existing `API_FOOTBALL_KEY` and either `SPORTMONKS_API_TOKEN` or `THIRD_FOOTBALL_API_KEY` Actions secrets. The collector considers one eligible fixture per run, uses at most six requests per run, and defaults to 12 requests per UTC day. `V45_LINEUP_DAILY_REQUESTS` accepts 0–24. Derived receipts include timestamps, provider identities, starter counts and starting-XI digests. A conflict or a single-provider receipt earns zero lineup points. Player names are saved only to private PostgreSQL when `DATABASE_URL` is configured.
- To diagnose Sportmonks access, set `V45_SPORTMONKS_FIXTURE_ID` to a recent completed fixture in a target competition, then run **Capture verified pre-match evidence** manually. The probe separates account authentication, access to the sample fixture, expected-goal data for that fixture and a separately sampled team xG feed. Use one sample per relevant competition when assessing coverage. Current-match xG comes from played minutes and can only serve as historical evidence for later matches. [Sportmonks xG product](https://www.sportmonks.com/football-api/xg-data/) · [Sportmonks xG coverage](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/expected/coverage)
- Sportmonks explicitly marks lineup confirmation. API-Football lineup publication varies by competition and can arrive very close to kickoff. An unconfirmed, partial, late or conflicting observation remains visible as a collection gap. Team and player names that cannot be reconciled safely can reduce verification coverage. [Sportmonks lineup schema](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/includes/lineups) · [API-Football documentation](https://www.api-football.com/documentation-v3)

## Historical recovery and verification

`python v45_history.py` audits genuine retained receipts and independently collected regulation outcomes. It records which original pair legs have exact history at the decision time, later recovery and missing evidence. Later recovery never fills the original cutoff retroactively. Conflicting historical results are quarantined. A recovered score and a separately saved final from the same provider are separate observations of the same source.

`python v45_history.py --database /path/to/private/football-history.sqlite3` stores a content-addressed recovery snapshot locally. `python v45_history.py --postgres` uses the existing authenticated `DATABASE_URL`. Set `V45_HISTORY_DATABASE_ENABLED=true` to enable the daily snapshot in **V45 data intelligence audit**. It runs against a private database and uploads aggregate diagnostics as workflow artifacts. The SQLite and PostgreSQL paths both store a snapshot once and validate its digest before `load_database_snapshot` returns filtered rows.

## Measuring source impact

### Automatic near-kickoff odds trial

The **V45 automatic near-kickoff market trial** runs on a dedicated ten-minute schedule so that longer football research jobs cannot hold up collection. It reserves up to four requests per UTC day from the existing V24 odds allowance, which defaults to 12. Each pass can request 1X2 and goal-total odds for the two fixtures of one existing frozen pair. It checks the pair at 35–20 minutes and again at 20–5 minutes before the earlier kickoff, based on actual runner time. The bookmaker quote update must be at most three hours old at the decision. The collector uses the configured API-Football key and `V45_MARKET_ENABLED=false` pauses the automation and restores the broad scan's full allowance.

At the first decision in the later window, the workflow freezes the V43 control and the same V43 grids with the market adjustment if both fixtures qualify. The trial records a skip when data is missing or a scheduled run misses the window. It authenticates the original forecasts, quote receipts and any independently captured regulation finals. **V45 Data** shows paired-market coverage, skipped and pending pairs, matched hit counts, and exact-score log loss. The trial uses the supported leagues represented in the existing frozen pairs, with their selection made before the odds check.

`python v45_source_impact.py` compares market and historical shot xG ablations within the same fixtures, decisions and settlement cohort. It keeps the older archived market cohort separate from the fixed-cutoff single-fixture and five-double cohorts. Compare common sample counts, score log loss, outcome Brier scores and descriptive five-double hits. Starting XIs remain contextual until a player-impact model and matched evaluation are available. Historical recovery establishes traceability. Its incremental value requires forecasts with and without the recovered inputs at the same historical cutoffs.

Existing on-demand AI reviewers receive the same timestamped football evidence. The prompt asks them to explain the strongest supplied fact, flag gaps or inconsistencies, and describe a plausible alternative scenario. Deterministic source checks authenticate facts. AI opinions retain zero numerical forecast weight.

The 28 November archive audit recovered 4,644 distinct regulation scores. It found exact original-cutoff history for 2 of 40 original pair legs, later recovery for one and no exact receipt for 37. The old market cohort had 86 comparable fixtures and its score log loss was 2.956 without odds and 2.914 with odds. The sample supports an archive comparison. The live five-double and xG samples are still too small to establish a forecasting gain or a 70% five-double hit rate.

The next data gaps are target-league Sportmonks entitlement and coverage, dependable cross-provider team and player ID mappings, and independently corroborated injury and suspension feeds. Measure lineup publication time and agreement by competition before expanding the collection budget. Rest-of-season history obtained later can support fresh training and future predictions with its actual observation date.
