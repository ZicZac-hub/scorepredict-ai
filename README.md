> **Accuracy checkup (21 Nov 2026):** Fixed false result conflicts from equivalent UTC timestamps, blocked stale dynamic API fallback and future/duplicated evaluation evidence, authenticated saved research doubles, and supplied one saved evidence dossier to configured AIs after source collection. The corrected original archive has **0/3 successful verified doubles, 3 pending**. **70% remains unsupported; no forecasting gain is claimed from software tests.** See [measured results, miss analysis and remaining limits](data/diagnostics/nov21_accuracy_checkup.json).

> **Matched evidence update (19 Nov 2026, Sydney):** Align bounded near-kickoff collection with both original fixtures and add fixed V8 / market-only / xG-only / xG-plus-market prospective comparisons. Missing evidence remains an explicit abstention; the website authenticates original sources before displaying results. **No new verified accuracy gain or 70% result is established.** See [methods, collection fixes and limits](MATCHED_EVIDENCE_UPDATE.md).

> **Professional five-double research (18 Nov 2026):** Added a source-qualified optimizer, automatic prospective research capture, common AI evidence, source-date quarantine and an eight-model comparison. The frozen candidate achieved **20/166 = 12.05%** retrospectively; development results range from **7.83% to 16.27%**. The new ledger has **0 settled, 2 pending**; premium abstains without verified calibration. **70% is not demonstrated.** See the [complete upgrade and evidence report](PROFESSIONAL_UPGRADE_REPORT.md).

> **Five-double accuracy repair and diagnosis (15 Nov 2026):** Removed unvalidated AI ranking boosts from the default Studio selection, corrected provider response timestamps and live fixture eligibility, and added a scheduled source-validated miss report. The six original sets cover only **5.92–7.62% estimated model mass**, with **0/1 verified sets successful and 5 pending**. A 70% exact-score-double hit rate is **not established**. See [root causes, reproduced evidence and fixes](FIVE_DOUBLE_ROOT_CAUSE.md).

> **V24 wider real-data collection (16 Nov 2026):** The existing owner-enabled `V23_EARLY_ODDS_ENABLED=true` now also enables a paginated, date-wide API-Football market catalog for all provider-returned fixtures in the next 14 days, even before the statistical model selects them. Market evidence is timestamped and immutable; it is joined to V23 only after matching a genuinely frozen V8 fixture. The earlier V23 collector also searches all individually saved V8 forecasts, not just existing five-double pairs. **Default 12 additional date-page requests per UTC day (up to 2 per run)**, subject to provider quotas and coverage; no unrestricted internet scraping or free xG invention. See [V24 source collection and limits](V24_BROAD_DATA_CAPTURE.md).

> **V23 source-backed goal probabilities (15 Nov 2026):** The AI Five-Double Studio now has **V23 Market + Goal Form** and **V23 Auto Select**. V23 checks real, timestamped winner and total-goal bookmaker prices for BOTH matches, applies a bounded adjustment to the original V8 full score grid, preserves each five-double set before kickoff and automatically compares results. Genuine historical shot xG is optional; no invented odds, xG, injuries or numeric AI boosts. The latest matched 393-fixture retrospective found V8's score log loss slightly better than V20 goal-form, so V23 does not automatically weight the weaker goal-form variant. Source capture remains **owner opt-in** and API quota may apply. **Accuracy improvement is not yet established.** See [V23 setup, verification and limitations](V23_MARKET_GOAL_RESEARCH.md).

> **V21 verified research and calibrated score tournament (15 Nov 2026):** Source-matched API-Football xG/starting-XI/injury/odds packets with pre-match timestamps (optional/owner-approved), chronological Poisson versus negative-binomial versus bivariate-Poisson goal-model comparisons, same-day evidence-qualified AI Auto Select with abstention, and immutable verified future five-double scoring. V21 research does **not** weight unvalidated AI votes or replace the original model. No claim of 70% observed exact-score accuracy. See [V21 setup and limitations](V21_RESEARCH_PIPELINE.md).

> **V20 evidence-driven goal research (15 Nov 2026):** The AI Five-Double Studio now offers an experimental **Research goal-form challenger** when BOTH selected matches have a full original probability grid, at least eight verified previous games per team and four relevant home/away results. It compares venue and overall scoring/conceding rates, last-five scorelines, 2+ goals, high-total probabilities, H2H and real API-Football season/lineup/injury context when available. The AI prompt now receives those facts for optional owner-approved providers. It displays 0/1/2/3+ goal probabilities and five combinations ranked using a separate, conservative original-plus-form score grid; no famous-team win or 1-1 outcome is forced. This is a **shadow model only**; no proven accuracy improvement, no unapproved AI spending, no claim to scrape every website, and no retroactive changes to saved forecasts. If data are missing, the original no-cost baseline remains available. See [V20 scoring source](goal_research_v20.py) and [time-safe tests](tests/test_goal_research_v20.py).

> **V19 verified match intelligence:** Research-only genuine market-source five exact-score challenger on the same immutable original two-match pair, with timestamp checks, no-source abstention, independently verified results and no additional provider calls. It does not claim 70% accuracy or purchase xG/lineup data. See [V19 source audit](V19_VERIFIED_MATCH_INTELLIGENCE.md).
 
> **V18 real-data reliability research:** Automatically audits original frozen V8 predictions against independently verified full-time outcomes, counts missing evidence, shows confidence-bin accuracy and tests time-safe calibration without promoting unproven forecasts. See [V18 real-accuracy research](V18_REAL_ACCURACY_EVIDENCE.md).

> **V17 evidence-gated confidence research:** a separate, prospective 1X / X2 / 12 outcome mode is available in **Five Doubles** and **Performance**, with automatic source validation, abstentions and verified 70% reporting gate. This is **not** 70% proven accuracy or a replacement for five exact-score doubles. See [V17 evidence research](V17_EVIDENCE_CONFIDENCE_RESEARCH.md).\n\n# ScorePredict AI Worldwide Pro

A free, transparent football score forecasting Streamlit application. No fabricated fixtures, paid services or hidden prediction guarantees.

## V43 integrated football forecast

Open **V43 Forecast** in Matchday Studio to generate a single-match score distribution or five exact-score doubles for two eligible matches on the same Sydney matchday. V43 blends the V41 grouped statistical model with the four V42 score-shape hypotheses. Pre-kickoff xG and market data enter only when they pass the time and fixture checks. For doubles, both matches must qualify for a given adjustment. The combined forecast is experimental.

**Research and validation** shows the version history, archived data sources and matched replay results. The replay compares common independently settled regulation scores within each cohort. It runs read-only with `python v43_replay.py`.

The five-double target panel checks verified finals and shows why selections missed. It calculates the maximum modeled coverage for five exact-score combinations and the number of combinations the same distributions would need for 70% coverage. The prospective study freezes one authenticated pair per Sydney matchday before kickoff and grades it against independently verified finals. Run `python v43_prospective.py` to recheck the study without writing records. The first review uses 100 independently settled pairs and a 95% Wilson lower bound of at least 70%. The two-match model still uses statistical probabilities, with no target-driven increase in confidence.

The existing Private Access gate and server-side configuration control optional OpenAI, Claude and other provider reviews. Each configured provider reviews the same available match context on request. Their opinions carry zero numerical forecast weight. See [on-demand provider setup](ON_DEMAND_AI_SETUP.md).

## Matchday Studio 2.0 interface

The existing website is https://ziczacscorepredict-ai.streamlit.app/ . The updated interface has a compact Version 2.0 header, route-specific content, and Home buttons for AI Auto Select, Manual Select, Deep Research, prediction history and real performance. Choosing a section updates `?view=` in the URL; Home cards no longer appear above every other page. Direct doubles links also accept `&mode=auto` or `&mode=manual`.

The forecast models, provider keys and original saved records are preserved. Streamlit hosting controls account access; this interface update does not introduce a custom password or change host sharing settings. The separate iPhone launch page in `pwa/` links to this existing service.

## Optional ten-provider AI + Sportmonks Predictions research

A non-destructive research panel is wired to the scheduled collection workflow
and the Deep Research / Performance pages. It supports OpenAI, Claude,
Gemini, Grok, DeepSeek, Perplexity Agent, Mistral, Alibaba Qwen, GroqCloud
(Llama) and Cohere, plus a separate Sportmonks **Predictions add-on**
correct-score probability comparison. Existing Sportmonks *fixture* access
does not automatically include the paid Predictions add-on.

**All external research adapters are opt-in and currently disabled until
individual provider keys and model IDs are configured in GitHub Actions.**
They do not retrieve proprietary ScoreGPT predictions. Forecasts and doubles
remain immutable and none of these research opinions is automatically promoted
to the live numerical model. See [10 AI + Sportmonks setup guide](
TEN_AI_AND_SPORTMONKS_SETUP.md) for secure configuration and cost limits.

## Features
- Upcoming real fixtures from football-data.org competitions accessible to your key
- Recent and venue-weighted attacking/defensive rates, Poisson exact-score probability matrix
- Ten exact-score candidates per match, 1X2 probabilities, context-aware narrative
- Top five **double-chance selections** (1X / X2 / 12) across distinct fixtures
- Five **two-leg double** combinations with independence-assumption probabilities
- Chronological walk-forward backtesting with exact-score accuracy, 1X2 accuracy, Brier score, goal MAE
- Write-once prediction snapshots in local SQLite, settlement from final results, downloadable CSVs

## Run locally
1. Install Python 3.10+.
2. Extract this folder and run `pip install -r requirements.txt`.
3. Register for an API key at https://www.football-data.org/client/register .
4. Set `FOOTBALL_DATA_API_KEY` as an environment variable, or create `.streamlit/secrets.toml` containing `FOOTBALL_DATA_API_KEY = "your_key"`.
5. Run `streamlit run app.py`.

## Free hosting
Push the project to GitHub, then deploy `app.py` on https://share.streamlit.io . Set `FOOTBALL_DATA_API_KEY` in the app's Secrets dashboard. Never commit secrets to GitHub. Streamlit free hosting and API free plans have quotas; availability and terms can change.

## Coverage and accuracy caveats
- "Worldwide" means all competitions available through the configured provider; **not** every league on Earth. A free football-data.org key does not unlock every league or every historical season.
- A missing team history produces **no prediction**, rather than a fabricated number.
- Live baseline forecasts use statistical goal models. Optional source-qualified xG, injury, lineup and market evidence depends on provider coverage. Gradient boosting and ensembles have separate historical research results; they are not automatically promoted to the live model. AI explanations are not measured accuracy.
- A double-chance market is one match with two possible outcomes; a two-leg double is two selections across different matches. Both are shown separately.
- A probability is **not** a promise. Two-leg joint probabilities assume independence; correlated outcomes can violate this.
- SQLite data on free cloud hosting can disappear after app restarts or redeploys. Export CSVs frequently. For persistent automated audit history, migrate to a free hosted Postgres database and schedule collection outside the interactive app.
- Interactive forecasts and scheduled repository-backed evidence have separate storage paths. GitHub Actions collect and verify supported fixtures automatically, subject to provider coverage, credentials and quotas; private local SQLite records are not automatically visible to GitHub.
- Historical walk-forward testing uses only prior completed matches, but is limited by available match history and should be validated out of sample before any accuracy claims.

## Tests
Install `pytest` and run `python -m pytest -q`. This includes both unittest
classes and function-based regression tests. GitHub runs these checks on pull
requests and pushes to main.

## Prediction and result integrity repair
The v6 live baseline separates provider and competition histories, uses genuine
league-wide inputs in worldwide analysis, and supports four-plus modal scores
when goal-rate evidence warrants them. Research context remains separate from
the numerical model. See [validation evidence](MODEL_REPAIR_VALIDATION.md).

Existing SQLite archives migrate automatically on first connection. The original
table, original forecasts, saved timestamps, and previous results are retained.
Older results without provider identity are excluded from accuracy until their
provider, teams, kickoff, and regulation-time score are verified again. On the
Finished page, use **Check completed results**; each click checks a rotating
batch of ten matches, so unavailable recent results cannot block older records.
The Finished count may change during this revalidation. Archive CSV exports
include the previous result and verification status for inspection.

New predictions are saved only before kickoff and never overwrite the original
forecast. Scheduled verification also preserves old result evidence before
rechecking it. Neither these repairs nor the experimental model comparison
establish an improvement in the separate 236-match live archive.


## Reference-informed upgrade and complete archive revisions
The original code is preserved on
[`archive/pre-reference-upgrade-v6-2026-11-13`](https://github.com/ZicZac-hub/scorepredict-ai/tree/archive/pre-reference-upgrade-v6-2026-11-13).
The unchanged v6 engine also remains available in this version.

Public [AI/site references](REFERENCE_SOURCES.md) inform the new fitted
opponent-strength and Dixon–Coles research model. These sources do not supply
actual provider API votes. [Validation](REFERENCE_MODEL_VALIDATION.md) found
better probability scores but fewer exact hits in an independent 1,385-match
replay, so the preserved scorer stays the default. Select **New fitted model
(research)** in the sidebar to use the candidate for live forecasts.

The upgrade automatically queues the server's old SQLite forecasts and saved
doubles for revision on first load. The Finished page shows progress, upcoming
revisions, separate finished retrospective comparisons, revised five doubles,
and skipped-record reasons. **Update all saved forecasts and five doubles**
retries the complete archive. New-model comparisons use `reference_v7`; original
predictions, original accuracy and original five-double sets remain unchanged.
Finished comparisons reconstruct information available at original capture;
where a final-whistle timestamp is absent, prior games require a conservative
three-hour interval after kickoff. They cannot be advertised as contemporaneous
predictions or improvements to the original hit rate.

For an unattended pass on repository-backed records or a local server archive:

```sh
python update_saved_forecasts.py --archive-db predictions.sqlite3
```

The existing provider keys stay in environment/Streamlit settings. GitHub
Actions cannot read a private Streamlit instance's local SQLite; the server's
background pass handles those records. An incomplete run explicitly reports
unavailable histories/identities and retains originals. New five-double
selection ranks five distinct score pairs by their actual joint mass, under the
stated independent-match assumption; probabilities across different fixture
pairs must not be added as portfolio coverage.

## Shared evidence and central prospective ledger
See [EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md) for the common history policy,
PostgreSQL schema, restart recovery, full-grid v6/v7 comparisons, forecast
horizons, and outbox retry rules. The app uses `DATABASE_URL` already configured
in Streamlit Secrets. The Archive page shows central synchronization status and
fresh same-evidence model comparisons; old originals and retrospective revision
results stay separate. Scheduled collection journals all four models to GitHub
and optionally the same database, with explicit budget/coverage reports.


## V16 verified two-match AI evidence (research only)

The no-cost V16 [paired AI evidence audit](V16_PAIRED_AI_EVIDENCE.md)
matches each AI response to the exact immutable statistical forecast
and input digest for **both** fixtures. The scheduled research job
publishes source-validation diagnostics in Deep Research and the ledger.
Future verified, original-versus-AI exact-score double comparisons
include Wilson 95% uncertainty intervals. No old forecasts or original
five-double picks are rewritten, no new paid requests are enabled,
and the V8 production selector remains unchanged.
