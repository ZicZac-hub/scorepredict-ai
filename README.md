# ScorePredict AI Worldwide Pro

A free, transparent football score forecasting Streamlit application. No fabricated fixtures, paid services or hidden prediction guarantees.

## Matchday Studio 2.0 interface

The existing website is https://ziczacscorepredict-ai.streamlit.app/ . The updated interface has a compact Version 2.0 header, route-specific content, and Home buttons for AI Auto Select, Manual Select, Deep Research, prediction history and real performance. Choosing a section updates `?view=` in the URL; Home cards no longer appear above every other page. Direct doubles links also accept `&mode=auto` or `&mode=manual`.

The forecast models, provider keys and original saved records are preserved. Streamlit hosting controls account access; this interface update does not introduce a custom password or change host sharing settings. The separate iPhone launch page in `pwa/` links to this existing service.

## Optional independent five-AI panel

A non-destructive research panel is wired to the scheduled collection workflow and
the Deep Research / Performance pages. It can independently consult OpenAI,
Anthropic Claude, Google Gemini, xAI Grok and DeepSeek using genuine, frozen
pre-match evidence. **It is disabled unless the owner opts in and configures
individual API keys and supported model IDs.** It does not download predictions
from ScoreGPT or replace the saved numerical scores. Its same-fixture accuracy
comparison begins only after verified regulation-time finals. See
[FIVE_AI_SETUP.md](FIVE_AI_SETUP.md) for exact secure setup and cost controls.

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
- This version does **not** have a trained ML model, premium xG, injuries or lineups. "Human-style" means explainable statistical heuristics, not human intuition or an LLM.
- A double-chance market is one match with two possible outcomes; a two-leg double is two selections across different matches. Both are shown separately.
- A probability is **not** a promise. Two-leg joint probabilities assume independence; correlated outcomes can violate this.
- SQLite data on free cloud hosting can disappear after app restarts or redeploys. Export CSVs frequently. For persistent automated audit history, migrate to a free hosted Postgres database and schedule collection outside the interactive app.
- Forecasts are created when a visitor opens the site, not on a guaranteed background schedule. For unattended updates, add a scheduled job with a persistent database.
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
