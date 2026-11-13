# ScorePredict AI Worldwide Pro

A free, transparent football score forecasting Streamlit application. No fabricated fixtures, paid services or hidden prediction guarantees.

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

