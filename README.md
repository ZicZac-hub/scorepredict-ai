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
Run `python -m unittest discover -s tests -v`.
