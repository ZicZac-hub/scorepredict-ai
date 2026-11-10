# ZicZac forecast accuracy audit and improvement plan (11 Nov 2026)

## What the current result means
The user observed 3 exact-score hits in 38 verified pre-kickoff forecasts (7.9%). Exact-score forecasting is substantially harder than 1X2. If the average top-score probability is near 10%, the expected count is about 3.8/38; three hits is not by itself evidence of a broken model. **Compare the sum of archived top-score probabilities to the number of hits on the same subset**. The Finished page now calculates this, top-three coverage and competition-level audit. Never equate 1X2 accuracy with exact-score accuracy.

## Confirmed code findings
- engine.py uses an independent Poisson score matrix (Dixon-Coles rho = 0), 10 recent overall games and up to 10 venue games, a recency-by-match-index weighting, heuristic attack/defence blend and shrinkage to league goals. None of those parameters is empirically tuned on a documented held-out test set.
- Current score probabilities may be reasonable even when most top-score picks miss; the full score distribution is more informative than one modal score.
- worldwide_model.py now retrieves additional Pro season statistics, league standings, injuries and available formations, but they are contextual; the live goal-rate estimator does not incorporate them as calibrated covariates.
- historical data may mix league and cup competitions; team quality/opponent strength, competition scoring environments, season transitions and youth/women's league distributions can differ.
- Pro availability is not equivalent to complete player xG, starting XI, confirmed injuries or internet-wide news coverage. Missing fields must be explicitly marked unknown.
- the archive uses SQLite on potentially ephemeral Streamlit storage; this threatens the integrity of long-term performance measurement. Move to durable storage before making accuracy claims.
- Only snapshots saved BEFORE kickoff should be evaluated. Do not backfill historical predictions as if they were contemporaneous.

## Experiment plan (do not deploy an unvalidated challenger)
1. **Audit data and baselines:** deduplicate fixtures by provider and ID; check UTC timestamps and Australia/Sydney display; validate provider status and full-time scores; separate friendlies/cups/leagues; verify home/away team IDs. Quantify missingness by competition and age.
2. **Backtest:** expanding or rolling walk-forward splits, strict pre-kickoff data cutoffs, minimum ~100–300 completed games per segment before interpreting segment results. Include a season holdout and uncertainty intervals. Avoid tuning on the same matches used for final evaluation.
3. **Baselines:** current venue-form Poisson; independent league-rate Poisson; fitted time-decay Dixon-Coles; opponent-adjusted attack/defence Poisson; optional bivariate Poisson. Fit low-score rho from training data; do not assume a fixed negative value.
4. **Goals/xG:** where *licensed and verified* per-match xG exists, test xG-adjusted scoring rates against goals-only. Do not label goals-based estimates as observed xG. Add opponent quality, rest days, home advantage and promotion/season transitions as testable features.
5. **Team/player context:** use confirmed lineups, injury reports, suspension status and squad minutes ONLY with reliable timestamps. Apply a player adjustment only if out-of-sample tests show improvement. Rumours and generic news are not quantitative evidence.
6. **Odds benchmark:** if appropriately licensed bookmaker pre-match odds are available, remove overround, store observation timestamps, compare model vs market; test blending only on past data. Never claim to outperform odds without held-out evidence.
7. **Metrics:** exact-score hit rate; sum of predicted top-score probabilities vs observed hits; top-three coverage; full score-grid negative log likelihood; 1X2 multiclass Brier and log loss; reliability/calibration plots; MAE of home/away goals; compare by league, competition type and evidence quality.
8. **Decision rule:** replace the live model only if challenger improves *held-out* full score-grid log loss and calibration without material degradation on 1X2 or large competitions. Report sample size, confidence intervals, model version and date range.
9. **Five Doubles:** a two-leg exact-score hit probability is small; multiplication assumes independence and can overstate certainty for related fixtures. Show joint probability, source and uncertainty, not 'sure picks'. Avoid ranking solely by top-score probability when evidence is weak.
10. **Continuous operations:** cache API calls, schedule off-peak research with provider rate limits, retain source provenance and research timestamps, add monitoring for failed fixtures and stale lineup/injury information.

## Evidence consulted
- Wilkens (2026), *Can simple models predict football—and beat the odds?* https://journals.sagepub.com/doi/10.1177/22150218261416681 — walk-forward xG, isotonic calibration, market comparison.
- Football Data Science, Dixon-Coles implementation and limits: https://www.footballdatascience.co.uk/models/dixon-coles
- Football Data Science, model evaluation: https://www.footballdatascience.co.uk/learn/model-evaluation
- Footlab AI methodology (2026): https://footlab.ai/en/methodology — disclosed market-odds blending and measured performance; provider claims are not independently verified.
- Madsen (2025), *Goal-line oracles*: https://pmc.ncbi.nlm.nih.gov/articles/PMC11785260/ — difficulty of exact goal prediction.

## Implementation status
DONE: finished-page expected-vs-observed exact hits, top-three coverage, per-competition audit.
NOT DONE: retrained opponent-adjusted model, calibrated player/tactical adjustments, licensed external news/odds ingestion, full historical walk-forward head-to-head, durable forecast database. These must not be described as deployed.
