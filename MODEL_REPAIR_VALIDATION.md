# Prediction model repair validation

The live forecast remains the baseline variant. Its model version is now
`venue-form-ensemble-v6-competition-history`: provider IDs and competition
histories are separated, a genuine full league population supplies the league
baseline, and an explicit missing league history uses disclosed generic priors.
The old 3.8 expected-goals ceiling is removed; Poisson grids extend until their
omitted joint tail is below 1e-10. Probability mixtures pad different grid sizes
before combining them. These are input and numerical corrections, not a claim
that exact-score accuracy has improved.

## Reproducible historical comparison

Run `python model_repair_benchmark.py --download` after installing requirements.
Downloads are opt-in; cached data are reused without `--download`. Results and
CSVs default to `work/`, and paths can be changed with `--output`/`--data-dir`.

Public sources: [football-data.co.uk](https://www.football-data.co.uk/data.php),
CSV URLs `https://www.football-data.co.uk/mmz4281/{season}/{league}.csv` for
seasons `2223`, `2324`, `2425` and leagues `E0` (Premier League), `D1`
(Bundesliga). Results use regulation full-time goals. These public results are
separate from the app's live provider and its 236-match Finished-page archive.

The existing league-opponent model's hyperparameters were unchanged. Inputs are
2022/23 and 2023/24 plus strictly earlier results during the 2024/25 holdout;
all same-day results are excluded. Only paired forecasts with sufficient team
history are evaluated: 375 Premier League and 296 Bundesliga matches.

| Holdout metric | Baseline | League-opponent challenger |
| --- | ---: | ---: |
| Premier League exact hits | 54/375 | 49/375 |
| Premier League score log loss | 3.00844 | 2.98048 |
| Premier League outcome Brier | 0.60384 | 0.59387 |
| Bundesliga exact hits | 22/296 | 26/296 |
| Bundesliga score log loss | 3.15115 | 3.12597 |
| Bundesliga outcome Brier | 0.63000 | 0.61956 |
| Combined exact hits | 76/671 (11.33%) | 75/671 (11.18%) |
| Combined 1-1 forecasts | 543/671 (80.9%) | 392/671 (58.4%) |

Lower log loss and Brier are better. Probability quality improved in these two
leagues, while exact-score improvement was not established. No model promotion
or forced score diversity follows from this comparison. It covers two leagues,
one holdout season, and fuller histories than many live fixtures have. The
approximate uncertainty intervals in the JSON assume independent matches;
shared teams and histories weaken that assumption. A fresh prospective paired
evaluation with verified settlement is still required before promotion.

Regression coverage includes provider/competition collisions, future-result
exclusion, complete league populations, missing league priors, supported four-
plus modal scores, tail mass, and mixtures with different matrix sizes.
