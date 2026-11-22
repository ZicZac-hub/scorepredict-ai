# V34 — Why five exact-score doubles miss (read-only research)

**23 November 2026, Sydney.** Research only. This update does not change the live
prediction model, score grids, five originally issued selections, fixture
selection policy, provider request budget, API keys, or any original result.

## Verified baseline and the remaining problem

- Original V8 five-score doubles: **0 wins from 6 independently verified daily
  pairs**, with one additional pair pending in the original archive.
- Original V25 robust versus V8 on the *same* five verified V25-selected fixture
  pairs: **V25 1/5, V8 0/5**; one newer pair is pending. The earlier 1/4
  report was superseded as more identity-matched final scores became available.
- V28/V30 contains **216 frozen overlapping candidate pair rows**, of which
  **156 have independently verified two-leg scores** and **60 remain pending**
  at the latest baseline. Only one entire Sydney-day slate was complete.
  Candidate pairs share physical matches; **3 hits across 156 correlated
  pair rows is not an independent win-rate sample**.

None of these results proves that any challenger improves forecast accuracy
or achieves a 70% five-double hit rate. This is a diagnostic study with a
pre-existing, outcome-observed exploratory dataset, *not* new prospective
selection performance.

## New independent read-only diagnostic

`v34_score_error_research.py` re-verifies the complete original immutable V28
candidate catalogs and all original V8 forecast source IDs through V30 and
`verify_decision`. Only regulation-time scores already independently
identity-matched in V30 are used. It NEVER issues replacement tickets.

For each **original frozen pair**, both V25 robust and V8 optimal five tickets
are classified separately as:

1. `hit`: both exact regulation-time scores appear on the **same** originally
   frozen joint ticket;
2. `one_leg_score_absent`: only one of the two actual scorelines occurs
   anywhere among the five originally selected leg-score values;
3. `both_leg_scores_absent`: neither actual scoreline occurs in its leg's
   selected scores;
4. `scores_present_but_wrong_joint_ticket`: both actual scores occur
   individually in the frozen selections but were **not combined on the same
   joint ticket**.

The last category is crucial: it is a *combination selection* failure, not a
failure to consider the actual score in either match. Neither that category
nor high-score retrospective patterns justifies revising previously issued
predictions after observing finals.

## Calibration without double-counting football matches

Each original physical fixture is deduplicated by **provider + fixture ID**.
Repeated candidate pair appearances are not treated as new football games.
Inconsistent final scores for one fixture cause an error instead of a silent
overwrite. An original V8 probability matrix is evaluated at most once per
verified fixture, using:

- top-1 / top-3 / top-5 *single-fixture* exact-score hits;
- full normalized exact-score log loss and multiclass Brier score;
- goal prediction error, 4+ total-goal and 0–1 total-goal calibration.

A distinct section repeats the fixed Poisson, negative-binomial,
bivariate-Poisson and equal-sensitivity transforms of the archived V8 grids.
**These are explicitly POST-HOC mathematics, not contemporaneously frozen V25
forecasts, and they are ineligible for claims of live improvement.**

A complete day counts as one unit for future ranking inference. Partial
slates contribute only descriptive **verified** diagnostics; missing finals
remain pending, never losses. The initial day-level review gate remains 30
fully verified Sydney-day slates, while original frozen V25 calibration needs
its separate genuine prospective paired cohort.

## Run, inspect, repeat

```bash
python -m pytest -q tests/test_v34_score_error_research.py
python v34_score_error_research.py
cat data/v34_score_error_research/status.json
```

A passing full GitHub `prediction-tests.yml` regression is required before
merging. The scheduled `prematch-evidence.yml` and
`backfill-verified-results.yml` flows recompute and save the V34 report after
original V28 final-score verification, without additional provider/API calls.

## Research-first next action

Use the report's *deduplicated physical-fixture* high/low goal calibration to
state a narrowly prespecified, prospective scoreline challenger hypothesis.
Do not tune probabilities on the same verified matches that suggested the
hypothesis, do not treat V25 heuristic lower-envelope scores as normalized
probabilities, and do not promote a model before independent time-forward
scoring-rule and frozen five-double comparisons.

Scientific context:

- Wheatcroft, `Evaluating probabilistic forecasts of football matches: the
  case against the ranked probability score` (2021),
  https://doi.org/10.1515/jqas-2019-0089 — motivation for proper probability
  scoring rather than win-rate alone.
- `Bayesian state-space models for the modelling and prediction of the results
  of English Premier League football` (2025),
  https://doi.org/10.1093/jrsssc/qlae075 — benchmark context for log-score
  and Brier-based football forecast evaluation.
- `Deployability of machine-learning prediction models in elite and
  high-level team sports` (2026),
  https://doi.org/10.3389/fphys.2026.1920214 — caution around leakage,
  time-respecting validation, and overclaiming predictive generalisation.

**No observed 70% five-double win rate has been demonstrated.**
