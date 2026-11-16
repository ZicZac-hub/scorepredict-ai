# V25 professional analyst shadow trial — November 2026

**This is a research-only experiment, not a 70% hit-rate claim.** Do not replace the published five doubles, promote V25, backfill scores, or increase paid API usage based on this document.

## Why this trial exists

The archived V8 retrospective 2026 replay found **0/34** winning five-double sets, the fixed negative-binomial sensitivity model found **1/34**, and the current original prospectively verified V8 five-double sample is **0/1** with five pending. These denominators are too small for a stable claim about future five-double prediction accuracy.

The existing source-grid audit estimates the five most probable V8 joint outcomes cover roughly **5.9–7.6%** of its own outcome mass, not 70%; those masses have not been independently calibrated as live success probabilities. Maximizing the same five cells over and over does not improve bad input probabilities.

Relevant literature includes Dixon & Coles (1997), *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, [DOI](https://doi.org/10.1111/1467-9876.00065), and Reade, Singleton & Vaughan Williams (2020), *Betting markets for English Premier League results and scorelines*, [open paper](https://centaur.reading.ac.uk/89738/1/reade_singleton_scorelines.pdf). These emphasize proper goal models, market comparisons and rigorous evaluation. Neither establishes a 70% hit rate for two exact scores.

## Implemented V25 research workflow

File: `professional_analyst_shadow_v25.py`

1. Read original immutable V8/V8-near fixtures from `data/v10_prospective/forecasts`; use only source records timestamped before the evaluation cutoff and both kickoffs.
2. Enforce provider, fixture ID, home/away IDs, regulation-time score basis, input digest, minimum documented match history, full normalized score matrix, 72-hour research source-freshness guardrail, 45-minute minimum to kickoff, and maximum 24-hour lead. Separate fixture pairs must have four distinct teams and the same **Australia/Sydney** calendar day.
3. Evaluate **four predefined mathematical sensitivity distributions**, all derived from the same original score-grid means: V8, independent Poisson, negative-binomial with fixed shape, and correlated bivariate Poisson. **These are not independent evidence sources.**
4. Rank the five exclusive score-pair combinations with the highest `min(model joint mass)` across these four variants. This is a robustness criterion, **not a mathematically proven probability lower bound**. Separately freeze the original V8 top-five combinations on **those exact same two games** to enable a fair paired head-to-head.
5. Store each date's set once in `data/pro_analyst_shadow/sets/YYYY-MM-DD.json`. Maintain a `status.json` containing pending/verified sets, hit/loss outcomes and exact same-pair baseline V8 hits. An unverified result is **pending**, not a loss.
6. A descriptive per-leg human research note is compiled only from previously saved schedule/rest, source age, genuine xG availability, lineup-adjustment status and bookmaker-evidence availability. **Context fields have zero numerical weight until independent testing supports an effect.** No late team news, tactics, scores, odds or xG are invented.

The collector adds **no provider/API requests**. The existing `prematch-evidence.yml` job generates frozen shadow picks on eligible fixture days, appends the report to the existing research artifact, and commits the report together with the existing immutable archive. Read the V25 section under the website's V10 Research panel.

## How to decide whether to replace the old method

- Verify at least 100 independent, genuinely prospective, matched same-pair five-double sets before initial assessment; more are needed for reliable precision. Do not tune thresholds retrospectively on the prospective evaluation data.
- Report `robust_wins/verified_sets` **and** `baseline_wins_on_same_pairs/verified_sets`, pending sets, abstentions, selected source timestamps, changes in estimated mass, and uncertainty intervals.
- If performance is statistically indistinguishable or worse, leave the existing published method untouched. Do not promote based solely on a few wins, high synthetic confidence or model consensus.
- Investigate credible improvements next via **genuinely timestamped, licensed historical shot-based xG**, confirmed XI/injury evidence and properly calibrated bookmaker lines. At the time of implementation the archived V22 source audit showed 0/10 packets with historical xG and 0/10 with both starting XIs. Missing signals are not substitutes for human intuition.

Current workflow job: `python professional_analyst_shadow_v25.py`. Tests: `tests/test_professional_analyst_shadow_v25.py`.

**Limitations:** The lower envelope does not account for unmodelled dependence or match-state effects. Four models sharing expected-goal parameters are highly correlated. A high rank under these models does not establish a good bookmaker price, profitable strategy, or guaranteed hit probability.
