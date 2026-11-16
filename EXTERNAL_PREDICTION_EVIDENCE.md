# ZicZac external prediction research — 16 November 2026

This is a research checklist, **not** an assurance of winning or a claim that any competitor outperforms ZicZac on five two-match exact-score combinations.

## Define the target before comparing claims

For five-double exact-score prediction, each daily set contains **two distinct football fixtures** and **five distinct ordered (exact score A, exact score B) combinations**, frozen before either kickoff. A WIN requires that one of those five combinations matches both regulation-time scores. A correct 1X2 outcome, one correct leg, a top-five single-match score, and profit at arbitrary betting odds are **different metrics**. Never compare their advertised percentages as equivalent.

The present source-validated original V8 five-double diagnosis in `data/accuracy_diagnosis/report.json` has **six frozen sets, one independently verified loss and five pending sets**. The original models' estimated five-cell joint coverage is approximately **5.9%–7.6%**. This is a model estimate, **not** a measured frequentist hit probability. A single loss cannot establish long-run effectiveness. The existing source-grid optimizer already maximizes five-cell mass for those pairs; random reranking of the same scores is not expected to help.

## External evidence and limits

- [ScoreGPT accuracy](https://scoregpt.app/accuracy): The site publishes claims about 1X2 match outcomes and picks in different markets. Its 15 November 2026 page reported 478 of 594 high-conviction **match results**, not a five-double exact-score record. [Model detail](https://scoregpt.app/accuracy/scoregpt) keeps exact-score rates behind Pro. **Do not interpret a high 1X2 win rate as proof of exact-score double skill.**
- [Football Prediction AI results](https://www.footballprediction.ai/results): Self-reported published-card correct-score rate **10.8% (4,163 correct-score picks)**, compared with higher rates for double chance and other easier markets. Verify the dataset and timestamp rules before ranking externally.
- [Football-aware match-state simulation paper (arXiv:2608.05030)](https://arxiv.org/abs/2608.05030): A 2026 exploratory comparison on 150 English Premier League matches reports 10.0% baseline Top-1 correct scores versus 14.7% in a contextual variation, while acknowledging development-set and possible LLM look-ahead limitations. Treat as a *research hypothesis*, not a transferable estimate or proof of benefit.
- [Prediction-model comparison and chronological validation](https://www.sportsignals.com/resources/football-statistics-betting/statistical-models-overview): Compare like targets, same fixture evidence cutoffs and out-of-sample proper scores. A more complicated model is not guaranteed to forecast better.
- [Auditing free prediction sites](https://tipsaudit.com/blog/are-free-football-prediction-sites-accurate-we-started-measuring): This site's 16 November 2026 report warns that hit rates are not profit, and most tracked services had not shown a statistically reliable advantage over closing odds. Its findings are **their** reported dataset, not independently verified by ZicZac.

No competitor predictions should be scraped behind paywalls, cloned, or represented as original pre-match ZicZac forecasts. Respect provider terms and attribution.

## Engineering research priorities

1. **Integrity of pre-kickoff observations:** provider/fixture/team/kickoff identity, model version, source event timestamps, archived forecast IDs, snapshots; exclusion of stale, late, missing, regraded or retrospective data.
2. **Data availability and realistic abstention:** Identify missing independently sourced starting-XI, injuries/suspensions, verified shot-based xG/xGA, fixture congestion/rest and independently time-stamped bookmaker market snapshots. Missing evidence is a reason to abstain or reduce claimed confidence, not to invent inputs.
3. **Modelling as shadow experiments:** Compare dynamic opponent-strength models (Poisson/Dixon–Coles), league priors, proven tactical/context adjustments, and bookmaker-anchored probability calibration on shared pre-kickoff inputs. Constrain contextual/LLM adjustments; do not retroactively fit to known scores.
4. **Paired match selection:** Test whether choosing less correlated/less uncertain fixture pairs and a defensible no-pick option raises ***prospective*** five-set hit rates. Preserve original selections and all abstentions. If there are insufficient strong pairs, report no play.
5. **Probability-quality tests:** Use full exact-score log loss/Brier, top-1/top-5 **single-match** score coverage, calibration and actual five-cell two-fixture joint coverage separately. For doubles, report frozen daily sets, pending, WIN/LOSS, empirical hit rate and intervals. Report the theoretical top-five attainable mass under each model and the number of combinations the model needs for 70% mass. Model probability is not an empirical guarantee.
6. **Real-value validation:** Only where timestamped real market odds exist, measure fair probabilities, margin, quoted prices, closing-line value and realized flat-stake outcomes *separately* from accuracy. Never fabricate a bookmaker price, promise 80+ odds, or maximize odds by combining markets without calibrated joint probabilities.
7. **Promotion gate:** Hold out future seasons and independently evaluate at least 100 genuinely settled, identity-verified double sets **before initial review**; larger samples and robust uncertainty bounds are needed before any bold percentage claim. Report all selected and skipped days, protocol changes, and drift. Never auto-promote an experimental model solely on its first wins.

## What “think like a human” means technically

Write a pre-kickoff, human-readable evidence memo for each candidate: travel/rest, confirmed absences and lineup changes, tactical formations and matchup mechanisms, home/away asymmetries, incentive/context, bookmaker disagreement, and credible late news. Timestamp every claim and its source. **Only validated, causally plausible features** may modify the numerical goal distribution, with explicit before/after change and later prospective verification. LLM commentary without source evidence is explanation, not additional numeric precision.

Always keep the automatic verified-result watcher distinct from the improvement task; suppress duplicated notifications. The target of 70% five-double exact-score success is **aspirational**, currently unsupported, and should never be implied by a competitor's easier-market track record.
