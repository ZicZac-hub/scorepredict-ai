# Matched five-double evidence update — 19 November 2026, Sydney

This implements the next recommendation from the [professional upgrade report](PROFESSIONAL_UPGRADE_REPORT.md): collect fresh source-matched evidence for the original two fixtures and measure which inputs improve forecasts. It does **not** establish a 70% hit rate or promote an unvalidated model.

## Measurable collection gaps

The existing near-kickoff refresh ran only 8–35 minutes before a fixture. Confirmed lineups can be available around an hour before kickoff. The collector also prioritized individual fixtures; the later match in an original double could remain outside its refresh window until the first match had started. Its old completeness check counted total xG rows and lineup rows, which did not prove adequate samples or confirmed lineups for both distinct teams.

At the checked repository checkpoint (`a75357cb`, source report 19 November 08:25 Sydney), the provider archive contained **64 packet records**, eight with six historical xG fixtures per team, and **zero with both confirmed starting XIs**. These are packet counts, not 64 distinct researched matches. Four of six daily source attempts had been used, with two reserved for near-kickoff collection. The fixed pair's available evidence was older than the new three-hour component limit. Collection activity and source availability are not accuracy gains.

## Bounded source refresh

- Expand the one-time near-kickoff refresh to **8–65 minutes** before kickoff.
- Within that window, prioritize the authenticated original professional pair when both matches are still future and the later kickoff is within three hours. This permits the later match to use the reserved allowance before the common decision cutoff.
- Keep the existing **two attempts per run / six per UTC day** defaults and existing single-refresh markers. Attempts are reserved before requests; failures consume the same attempt allowance. These are capture-attempt limits, not HTTP-request limits: one attempt can use multiple existing endpoints.
- Preserve the original packet and write a separate version. Recheck the actual completion clock and reject late observations.
- Diagnose recent observations, fresh normalized winner/total markets, per-team unique historical xG counts and identity-qualified confirmed XIs. Missing injury reports never establish fitness. Conflicting lineup rows and repeated players do not establish a complete XI.

The refresh does not guarantee a lineup: provider timing, coverage, credentials, remaining quota and the single allowed attempt still matter. No new API subscription, AI provider or unrestricted request budget is enabled.

## Fixed prospective experiment

The study takes its pair from the previously frozen, authenticated professional raw-baseline selection. The new challenger cannot select an easier pair. On the first scheduled pass **more than five and at most 60 minutes before the earliest kickoff**, it saves all available arms at one common evidence cutoff:

| Arm | Numerical inputs |
| --- | --- |
| Original V8 control | Unchanged original full score grids |
| Market only | Original grids plus existing bounded market transform |
| Historical xG only | Original grids plus the existing 25% historical xG blend |
| Historical xG + market | Fixed xG blend followed by the same market transform |

The fixed market settings are strength 0.35 and maximum total-variation movement 0.15. xG requires six genuine historical fixtures per team and three relevant venue fixtures. No parameters are fitted to the newly arriving outcomes. Lineups and AI explanations have **zero numerical weight** until separately validated.

Each component must come from an identity/hash-qualified packet observed within three hours of the decision; market source updates must be no older than 12 hours. Select the newest usable packet independently for each component, so a newer odds-only packet does not hide valid xG in a separate fresh packet. Every component retains its own exact packet hash, reference and observation time. Packets are never merged under a fabricated timestamp.

Missing components produce an explicit arm abstention. The original baseline remains an opportunity even when every challenger abstains. The trial is write-once: later lineups, prices, missing inputs or final scores cannot rewrite it. The original baseline may be older than the new evidence; this measures the entire evidence refresh, not a model-only effect at equal information age.

The immutable protocol records decision rules, settings, component rules and implementation/dependency hashes. Publication provides an external evidence boundary; local timestamps alone are not independent preregistration. A later algorithm/dependency change requires a new study version rather than rewriting a registered protocol.

## Verification and website reporting

Evaluation reloads the original pair, exact source forecasts and provider packets; reconstructs grids and the five optimal distinct doubles; and grades only identity-matched regulation-time results observed by the evaluation cutoff. Conflicting or missing results remain pending. Reused fixture/event trials are rejected. Reading results never creates a missing protocol or substitutes a cached success report.

For every arm, report issued, abstained, pending and verified sets separately, plus wins, observed hit rate, descriptive Wilson intervals, single-score log loss and outcome Brier. Matched comparisons use only the arm's common verified cohort with the original baseline. Positive loss advantage means lower challenger loss. Confidence intervals are descriptive and rely on independence assumptions that football data may violate.

Single-score log loss and 1X2 Brier compare probabilities for the **same actual event**. Each arm's binary set Brier concerns its own five selected score combinations; those events can differ, so cross-arm set Brier is descriptive and cannot prove proper-score superiority. Individual exact-score accuracy is never relabelled as five-double accuracy.

Model Benchmark adds an authenticated prospective display and downloads. It recomputes from immutable evidence instead of trusting `performance.json`; zero settled trials display accuracy as unavailable. The scheduled workflow runs the study after source collection and archives its protocol/trials. Original historical models, source files and five-double sets remain unchanged.

## Results and remaining limits

At this implementation checkpoint the original professional ledger still has **zero settled sets and two pending sets**. No new verified live improvement, ROI or 70% result is established. The earlier historical result remains **20/166 = 12.05%** for the frozen candidate; the best later exploratory development result remains **27/166 = 16.27%**. Those retrospective figures are not prospective evidence for this new study.

The first no-API run registered the protocol before either pair's decision window; it issued **zero new trials**, with no source rejections. See the [fixed protocol](data/professional_paired_study/protocol.json), [launch source review](data/professional_paired_study/launch_review.json) and [initial evaluation](data/professional_paired_study/performance.json).

Local software validation: **902 passed, 17 database-dependent skips, 8 subtests passed**; compilation and diff checks passed. The no-API study run blocked outgoing provider requests explicitly. Final GitHub CI evidence is recorded in the pull request. Regression coverage includes source-budget preservation, immutable versions, genuine per-team coverage, delayed completion, original-pair authentication, missing-arm denominators, future-input/result rejection, tampered grids/protocols and website handling of unavailable accuracy.

The next required evidence is settled future matched trials. A complete-slate historical protocol applying the separately verified missing outcomes remains separate work. Executable accepted double prices, stakes and returns are still needed for ROI. No real iPhone/MacBook browser session or guaranteed future accuracy is claimed.
