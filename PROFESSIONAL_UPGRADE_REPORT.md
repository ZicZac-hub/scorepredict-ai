# Professional five-double upgrade: evidence and release report

Reviewed against `main` commit `671a8aed`, 18 November 2026 UTC.

**The 70% exact-score five-double target is not supported by the available results.** The frozen validation-selected candidate achieved **20/166 = 12.05%** in its 2024–25 retrospective. Later development comparisons achieved **7.83%–16.27%**. These do not establish a prospective improvement over live V8. The new professional automatic ledger has **zero settled sets and two pending sets**; premium selection therefore abstains.

A success requires one of five distinct rows to match the regulation-time final scores of **both** games. Single-match exact scores, top-five single scores, double-chance outcomes, estimated coverage and betting profit are different measurements.

## Verified weaknesses and changes

| Finding | Change and practical limit |
| --- | --- |
| The pinned `2019-20/en.1` source has 66 calendar anomalies; two official reports independently confirm the wrong year. | The new research runner quarantines the whole 380-match file. Original sources and experiments remain unchanged; no guessed date repairs. |
| Twenty 2024–25 final-round scores were missing, affecting daily pair opportunities. | Verified all 20 against official sources in a separate supplement, observed now. No original evaluation was rewritten. A new full-slate evaluation is still required. |
| High estimated coverage could be confused with demonstrated selection skill. | Added explicit professional optimization, raw probability labels, calibration-gated premium abstention and separate automatic research capture. No unproven model promotion. |
| Owner-supplied probabilities/timestamps cannot authenticate calibration trials. | Reconstruct choices and coverage from immutable source grids; verify digests, identities, source times, model versions and policy; reject reused events. |
| Persistence could receive a backdated forecast after kickoff. | PostgreSQL checks its receipt clock against the earliest kickoff before inserting. Existing originals remain preserved. |
| The unified evaluator omitted existing V25 private records. | Register V25 and the professional policy for result verification. |
| V17 outcome research could count relabelled fixtures or altered selections toward its claim gate. | Validate original source, Sydney date, probability and predeclared choice; prohibit fixture reuse and explicitly distinguish outcome research from exact-score doubles. |
| AI providers received inconsistent or partly unqualified context. | Use a common evidence dossier with reproducible evidence/prompt hashes. AI opinions retain zero numerical weight. |
| Existing responses discarded correct-score odds and historical shots; collection diagnostics miscounted pages. | Retain source-qualified prices/statistics without additional requests; add rejection categories and fix cumulative page counting. |
| Pro-only navigation depended on the primary catalog; featured forecasts could change during live play. | Allow supported routes with Pro access; preserve the original featured forecast and validate result identity. |
| Feature-branch dispatch could enter a publishing workflow. | Guard relevant collection/audit jobs to `main`; add separate manual, read-only model research. |

The previously merged repair removed unsupported AI ranking boosts and fixed response-completion timestamps and historical pair cutoffs. Its findings remain in [FIVE_DOUBLE_ROOT_CAUSE.md](FIVE_DOUBLE_ROOT_CAUSE.md); these are not newly claimed accuracy gains.

## Data, source research and chronological safety

The CC0 OpenFootball snapshot is pinned to `e6744429ee395bc86f247348c6184bb08d4eb361`: **30 hash-verified files**, five leagues (England, Germany, Spain, Italy, France), six seasons (2019–20 through 2024–25). It contains **10,808 schedule rows**, **10,687 played matches**, 101 cancelled French fixtures and 20 missing finals. The original split was **7,203 training / 1,752 validation / 1,732 holdout matches**.

The frozen model was selected by 2023–24 exact-score log loss before scoring 2024–25. Monthly fits use only dates strictly before the month start minus one day. The source supplies calendar dates, not verified kickoff/result-publication times. Historical slates use source dates, not invented Sydney kickoffs; exact real-time availability of every historical result is unproven. The [post-holdout audit](data/target70_research/evaluation_audit.json) checks nine provenance/selection conditions, raw hashes, fixture/result identities, distinct choices, training cutoffs and computed wins. Local hash ordering supports reproducibility, not an independent public preregistration.

Official Premier League reports confirm [Bournemouth 1–4 Newcastle on 1 July 2020](https://www.premierleague.com/en/news/1697163/match-summary-afc-bournemouth-v-newcastle-united) and [City 4–0 Liverpool on 2 July 2020](https://www.premierleague.com/en/news/1701434/man-city-cruise-to-victory-over-liverpool), while the raw file says July 2019. Two events were independently checked; all 66 anomalies were not individually verified. This can leak later results into a 2019 replay and distorts age weights. Their confirmed actual dates still precede the evaluation seasons, so future-test outcome leakage into 2023–24/2024–25 is **not demonstrated**. Quarantine retains **10,307** played matches. See [source audit](data/source_quality_audit_v1/audit.json).

The raw holdout spans **185 source dates**, versus **183 played-only dates**: May 23/24 disappear and May 25 has a partial slate. Filtering missing outcomes can change pair selection. The [official-source supplement](data/source_quality_audit_v1/final_round_supplement.json) now supplies the 10 Spanish and 10 Italian missing finals, with source URLs and row identities. It records observation in November 2026 and leaves historical availability unknown. It has not been applied to the frozen experiment or sensitivity; post-match reports never become pre-match features.

## Historical model results

The source-quality sensitivity below excludes the affected English file without changing hyperparameters. All models share **1,710 evaluated single fixtures**; each pair policy issued **166 sets**, abstained on **17 of 183 played-source dates**, and may choose different pairs. Lower single-score log loss is better. These 2024–25 outcomes have already been explored: this is development evidence, not new unseen or prospective validation.

| Model | Five-double hits / issued | Set hit rate | Single-score log loss |
| --- | ---: | ---: | ---: |
| League independent Poisson | 14/166 | 8.43% | 3.036104 |
| Time-weighted strength Poisson | 22/166 | 13.25% | 2.914630 |
| Dixon–Coles, ridge 6 / half-life 240 days | 20/166 | 12.05% | 2.914554 |
| Dixon–Coles, ridge 18 / half-life 365 days | 27/166 | 16.27% | 2.913225 |
| Bivariate Poisson, fixed common component 0.12 | 20/166 | 12.05% | 2.918461 |
| Negative binomial, fixed dispersion 3 | 13/166 | 7.83% | 2.996118 |
| Gradient boosting with Poisson loss | 18/166 | 10.84% | 2.930319 |
| Validation-fitted ensemble / temperature comparison | 23/166 | 13.86% | 2.914787 |

Ensemble weights use earlier 2023–24 data; temperature uses its later 900 rows, selecting **1.0**, rejecting extra sharpening. Clean gradient boosting uses **3,521 earlier training labels** and fixed settings. Bivariate/negative-binomial parameters are fixed challengers, not validated production effects. Complexity did not consistently improve scores or hits. Quarantine leaves seven hit counts unchanged; gradient boosting falls from **19/166 to 18/166**. This does not justify retaining incorrect dates. See [sensitivity deltas](data/source_quality_sensitivity_v1/sensitivity_delta.json), [full compressed output](data/source_quality_sensitivity_v1/comparison.json.gz) and [summary](data/source_quality_sensitivity_v1/summary.json).

### Calibration and the frozen candidate

The original validation-selected `opponent_dc_r6_h240` remains **20/166 = 12.05%**, descriptive Wilson 95% interval **7.94%–17.88%**. Football trials may be serially dependent; this is not a dependence-adjusted interval. Its mean estimated coverage was **8.92%**. Binary set Brier was **0.108747**, worse than **0.106519** for a validation-prevalence constant. Descriptive post-holdout bin error was **0.057248**; no calibration was fitted to those bins.

Requiring at least 10% estimated mass produced **4/53 = 7.55%**, with 130 abstentions. A 20% threshold issued nothing; zero selections do not establish 70%. The 27/166 result was examined after frozen selection and cannot replace that candidate based on this holdout.

### Old versus new, with separate denominators

| Cohort | Original/control | Challenger | Interpretation |
| --- | --- | --- | --- |
| Earlier 122-date replay | V8 5/122 = 4.10% | Previous variants also 5 hits | Existing development data. |
| Earlier 34 common-decision dates | V8 0/34 | Negative binomial 1/34 | Separate small historical source. |
| New source-screened 166-date comparison | League Poisson 14/166 | DC 20/166; exploratory ridge-18 DC 27/166 | Different policies may choose different pairs; no matched live-V8 improvement claim. |
| Available original prospective checkpoint | V8 0/1 verified, 5 pending | No settled matched comparison | Insufficient live evidence. |
| New professional automatic ledger | No settled control comparison | 0 verified, 2 pending | No observed rate or confidence interval yet. |

Do not pool cohorts or claim the website improved from 0%/4.10% to 16.27%.

## Optimizer and prospective recording

Under independent matches, exact-score joint probabilities multiply and distinct final-score pairs are mutually exclusive. The sum of the **largest five joint cells** maximizes coverage. Each leg's leading five cells suffice to find these products. Full normalized grids are validated; no diversity boosts, forced high scores or AI bonuses are added. Known dependence without a validated joint model causes abstention; otherwise independence remains an explicit assumption.

Manual selection supports qualified upcoming fixtures. Automatic research compares all eligible same-Sydney-date pairs with four distinct teams and explicit source times within 24 hours. Premium also requires trials for the same model versions and exact raw automatic policy: **50 prior training / 50 prequential test pairs**, **20 in the current fixed probability bucket**, and a positive 95% lower bound for prequential Brier improvement. These are readiness rules, not proof of 70%, profitability or independent trials. Manual and premium-scope records do not silently bootstrap the raw-policy cohort.

Public write-once sets record source references, digests, five choices, capture time, estimated coverage and versions. Evaluation reloads original grids and verifies regulation results; missing/conflicting results remain pending. The two new sets are for Sydney **November 19 and 20**, with raw coverage **7.09% and 6.70%**. Local capture times are recorded; publication is the externally visible evidence boundary. Premium remains unproven.

## Research engine, AI and market coverage

The common dossier includes actual-count last 5/10/20 windows, venue form, scored/conceded goals, descriptive H2H, known schedule gaps, source-qualified historical xG/xGA and shots/on-target when available, reported injuries/suspensions, lineup status, formations and timestamped markets. Features carry sources/observation times; conflicting, wrong-competition and late evidence is excluded. Partial history is labelled. Schedule gaps do not prove full rest, missing injury reports do not establish fitness, and model scoring means are not shot xG.

Configured OpenAI/ChatGPT, Claude, Gemini and existing adapters receive the same qualified dossier for a fixture. Their explanations can discuss tactics and market disagreements; votes have zero numerical weight until independently validated. No extra providers or paid budgets are activated.

Historical xG, market-implied and AI variants cannot be fairly tested on this result-only CC0 source. Existing V22/V23 source-backed models remain for prospective comparison. Collection is currently **enabled**: at the November 18 checkpoint, 53 packets existed; six had genuine historical xG/statistics; none had both confirmed XIs or reported injuries. V24 matched markets to 8 of 211 eligible forecast records, leaving 203 unmatched. Sportmonks xG discovery returned `provider_access_or_subscription` despite a configured token. The [source review](data/diagnostics/source_evidence_review.json) retains dated older and current snapshots; collection activity alone is not an accuracy gain.

Correct-score prices from existing responses retain bookmaker, identity and timing. Partial menus are not normalized into complete score distributions. Leg-price products/expected returns are labelled illustrative and non-executable, using uncalibrated individual probabilities even if aggregate coverage is later calibrated. **ROI is unavailable** without actual accepted double prices, stakes and returns.

## Automation, interface and validation

- Scheduled collection adds professional shadow capture/evaluation using **zero additional sports/AI requests**; originals remain write-once.
- Manual research has read-only permissions, no provider secrets, source quarantine by default and artifact upload. It cannot deploy a model.
- Studio retains existing engines and adds the professional optimizer; Model Benchmark displays the screened development comparison. Existing routes, Sydney timezone handling and old versions remain.
- Regression tests exercise source/analysis delays, identity, grid optimality, same-day pairing, real price-parser packets, dossiers, calibration provenance, persistence and archive preservation. Streamlit AppTest checks rendering/navigation. No real iPhone/MacBook browser session was available.
- PostgreSQL integrations run in GitHub CI. Local database skips are not reported as passing. Final test counts and CI evidence are recorded in the PR.

Final local regression run: **864 passed, 17 database-dependent skips, 8 subtests passed**. All modules compiled and `git diff --check` passed. A fresh read-only provenance check passed all nine frozen-experiment conditions; compressed report hashes and all 20 separate official-source label identities were checked again. The optional research dependency was installed locally; the production CI environment may skip optional gradient-boosting cases when it is absent.

## Remaining gaps and highest-value next improvement

Still missing: adequate prospective samples; a separately fixed complete-slate protocol applying the new labels; broader verified dates and real kickoff/availability times; enough matched xG/market/lineup coverage; validated keeper/player/tactical/weather/travel effects; executable prices/stakes; and matched-pair ablations over an untouched future period. Provider coverage/subscription responses constrain existing credentials. Unavailable private Streamlit/PostgreSQL records were not claimed as inspected.

The highest-value next step is fresh, identity-matched odds and genuine xG/lineups for the **same immutable upcoming fixtures**, with a predeclared challenger versus preserved V8/simple baselines on the same pairs. Accumulate future settled trials before recalibration or promotion. The new shadow path starts that record. Repeated exploration of 2024–25 cannot become unseen evidence; 70% remains unsupported.

## Reproduction

```sh
python -m pip install -r research-requirements.txt 'pytest>=8,<10'
python -m pytest -q
python run_professional_research.py --include-gradient-boost
```

Choose a new empty `--output` for each run. Default research quarantines the defective file; `--reproduce-original` explicitly reproduces the preserved affected control. Downloads/uncompressed large reports stay untracked; compressed evidence and hashes remain reviewable. No command promotes a live model. Software tests demonstrate tested behavior, not forecast accuracy or error-free future operation.
