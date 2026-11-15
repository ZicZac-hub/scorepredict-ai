# Five exact-score doubles: root cause and repair evidence

Audit of `ZicZac-hub/scorepredict-ai` at source commit `03e4298d`, 15 November 2026 UTC.

The requested **70% success rate is not demonstrated**. The repair fixes verified selection and source-timing errors; it does not manufacture a successful model. A successful set means **at least one of its five rows matches both final regulation-time scores**. It is different from predicting a winner, a single match's double-chance outcome, or having either score appear somewhere in the table.

## What failed

### 1. Five combinations cover too little of these score distributions

The new [source-validated audit](data/accuracy_diagnosis/report.json) follows each frozen pair back to its exact archived V8 forecast IDs and complete probability matrices. All six saved sets already maximize the joint probability of five combinations under the independent-match assumption. Their estimated coverage is **5.9238–7.6198%**, with **none reaching 70%**. Reordering five combinations cannot raise coverage beyond the largest five cells of the same joint grid.

For the original Sydney-date **15 November** set:

| Match | Verified actual score | Actual score rank in its leg | Scores included in the five selected combinations |
| --- | --- | --- | --- |
| Chapecoense-sc vs Vasco DA Gama | 0–2 | 4 | 0–1, 1–1, 1–2 |
| Mirassol vs Atletico-MG | 0–0 | 4 | 1–0, 1–1 |

The actual pair ranked **31st**, with **0.6346% model probability**. Neither actual score was among the selected leg options. The five selected rows covered **5.9238%** of the source model's probability. Reaching **70% of this same model's estimated mass would require 156 combinations**, rather than five. That is a diagnostic of the present distribution, not a recommendation to buy more combinations or a promise of 70% observed success.

No market, validated lineup adjustment or verified shot-quality xG was applied to either original forecast. Adding arbitrary boosts cannot substitute for those missing measurements. Nor does their absence prove which particular feature would have predicted these results.

### 2. The Studio's default could quietly sacrifice model coverage

`merged_five(selection_mode="probability")` previously multiplied ranking scores by up to **1.12 per leg** for AI endorsements. The UI called this the original top-probability baseline even though it could replace more probable combinations. The regression reproduces a drop from **7.3840% optimal modeled coverage to 7.3545%** under the former ranking.

**Fixed:** the default now uses pure model joint probabilities. AI endorsements remain visible as annotations and have **zero numerical weight**. The comparison baseline also receives no AI votes. The optional balanced selection remains explicitly experimental; diversity is not evidence of higher accuracy. These changes fix a selection inconsistency, not a measured gain in future hit rate. The already-frozen November 15 miss used the optimal original five, so this bug is **not** a causal explanation for that miss.

### 3. Slow provider responses used the wrong observation time

V21, V23 early odds and V24 broad capture could stamp source evidence with the request/job start rather than response completion. A slow request could cross kickoff and still appear pre-match. Conversely, genuine first-party fetch timestamps just after job start could be rejected as future observations, suppressing valid market evidence.

**Fixed:** collectors advance the observation clock while requests run, timestamp completed responses honestly, reject data received after kickoff, and validate the Studio against its refreshed clock. Packet validation now also enforces the API-Football provider identity. The Studio checks its actual analysis completion time too, and will not generate a new prediction or start another AI leg request after the eligible pre-match window expires. Request accounting counts malformed or failed requests once. Deterministic delayed-response tests cover both late-data rejection and valid after-start odds acceptance.

This is a correctness defect demonstrated in tests. The audit does not assert that previously committed packets exploited it or that fixing it increases accuracy by a known amount.

### 4. Historical pair comparisons could include later information

The V21 benchmark originally reused each fixture’s individual forecast made one hour before its own kickoff when choosing a pair across the day. A later fixture’s evidence could therefore become available after the earlier selected match had started. Individual-match metrics had a valid individual cutoff, but the pair comparison did not ensure a single decision time before both matches.

**Fixed:** pair candidates are now reconstructed at one common cutoff, one hour before the earliest kickoff in that daily slate, with the three-hour prior-result delay applied there. Eligibility is recalculated at that time. The rerun reports every day’s cutoff, selected forecast times and exclusions. Its 34 eligible pair days replace the invalid prior 37-day comparison; individual-match metrics remain unchanged.

### 5. Cached started fixtures could enter a new live pair

The legacy live `rank_pairs` path checked dates but not the current clock or four distinct teams. A cached fixture could therefore appear in a new recommendation after kickoff, or a pair could share a team.

**Fixed:** live callers supply the current UTC time, require more than five minutes until both kickoffs, and require four known, distinct team names. Historical replay remains possible without applying today's clock. The date rule remains Sydney-local, including daylight saving.

## Measured evidence, with separate denominators

| Evidence | Actual result | Meaning |
| --- | --- | --- |
| Original prospective five-double archive | **0 / 1 verified sets**, **5 pending** | Too small to estimate a dependable future rate. Wilson 95% interval: 0–79.35%, under independent-trial assumptions. Pending sets are excluded. |
| Prior archived 122-day chronological replay | **5 / 122 = 4.10%** for original V8 | Retrospective development data; not independent future validation. All reported V7/V14 selection variants also had 5 hits. Mean estimated V8 coverage was 7.45%, maximum 11.80%. |
| Fresh reproduction from the preserved 2,255-match source, last 500 targets | **436** eligible V8 forecasts; **49 / 436 = 11.24%** single exact-score hits; top-five **198 / 436 = 45.41%** | Single-match metrics must not be presented as double-set accuracy. Predictions use history available one hour before kickoff with a conservative three-hour prior-result delay. |
| Same fresh replay, selected two-match sets | V8 **0 / 34**; negative-binomial challenger **1 / 34**; other variants 0 wins | No tested variant approaches 70%. V20 issued only 28 of these 34 sets; its missing sets are not wins. The comparison is retrospective and cannot justify promoting the best-looking variant. |
| Strict same-fixture V8/V20 cohort | V8 **43 / 393** top exact hits; V20 **41 / 393**; score log loss **3.01183 vs 3.01486** | Additional goal-form weighting did not improve the proper probability score. This reproduces the existing decision to retain V8 as the control. |

See the full [reproduced model comparison](data/accuracy_diagnosis/historical_replay.json) and [artifact provenance, checksums and original 122-day summary](data/accuracy_diagnosis/historical_provenance.json). The source ZIP SHA-256 matches the digest returned by GitHub Actions. No results were edited or selected to improve the displayed rate.

## Research supporting the approach

- [Dixon and Coles (1997), Lancaster University publication record](https://research.lancaster-university.uk/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-f/): Poisson regression can account for changing team performance; it does not establish 70% exact-score-double accuracy.
- [Gneiting and Raftery (2007), original paper](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf): evaluate full distributions with proper scoring rules and compare models on the same forecast situations. Sharpening probabilities alone does not establish calibration.
- [UC Berkeley probability foundations](https://stat20.berkeley.edu/summer-2026/3-generalization/01-prob-foundations/lecture1-probability.html): independent events multiply and mutually exclusive alternatives add. Here the modeled coverage is the sum of five distinct products, not five independent chances of winning.
- [API-Football official guide](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide): odds availability and lineup coverage vary, prices update periodically and historical availability is limited. Missing archived pre-match information cannot be reconstructed from today's prices.

These sources justify methods and constraints. None is evidence that this app has reached the requested target.

## Operational update and reproducibility

- `python five_double_accuracy.py` regenerates the read-only miss/coverage report without provider requests. It rejects mismatched source grids, duplicate combinations, invalid probabilities, later source snapshots and unverified/wrong-identity results. Conflicting results remain pending.
- Both Five Doubles and the AI Studio expose the scheduled diagnosis and its downloadable raw evidence. The existing collection workflow refreshes it with no extra sports or AI API calls.
- The original score grids, frozen selections, verified outcomes and private account settings remain intact. Provider opt-ins and request budgets are unchanged.
- Reproduce the historical comparison by downloading the source artifact linked in `historical_provenance.json`, verifying its hashes, and running its exact command.

The public repository does **not** contain the private Streamlit/PostgreSQL prediction archive. This investigation covers the available public frozen records, authentic historical artifact and source code; it cannot claim to have inspected unavailable private results. The existing live collection and verified-result pipeline must accumulate prospective evidence before any future accuracy claim. A zero-error or 70% future guarantee is not supported by these tests.

## Validation of this repair

Final local run: `python -m pytest -q -rs` — **572 passed, 16 PostgreSQL integration cases skipped, 8 subtests passed**. The isolated PostgreSQL service is supplied by the existing GitHub prediction-test workflow; the local workspace has no test database. All Python modules compile and `git diff --check` passes. The skipped tests are storage integrations, not forecast accuracy measurements. The 22 new source-audit cases, delayed-response/analysis timing regressions, full-grid geometry check and historical pair-cutoff regressions are included in the passing total.

Software regression success establishes the tested behavior. It is not a football hit-rate measurement or a guarantee of error-free future operation.
