# Reference sources and implementation boundaries

## AI references

Checked 13 November 2026. These are public primary sources, not independent
verification of marketing claims. No other provider's model was called and no
proprietary predictions or code were obtained. The likely intended ScoreGPT
product is scoregpt.app; the user's unqualified name could refer to another
product, so identify this reference explicitly.

## Verified sources and useful lessons

1. **ScoreGPT methodology** — https://scoregpt.app/methodology
   The service describes separate LLM calls on the same API-Football match data,
   structured outputs, averaged home/draw/away probabilities, per-model
   agreement, and a complete graded record. Its scoreline is supplied by a
   writing step consistent with the aggregate favourite; it is not described
   as the maximum of a calibrated full score distribution. Use the comparable
   evidence dossier, explicit contributor identity, recorded disagreement and
   immutable grading concepts. Keep our actual contributors accurately named.
   Its public methodology says published predictions lock before kickoff.
   Rescored versions should therefore remain separate from originals.

2. **ScoreGPT accuracy definitions** — https://scoregpt.app/accuracy
   Public pick and match-outcome accuracy are distinguished from exact-score
   accuracy, which requires its Pro app. Sample sizes and windows vary. Do not
   copy its displayed pick percentages as exact-score performance or treat
   the service as an independently established market leader.

3. **OpenAI structured outputs** —
   https://developers.openai.com/api/docs/guides/structured-outputs
   JSON Schema constrains completed non-refusal responses, while applications
   must still handle API failures, incomplete responses and business rules.
   Transferable lesson: validate contributor, timestamp, fixture identity,
   finite probabilities, normalization, score-grid support and evidence fields.
   Schema adherence does not establish calibrated football probabilities.

4. **OpenAI web search** —
   https://developers.openai.com/api/docs/guides/tools-web-search
   Its search integration supplies current information with source citations.
   Keep clickable supporting sources and retrieval timestamps in any future
   evidence-based explanation. This is API functionality, not an existing
   connection in the app or a licensed football feed.

5. **Claude citations** —
   https://platform.claude.com/docs/en/build-with-claude/citations
   Claude supports claims linked to passages in supplied documents. The useful
   design principle is traceable evidence, rather than treating generated
   prose as new match data. Actual Claude participation requires a real
   configured integration and a stored response; a local statistical model
   must not be labelled Claude.

6. **DeepSeek JSON Output** —
   https://api-docs.deepseek.com/guides/json_mode/
   JSON mode provides valid JSON strings. Documentation notes token truncation
   and blank-content cases that callers must handle. Parse and validate all
   fields; never turn an empty response into a fabricated contributor vote.

7. **Gemini structured output** —
   https://aistudio.google.com/docs/structured-output
   The official ai.google.dev URL redirects here. JSON syntax and schema
   constraints do not ensure semantic correctness; client-side validation is
   necessary. Enforce probability sums, supported score targets and chronology
   independently of generated output.

8. **Gemini grounding with Google Search** —
   https://aistudio.google.com/docs/google-search
   The official ai.google.dev grounding URL redirects here. Grounding connects
   Gemini to current web information and returns cited sources. Relevant
   future use is timestamped verification of factual team news, with explicit
   unknowns; a grounded explanation alone is not a fitted player adjustment.

## Recommended implementation boundaries

- Treat ChatGPT/OpenAI as one provider family, not two independent votes.
- Describe a local statistical ensemble as such. Only show OpenAI, Claude,
  DeepSeek or Gemini as contributors after actual successful model calls.
- Keep outcome-favourite and most-probable exact-score labels separate.
  Selecting a score to agree with the favourite is a different objective.
- Use identical pre-kickoff evidence for comparisons. Frozen original records
  and later revisions need model versions, evidence cutoff and creation time.
- Completed-match rescoring is retrospective evaluation using only earlier
  fixtures. It cannot be added to the original prospective accuracy tally.
- Evaluate each method and ensemble on the same fixtures, showing exact hits,
  top-three coverage, score-distribution loss, 1X2 calibration and sample size.
- Do not claim model independence merely because related statistical variants
  have different names, or claim any unmeasured exact-score improvement.
- These public methods and engineering patterns do not reveal proprietary
  prompts, weights or trained models, and do not establish a ranking of sites.


## Football prediction references

Reviewed 13 November 2026 (Australia/Sydney). These are established public examples, not a ranking of independently proven exact-score accuracy. No proprietary models, source code, paid data, or prediction feeds were accessed or copied. Source descriptions below are paraphrases.

| Reference | What its own public material establishes | Useful implementation direction | Limits |
| --- | --- | --- | --- |
| [Forebet FAQ](https://www.forebet.com/faq/) | Forecasts are generated from historical sports statistics using mathematical/statistical analysis; the publisher says no human adjustments alter them. It distinguishes upcoming, live, and finished forecasts and displays average-goal estimates. | Publish score probabilities alongside recent history and a separate result ledger. Keep numerical forecasts reproducible; keep editorial context distinct. | The FAQ does not disclose full estimation equations or independently verified exact-score performance. Do not say this app reproduces Forebet. |
| [PredictZ](https://www.predictz.com/) | The public page presents exact-score selections, form/statistics, league tables, results, and accumulator selections. | Put one score pick in context with form, alternatives, and the outcome probability. Show doubles separately from single-match exact scores. | No reproducible fitting algorithm was disclosed on the inspected public page. Its dated picks cannot substantiate an accuracy ranking. |
| [WinDrawWin statistics](https://www.windrawwin.com/statistics/) | Official search extracts describe a large statistics database used to formulate tips, including goal and correct-score statistics. | Show competition-specific goal distributions and evidence behind selections. | Direct pages returned HTTP 403 to the browsing tool; only official search extracts were available. No formula or exact-score success rate was verified. |
| [Opta Analyst model explanation](https://theanalyst.com/articles/opta-football-predictions) | Its league model derives match-outcome probabilities from market odds and Opta Power Rankings, accounts for opponent strength, then simulates remaining fixtures. | Fit opponent-adjusted team ratings. If licensed timestamped odds become available, compare against overround-adjusted odds before considering an independently validated blend. | The explanation concerns outcome/season probabilities; it does not reveal a complete exact-score model. This project has no Opta feed or licensed market feed by default. |
| [Stats Perform Opta Predictions](https://www.statsperform.com/products/opta-predictions/) | The provider describes probabilities informed by historical/current sports data, team/player strength, venue, and game state. | Treat data quality and timestamped context as inputs to test, not reasons for arbitrary multipliers. | Provider product claims are not independent validation; paid player/event data were not obtained. |

## Primary modelling sources

- [Dixon and Coles (1997), Modelling Association Football Scores and Inefficiencies in the Football Betting Market](https://academic.oup.com/jrsssc/article-abstract/46/2/265/6990546), DOI [10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065). The published abstract establishes a Poisson-regression approach accommodating changing team performance. Practical direction: estimate league-specific attack/defence and home effects from past data, decay older observations, and estimate low-score dependence rather than hardcode a correction. The publisher abstract was searchable; full article access was not required or bypassed.
- [Karlis and Ntzoufras (2003), Analysis of sports data by using bivariate Poisson models](https://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf), DOI [10.1111/1467-9884.00366](https://doi.org/10.1111/1467-9884.00366). Shared goal dependence can affect draw forecasts. This is a candidate to compare with independent Poisson/Dixon–Coles, not evidence that adding correlation necessarily improves this app.
- [Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), DOI [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437). Proper scores reward honest probability distributions. Evaluate full-grid log loss, outcome Brier/calibration and exact/top-three hits on chronological held-out matches; do not optimise score diversity as if it were accuracy.

## Recommendations for this repository (our inferences, not third-party formulas)

1. Build an independently implemented fitted opponent-adjusted goal model using complete provider-qualified competition histories. Estimate home advantage and time-decayed attack/defence, with regularisation selected only before the held-out period. Estimate a valid low-score correction from training data, and report when data are insufficient.
2. Return a coherent normalised score grid. Derive the top score, top-three coverage, 1X2, BTTS and totals from that same grid. Expected goal counts estimated from results must not be described as observed shot-quality xG.
3. Freeze candidate parameters before chronological evaluation. Report exact-hit count, full-grid log loss, Brier, coverage and calibration on identical paired subsets; a reduction in 1–1 repetition alone is not a successful validation.
4. Preserve every original prediction and original five-double snapshot. Add model-versioned revisions with generation timestamp, feature cutoff and input provenance. Label finished-match reconstructions as retrospective and exclude them from original live accuracy. Historical team news/odds require historical observation timestamps; present-day fields cannot reconstruct what was known before kickoff.
5. For five doubles, recompute each leg from the revised grid, retain the original leg/selection identity, show joint probability and the independence assumption, and avoid presenting overlapping pairs as independent evidence. Ranking/coverage changes need their own paired tests.
6. Treat external AI names and prediction sites as documented references unless an actual authorised API call supplies a recorded, timestamped response. Do not manufacture provider votes, label several variants as independent AIs, or infer access to proprietary models.

No evidence found establishes which public website is most accurate for this user's 236-match sample. Those original records need verified provider/team/kickoff/result joins before any such comparison.
