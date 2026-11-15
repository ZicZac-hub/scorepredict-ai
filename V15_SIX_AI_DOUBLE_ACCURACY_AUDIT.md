# ZicZac V15 — Why 6+ AI connections do not yield 7/10 exact-score doubles

Date: **15 November 2026**. Audit of this repository's actual saved records,
not hypothetical provider responses. Original forecast archive unchanged.

## Current facts observed in the real repository

- `data/llm_research/status.json` showed six **configured** providers:
  OpenAI, Claude, Gemini, Grok, GroqCloud and Cohere. Its latest scheduled pass
  reported **0 attempted** fixtures / **0 captured**; this alone does not mean
  all six are sending valid responses.
- The **two** archived individual AI research records for API-Football
  fixtures 1492312 and 1492455 each contain **one actual Gemini vote**.
  The other configured providers have no saved successful votes in those two
  records. These old requests were made before kickoff but are NOT six-model
  consensus, and neither builds a genuine two-match five-double panel.
- `data/llm_research/evaluation.json` reports **0 verified complete
  five-provider panels and 0 verified ten-provider panels**. No empirical
  multi-AI exact-score hit rate, let alone five-double hit rate, exists.
- The GitHub scheduled panel had `MAX_MATCHES_PER_RUN = 1`, and heavy
  multi-AI research occurs on the six-hour schedule. A five-double needs
  matched evidence on **both distinct fixtures**, so this is an important
  coverage limitation. It does not prove that a single API call caused
  previous losing doubles.
- The Streamlit website has a separate Secrets store. Keys configured
  in GitHub Actions are **not automatically shared** with Streamlit.
- Previous prospective performance:
  `data/v10_prospective/performance.json`: 1 verified V8 two-match set
  (0 wins), 5 pairs pending at the audited snapshot. This is **insufficient
  to estimate a stable live win rate**.
- The existing 122-day retrospective 2026 tournament on up to 1,300 target
  fixtures: **5/122 winning V8 five-double sets (4.10%)**. V7 diversified
  and V14 robust also recorded **5/122**. This replay is development
  evidence, **not independent prospective proof**.

## Why 70% five-double accuracy is not demonstrated

One five-double set covers only FIVE distinct outcomes from the Cartesian
product of two exact final football scores. Under a calibrated independent
score model, the probability of winning such a set is the **sum of the five
joint scoreline probabilities**, not the probability that either one leg is
approximately right.

For the unchanged, frozen 19 November 2026 example (Corinthians–Botafogo
and Cruzeiro–Palmeiras), the full original set has a model-estimated
coverage of **7.1053%**. It is not the true win rate; the model requires
calibration. Achieving a reliable **7/10** would require around **70%** true
coverage per set, a factor of almost ten above this example's probability
estimate. Having 6 or 10 language models does not increase the number
of joint score cells the five options actually cover.

If a hypothetical set truly won with chance 7.1053% on each independent
day, the probability of **at least seven wins in ten** would be about
**0.0000905%**, a modelling illustration NOT a prediction of actual
results. Repeated samples can be correlated.

## Missing source/data/evaluation capabilities

1. **Actual AI availability, not credentials:** validate real HTTP response,
   model ID, available credits, schema and returned score independently for
   every provider. Names of configured services ≠ six successful votes.
2. **Real football chance quality:** there is no verified timestamped
   team/shot-level xG/xGA feed with pre-match opponent adjustments. The
   current `home_xg`/`away_xg` values are MODEL-ESTIMATED GOALS,
   not Opta/Understat shot-based xG.
3. **Lineups and absences:** no independently time-stamped confirmed XI,
   injuries, suspensions, player minutes, substitute depth, rest/travel
   adjustment, or evidence-calibrated player coefficients for the forecast.
4. **Market comparison:** V9/V10 exist but have not had enough genuine
   pre-match odds quotes / complete independent bookmaker consensus. A
   verified, de-vigged market 1X2 + totals + Asian handicap snapshot is
   useful as a benchmark, not a guarantee.
5. **Probabilistic combination:** LLMs read essentially the SAME statistical
   fixture packet. Six answers are correlated interpretations; giving them
   six independent 20% votes or simply averaging chosen scores is not a
   valid calibrated full score distribution.
6. **Same-day two-fixture research:** the GitHub collector spent at most
   one paid per-provider request group per heavy pass. The database may
   contain a match picked days before the other fixture's AI calls. Both
   pre-kickoff packets must be preserved with correct provider, fixture,
   home/away IDs, kickoff and source forecast ID.
7. **Prospective evaluation:** no independently verified completed double
   outcomes for a COMPLETE multi-provider two-match forecast exist. It is
   invalid to report post-match rescoring as a win for a previously frozen
   forecast or choose a lucky pair after final scores.

## Evidence-based research

- Dixon & Coles (1997): Poisson team-strength and low-score dependence,
  **Journal of the Royal Statistical Society C**, 46(2), 265–280.
  <https://ajbuckeconbikesail.net/wkpapers/Airports/MVPoisson/soccer_betting.pdf>
- Wilkens (2026), journal study of Bundesliga xG models against book odds:
  calibration matters; <https://doi.org/10.1177/22150218261416681>.
- Simiyu (2026), 8,982-match independent goals/xG/market comparison:
  historical sharp bookmaker closing prices outperform both standard
  evaluated statistical variants on probability quality, while simple
  xG alone did not yield reliable wagering advantage.
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7549503>.
  Treat as working-paper evidence, not a universal theorem.
- The independent ZicZac **122-day** 2026 replay:
  <https://github.com/ZicZac-hub/scorepredict-ai/actions/runs/43685699587>.

## V15 implementation

**Scheduled AI collector** in `llm_research_panel.py`:

- New `paired_candidates()` prioritises two distinct same-Sydney-day
  matches with four different teams; no observed scores are used in selection.
- **Paid cost opt-in required:** default remains ONE researched fixture
  per heavy batch. Set the Actions variable
  `LLM_RESEARCH_MAX_MATCHES=2` only after reviewing per-provider
  billing/caps. This permits up to **2 × configured providers** model
  requests per heavy batch, e.g. 12 attempts for six providers. If no
  qualified same-day pair exists, do not spend the second match batch.
- Status reports qualified fixtures, pair selection, attempted API budget
  and specific error reasons without disclosing keys.

**Research-only joint-score challenger** in `research_double_v15.py`:

- Reads ONLY the original genuinely frozen V8 two-match pair, authentic
  archived numerical top-ten score options for both legs and real,
  source-matched, pre-kickoff provider votes for both matches.
- Requires **at least two successful providers per leg** before calling
  the output “AI-assisted”. Two is an availability gate, NOT evidence
  of sufficient statistical accuracy.
- Validates source fixture IDs, team IDs, timestamps, matched frozen
  source forecast ID, provider identity and vote schema.
- The existing ZicZac model supplies all joint **numerical probabilities**.
  AI agreement can only gently rerank candidates (12% cap per leg);
  opinions are not turned into fabricated statistical probabilities.
- Freezes a **separate write-once** `data/v15_ai_double/forecasts/<date>.json`,
  with exactly five scores when eligible. There is no post-match backfill.
- Independently verifies BOTH final regulation-time scores and compares
  actual winning rows against the original five-score selection on the
  SAME TWO matches, otherwise reports *pending*.
- Shows counts, abstention reasons and real paired five-score research
  in Deep Research → Ten-Provider Research Panel.
- No extra AI requests are made by the V15 joiner. Existing V8,
  V7, V12, V13, V14 and manually chosen predictions are preserved.

## Testing and promotion

Unit tests simulate authentic two-match saved records to validate
no-future evidence, no fake votes, correct time/identity, five distinct
double outcomes, immutable write-once snapshots and independently
verified hit/miss/pending settlement. **Synthetic tests prove code
integrity, not predictive accuracy.**

No V15 real two-fixture pairs satisfy the 2-provider-per-leg
requirement in the audited source snapshot. Consequently **V15 live
accuracy is unavailable**, not 0%, 70%, or any invented percentage.
Historical V8/V7/V14 results must not be mislabelled V15.

Do not replace the production selector until a representative,
prospectively frozen cohort demonstrates better same-pair hit rates and
probability calibration with uncertainty intervals.

## Next cost/sourcing steps (not performed)

1. Audit provider model IDs, 400/401/402/429 responses and credits using
   private GitHub Actions logs, without sharing API keys.
2. Opt into two-match calls by setting `LLM_RESEARCH_MAX_MATCHES=2`.
   Do not enable automatic paid calls without accepting their cost.
3. Supply and validate a timestamped real xG source and official lineup,
   injury and bookmaker market data if their paid API plans allow.
4. Monitor the pre-kickoff frozen paired results across many days.
5. If 70% success is a strict requirement, consider predicting a
   fundamentally broader market/outcome type, rather than five specific
   combinations of two exact football scores.
