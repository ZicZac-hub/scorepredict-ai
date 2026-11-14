# ZicZac prediction v9 — reviewed sources and deployment notes

## What changed

- The **daily prediction model** now defaults to the existing, improved
  Evidence Ensemble **v8** rather than v6. v6 is still one click away in the
  sidebar, and all original saved predictions are preserved. v8's advantage
  on the 632-game retrospective replay was modest (77 exact hits vs 70).
  It was **not statistically conclusive** and may not repeat.
- The **Deep Research** page has a new first-choice
  **Market-Informed v9** model. This is a hybrid of v8's opponent-adjusted
  Dixon–Coles/Poisson/Elo score distributions and **real pre-match 1X2 plus
  Over/Under 2.5 prices**, when the existing API-Football subscription returns
  complete current prices with valid timestamps.
- Bookmaker overround is removed from the market probabilities. A minimum
  distribution-change / KL-style partition projection conditions the score
  matrix on those market probabilities. The v9 preview blends 55% of
  original v8 probabilities with 45% of the market-conditioned matrix.
- **No live betting prices are guessed or fabricated.** An unavailable or
  stale market displays a warning and v9 shows exactly the v8 distribution.
  The engine will not fetch odds automatically when you open the site:
  odds are fetched **only when you click Run Deep Research AI**.
- No historical v9 performance improvement is claimed. API-Football retains
  only **seven days of pre-match odds history**, so the previous long-range
  match dataset cannot honestly be replayed with historical market odds.
  v9 needs a new **prospective, timestamped validation** across a meaningful
  number of fixtures before it can be claimed to beat v8.
- Your existing **five-double selection**, archived predictions and
  six-hour verified capture workflow are preserved. Research v8 doubles
  remain available under the Doubles tab.

## How to use

1. **Daily Prediction:** Open the sidebar Score model control. The default
   is now **Evidence Ensemble v8 · enhanced (default)**.
   Select **Preserved v6 · original** to return to the earlier predictor.
2. **Market model:** Navigate to Deep Research AI, select
   **Market-Informed v9 · odds when verified**, choose the match,
   then click **Run Deep Research AI**. If actual supported pre-kickoff
   odds are available, the resulting forecast will display the bookmaker,
   timestamp and normalised market details. Otherwise it clearly states
   it fell back to the historical v8 score model.
3. **Five doubles:** Open the dedicated Doubles navigation.
   The experimentally ranked v8 automatic/manual options are separate from
   the old production list. Under **Market-Informed v9 · Manual five-double
   comparison**, choose two researched, same-Sydney-day, four-distinct-team
   API-Football fixtures and click its research button. Only **two market
   requests** are made, and missing odds cause an explicitly labelled
   historical v8 fallback. Broad automatic v9 market requests are intentionally
   disabled to protect API quotas and first-page load time.
   Every double requires *both* exact scores to match; estimated probability
   mass is never a verified hit rate.

## Publicly documented methodological references

- [ScoreGPT methodology](https://scoregpt.app/methodology):
  independent model opinions and consensus 1X2 probabilities with frozen
  pre-match picks and transparent grading. ScoreGPT's published 1X2/pick
  accuracy figures **cannot** be used as a correct-score accuracy benchmark.
  Proprietary ScoreGPT outputs have not been copied.
- [ScoreGPT public accuracy](https://scoregpt.app/accuracy):
  separately grades match outcome versus exact score.
- [Domain-Driven Identification of Football Probabilities (2025)](https://www.mdpi.com/2227-7390/13/24/3976):
  model-integrated margin removal and multi-market probability estimates.
- [Dixon and Coles (1997)](https://doi.org/10.1111/1467-9876.00065):
  low-score football probability dependence.
- [API-Football V3 docs](https://www.api-football.com/documentation-v3):
  pre-match /odds available one to 14 days before a fixture,
  seven-day price-history retention, updated approximately every three hours;
  provider coverage and subscription access vary.
- [ImpliedScore method](https://impliedscore.com/correct-score-calculator-explained/):
  de-vig 1X2 and O/U then fit goal-rate distributions; not a guarantee.

## Limitations and privacy

Neither these references nor any paid AI subscription licenses an assertion
that the website connects to ChatGPT, Claude, Gemini, Grok or DeepSeek
predictions. ZicZac uses its own disclosed calculations.

Only 90-minute regulation scores are evaluated. Do not treat historical
replay results as the verified prospective hit rate. No first-page market
calls, API-key storage, invented injuries or historical result rewriting.

When the repo is public, all files committed there are visible.
Store API keys, database URLs and access passwords **only** in Streamlit
Secrets and GitHub Actions Secrets.
