# Evidence Ensemble v8 — experimental challenger (14 November 2026)

This research release **does not replace production**. ZicZac's preserved
`engine.py` v6 baseline, `forecast_model.py` v7 candidate, pre-match ledger,
existing smart-double workflow and six-hour job remain unchanged. New code lives
in separate modules.

## Evidence-based techniques

- **Recency-weighted home and away goal modelling:** retains the production v6
  model with robust effective sample sizes.
- **Opponent-aware attack/defence:** uses the existing league-fitted,
  regularized `reference_model.py` with a Dixon–Coles correction fitted only
  on pre-match completed results when a sufficiently broad league population
  exists.
- **Historical Elo:** updates ratings in kickoff order, conditional on
  results already available. Its home/away strength difference provides a
  third goal-rate distribution when at least seven historical games for each
  target team and meaningful league-wide history exist.
- **Distribution mixture:** predeclared candidate weights are 45% v6 / 35%
  fitted opponent model / 20% Elo goal-contrast. If evidence for a component
  is insufficient, use explicitly identified fallback weights. These are
  **untuned trial settings**, not empirically optimal parameters.
- **Exact scores:** calculate full normalized goal-count probability matrices.
  The top score is the most likely *single* outcome; this does not mean it is
  likely to occur, or that an exact-score double is safe.
- **Five doubles:** only same-Sydney-date match pairs with four different
  teams are eligible for v8 Auto Select. Compute the five highest-probability
  distinct score pairs, using the product of modelled score probabilities
  under an explicit **independence assumption**. The result is estimated
  coverage, **not measured historic win rate**. Manual Select remains possible.
- **Pre-match only:** provider-qualified same-competition match history,
  verified regulation-time scores, availability timestamps (or conservative
  three-hour delay) and cutoff before the fixture kickoff. Backtests share
  the same 730-day, two-season history policy as other ZicZac validation.

The new features are on demand to limit provider usage and Streamlit startup
delays. Inactive games, missing histories and unverified injury, lineup, xG,
odds and formation records are never replaced by invented data.

## Live UI

`?view=deep` → choose **Evidence Ensemble v8** or **Preserved Deep Research
v2**, choose a researched match, click **Run Deep Research AI**. The report
shows model agreement, preserved modal score, new modal score, Elo signal,
component weights and estimated probabilities.

`?view=doubles` → **Deep Research AI v8 five-double challenger** → choose
Auto or Manual → click its research button. This does not overwrite the
existing preserved Five Doubles below it.

## How to validate (no guaranteed accuracy)

GitHub Actions → **Evidence v8 honest historical comparison**. It loads
provider-verified 2025/2026 results using the existing GitHub Actions
`API_FOOTBALL_KEY` secret, computes predictions only from evidence available
before kickoff, and writes `work/v8_challenger_evaluation.json` as a workflow
artifact, not a public guessed accuracy badge.

The report includes paired exact-hit rates, model probability log loss,
outcome Brier score, top-three accuracy, concentration of 1–1 modal
predictions, and a small common-snapshot same-pair five-double comparison.
A historical reconstruction is **not** a new prospective win rate. Keep
new future predictions frozen before kickoff and measure every win and loss.

Do **not** automatically switch production to v8 just because individual
matches look more plausible. Compare sufficient independent results and
calibration first. Refitting model weights on the same holdout invalidates
that holdout.

## Related published methodologies (not copied proprietary models)

- Dixon & Coles (1997), `https://doi.org/10.1111/1467-9876.00065`.
- Boshnakov, Kharrat & McHale (2017), bivariate goal count models,
  `https://doi.org/10.1016/j.ijforecast.2016.11.006`.
- Football match score modelling and bivariate Poisson benchmark:
  `https://journals.sagepub.com/doi/full/10.3233/JSA-200462`.
- Forebet statistical prediction methodology,
  `https://www.forebet.com/en/faq`.
- ScoreGPT public panel consensus and mandatory grading,
  `https://scoregpt.app/methodology`.
- Independent AI model prediction grading and disagreement,
  `https://turingstats.com/ai-prediction-leaderboard`.

**Important:** References to ChatGPT, Claude, Gemini, Grok, DeepSeek, ScoreGPT,
BetGPT and other brands are methodology references only. This update does
NOT scrape proprietary algorithms or connect those vendors' prediction
services. Doing so would require supported official APIs, separate user
credentials, permission, ongoing cost and separate historical validation.

## Frozen replay evidence (14 November 2026)

A larger pre-kickoff reconstruction from 2,255 verified source games evaluated
632 paired fixtures (70/632 exact v6 versus 77/632 exact v8). Mean outcome
Brier decreased from 0.6083 to 0.5946, and modal 1–1 concentration fell from
80.38% to 68.51%. However, top-three exact-score coverage decreased slightly
(30.38% to 29.91%). In the common-snapshot same-match-pair comparison across
59 Sydney dates, v8 hit two of five-double selections and v6 hit one. These
are too few winning doubles to support a reliable improvement claim.

The exact-score paired discordance is 17 v8-only hits versus 10 v6-only hits
(two-sided exact paired p≈0.248). Thus v8 remains a **manual research option**
instead of automatically replacing v6 or rewriting historical predictions.

Evidence: `data/research_v8/extended_historical_validation.json`, captured
from [the successful CI run](https://github.com/ZicZac-hub/scorepredict-ai/actions/runs/43532480039).
Earlier 160-match replay remains available separately as an audit.

## Credentials and security

Do not commit keys or passwords to GitHub. Keep secrets in Streamlit Secrets
and GitHub Actions Secrets. As of this development, the repository can be
public; that visibility does not make the Streamlit site private.
