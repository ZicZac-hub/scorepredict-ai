# ZicZac · Highly Researched Prediction of the Day

Implemented on the **Daily Predictions** page, immediately above Live scores.
The card shows one real upcoming match and a transparent exact-score
forecast, not an invented or guaranteed result.

## Selection

- Date grouping uses **Australia/Sydney**, including daylight-saving time.
- Prefer a real prediction-ready upcoming match **today**. If today's
  candidates do not have adequate history, show the **next** eligible date
  explicitly (never call tomorrow's match today's prediction).
- Rank candidates by available **last-10 overall form**, **home and away
  form**, independent statistical-model contributor coverage, then estimated
  outcome strength and modal score probability.
- A candidate requires at least **five completed historical results for
  both teams**; the durable capture workflow is stricter: it requires at
  least three relevant home/away games and two actual statistical
  contributors, with source evidence recorded.
- Historical H2H counts and model disagreement are disclosed. Missing
  xG, lineups, injuries and other facts are labelled unknown rather than
  inserted as numerical adjustments.

## New prediction method

The foundation is **Evidence Ensemble v8**, combining preserved v6 venue
form, regularized opponent strength/Dixon–Coles and historical Elo where
actual league history supports each contribution.

On first session selection, the app may obtain **one** timestamped,
margin-corrected, pre-kickoff 1X2 market from API-Football for that match.
Only when the provider actually supplies qualifying data does it show a
**Market-Informed v9** modal exact score, with the original v8 score
preserved alongside it. No odds are invented and v9 has not yet been
validated on a comparable prospective dataset.

## Pre-kickoff freeze across devices

The existing **Capture verified pre-match evidence** GitHub Actions
workflow also generates write-once research picks, without any additional
provider calls. It uses the already-fetched, timestamped, verified evidence
for upcoming Sydney today/tomorrow games. After the first qualified
snapshot is committed under `data/daily_picks/YYYY-MM-DD.json`, the app
prefers that **same immutable v8 prediction across iPhone and MacBook**,
even when the currently selected league filters differ. The exact score
is frozen before kickoff; it is *never regenerated after kickoff*.

The scheduled record is a **v8 historical-research** score only; live
bookmaker quotes can be inspected separately but never rewrite that
globally frozen prediction. If the scheduled capture has not yet created
a valid dated file, the app falls back to a session-frozen v8/v9 preview
and states that it is not yet shared across sessions.

Original pre-match snapshots, six-hour capture jobs, saved predictions
and historical accuracy figures are unchanged. The new card does not
mark a prediction correct until a trusted final score has been verified.

## Validation

`tests/test_daily_pick_ui.py` and `tests/test_daily_pick_capture.py`
validate Sydney dates, eligibility, no post-kickoff forecasts, genuine
market timestamps, fallback, write-once semantics, and reloading the
same file in a fresh session.

**Accuracy caveat:** The first v8 replay found 77/632 exact hits versus
70/632 for v6; that difference is not statistically decisive, and
the v9 market method is not yet proven better for correct scores.

## Usage

Navigate to **Daily Prediction** and find the card immediately above
**Live scores · Your predicted matches**. Expand *Why this match?* for
research coverage, contributors and five possible scorelines. Use the
optional market comparison before kickoff if available.
