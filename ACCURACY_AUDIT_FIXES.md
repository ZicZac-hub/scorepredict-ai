# ZicZac accuracy bug audit — verified fixes (14 November 2026)

This is a **bug/data-integrity** release. These fixes improve the credibility
of exact-score probabilities and five-double selection but are **not a
measured increase in win rate**.

## Confirmed fixes

| Actual issue | Change | Automated check |
|---|---|---|
| V8 prediction captured several days early prevents later bookmaker updates from being saved | `capture_market_followup` independently checks existing frozen v8 matches within 30 minutes to 48 hours of kickoff, on the six-hour heavy research run; records a separate v9 and (only with three independent real bookmakers) v10 challenger | `tests/test_v10_market_refresh.py` |
| Apples-to-oranges comparison: v8 old history vs newer v9 market history | Write-once `v8_control` uses the precise newer evidence digest, then `v10_evaluate.py` compares only matched historical-source digests; mismatched observations are excluded and labelled | `tests/test_v10_market_refresh.py` |
| Some five-double sets were frozen days before all same-day eligible games had been researched | New sets wait until the first eligible kickoff is within 24 hours, require two distinct matches each at least 45 minutes before kickoff, rank all available qualified same-Sydney-date games, then preserve the first original forever | `tests/test_v10_prospective.py` |
| Five-double scheduler raised NameError when time-window gating was introduced | Explicit timedelta import; CI regression suite validates the path | `tests/test_v10_prospective.py` |
| Market source selection gave preference to older odds with more markets | V9 uses only quotes at most 12 hours old and considers completeness/preferred source only among quotes within three hours of freshest verified snapshot | `tests/test_market_score_v9.py` |
| Multi-bookmaker averaging could blend stale prices and fresh prices | V10 requires three real, contemporaneous bookmaker quotes from the same fixture, rejecting historical quotes outside the three-hour window | `tests/test_multi_market_v10.py` |
| Can't tell whether scheduled market refresh ran | Performance page now shows the latest collection's actual saved forecasts, market refresh checks, provider calls and errors | `v10_status_ui.py` |
| GitHub tests passed even when Streamlit Cloud had not deployed a matching revision | New **Check ZicZac live Streamlit hosting** workflow performs simple HTTP checks on the live URL on source pushes and every six hours | `.github/workflows/streamlit-live-smoke.yml` |

## Automatic workflow

- Pre-kickoff capture: scheduled at :02, :17, :32 and :47 UTC each hour.
- Heavy research and independent result verification: at least every six
  hours or on relevant manual/source pushes, subject to GitHub schedule delays.
- Additional near-kickoff market requests: maximum **4 per heavy run**,
  budgeted alongside the existing collector's API call limit.
- The original v8 selection, archived five doubles and original v6/v7
  predictions are never rewritten. New contenders and fair-control forecasts
  receive separate immutable identities.
- Real verified scores—not an AI's suggested final score—are used when
  calculating actual wins. A pending game is neither a win nor a loss.

## Still not verified / not promised

- **Production website:** a healthy URL is not proof Streamlit Cloud
  deployed the newest GitHub commit or that the app Python session has no
  exceptions. If the source-host smoke workflow fails or the new diagnostics
  do not appear after refresh, inspect Streamlit Cloud deployment logs.
- **No proven best model:** v8/v9/v10 weights have not been re-optimised on a
  new independent prospective sample. Existing v8 remains the production
  default; candidates can lose against it.
- **True xG/xGA, independently measured injury strength and verified
  tactical coefficients** remain unavailable. The models do not invent them.
- **Betting combinations:** the five exact-score pairs maximise modelled
  coverage conditional on independent matches and calibrated probabilities.
  Real-world wins remain rare; the 4% modelled-coverage operating guardrail
  is not empirically tuned or a guarantee.
- **Live bookmaker availability:** odds are used only when provider coverage,
  freshness, applicable regulations, and API quota permit; missing odds
  produce explicit missing/untested status, not synthetic probabilities.

## How to verify

1. GitHub Actions → **Prediction integrity tests** (unit and regression suite).
2. GitHub Actions → **Capture verified pre-match evidence** (actual saved
   snapshots and diagnostics).
3. GitHub Actions → **Check ZicZac live Streamlit hosting** (HTTP health only).
4. ZicZac → **Performance** → **Automatic forecast & odds-refresh diagnostics**.
5. Future proof of score accuracy is in
   `data/v10_prospective/performance.json`, not the historical replay.
