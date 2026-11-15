# ZicZac V13 Hybrid Five-Doubles — Combined Experimental Model

**Built 15 November 2026. The release does NOT replace V8 and does NOT claim
a demonstrated improvement in exact-score or five-double accuracy.**

## What was combined

The distinct stored full-score grids are blended with fixed declared weights,
then renormalised whenever an optional model is legitimately unavailable:

| Component | Initial weight | Evidence |
| --- | ---: | --- |
| V8 Evidence Ensemble | 60% | venue form, opponent Dixon–Coles, Elo when available |
| V7 fitted Dixon–Coles | 20% | opponent-adjusted goals and score dependence |
| V6 preserved original | 15% | venue-form score distribution |
| V12 shrunken league score residual | 5% | pre-match historical league scoring only, if at least 80 valid matches |

**The components are statistically correlated** — V8 itself includes V6
and V7-type signals. These are NOT four independent expert votes. Weights
were selected conservatively before this tournament rerun, not trained on
2026 outcomes or claimed as optimal. V12 received a low weight because its
first-choice score accuracy on the explored holdout was lower than V8.

A V13 prediction requires a fixture at least five minutes from kickoff,
at least 30 genuinely completed provider/competition-matched league matches,
at least two legitimately available model components, and a V8 fit with more
than one contributor. Otherwise it abstains. No fake xG, injury reports,
market odds, proprietary AI forecasts, win probabilities or historical hit
rates are introduced.

## Five exact-score double selectors

For a valid two-match pair with four different teams on one Sydney date:

- **Top five by joint model probability:** mathematically maximises estimated
  probability mass over five distinct score pairs if both calibrated grids
  and the independence assumption are valid
- **Balanced:** predeclared score-diversity weight 0.18, sometimes reducing
  model-estimated coverage
- **Fully diversified:** maximises novel per-leg score coverage at a cost to
  estimated probability mass

Changing the selection method does not alter a model forecast, rewrite an
old prediction, or guarantee more wins.

## Historical SAME-DATA comparison (15 November 2026)

Source: GitHub Actions all-methods tournament, genuine API-Football
2025/26 completed history. 2,255 source matches, latest 220 2026 target
fixtures, **219 same-fixture qualified comparisons**. Scores are replayed
at least 60 minutes before kickoff; results at that time have a conservative
3-hour availability delay. This is **historical reconstruction**; 2026 had
already been inspected during model development, so it is NOT a clean
unseen V13 validation.

| Model | Exact hits / 219 | Top-three | Full-score log loss (lower better) |
| --- | ---: | ---: | ---: |
| V6 | 24 / 219 | 28.77% | 3.07223 |
| V7 | 24 / 219 | 31.05% | 3.05177 |
| V8 | 25 / 219 | 29.68% | 3.04597 |
| V12 | 22 / 219 | 31.96% | 3.05944 |
| **V13** | **24 / 219** | **30.59%** | **3.04623** |

Five-double selection evaluated on **20 same-day pairs selected by V8**
before kickoff. Every model used exactly the same pair:

| Model | Highest 5 | Balanced | Fully diversified |
| --- | ---: | ---: | ---: |
| V6 | 0/20 | 0/20 | 1/20 |
| V7 | 0/20 | 1/20 | 2/20 |
| V8 | 0/20 | 0/20 | 1/20 |
| V12 | 0/20 | 0/20 | 1/20 |
| **V13** | **0/20** | **0/20** | **1/20** |

**Result: V13 did not beat V8 on first-choice exact scores or V7 on
diversified five doubles. It is not promoted.** V7's 2/20 is extremely small
evidence and does not prove a better future strategy.

## How to use

1. Open **Five Doubles** in the Streamlit ZicZac website.
2. Find **V13 Hybrid AI · Combined Five Doubles**.
3. Choose **AI Auto Select** or **Manual Select**.
4. Select **Top five**, **Balanced**, or **Fully diversified**, and run the
   research button.
5. The interface displays the five score pairs, estimated joint coverage,
   each match's actual participating component weights, and CSV download.

Auto Select searches the already-loaded fixture list, limited to 12
researchable matches in three Sydney matchdays; it will not silently call
unlimited APIs. Manual Select uses only the two explicitly chosen matches.
Both modes are session-only, separate from previously saved user picks.

## Prospective evaluation and protection

The existing pre-match collector can now save a separate, immutable V13
full score grid alongside a V8 original forecast when eligible.
`v10_evaluate.py` tracks V13 exact hits, top-three accuracy, full-score
log loss, 1X2 Brier, and paired same-evidence results.

During the **existing V8-controlled** daily pair freeze, V13's top-five,
balanced and diversified alternatives are frozen and scored on the **same
two matches** ONLY if both V13 and selected V8 snapshots have the same
verified evidence digests and were frozen before selection. Later model
outputs are never backdated into a past pair.

No current or past V13 performance should be claimed from simulated tests.
A meaningful promotion review should require a much larger sample of
prospectively frozen, independently verified match pairs and strong
probability-calibration results. Until then, V8 remains the supported
baseline and V13 an optional challenger.

## Code and repeatability

- `hybrid_five_v13.py` — strictly time-qualified, model-based mixture
- `hybrid_v13_ui.py` — manual/auto interface with three selectors
- `all_methods_tournament.py` — same-evidence historical comparison
- `v10_prospective.py` and `v10_evaluate.py` — write-once shadows and
  independently verified performance
- `tests/test_hybrid_five_v13.py` and
  `tests/test_v13_prospective.py` — model, time and ledger regressions
- <https://github.com/ZicZac-hub/scorepredict-ai/actions/runs/43681130971>
  — V13 historical comparison execution and downloadable outputs
