# V14 Robust Agreement — Evidence-Led Five Exact-Score Doubles

**Implementation date: 15 November 2026.** Experimental, NOT promoted. The
live V8 automatic five-double logic and existing immutable historical records
are unchanged. Old results were not used to backdate forecasts.

## Research motivation

- **Dixon & Coles (1997)** modeled correlated football goal events and dynamic
  performance; the adjustment is useful but must be tested for each league:
  <https://doi.org/10.1111/1467-9876.00065>
- **Reade, Singleton & Vaughan Williams (2020)** found exact-score
  bookmaker favorite/long-shot biases and warned that good probability
  forecasts do not automatically deliver reliable financial gains:
  <https://ideas.repec.org/p/rdg/emxxdp/em-dp2020-03.html>
- **Wilkens (2026)** compared xG-based football probabilities with market
  forecasts and found that market probabilities were often better calibrated:
  <https://doi.org/10.1177/22150218261416681>

This release does **not** claim to use real shot-quality xG, verified odds
from the 2026 replay, premium football AI models or a calibrated win-rate
formula. The available V7 and V8 models use correlated football result data.

## Code change

`robust_double_v14.py` independently checks the **entire** normalised
exact-score probability grid for V7 and V8 in each match. It pairs the
two matches using a rectangular product (the two matches can have
different high-goal grid dimensions), and evaluates every exact-score
pair, not just the ten most likely cells from one model.

For an exact-score pair `s` across two matches:

```
p7_joint(s) = p7_match1(s1) * p7_match2(s2)
p8_joint(s) = p8_match1(s1) * p8_match2(s2)
robust_score(s) = min(p7_joint(s), p8_joint(s))
```

The V14 five-double selector chooses the **five different outcomes**
maximising the sum of their `robust_score`; auto pair selection seeks
the match pair with the highest such conservative envelope among
eligible same-Sydney-day, four-team fixtures. The envelope is a
*ranking score based on two models*, not a calibrated lower bound on the
true chance of winning. V7 and V8 are correlated, not two independent
witnesses. Arbitrary scoreline diversification has not been proven
more accurate than a top-probability five-double selection.

An important bug was also fixed: a powerful attack can produce score grids
larger than 11x11. Those genuine high-scoring distributions are now aligned
by **zero-padding**, preserving their existing probabilities and retaining
the match instead of discarding it.

## Historical research — source and comparison

GitHub Action:
<https://github.com/ZicZac-hub/scorepredict-ai/actions/runs/43685699587>

- 2,255 verified completed fixtures retrieved across PL, PD, BL1,
  SA and FL1 from provider seasons 2025/2026.
- The 1,300 most recent available 2026 match results yielded 124
  multi-game Sydney days, of which **122** had two qualified matches.
- Historical predictions for all methods used only provider- and
  competition-matched results available **at least 60 minutes before**
  the day's earliest kickoff. Observed scores are revealed only when
  scoring the completed historical slate.
- No method was selected or tuned to the result of an evaluated game
  after kickoff. The V7/V8/V13/V14 models **were previously developed
  using 2026 aggregate feedback**, so this replay is NOT a clean,
  untouched holdout and NOT live/prospective win-rate evidence.

| Exact five-double strategy | Historical wins / 122 eligible days |
| --- | ---: |
| V8 original highest-probability pair and five options | **5 / 122 (4.10%)** |
| V7 diversified on V8-selected pair | **5 / 122 (4.10%)** |
| V7 diversified on pair selected by V7 top-five coverage | **5 / 122 (4.10%)** |
| V7 diversified on pair selected by V7 diverse coverage | **5 / 122 (4.10%)** |
| V14 robust score options on same V8 pair | **5 / 122 (4.10%)** |
| V14 independently selected pair with five robust scores | **5 / 122 (4.10%)** |

V14 chose the same fixtures as V8 on 114/122 available days and a
different pair on 8. Equality in the aggregate win totals **does not**
mean identical winning days. A previous 65-day test showing 3/65 for V7
versus 2/65 for V8 was an overlapping smaller retrospective sample; the
larger 122-day test does not support that advantage.

**Conclusion: no demonstrated V14 or V7 accuracy improvement. Do not
promote, retroactively alter archives, or quote this as prospective
accuracy.** The probability of a correct two-leg exact-score result
remains low even with extensive research.

## Availability and prospective comparison

- **Five Doubles → V13 Hybrid Research** now exposes
  **V14 Robust Agreement** alongside V7, V8, and V13. It can
  automatically choose two already-loaded matches or evaluate
  your manually selected two matches.
- The UI shows all five combinations, individual model V7/V8
  estimated joint probabilities, the conservative envelope and CSV.
- The existing pre-match collection pipeline captures genuinely
  independent, write-once V7/V8 component grids in **new V13 shadow
  records**. No prior snapshots are rewritten or backfilled.
- When the original V8 pair is frozen, the future evaluator also
  freezes two experimental alternatives on the **same pair**:
  `v7_diversified_same_pair` and `v14_robust_same_pair`.
- Only when the complete day was already present, the relevant
  V13/V8 forecasts share identical input digests, and all timestamps
  precede kickoff, the evaluator may also freeze **its own independent
  V14 pair** in `data/v10_prospective/robust_doubles`.
  The operational threshold is a predeclared 4% lower-envelope score,
  NOT a measured accuracy cutoff; abstention is permitted.
- `v10_evaluate.py` settles the V14 auto-selected pair using **the
  actual two V14 fixtures**, which may differ from the original V8
  fixtures. It additionally compares performance on identical Sydney
  dates when both are independently verified.
- **Performance** and **Five Doubles** show V14 frozen selections,
  verified hits, losses and pending games separately. Unfinished matches
  are **not** counted as failures.

## Future promotion gate

Keep V8 frozen as the existing production model. The ability to rank a
higher-scoring pair on paper is not proof of a higher true hit rate.
Review prospects with several hundred **genuinely future frozen** pairs,
same-day paired win/loss differences, league-level breakdowns,
probability calibration and uncertainty intervals. Never tweak model
weights based on the same evidence used for evaluation.

## Implemented files

- `robust_double_v14.py` — full-grid V7/V8 robust 5-score selector
- `hybrid_five_v13.py` — genuine component grids, no fake scores
- `hybrid_v13_ui.py` — V14 Manual / Auto choice and model explanations
- `v14_robust_backtest.py` — reproducible 1,300-fixture evaluation
- `.github/workflows/v14-robust-validation.yml` — genuine provider
  source and statistical regression test in GitHub Actions
- `v10_prospective.py`, `v10_evaluate.py`, `v10_status_ui.py`,
  `automatic_doubles_ui.py` — immutable future archive and display
- `tests/test_robust_double_v14.py`,
  `tests/test_v13_prospective.py` — no-leakage, 5 distinct scores,
  stronger-attack grid shapes and protected original forecasts.
