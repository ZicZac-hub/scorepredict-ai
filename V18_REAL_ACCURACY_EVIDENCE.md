# V18 — Real-source accuracy and time-safe probability calibration

**15 November 2026. Research-only update.** No production prediction methods,
historical picks, five exact-score doubles, original V8 matrices or already
verified full-time results are modified.

## What has actually changed

1. **An authentic source-matched audit** reads immutable original V8 forecasts
   and independently verified full-time scores. Every counted record must
   have a real original archived fixture/provider ID, matching home/away IDs,
   kickoff, source forecast ID, input digest, unmodified complete score matrix,
   capture timestamp and verified result. Missing scores are **pending**, not
   losses. No other model's later refresh or result-based chosen predictions
   enter the original V8 cohort.
2. **Quality of probability estimates:** The original complete grid generates
   single-match **1X**, **X2**, and **12** outcome probabilities. The winner is
   predeclared by the highest estimated probability. The report measures
   actual observed wins, 95% Wilson intervals, Brier error, model-predicted vs
   observed hit rates and confidence-band calibration gaps.
3. **Honest feature coverage** counts missing verified shot-quality xG,
   validated lineup coefficients, verified market quote evidence, short
   last-ten samples, invalid/tampered forecast sources, and not-yet-verified
   results. A username, API key or configured model is not proof that
   validated features were actually used.
4. **Time-safe recalibration research:** An experimental probability correction
   is evaluated only when at least **100 independently verified earlier**
   original fixtures exist and **20** are in the same probability band.
   Every previous final result must have been *verified strictly before the
   candidate's original forecast capture*, preventing future leakage. The
   candidate correction is shrunk toward its original probability using
   a fixed pseudo-sample strength of 30. At least 50 eligible future-tested
   fixtures and an improvement in **Brier error** are required merely to
   trigger *review*. No auto-promotion occurs. All original model scores
   stay intact.
5. **Competition-level monitoring:** Separate league-level, verified outcome
   accuracy and Brier quality are shown with a small-sample warning.
6. **Website dashboard:** The V17 Five Doubles and Performance research panel
   now includes "V18 Real-Accuracy & Calibration Audit" with observed
   accuracy, expected accuracy, confidence buckets, pending records and
   evidence gaps.
7. **Scheduled persistence:** The established capture workflow runs the V18
   calculation **using already saved files**, with no extra paid AI or
   football API requests. It preserves the read-only report at
   `data/v18_accuracy/accuracy_report.json` in GitHub's data history and
   run artifacts.

## Critical interpretation

- The current original prospective five **EXACT-SCORE** double archive has
  1 verified set with 0 wins, 5 pending (snapshot of 15 November 2026);
  retrospective V8/V14 122-day research had 5/122 (4.10%). **No
  demonstrated 70% exact-score double success.**
- **V18 single-match double-chance** diagnostics are not five-exact-score
  double accuracy and must never be reported as such.
- The V17 70% model selection threshold is an **uncalibrated probability
  estimate**, not 70% proven success. V18 checks actual calibration but
  cannot train a credible correction when the genuine sample is small.
- Independent shot-based xG/xGA, verified player-value/absence impacts,
  and a time-stamped de-vigged bookmaker reference are not automatically
  created by this update. These require authorised feed access, match-ID
  alignment, sufficient archival coverage, tested coefficients and
  additional costs as applicable. **None was enabled or purchased.**
- The existing six configured LLM providers also lack qualifying real
  source-matched two-fixture research. Configured secrets are not
  successful votes. Paid pair AI collection stays opt-in.

## What must happen before a factual improvement claim

1. Real licensed xG/lineups/odds sources are independently verified and
   incorporated, with proper as-of timestamps and ability to abstain.
2. Feature selection and any model weights are trained using earlier seasons
   only; an untouched later season or future prospective cohort is used
   for independent validation, with league/season ablations.
3. At least hundreds of genuine pre-kickoff forecasts accumulate, letting
   the team compare model calibration, Brier/log loss, outcome accuracy and
   five-double hit coverage on exactly matched games, with uncertainty.
4. Source-matched evidence is required for BOTH legs of any AI-assisted pair.
5. Only a predeclared challenger that demonstrably improves independent
   future performance is eligible for release; no backdating of wins,
   manufactured odds, invented scores or arbitrary 70% guarantees.

### Operator access

Run locally (no network): `python v18_accuracy_audit.py`.

See `tests/test_v18_accuracy_audit.py` for duplicate-source, genuine
result identity, late-score, probability calibration, chronological
training and no-auto-promotion regression tests.
