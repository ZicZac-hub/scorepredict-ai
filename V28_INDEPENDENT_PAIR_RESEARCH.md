# V28 — Independent prospective pair-choice trial

## Objective

Remove the **V26/V27 dependence on a newly frozen V25 daily packet**. The original
V25 sets dated 19–23 November 2026 cannot be retroactively attributed to later
research versions. In November's archive, V27 had **zero** frozen decisions,
and V26 had zero verified prospective pairs despite 59 independently verified
V8 match outcomes available for later calibration.

V28 enrolls NEW matches genuinely before their kickoffs, using currently
available immutable V8/V8-near score grids. It never claims to have selected
earlier historical matches.

## Two arms, equal information and scoring rule

On each eligible Sydney match day, V28 compares every independently recorded
four-distinct-team pairing across the complete frozen source slate.

- **Quality/freshness arm:** Rank by `(0.60 * V8 top-five model mass +
  0.40 * robust top-five mass evaluated under V8) * freshness multiplier`.
  These ranking coefficients are fixed before observing future outcomes.
- **V25-style robustness control:** Rank by the original V25 lower-envelope
  robustness score. This is an uncalibrated ranking, **not** a calibrated
  probability lower bound.
- **Both arms select five combinations using the EXACT SAME original V25
  robust-score-picking algorithm.** This isolates pair selection from
  five-score-choice policy. The saved pairs may differ.

Neither arm uses a new stochastic football model, AI votes, injury guesses,
market offers, or result information.

## Eligibility, fairness and abstention

Require original pre-kickoff V8 score grids, at least 45 original historical
matches per fixture, correct provider/home/away identity, distinct teams,
both matches on the same Sydney day, source age at most 14 days, and real
prediction issue at least 45 minutes before kickoff and at most 24 hours
before kickoff.

For every genuinely eligible source:
1. Record the independent original forecast ID, input digest, timestamp and
   normalized score-grid receipt hash.
2. Evaluate the **entire** slate, without quietly restricting to a top subset.
   A slate over 2,500 candidate pairs records an explicit abstention.
3. Require at least two valid pair options. Insufficient fixtures are
   **retryable**, not a permanently frozen miss: a later capture can qualify.
4. Fix a minimum 5.5% V8-model top-five mass for the quality-selected pair as
   a research coverage gate, not a measured-win threshold.
5. Save both pair choices and the complete catalog in a write-once dated
   source packet, no later modifications. A 20-minute maximum capture window
   and a second lead-time check prevent backdated selections.

## Independent evaluation

The audit rereads each original source file, checks fixture identity and
capture time, reproduces the complete candidate slate and both selections,
and then uses only independently identity-matched regulation-time verified
final scores. It reports:

- Completed **paired Sydney match days** (both arms' selected fixtures
  verified), still-pending days, abstentions and invalid-source exclusions
- Five-double wins for both policies and 95% Wilson intervals
- Days when the policies selected the same fixture pair versus different pairs
- Discordant-win counts and an exact two-sided McNemar/sign-test p-value
- At least 100 future verified paired days as the first research review
  milestone, not a claim or guaranteed 70% target

**No live production model is promoted**, regardless of early test results.
Model-estimated five-score mass is not verified real-world accuracy. A
same-day comparison on **different pairs** is not same-fixture calibration.

## Execution

```bash
# Pre-kickoff collector: create fresh selections, then grade known results.
python v28_independent_pair_trial.py

# Result backfill / PR tests: independently grade, never issue new picks.
python v28_independent_pair_trial.py --verify-only
```

Permanent artifacts:

- `data/v28_independent_pair_trial/decisions/YYYY-MM-DD.json`
- `data/v28_independent_pair_trial/status.json`

Runs on the existing scheduled pre-match evidence collector and hourly verified
results backfill, requiring **no new external API or LLM calls**.

The archive proves the content and event timestamps stored by the provider and
the project's write-once file policy; it is not third-party notarization of
Git commit capture times. Comparisons require sufficient prospective verified
results before any evidence-backed accuracy claim.
