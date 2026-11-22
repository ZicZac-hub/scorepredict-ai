# V29 — Prospective distinct-pair sensitivity and source-verified result readiness

**Research only, 22 November 2026. Live prediction model unchanged.**

## Grounded first evidence

The V28 scheduled collector successfully froze **two original new pairs** on
22 November 2026 UTC, for Sydney dates **22 and 23 November**, with full
qualified candidate slates of **3 and 210 four-team pairs**. Both V28 selection
policies chose the SAME two fixtures on each day, leaving **zero** observed
pair-policy disagreement. Both original pairs were pending at inspection.

The working original V8 and V25 source records and results must never be
rewritten after decisions. The original V28 results are also preserved.

## What V29 adds

**First: evidence-state audit of existing V28 predictions.** Each recorded
V28 decision is independently revalidated against its immutable original V8
forecast source, original capture time, complete candidate pair catalog,
and both saved policies. Then each unique selected physical fixture is
classified as:

- `waiting_for_kickoff`: genuinely before the saved kickoff time
- `awaiting_independently_verified_result`: kickoff has passed, but the exact
  source/team/kickoff-matched regulation final is not verified
- `verified_final`: source-matched final has independent verified evidence

This reports how many forecasts and fixture results are actually ready.
Pending and missing results never become losses. It shows whether the
two V28 arms selected the same fixture pair, INCLUDING pending days.

**Second: one additional truly future-only pair-selection sensitivity arm.**
For any *new* V28 daily decision, V29 may freeze the highest V28-quality-ranked
four-team pair that is **not the same pair as either original V28 policy**,
provided that:

1. The genuine V28 packet was created within the previous 20 minutes.
2. It passes independent source, timestamp, and full-catalog reproduction.
3. The alternative is recorded at least 45 minutes before BOTH kickoffs.
4. The alternative uses the SAME unchanged original V25-style five-score
   selection algorithm and original V28 candidate slate.
5. Both the alternative and independent source packet are frozen write-once
   with a cryptographic source-receipt digest. Old V28 days cannot acquire
   V29 predictions retrospectively.

V29 is a **predeclared sensitivity check**, not a supposedly more accurate
model: the top-ranked alternative may have LESS estimated coverage than V28's
top choice, and it may share one fixture with the original pair. Shared
fixtures introduce dependence; the report discloses overlap. Do not treat
match-day samples as independent team/game observations.

## Evaluation

Only when the two fixtures for the V29 alternative AND the original V28
quality/control arms finish with independently authenticated regulation
scores may the study count a paired day. The report includes:
- Number of source-valid original days, future fixtures, post-kickoff
  missing verifications, and settled original fixtures.
- New V29 decisions, abstentions, pending trials and excluded tampered sets.
- Paired exact-score five-double hits, Wilson uncertainty, and the
  two-sided paired exact-test p-value.
- 100 verified paired days as an initial review milestone, NOT an
  accuracy guarantee or an automatic promotion criterion.

This cannot establish a 70% hit rate from a few lucky games, or from
model-estimated five-score probability mass alone.

## Workflow

```bash
python v29_pair_sensitivity.py
# Creates NEW alternative research-only decisions, within the V28 freeze window.

python v29_pair_sensitivity.py --verify-only
# Read-only: independently validate existing decisions and refresh result readiness.
```

The scheduled pre-match workflow runs V29 immediately after V28. The hourly
verified-result backfill and GitHub CI only use `--verify-only`.
Output paths:
- `data/v29_pair_sensitivity/decisions/YYYY-MM-DD.json` — immutable future picks
- `data/v29_pair_sensitivity/status.json` — independent readiness and results

The research dashboard displays both original V28 pending-result reasons and
V29 alternative study results. Neither external sports API nor AI calls are
added. No live model is changed.
