# V30 — Prospective full-slate five-double ranking audit

**22 November 2026 · Research-only · Original live prediction model unchanged**

## Grounded problem

The first real independent V28 research captures saved **two** genuine daily
slates (Sydney 22 and 23 November 2026). Their original frozen catalogs
contain respectively **3** and **210** same-day, four-team match pairs.
The existing V28 quality/freshness and robustness rules selected the same
two games on both days. The verified-results log did not yet have completed
results for those two selected pairs at the time of the V29 audit.

A five-double accuracy comparison is not possible until independent
regulation-time finals are recorded. V29's alternative-pair arm must
never be backfilled against the existing days.

## What V30 does now

V30 uses the write-once candidate catalog originally stored by V28
**before kickoff**. It grades EVERY saved candidate against original
source-matched independently verified final scores, without changing or
creating original predictions:

1. Revalidate V28's full original source forecast IDs, timestamp provenance,
   digest, entire candidate slate, ranking fields and five selected score
   combinations.
2. Verify the provider/fixture/team/kickoff identity for every original source
   using independently recorded regulation-time final results. A future
   result-verification timestamp is never accepted early.
3. Keep missing and partially verified candidate rows **pending**, not
   losses. Report exact distinct fixture counts and candidate pair counts
   separately.
4. Only when **every original fixture in the daily slate** has a verified
   final, perform within-day ranking diagnostics: robust five-double hits
   and original V8 baseline hits, quality rank quartiles, expected five-score
   model mass evaluated under V8, and equal-budget top 1, 5 and 10-pair
   comparisons.
5. With 5 or 10 candidate pairs, the ticket budget is 25 or 50 possible
   exact-score combinations (up to the number of pairs actually available).
   This must NEVER be presented as accuracy of one five-double set.
6. Treat each complete **Sydney day** as the independent research unit for
   a first review target of 30 whole-day slates. The 3 and 210 overlapping
   pair catalogs are NOT 213 independent experimental observations.
7. V30 records any invalid source, modified candidate catalog, disputed
   verified final, or source-identity mismatch and excludes that evidence.

Existing V28 and V29 selected-pair hit rates remain separate experiments.
This study is **exploratory retrospective ranking analysis of genuinely
pre-frozen candidate predictions**; the catalog is prospective, but V30's
new ranking report does not retroactively imply that any other pair was
actually selected for the original five-double research ticket.

## Intended outputs

```bash
python v30_full_slate_audit.py
```

Writes `data/v30_full_slate_ranking/status.json`.

The scheduled collector and independently verified-result backfill refresh
this **read-only** analysis; no new football-data provider or LLM calls.
GitHub CI runs the same independent audit.

## Uncertainty and promotion guardrails

- Model-estimated five-score probability masses are **not** validated hit
  rates. The V25 robustness minimum is not a calibrated probability bound.
- Partially verified days cannot establish the direction of rank calibration.
- Many pairs reuse one match. They must **not** be treated as iid samples for
  binomial confidence intervals or statistical significance.
- An apparent lucky top-five result from more purchased pairs is NOT proof
  that a single five-double accuracy improved.
- No version is automatically promoted to live predictions.
- The user's aspirational 70% five-double exact-score hit target is **unproven**.

The next step is independent chronological collection of results, with
full-day verification and a meaningful number of future daily slates.
