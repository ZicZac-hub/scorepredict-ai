# Double 5 evidence

The original automatic Double 5 archive had **0 hits in 7 verified sets**, with **1 pending**, as of 24 November 2026 at 07:20 UTC. A set hits when one of its five rows matches the regulation-time scores of **both** fixtures. Every archived set already had the five highest joint probabilities for its chosen fixtures under the frozen V8 grids. The average modeled coverage was **7.19%**, and the observed results do not establish the 70% target. [Original archive audit](../five_double_accuracy.py)

| Sydney date | Verified status | Five-row modeled coverage | Actual joint rank | Rows covering 70% of modeled mass |
| --- | --- | ---: | ---: | ---: |
| 15 Nov | Miss | 5.92% | 31 | 156 |
| 19 Nov | Miss | 7.11% | 211 | 119 |
| 20 Nov | Miss | 7.26% | 31 | 118 |
| 21 Nov | Miss | 7.53% | 22 | 115 |
| 22 Nov | Miss | 7.62% | 60 | 117 |
| 23 Nov | Miss | 6.54% | 131 | 140 |
| 24 Nov | Miss | 7.72% | 617 | 110 |
| 25 Nov | Pending | 7.81% | — | 105 |

In three verified misses both actual leg scores were absent from the five rows. In four misses one actual leg score was absent. The descriptive Wilson 95% interval for the seven settled sets is **0.0%–35.4%**, under its independent-trials assumption. The source audit found no applied market or validated lineup adjustments and no archived verified shot xG for the 16 original legs. [Archived forecasts and finals](../data/v10_prospective/)

## What comparing every candidate can achieve

Under match independence, a row pairing score `a` with score `b` has modeled probability `P(a) × P(b)`. Distinct rows cannot both happen. Their probabilities add, so replacing any selected row with a higher-probability unused row increases modeled coverage. Selecting the five largest joint cells is therefore optimal for the given grids. The counts in the table use those same distributions.

Even a different dependence assumption cannot get to 70% while retaining these marginal score distributions for the fixed pairs. Any five joint rows use at most five distinct scores per leg. The chance of hitting a row is bounded by the smaller of the two legs’ top-five marginal masses, because every hit requires both leg scores to be in those sets. This upper bound ranged from **41.98% to 52.82%**. It applies to the archived marginals and leaves open whether better, independently validated score distributions could improve real accuracy.

## One pair per day on complete slates

The V28 archive froze every eligible pair and its five V8 combinations before kickoff. The V30 audit independently graded all fixtures on three complete Sydney slates. Comparing the first eligible pair in canonical fixture-ID order with the highest-mass pair changes only the pair-selection rule. Both rules use the fixed five V8 rows for their chosen pair. [Authenticated slate audit](../data/v30_full_slate_ranking/status.json)

| Sydney date | Eligible pairs | First pair modeled coverage | Highest-mass pair modeled coverage | Hits, first / highest-mass |
| --- | ---: | ---: | ---: | ---: |
| 22 Nov | 3 | 5.59% | 6.60% | 0 / 0 |
| 23 Nov | 210 | 5.55% | 7.59% | 0 / 0 |
| 24 Nov | 3 | 5.01% | 5.95% | 0 / 0 |
| Mean | | 5.38% | 6.71% | 0/3 vs 0/3 |

A uniformly drawn eligible pair had an equal-day average modeled mass of **5.17%** and an expected **1.43%** hit rate conditional on these three completed catalogs and their observed outcomes. This is an exploratory comparison of pre-frozen candidates. The 216 graded pair rows share fixtures and represent three complete day-level comparisons. Another 28 pair rows remained pending. [Slate protocol](../V30_FULL_SLATE_RANKING_RESEARCH.md)

The existing professional automatic shadow policy had **0/6 verified hits and 7 pending**. Its cohort overlaps the original archive. The separately registered V41 unified ensemble had zero verified opportunities. [Professional shadow results](../data/professional_double_shadow/performance.json) · [V41 results](../data/v41_unified_ensemble/performance.json)

The frozen historical research candidate hit **20/166 (12.05%)** on its own retrospective dates. Feature rules and model weights need dedicated time-forward checks because concentrating a model’s score probabilities can increase its reported five-row mass. [Historical comparison](../PROFESSIONAL_UPGRADE_REPORT.md)

## Replay of the new Final Five Doubles policy

The new policy compared all **244** eligible pairs across the four archived V28 slates, using each slate’s original cutoff. It chose the same pair and the same five score combinations as the exhaustive V8 control on every slate. On the three fully verified slates, each policy had **0/3 hits**. The final policy’s average coverage under its own model was **6.56%**. Paired market evidence changed the distribution for three comparison candidates on 23 November. The selected pairs used no qualified paired market or xG data. The fourth slate had 28 candidates and remained pending. These retrospective selections contribute zero trials to the final policy’s prospective release gate. [Final comparison code](../final_five_doubles.py)

## Prospective release rule

Final Five Doubles compares every eligible fixture pair in the authenticated V28 slate and issues one five-row candidate set in the research ledger before the kickoff window closes. The fixed V41 ensemble can use verified shot xG or bookmaker markets when both legs qualify at the common cutoff. An exhaustive V8 control receives the same one-pair and five-row budget. A failed or incomplete comparison withholds the final set. [Prospective selection code](../final_five_doubles.py)

Public release requires the first **100** issued sets from prior Sydney dates to have independently verified results, with a **95% Wilson lower bound of at least 70%**. The same lower bound must hold cumulatively with pending issued sets contributing zero wins. The policy must issue on at least **80%** of its past opportunities and have zero invalid evidence rows. The denominator covers published V28 slates since activation, including missed comparison windows as abstentions. Dates with no published V28 slate are outside that denominator. Each Sydney day contributes at most one final set. The intervals assume independent daily results. Repeated clubs, competitions, and common forecasting errors can still create dependence even when physical fixtures are deduplicated. The test applies to eligible V28 slates and describes the recorded history.

## Reproduce

Run from the repository root with the project requirements installed. The archived inputs are from source commit `8a15c450`. The script authenticates the frozen archives against stored forecast and result records and replays the new policy read-only. It makes no provider calls.

```bash
python - <<'PY'
from datetime import datetime
import json
from pathlib import Path
from statistics import mean

from final_five_doubles import PROVIDER_ROOT, _outcome, compare_slate
from five_double_accuracy import ROOT, _sources, build_report, utc
from professional_double_shadow import evaluate
from v28_independent_pair_trial import ORIGINAL_ROOT, OUT, PREMATCH, verify_decision
from v30_full_slate_audit import analyze
from verified_result_index import index_verified_results

cutoff = datetime.fromisoformat("2026-11-24T07:20:00+00:00")
original = build_report(now=cutoff)
print(original["observed"], original["diagnosis"])
for row in original["sets"]:
    packet = json.loads(Path(row["source_file"]).read_text())
    sources = _sources(ROOT, packet, datetime.fromisoformat(row["captured_at_utc"]))
    marginal_bound = min(
        sum(sorted((p for scores in source["score_matrix"] for p in scores), reverse=True)[:5])
        for source in sources
    )
    print(row["sydney_date"], row["status"], row["selected_five_model_coverage"],
          (row["verified_result_attribution"] or {}).get("actual_joint_score_rank_min"),
          row["full_grid_diagnosis"]["minimum_combinations_for_target_model_coverage"],
          marginal_bound)

shadow = evaluate(now=cutoff)
print("Professional shadow", shadow["hits"], shadow["verified_sets"], shadow["pending_sets"])
report = analyze(now=cutoff)
mass = "reference_v8_optimal_five_mass"
hit = "v8_baseline_five_hit"
for day in report["days"]:
    if day["complete_slate"]:
        rows = day["candidate_rows"]
        first = min(rows, key=lambda row: row["fixture_keys"])
        best = min(rows, key=lambda row: (-row[mass], row["fixture_keys"]))
        print(day["sydney_day"], len(rows), first[mass], best[mass], first[hit], best[hit],
              mean(row[mass] for row in rows), mean(row[hit] for row in rows))

finals, _, _ = index_verified_results(ORIGINAL_ROOT, PREMATCH, cutoff)

def tickets(selection):
    return {(row["score1"], row["score2"]) for row in selection["five_doubles"]}

for path in sorted((OUT / "decisions").glob("*.json")):
    source, items = verify_decision(path, root=ORIGINAL_ROOT, clock=cutoff)
    lookup = {tuple(map(str, item["key"])): item for item in items}
    result = compare_slate(items, provider_root=PROVIDER_ROOT,
                           cutoff=utc(source["policy_cutoff_utc"]))
    chosen, control = (result[key] for key in ("selection", "baseline_selection"))
    print(path.stem, result["compared_pairs"], result["evidence_counts"],
          chosen["estimate"], control["estimate"],
          chosen["fixture_keys"] == control["fixture_keys"] and tickets(chosen) == tickets(control),
          _outcome(chosen, lookup, finals, cutoff), _outcome(control, lookup, finals, cutoff))
PY
```
