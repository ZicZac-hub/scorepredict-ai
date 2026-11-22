"""V30: strictly read-only, full-slate prospective rank/coverage research.

V28 froze every valid pair, with its five exact-score tickets and original
V8 reference-model mass, BEFORE kickoff. V30 can grade these already frozen
choices after independent verified final scores arrive. It NEVER retrofreezes,
changes ranks, or creates extra past predictions.

Every candidate pair shares underlying matches with many others. Candidate
pair hit fractions are correlated catalog diagnostics, not independent trials
and not the win rate of one selected five-double set. The independent unit
for initial comparative analysis is a COMPLETE Sydney-day candidate slate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from five_double_accuracy import _final, utc
from verified_result_index import index_verified_results
from v28_independent_pair_trial import (
    ORIGINAL_ROOT, PREMATCH, OUT as V28_OUT, _key, _hit, verify_decision,
)

VERSION = "v30-verified-full-slate-ranking-audit-v1-research-only"
OUT = Path("data/v30_full_slate_ranking")
CHECKPOINTS = (1, 5, 10)
QUARTILES = 4
INITIAL_COMPLETE_SLATE_DAYS = 30


def _ranked(catalog, metric):
    return sorted(catalog, key=lambda x:(-x[metric], x["fixture_keys"]))


def _rank_table(catalog, metric):
    """Deterministic rankings that read ONLY frozen pre-kickoff fields."""
    return {
        tuple(tuple(key) for key in candidate["fixture_keys"]): rank
        for rank, candidate in enumerate(_ranked(catalog, metric), 1)
    }


def _metric(items):
    if not items:
        return {
            "verified_pair_rows": 0,
            "observed_robust_five_hits": 0,
            "observed_v8_optimal_five_hits": 0,
            "observed_robust_five_fraction": None,
            "mean_v8_reference_mass_for_robust_five": None,
            "model_expected_robust_hits_sum_correlated": None,
        }
    n = len(items)
    total = sum(row["robust_five_hit"] for row in items)
    reference = sum(row["v8_baseline_five_hit"] for row in items)
    mean = math.fsum(row["reference_robust_five_mass_under_v8"] for row in items)/n
    return {
        "verified_pair_rows": n,
        "observed_robust_five_hits": int(total),
        "observed_v8_optimal_five_hits": int(reference),
        "observed_robust_five_fraction": total/n,
        "mean_v8_reference_mass_for_robust_five": mean,
        "model_expected_robust_hits_sum_correlated": n*mean,
    }


def _top_k_comparison(rows, *, k):
    """Equal ticket budgets for robust/quality selection on COMPLETE slates."""
    if not rows:
        return None
    s = min(k,len(rows))
    quality = sorted(rows,key=lambda x:(x["quality_rank"],x["fixture_keys"]))[:s]
    control = sorted(rows,key=lambda x:(x["robustness_rank"],x["fixture_keys"]))[:s]
    def stats(items):
        sources = {tuple(key) for x in items for key in x["fixture_keys"]}
        return {
            "pair_count":len(items),
            "ticket_budget_if_all_pairs_used":5*len(items),
            "unique_underlying_fixtures":len(sources),
            "any_pair_five_double_hit":any(row["robust_five_hit"] for row in items),
            "number_of_hit_pairs":sum(row["robust_five_hit"] for row in items),
            "model_expected_hits_sum_not_independent":
                math.fsum(x["reference_robust_five_mass_under_v8"] for x in items),
        }
    return {
        "top_k":s,
        "quality_ranked_pairs":stats(quality),
        "robustness_ranked_pairs":stats(control),
        "not_comparable_to_one_five_double_ticket_set":s > 1,
        "correlated_matches_preclude_independent_pair_inference":True,
    }


def _band_stats(full_rows):
    """Pre-declared within-day quality-rank quartiles, full slates ONLY."""
    n = len(full_rows)
    buckets = [[] for _ in range(QUARTILES)]
    for row in full_rows:
        idx = min(QUARTILES-1,(row["quality_rank"]-1)*QUARTILES//n)
        buckets[idx].append(row)
    return [
        {"quality_rank_quartile":idx+1, **_metric(entries)}
        for idx,entries in enumerate(buckets)
    ]


def grade_day(file, *, root, indexed, clock):
    """Read/save NO forecasts. Recheck full V28 provenance before grading."""
    original, items = verify_decision(file,root=root,clock=clock)
    day = original["sydney_day"]
    meta = {
        "sydney_day":day,
        "original_cutoff_utc":original["policy_cutoff_utc"],
        "original_issued_at_utc":original["actually_issued_at_utc"],
        "original_policy":original["policy"],
        "original_candidate_catalog_sha256":
            original["candidate_catalog_sha256"],
        "candidate_count":original["eligible_four_team_pair_count"],
        "frozen_source_count":original["eligible_fixture_count"],
        "no_forecasts_created_during_grading":True,
    }
    if original["policy"]!="FROZEN":
        return {**meta,"status":"abstained_originally",
                "reason":original["reason"],
                "candidate_pairs_verified":0,
                "candidate_pairs_pending":original["computed_pairs"],
                "complete_slate":False}
    source_map = {_key(item):item for item in items}
    verified = {}
    missing = []
    for key,item in sorted(source_map.items()):
        source = item["record"]
        actual = _final(
            source,indexed.get((str(source["provider"]),
                                str(source["fixture_id"]))),clock)
        if actual is None:
            missing.append({
                "provider":key[0],"fixture_id":key[1],
                "kickoff_utc":item["kickoff"].isoformat(),
                "status":("before_kickoff" if clock<item["kickoff"]
                          else "verified_regulation_final_unavailable"),
            })
        else:
            verified[key] = actual
    catalog = original["candidate_catalog"]
    quality_ranks = _rank_table(catalog,"quality_rank_score")
    robust_ranks = _rank_table(catalog,"robustness_rank_score")
    observations = []
    waiting = 0
    for frozen in catalog:
        keys = tuple(tuple(x) for x in frozen["fixture_keys"])
        if not all(x in verified for x in keys):
            waiting += 1
            continue
        scores = [verified[k] for k in keys]
        observations.append({
            "fixture_keys":frozen["fixture_keys"],
            "quality_rank":quality_ranks[keys],
            "robustness_rank":robust_ranks[keys],
            "quality_rank_score_uncalibrated":frozen["quality_rank_score"],
            "robustness_rank_score_uncalibrated":frozen["robustness_rank_score"],
            "reference_robust_five_mass_under_v8":
                frozen["robust_five_under_original_v8_mass"],
            "reference_v8_optimal_five_mass":frozen["v8_optimal_model_mass"],
            "actual_scores":[f"{h}-{a}" for h,a in scores],
            "robust_five_hit":_hit(frozen,scores),
            "v8_baseline_five_hit":any(
                (x["score1"],x["score2"]) ==
                tuple(f"{h}-{a}" for h,a in scores)
                for x in frozen["five_original_v8_optimal_scores"]),
        })
    complete = len(verified)==len(source_map)
    if complete and (len(observations)!=len(catalog) or waiting):
        raise ValueError("complete_fixture_slate_inconsistent_with_candidate_rows")
    result = {
        **meta,
        "status":("complete_verified_catalog" if complete
                  else "partially_verified_catalog" if observations
                  else "awaiting_verified_finals"),
        "candidate_pairs_verified":len(observations),
        "candidate_pairs_pending":waiting,
        "unique_fixtures_verified":len(verified),
        "unique_fixtures_missing_verified_final":len(missing),
        "missing_fixture_evidence":missing,
        "complete_slate":complete,
        "exploratory_verified_candidate_rows":_metric(observations),
        "candidate_rows":observations,
        "candidate_hit_rows_are_correlated_not_independent_trials":True,
    }
    if complete:
        result["full_slate_diagnostics"] = {
            "all_candidate_pairs":_metric(observations),
            "predefined_quality_quartiles":_band_stats(observations),
            "equal_budget_top_k_comparisons":{
                str(k):_top_k_comparison(observations,k=k)
                for k in CHECKPOINTS},
            "original_quality_selected_pair_keys":
                original["quality_selection"]["fixture_keys"],
            "original_robustness_control_pair_keys":
                original["robustness_control"]["fixture_keys"],
            "full_slate_learning_only_not_independent_policy_win_rate":True,
        }
    return result


def analyze(*,root=ORIGINAL_ROOT,prematch=PREMATCH,v28_out=V28_OUT,now=None):
    clock = utc(now or datetime.now(timezone.utc))
    index,conflicts,rejections = index_verified_results(root,prematch,clock)
    rows,errors = [],Counter()
    for file in sorted((Path(v28_out)/"decisions").glob("*.json")):
        try:
            rows.append(grade_day(file,root=root,indexed=index,clock=clock))
        except (OSError,ValueError,TypeError,KeyError,OverflowError,IndexError) as exc:
            errors[str(exc) if isinstance(exc,ValueError)
                   else type(exc).__name__] += 1
    active = [x for x in rows if x["original_policy"]=="FROZEN"]
    fully = [x for x in active if x["complete_slate"]]
    exploratory = [x for x in active if x["candidate_pairs_verified"]]
    # Unlike the pair-row total, this count treats ONE date as ONE correlated
    # research unit, regardless of how many overlapping candidates it held.
    full_day_mean = (
        math.fsum(x["full_slate_diagnostics"]["all_candidate_pairs"]
                  ["observed_robust_five_fraction"] for x in fully)/len(fully)
        if fully else None
    )
    return {
        "version":VERSION,
        "research_scope":"original_pre_match_write_once_v28_full_candidate_catalogs",
        "evaluated_at_utc":clock.isoformat(),
        "original_frozen_days":len(active),
        "original_abstained_days":sum(
            x["status"]=="abstained_originally" for x in rows),
        "fully_verified_slate_days":len(fully),
        "partially_verified_slate_days":sum(
            x["status"]=="partially_verified_catalog" for x in active),
        "days_with_some_verified_pairs":len(exploratory),
        "days_still_missing_results":len(active)-len(fully),
        "frozen_candidate_pair_rows":sum(x["candidate_count"] for x in active),
        "candidate_pair_rows_independently_verified":sum(
            x["candidate_pairs_verified"] for x in active),
        "candidate_pair_rows_awaiting_verified_finals":sum(
            x["candidate_pairs_pending"] for x in active),
        "complete_day_unweighted_mean_candidate_hit_fraction":
            full_day_mean,
        "initial_full_slate_day_review_target":INITIAL_COMPLETE_SLATE_DAYS,
        "initial_review_ready":len(fully)>=INITIAL_COMPLETE_SLATE_DAYS,
        "verified_result_conflicts_excluded":len(conflicts),
        "invalid_independent_final_sources":dict(rejections),
        "invalid_or_tampered_frozen_days_excluded":dict(errors),
        "days":rows,
        "no_new_predictions_or_past_score_reconstruction":True,
        "no_retroactive_chosen_pair_alteration":True,
        "model_estimated_mass_is_not_real_win_rate":True,
        "overlapping_pairs_are_not_independent_samples":True,
        "no_production_model_promotion":True,
        "seventy_percent_five_double_hit_rate_not_proven":True,
    }


def run(*,root=ORIGINAL_ROOT,prematch=PREMATCH,v28_out=V28_OUT,out=OUT,now=None):
    result = analyze(root=root,prematch=prematch,v28_out=v28_out,now=now)
    output = Path(out)/"status.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    data = json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if not output.exists() or output.read_text(encoding="utf-8")!=data:
        output.write_text(data,encoding="utf-8")
    return result


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.parse_args()
    report=run()
    print(json.dumps({
        "original_days":report["original_frozen_days"],
        "fully_verified_slate_days":report["fully_verified_slate_days"],
        "frozen_pair_rows":report["frozen_candidate_pair_rows"],
        "verified_pair_rows":report["candidate_pair_rows_independently_verified"],
        "production_model_promoted":False,
    },sort_keys=True))
