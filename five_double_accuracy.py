"""Read-only diagnosis of original, frozen five exact-score doubles.

This audits the actual archived source grids and final-score evidence. It does
not generate new predictions, change selections, call providers, or train on
results. Model coverage and observed accuracy are deliberately separate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import heapq
import json
import math
from numbers import Real
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from forecast_ledger import _forecast

ROOT = Path("data/v10_prospective")
OUTPUT = Path("data/accuracy_diagnosis/report.json")
SYDNEY = ZoneInfo("Australia/Sydney")
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_COVERAGE_CELLS = 100_000


def utc(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone_required")
    return parsed.astimezone(timezone.utc)


def wilson(wins, total):
    if not total:
        return None
    z = 1.959963984540054
    rate = wins / total
    denominator = 1 + z*z/total
    center = (rate + z*z/(2*total)) / denominator
    margin = z * math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-margin), min(1., center+margin)]


def _load(path, default):
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _score(value):
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9]\d*)-(?:0|[1-9]\d*)", value):
        raise ValueError("invalid_exact_score")
    return tuple(int(x) for x in value.split("-"))


def _cells(grid):
    return sorted(((float(p), h, a) for h, row in enumerate(grid)
                   for a, p in enumerate(row)), key=lambda x: (-x[0], x[1], x[2]))


def _validated_grid(grid):
    try:
        rows = [list(row) for row in grid]
    except TypeError:
        raise ValueError("invalid_score_grid") from None
    if not rows or not rows[0] or max(len(rows), len(rows[0])) > 256:
        raise ValueError("invalid_score_grid_dimensions")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("score_grid_must_be_rectangular")
    if any(isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or p < 0
           for row in rows for p in row):
        raise ValueError("invalid_score_grid_probability")
    if abs(math.fsum(p for row in rows for p in row)-1) > 1e-6:
        raise ValueError("score_grid_must_sum_to_one")
    return [[float(p) for p in row] for row in rows]


def joint_geometry(first, second, target=.70):
    """Exact descending Cartesian products without a huge outer-product array.

    Every row is sorted. A heap merge gives the true top five and the smallest
    number of cells reaching the requested *model* mass. Extremely diffuse
    grids stop after a documented work limit, with no invented minimum.
    """
    if not 0 < target <= 1:
        raise ValueError("invalid_target")
    left, right = _cells(_validated_grid(first)), _cells(_validated_grid(second))
    pending = [(-p*right[0][0], h, a, right[0][1], right[0][2], i, 0)
               for i, (p, h, a) in enumerate(left)]
    heapq.heapify(pending)
    top = []
    coverage = 0.
    count = 0
    minimum = None
    while pending and count < MAX_COVERAGE_CELLS:
        negative, h, a, j, k, i, column = heapq.heappop(pending)
        probability = -negative
        if count < 5:
            top.append({"score1": f"{h}-{a}", "score2": f"{j}-{k}",
                        "joint_probability": probability})
        count += 1
        coverage = math.fsum((coverage, probability))
        if minimum is None and coverage >= target-1e-12:
            minimum = count
        if count >= 5 and minimum is not None:
            break
        following = column+1
        if following < len(right):
            p, h, a = left[i]
            q, j, k = right[following]
            heapq.heappush(pending, (-p*q, h, a, j, k, i, following))
    return {"optimal_five": top,
            "optimal_five_model_coverage": math.fsum(x["joint_probability"] for x in top),
            "minimum_combinations_for_target_model_coverage": minimum,
            "target_model_coverage": target,
            "target_computation_status": "exact" if minimum is not None else "work_limit_reached",
            "coverage_cells_examined": count,
            "full_joint_support_cells": len(left)*len(right)}


def _sources(root, packet, captured):
    models = []
    for side in ("first", "second"):
        leg = packet[side]
        parts = [str(leg[k]) for k in ("provider", "fixture_id", "forecast_id")]
        if any(not SAFE.fullmatch(x) or x in (".", "..") for x in parts):
            raise ValueError("invalid_source_path")
        path = Path(root)/"forecasts"/parts[0]/parts[1]/(parts[2]+".json")
        if not path.is_file():
            raise ValueError("missing_original_forecast")
        model = _forecast(_load(path, {}))
        if model.get("model_key") not in ("v8", "v8_near"):
            raise ValueError("source_is_not_original_v8_family")
        for key in ("provider", "fixture_id", "forecast_id", "home_team_id", "away_team_id"):
            if str(leg[key]) != str(model[key]):
                raise ValueError("source_identity_mismatch")
        if utc(leg["kickoff_utc"]) != utc(model["kickoff_utc"]):
            raise ValueError("source_kickoff_mismatch")
        if utc(model["captured_at_utc"]) > captured:
            raise ValueError("source_created_after_pair_selection")
        models.append(model)
    return models


def _result_index(root, prematch_root):
    by = {}
    conflicts = set()
    rejected = Counter()
    for path in (Path(root)/"verified_results.json", Path(prematch_root)/"verified_results.json"):
        try:
            source = _load(path, {})
        except (ValueError, OSError):
            rejected["unreadable_result_file"] += 1
            continue
        if not isinstance(source, dict):
            rejected["result_file_not_dictionary"] += 1
            continue
        for result in source.values():
            if not isinstance(result, dict):
                rejected["result_not_dictionary"] += 1
                continue
            key = (str(result.get("provider")), str(result.get("fixture_id")))
            signature = tuple(str(result.get(k)) for k in (
                "home_team_id", "away_team_id", "home_goals", "away_goals", "score_basis"))
            try:
                kickoff = utc(result["kickoff_utc"]).isoformat()
            except (ValueError, TypeError, KeyError):
                rejected["invalid_result_kickoff"] += 1
                continue
            signature += (kickoff,)
            if key in by and by[key][0] != signature:
                conflicts.add(key)
            by.setdefault(key, (signature, result))
    return {key: value[1] for key, value in by.items() if key not in conflicts}, conflicts, rejected


def _final(model, result, now):
    if not isinstance(result, dict):
        return None
    if result.get("evidence_version") != 2 or result.get("score_basis") != "regulation":
        return None
    if result.get("status") not in ("FT", "FINISHED", "AET", "PEN"):
        return None
    try:
        if any(str(model[k]) != str(result[k]) for k in (
                "provider", "fixture_id", "home_team_id", "away_team_id")):
            return None
        if utc(model["kickoff_utc"]) != utc(result["kickoff_utc"]):
            return None
        if not utc(model["kickoff_utc"]) < utc(result["verified_at_utc"]) <= now:
            return None
        goals = tuple(result[k] for k in ("home_goals", "away_goals"))
        return goals if all(type(g) is int and g >= 0 for g in goals) else None
    except (ValueError, TypeError, KeyError):
        return None


def _probability(grid, goals):
    h, a = goals
    return float(grid[h][a]) if h < len(grid) and a < len(grid[0]) else 0.


def _attribution(models, picks, actual):
    legs = []
    for index, (model, goals) in enumerate(zip(models, actual), 1):
        probability = _probability(model["score_matrix"], goals)
        chosen = sorted({pick[f"score{index}"] for pick in picks})
        score = f"{goals[0]}-{goals[1]}"
        legs.append({"fixture_id": model["fixture_id"], "match": model.get("fixture_name"),
                     "actual_score": score, "actual_model_probability": probability,
                     "actual_score_rank_min": 1+sum(p > probability for p, _, _ in _cells(model["score_matrix"])),
                     "selected_leg_scores": chosen, "actual_leg_score_covered": score in chosen})
    joint = legs[0]["actual_model_probability"]*legs[1]["actual_model_probability"]
    right = sorted(p for row in models[1]["score_matrix"] for p in row)
    greater = equal = 0
    def bound(probability, inclusive):
        low, high = 0, len(right)
        while low < high:
            middle = (low+high)//2
            product = probability*right[middle]
            if product < joint or (inclusive and product == joint):
                low = middle+1
            else:
                high = middle
        return low
    for row in models[0]["score_matrix"]:
        for probability in row:
            if probability > 0:
                left_bound, right_bound = bound(probability, False), bound(probability, True)
                greater += len(right)-right_bound
                equal += right_bound-left_bound
            elif joint == 0:
                equal += len(right)
    hit = any(_score(p["score1"]) == actual[0] and _score(p["score2"]) == actual[1] for p in picks)
    return {"legs": legs, "actual_joint_model_probability": joint,
            "actual_joint_score_rank_min": greater+1,
            "actual_joint_score_rank_max": greater+max(1, equal),
            "classification": "hit" if hit else (
                "both_leg_scores_outside_selected_options" if not any(x["actual_leg_score_covered"] for x in legs)
                else "one_leg_score_outside_selected_options" if not all(x["actual_leg_score_covered"] for x in legs)
                else "leg_scores_present_but_not_in_same_combination")}


def build_report(root=ROOT, prematch_root=Path("data/prematch"), *, now=None, target=.70, results=None):
    clock = utc(now or datetime.now(timezone.utc))
    if not 0 < target <= 1:
        raise ValueError("invalid_target")
    if results is None:
        finals, conflicts, result_rejections = _result_index(root, prematch_root)
    else:
        finals = {(str(r.get("provider")), str(r.get("fixture_id"))): r
                  for r in results.values() if isinstance(r, dict)}
        conflicts = set()
        result_rejections = Counter()
    paths = sorted((Path(root)/"doubles").glob("*.json"))
    rejected = Counter()
    records = []
    for path in paths:
        try:
            packet = _load(path, {})
            if packet.get("mode") != "prospective" or packet.get("status") != "frozen_pre_kickoff":
                raise ValueError("not_original_frozen_prospective_set")
            captured = utc(packet["captured_at_utc"])
            if captured > clock:
                raise ValueError("pair_capture_after_audit_clock")
            models = _sources(root, packet, captured)
            if captured >= min(utc(m["kickoff_utc"]) for m in models):
                raise ValueError("pair_not_frozen_before_both_kickoffs")
            days = {utc(m["kickoff_utc"]).astimezone(SYDNEY).date().isoformat() for m in models}
            if days != {packet["sydney_date"]} or packet["sydney_date"] != path.stem:
                raise ValueError("not_same_declared_sydney_date")
            if len({(m["provider"], m[k]) for m in models for k in ("home_team_id", "away_team_id")}) != 4:
                raise ValueError("four_distinct_teams_required")
            picks = packet["five_doubles"]
            if not isinstance(picks, list) or len(picks) != 5 or len({(p["score1"], p["score2"]) for p in picks}) != 5:
                raise ValueError("five_distinct_combinations_required")
            derived = []
            for pick in picks:
                probability = math.prod(_probability(m["score_matrix"], _score(pick[f"score{i}"]))
                                        for i, m in enumerate(models, 1))
                saved = pick["joint_probability"]
                if type(saved) not in (int, float) or not math.isfinite(saved) or abs(saved-probability) > 1e-9:
                    raise ValueError("saved_joint_probability_disagrees_with_source_grid")
                derived.append(probability)
            coverage = math.fsum(derived)
            if not math.isclose(coverage, packet["estimated_coverage"], abs_tol=1e-9):
                raise ValueError("saved_set_coverage_disagrees_with_source_grids")
            geometry = joint_geometry(*(m["score_matrix"] for m in models), target=target)
            actual = [_final(m, finals.get((m["provider"], m["fixture_id"])), clock) for m in models]
            attribution = _attribution(models, picks, actual) if all(x is not None for x in actual) else None
            records.append({"sydney_date": packet["sydney_date"], "source_file": str(path),
                            "captured_at_utc": captured.isoformat(),
                            "source_forecast_ids": [m["forecast_id"] for m in models],
                            "source_model_keys": [m["model_key"] for m in models],
                            "status": ("hit" if attribution["classification"] == "hit" else "miss") if attribution else "pending",
                            "selected_five_model_coverage": coverage,
                            "selected_five_are_probability_optimal": math.isclose(coverage, geometry["optimal_five_model_coverage"], abs_tol=1e-10),
                            "model_coverage_shortfall_to_target": max(0., target-coverage),
                            "target_model_mass_reached_by_five": coverage >= target,
                            "full_grid_diagnosis": geometry, "verified_result_attribution": attribution,
                            "source_feature_gaps": [{
                                "fixture_id": m["fixture_id"],
                                "verified_shot_xg_available": bool(((m.get("research_context") or {}).get("verified_xg_xga") or {}).get("available")),
                                "validated_lineup_adjustments_applied": bool(((m.get("research_context") or {}).get("verified_lineup_adjustments") or {}).get("applied")),
                                "market_used": bool(m.get("market_used"))} for m in models]})
        except (ValueError, TypeError, KeyError, OSError, AttributeError, OverflowError) as exc:
            rejected[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    verified = [r for r in records if r["status"] in ("hit", "miss")]
    wins = sum(r["status"] == "hit" for r in verified)
    interval = wilson(wins, len(verified))
    return {"schema_version": 1, "evaluated_at_utc": clock.isoformat(),
            "prediction_type": "five_exact_score_combinations_on_the_same_two_matches",
            "evidence_policy": "original_archived_v8_source_ids_before_selection_and_both_kickoffs_identity_matched_verified_regulation_results",
            "target": target, "original_set_files": len(paths), "source_valid_sets": len(records),
            "source_rejections": dict(rejected), "conflicting_result_identities": len(conflicts),
            "result_source_rejections": dict(result_rejections),
            "observed": {"verified_sets": len(verified), "hits": wins,
                         "misses": len(verified)-wins, "pending_sets": len(records)-len(verified),
                         "hit_rate": wins/len(verified) if verified else None,
                         "wilson_95": interval,
                         "minimum_100_sets_and_lower_interval_at_target": bool(len(verified) >= 100 and interval[0] >= target),
                         "independence_of_repeated_sets_not_established": True},
            "diagnosis": {"all_original_sets_already_maximize_five_cell_model_coverage": bool(records) and all(r["selected_five_are_probability_optimal"] for r in records),
                          "sets_with_target_model_coverage": sum(r["target_model_mass_reached_by_five"] for r in records),
                          "model_coverage_range": [min(r["selected_five_model_coverage"] for r in records), max(r["selected_five_model_coverage"] for r in records)] if records else None,
                          "verified_miss_attribution": dict(Counter(r["verified_result_attribution"]["classification"] for r in verified if r["status"] == "miss"))},
            "target_70_percent_proven": False, "model_updated_or_promoted": False,
            "original_forecasts_or_doubles_modified": False, "external_requests_performed": 0,
            "limitations": ["Model probability coverage is not empirical accuracy.",
                            "A five-double hits only when both exact scores occur in the SAME selected combination.",
                            "The product of the two score distributions assumes independent matches.",
                            "Minimum combination counts describe current model mass, not a way to guarantee 70% observed success.",
                            "Wilson intervals assume independent Bernoulli trials; one completed set cannot establish a reliable rate.",
                            "Unavailable, conflicting, wrong-identity and not-yet-verified results remain pending.",
                            "This audit does not inspect private Streamlit/PostgreSQL prediction archives."],
            "sets": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prematch-root", type=Path, default=Path("data/prematch"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build_report(args.root, args.prematch_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sets"}, indent=2))


if __name__ == "__main__":
    main()
