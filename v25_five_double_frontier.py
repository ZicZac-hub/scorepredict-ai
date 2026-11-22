"""Five-double 70% goal: inspect the probability-coverage frontier, never fake wins.

Five distinct joint exact-score choices have, under a given pair of frozen
goal-score matrices, a known probability-mass optimum: their five largest
joint cell probabilities. A different ranking policy cannot exceed this
mass *under those same original distributions*.

Model-estimated coverage != observed hit rate or an upper bound for any
better future model. Report a prespecified 70%-mass ticket count, but never
call it 70% actual accuracy.
"""
import argparse
import heapq
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from five_double_accuracy import _cells, _validated_grid, utc, wilson
from v8_v25_matched_audit import (
    _frozen_set, _v25_probability, build_report,
)

ROOT = Path("data/v10_prospective")
PREMATCH = Path("data/prematch")
SETS = Path("data/pro_analyst_shadow/sets")
OUTPUT = Path("data/accuracy_diagnosis/v25_five_double_frontier.json")
TARGET = 0.70
CHECKPOINTS = (1, 5, 10, 20, 50, 100, 150, 200, 500, 1000)
MAX_WORK = 100_000


def frontier(first, second, target=TARGET, checkpoints=CHECKPOINTS, *, max_work=MAX_WORK):
    """Exact top-k joint-probability sums; no random sampling or refitting.

    The product assumes that the two regulation-time score outcomes are
    independent conditional on their saved forecasts. Ties are harmless to
    the summed optimal mass. For large supports the non-completed frontier
    is explicitly marked as censored rather than inventing a target count.
    """
    if type(target) not in (int, float) or not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError("invalid_target")
    if (not checkpoints or any(type(k) is not int or k < 1 for k in checkpoints)
            or type(max_work) is not int or max_work < 5):
        raise ValueError("invalid_frontier_settings")
    left = _cells(_validated_grid(first))
    right = _cells(_validated_grid(second))
    support = len(left) * len(right)
    limit = min(max_work, support)
    targets = tuple(sorted(set(checkpoints)))
    heap = [(-p * right[0][0], i, 0) for i, (p, _, _) in enumerate(left)]
    heapq.heapify(heap)
    count = 0
    mass = 0.0
    points = {}
    min_for_target = None
    while heap and count < limit:
        neg, i, j = heapq.heappop(heap)
        # A compensated-style cumulative sum avoids losing tiny cells.
        mass = math.fsum((mass, -neg))
        count += 1
        if count in targets:
            points[str(count)] = min(1., mass)
        if min_for_target is None and mass >= target - 1e-12:
            min_for_target = count
        if j + 1 < len(right):
            heapq.heappush(heap, (-left[i][0] * right[j + 1][0], i, j + 1))
        if min_for_target is not None and count >= max(5, max(targets)):
            break
    for k in targets:
        if k > support:
            points[str(k)] = 1.0  # All supported cells are covered.
    achieved_five = points["5"] if "5" in points else None
    return {
        "goal": target,
        "model_max_five_probability_mass": achieved_five,
        "target_met_by_five_model_choices": (
            achieved_five >= target - 1e-12 if achieved_five is not None else None),
        "minimum_joint_combinations_for_target_model_mass": min_for_target,
        "target_count_status": (
            "exact" if min_for_target is not None else "not_reached_within_work_limit"),
        "checkpoints_model_mass": points,
        "joint_score_support_cells": support,
        "computed_cells": count,
        "scores_are_independent_conditional_on_model": True,
        "not_a_verified_hit_rate": True,
        "higher_probability_model_does_not_prove_better_accuracy": True,
    }


def build_frontier_report(*, root=ROOT, prematch=PREMATCH, sets=SETS, now=None):
    """Use the same independently authenticated archived pairs as the V8/V25 audit.

    Full provenance and independent result matching come from build_report().
    Missing/invalid original evidence never receives an invented replacement.
    Only genuinely originally frozen V25 probabilities qualify for V25 mass.
    """
    clock = utc(now or datetime.now(timezone.utc))
    independent = build_report(root, prematch, sets, now=clock)
    indexed = {row["sydney_date"]: row for row in independent["rows"]}
    rows, misses = [], []
    for file in sorted(Path(sets).glob("*.json")):
        canonical = indexed.get(file.stem)
        if not canonical:
            misses.append(file.stem)
            continue
        try:
            packet, sources, grids, _, _, _ = _frozen_set(file, root, clock)
        except (ValueError, KeyError, TypeError, OverflowError):
            misses.append(file.stem)
            continue
        base = frontier(*grids)
        v25_status = canonical["v25_probability_status"]
        probability = None
        if v25_status == "frozen":
            try:
                frozen = [
                    _v25_probability(packet[side], source, utc(packet["selected_at_utc"]))
                    for side, source in zip(("first", "second"), sources)
                ]
                if all(model is not None for model in frozen):
                    probability = frontier(*frozen)
            except (ValueError, TypeError, KeyError, OverflowError):
                v25_status = "invalid"
        row = {
            "sydney_date": packet["sydney_date"],
            "status": canonical["status"],
            "original_v8_five_double_hit": canonical.get("v8_five_hit"),
            "v25_robust_five_double_hit": canonical.get("v25_five_hit"),
            "original_v8": base,
            "v25_frozen_probability_status": v25_status,
            "v25_originally_frozen_probability": probability,
            "original_v25_robust_five_mass_under_v8": canonical[
                "v25_selected_five_mass_under_v8_model"],
            "never_rewrite_original_five_choices": True,
        }
        rows.append(row)
    archived_five = [row["original_v8"]["model_max_five_probability_mass"]
                     for row in rows]
    verified = [row for row in rows if row["status"] == "verified"]
    frozen = [row for row in rows if row["v25_originally_frozen_probability"]]
    target_counts = [
        row["original_v8"]["minimum_joint_combinations_for_target_model_mass"]
        for row in rows
        if row["original_v8"]["minimum_joint_combinations_for_target_model_mass"] is not None
    ]
    report = {
        "schema_version": 1,
        "study": "pre_match_original_v8_model_five_double_70pct_feasibility_frontier",
        "evaluated_at_utc": clock.isoformat(),
        "target_model_coverage": TARGET,
        "original_archived_sets": len(rows),
        "source_invalid_excluded_sets": independent["source_exclusions"],
        "frontier_missing_sources_excluded": misses,
        "five_model_coverage_min": min(archived_five) if archived_five else None,
        "five_model_coverage_max": max(archived_five) if archived_five else None,
        "five_model_coverage_mean": (
            math.fsum(archived_five) / len(archived_five) if archived_five else None),
        "sets_with_70pct_model_mass_in_only_five": sum(
            row["original_v8"]["target_met_by_five_model_choices"] for row in rows),
        "min_number_of_combinations_to_70pct_model_mass": (
            min(target_counts) if target_counts else None),
        "max_number_of_combinations_to_70pct_model_mass": (
            max(target_counts) if target_counts else None),
        "original_v8_verified": {
            "wins": independent["v8"]["five_double_hits"],
            "verified_pairs": independent["verified_matched_pairs"],
            "wilson_95": independent["v8"]["wilson_95"],
        },
        "v25_robust_verified": {
            "wins": independent["v25"]["five_double_hits"],
            "verified_pairs": independent["verified_matched_pairs"],
            "wilson_95": independent["v25"]["wilson_95"],
        },
        "pending_original_pairs": independent["pending_matched_pairs"],
        "separately_frozen_v25_probability_pairs": len(frozen),
        "verifiably_frozen_v25_completed_pairs": sum(
            row["status"] == "verified" for row in frozen),
        "rows": rows,
        "research_conclusion": (
            "target_five_model_mass_reached_on_at_least_one_source_pair"
            if any(row["original_v8"]["target_met_by_five_model_choices"] for row in rows)
            else "70pct_model_mass_not_reached_with_five_original_v8_combinations"
            if rows else "no_authenticated_frozen_pairs"),
        "real_70pct_five_double_hit_rate_proven": False,
        "promoted_production_model": False,
        "retrospective_extra_tickets_are_not_frozen_predictions": True,
        "caveats": [
            "A model-based frontier is not an empirically verified accuracy rate.",
            "Five highest joint probabilities are mathematically maximal only within that original pair of forecast grids.",
            "An improved future model might assign different score probabilities, but must prove calibration on new frozen predictions.",
            "More than five choices is a DIFFERENT experiment and not five-double accuracy.",
            "The two games are treated as independent conditional on the saved distributions.",
            "Pending or unmatched final scores are never counted as losses.",
        ],
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--prematch", type=Path, default=PREMATCH)
    p.add_argument("--sets", type=Path, default=SETS)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    report = build_frontier_report(root=args.root, prematch=args.prematch, sets=args.sets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != text:
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps({k:report[k] for k in (
        "original_archived_sets", "five_model_coverage_min", "five_model_coverage_max",
        "min_number_of_combinations_to_70pct_model_mass",
        "max_number_of_combinations_to_70pct_model_mass",
        "sets_with_70pct_model_mass_in_only_five",
        "separately_frozen_v25_probability_pairs",
        "real_70pct_five_double_hit_rate_proven",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
