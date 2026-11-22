"""Independent, read-only prospective V25 freeze-integrity and coverage monitor.

This consumes the output of v8_v25_matched_audit.py, which independently
rechecks the original immutable V8 forecast identities, frozen V25 matrices,
mixture policy and independently verified regulation-time finals.

Legacy V25 five-double picks are allowed to remain selection-only. They must
NEVER be retroactively supplied with probability matrices. Newly frozen sets
are required to include both original pre-kickoff probability grids and
their reproducible mixture-five picks.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from five_double_accuracy import utc

SOURCE = Path("data/accuracy_diagnosis/v8_v25_matched.json")
OUTPUT = Path("data/accuracy_diagnosis/v25_freeze_monitor.json")

# The prospective probability-audit upgrade was merged into main at this
# time. This immutable rollout boundary is fixed, not inferred from results.
PROBABILITY_REQUIRED_FROM = "2026-11-22T05:17:27+00:00"
INITIAL_EVALUATION_TARGET = 100


def build_monitor(matched):
    if not isinstance(matched, dict) or not isinstance(matched.get("rows"), list):
        raise ValueError("missing_independent_matched_audit")
    if type(matched.get("archived_sets_found")) is not int:
        raise ValueError("missing_archive_denominator")
    source_valid = matched.get("source_valid_sets")
    if type(source_valid) is not int or source_valid != len(matched["rows"]):
        raise ValueError("invalid_source_valid_set_count")

    cutoff = utc(PROBABILITY_REQUIRED_FROM)
    legacy = []
    prospective = []
    problems = []
    seen_days = set()
    for row in matched["rows"]:
        if not isinstance(row, dict):
            raise ValueError("invalid_matched_row")
        day = row.get("sydney_date")
        if not isinstance(day, str) or not day or day in seen_days:
            problems.append({"day": day or "unknown", "reason": "duplicate_or_missing_day"})
        seen_days.add(day)
        try:
            selection_time = utc(row["selection_time_utc"])
        except (TypeError, ValueError, KeyError):
            problems.append({"day": day, "reason": "missing_selection_time"})
            continue

        probability = row.get("v25_probability_status")
        mixture = row.get("v25_mixture_five_status")
        if selection_time < cutoff:
            cohort = "legacy_selection_only"
        else:
            cohort = "prospective_probability_required"

        record = {
            "sydney_date": day,
            "cohort": cohort,
            "result_status": row.get("status"),
            "probability_status": probability,
            "mixture_five_status": mixture,
        }
        if cohort == "legacy_selection_only":
            legacy.append(record)
        else:
            prospective.append(record)
            if probability != "frozen":
                problems.append({"day": day, "reason": "missing_or_invalid_frozen_v25_grids"})
            if mixture != "frozen":
                problems.append({"day": day, "reason": "missing_or_invalid_frozen_mixture_choices"})
            if row.get("status") == "verified":
                if (not isinstance(row.get("v25_score_probability_diagnostics"), list)
                        or len(row["v25_score_probability_diagnostics"]) != 2):
                    problems.append({"day": day, "reason": "missing_same_leg_score_metrics"})
                if type(row.get("v25_mixture_five_hit")) is not bool:
                    problems.append({"day": day, "reason": "missing_verified_mixture_grade"})
            elif row.get("status") != "pending":
                problems.append({"day": day, "reason": "invalid_result_state"})

    excluded = matched.get("source_exclusions") or {}
    if not isinstance(excluded, dict):
        raise ValueError("invalid_original_source_exclusions")
    excluded_count = sum(value for value in excluded.values()
                         if type(value) is int and value > 0)
    if source_valid + excluded_count != matched["archived_sets_found"]:
        problems.append({"day": None, "reason": "unaccounted_original_archived_sets"})
    if excluded_count:
        problems.append({"day": None, "reason": "invalid_original_source_archives",
                         "count": excluded_count})

    ready = [
        row for row in prospective
        if row["probability_status"] == "frozen"
        and row["mixture_five_status"] == "frozen"
    ]
    verified = [row for row in ready if row["result_status"] == "verified"]
    pending = [row for row in ready if row["result_status"] == "pending"]
    all_verified_probability = (matched.get("matched_probability_calibration") or {})
    actual_verified_pairs = all_verified_probability.get("verified_pairs")
    if type(actual_verified_pairs) is not int or actual_verified_pairs < 0:
        raise ValueError("invalid_verified_probability_pair_count")
    if actual_verified_pairs > source_valid:
        raise ValueError("verified_probability_pairs_exceed_saved_sets")
    # The only independent V25 calibration cohort excludes retrofitted legacy
    # probabilities. The matched audit already enforces source/time integrity.
    status = ("integrity_failure" if problems else
              "awaiting_new_frozen_sets" if not prospective else
              "awaiting_verified_two_leg_results" if not verified else
              "collecting_verified_probabilities")
    return {
        "schema_version": 1,
        "policy": "independent_original_frozen_v25_scores_only",
        "probability_required_from_utc": PROBABILITY_REQUIRED_FROM,
        "status": status,
        "archived_sets_found": matched["archived_sets_found"],
        "source_valid_sets": source_valid,
        "legacy_sets_not_backfilled": len(legacy),
        "prospective_sets_since_rollout": len(prospective),
        "new_sets_with_both_valid_grids_and_mixture": len(ready),
        "new_verified_pairs": len(verified),
        "new_pending_pairs": len(pending),
        "all_eligible_verified_probability_pairs": actual_verified_pairs,
        "initial_100_pair_evaluation": {
            "goal": INITIAL_EVALUATION_TARGET,
            "verified": actual_verified_pairs,
            "remaining": max(0, INITIAL_EVALUATION_TARGET - actual_verified_pairs),
            "goal_reached": actual_verified_pairs >= INITIAL_EVALUATION_TARGET,
            "goal_is_not_an_accuracy_guarantee": True,
        },
        "legacy_rows": legacy,
        "prospective_rows": prospective,
        "integrity_issues": problems,
        "check_passed": not problems,
        "archived_picks_modified": False,
        "old_probability_grids_reconstructed": False,
        "production_model_promoted": False,
        "70_percent_hit_rate_proven": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--require-provenance", action="store_true",
                        help="Fail on absent/invalid new frozen evidence")
    args = parser.parse_args()
    matched = json.loads(args.audit.read_text(encoding="utf-8"))
    report = build_monitor(matched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
        args.output.write_text(content, encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "legacy_sets_not_backfilled", "prospective_sets_since_rollout",
        "new_sets_with_both_valid_grids_and_mixture", "new_verified_pairs",
        "new_pending_pairs", "all_eligible_verified_probability_pairs",
        "check_passed")}, sort_keys=True))
    if args.require_provenance and not report["check_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
