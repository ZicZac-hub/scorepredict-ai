"""V34: read-only diagnosis of verified five-double score-selection failures.

Only original V28 pre-kickoff frozen candidate catalogs and V30 independently
verified regulation-time final scores may enter. Catalog pairs overlap; never
interpret their rows as independent bets or use hindsight-selected tickets.

The original V8 probability grid can be assessed once per unique physical
fixture. V25 sensitivity variants are explicitly POST-HOC mathematical replays
of that V8 grid, not genuine frozen V25 predictions or promotion evidence.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from five_double_accuracy import utc
from v25_score_accuracy_research import SENSITIVITIES, replay_sensitivities, summarize_scores
from v28_independent_pair_trial import ORIGINAL_ROOT, PREMATCH, OUT as V28_OUT, _key, verify_decision
from v30_full_slate_audit import analyze as verified_v30

VERSION = "v34-deduplicated-score-error-research-v1"
OUT = Path("data/v34_score_error_research/status.json")
SCORE = re.compile(r"^(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$")
POLICIES = {
    "v25_robust_original": ("five_original_v25_robust_scores", "robust_five_hit"),
    "v8_optimal_original": ("five_original_v8_optimal_scores", "v8_baseline_five_hit"),
}
CLASSES = (
    "hit",
    "one_leg_score_absent",
    "both_leg_scores_absent",
    "scores_present_but_wrong_joint_ticket",
)
MIN_COMPLETE_DAYS = 30


def score(value):
    match = SCORE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("invalid_independently_verified_score")
    return int(match[1]), int(match[2])


def pair_key(row):
    keys = row["fixture_keys"]
    if len(keys) != 2:
        raise ValueError("invalid_original_two_leg_pair")
    out = tuple(tuple(str(x) for x in item) for item in keys)
    if any(len(item) != 2 for item in out) or out[0] == out[1]:
        raise ValueError("duplicate_or_invalid_pair_fixture")
    return out


def classify(picks, actual):
    """Classify WHY the original five frozen joint-score tickets did not hit."""
    if len(picks) != 5 or len(set(picks)) != 5:
        raise ValueError("invalid_frozen_five_distinct_tickets")
    if actual in picks:
        return "hit"
    a = any(x[0] == actual[0] for x in picks)
    b = any(x[1] == actual[1] for x in picks)
    if a and b:
        return "scores_present_but_wrong_joint_ticket"
    if a or b:
        return "one_leg_score_absent"
    return "both_leg_scores_absent"


def tickets(frozen, field):
    raw = frozen[field]
    if not isinstance(raw, list) or len(raw) != 5:
        raise ValueError("missing_frozen_ticket_set")
    return [(score(item["score1"]), score(item["score2"])) for item in raw]


def _counts(items):
    return {label: int(items[label]) for label in CLASSES}


def _goal_group(goals):
    return "low_0_to_1" if goals <= 1 else "middle_2_to_3" if goals <= 3 else "high_4_plus"


def summarize(v30, catalogs, sources):
    """Sources are independently reverified original V28 receipt/grids per day."""
    if not v30.get("overlapping_pairs_are_not_independent_samples"):
        raise ValueError("missing_original_overlap_guardrail")
    observed_fixtures = {}
    per_day = []
    counts = {policy: Counter() for policy in POLICIES}
    global_pairs = 0
    duplicated_fixture_slots = 0
    for day in sorted(v30["days"], key=lambda row: row["sydney_day"]):
        if day["original_policy"] != "FROZEN":
            continue
        date = day["sydney_day"]
        catalog = catalogs[date]
        grid_map = sources[date]
        if (len(catalog) != day["candidate_count"] or
                day["candidate_pairs_verified"] != len(day["candidate_rows"]) or
                day["candidate_pairs_pending"] + len(day["candidate_rows"]) != len(catalog)):
            raise ValueError("v30_verified_counts_disagree_with_frozen_catalog")
        lookup = {pair_key(x): x for x in catalog}
        if len(lookup) != len(catalog):
            raise ValueError("duplicated_frozen_catalog_pair")
        day_counts = {policy: Counter() for policy in POLICIES}
        results_this_day = {}
        for row in day["candidate_rows"]:
            key = pair_key(row)
            if key not in lookup:
                raise ValueError("verified_pair_not_in_original_catalog")
            actual = tuple(score(x) for x in row["actual_scores"])
            if len(actual) != 2:
                raise ValueError("invalid_verified_pair_score_count")
            for fixture, result in zip(key, actual):
                previous = results_this_day.setdefault(fixture, result)
                if previous != result:
                    raise ValueError("conflicting_verified_scores_within_day")
            for policy, (field, expected_hit) in POLICIES.items():
                outcome = classify(tickets(lookup[key], field), actual)
                if (outcome == "hit") is not row[expected_hit]:
                    raise ValueError("original_frozen_ticket_grade_disagrees_with_v30")
                day_counts[policy][outcome] += 1
                counts[policy][outcome] += 1
            global_pairs += 1
        for fixture, actual in sorted(results_this_day.items()):
            if fixture not in grid_map:
                raise ValueError("verified_original_score_grid_missing")
            if fixture in observed_fixtures:
                if observed_fixtures[fixture]["actual"] != actual:
                    raise ValueError("contradictory_final_across_frozen_days")
                duplicated_fixture_slots += 1
                continue
            observed_fixtures[fixture] = {
                "actual": actual, "day": date,
                "original_grid": grid_map[fixture],
            }
        per_day.append({
            "sydney_day": date,
            "original_candidate_pairs": len(catalog),
            "verified_pair_rows_correlated": len(day["candidate_rows"]),
            "pending_pair_rows": day["candidate_pairs_pending"],
            "complete_original_slate": bool(day["complete_slate"]),
            "distinct_verified_physical_fixtures": len(results_this_day),
            "pair_failure_categories": {
                policy: _counts(group) for policy, group in day_counts.items()
            },
        })
    model_rows = {name: [] for name in SENSITIVITIES}
    goal_groups = Counter()
    for item in observed_fixtures.values():
        actual = item["actual"]
        goal_groups[_goal_group(sum(actual))] += 1
        metrics = replay_sensitivities(item["original_grid"], actual)
        for name in SENSITIVITIES:
            model_rows[name].append(metrics[name])
    summaries = {name: summarize_scores(rows) for name, rows in model_rows.items()}
    baseline = summaries["baseline_v8"]
    differences = {}
    for name in SENSITIVITIES:
        if name == "baseline_v8":
            continue
        candidate = summaries[name]
        differences[name] = {
            "retrospective_minus_original_v8_mean_score_log_loss": (
                candidate["mean_exact_score_log_loss"] - baseline["mean_exact_score_log_loss"]
                if model_rows[name] else None
            ),
            "retrospective_minus_original_v8_mean_multiclass_brier": (
                candidate["mean_multiclass_score_brier"] - baseline["mean_multiclass_score_brier"]
                if model_rows[name] else None
            ),
            "NOT_an_independently_frozen_challenger_comparison": True,
        }
    return {
        "version": VERSION,
        "source": "V30 independently checked V28 catalog, original immutable V8 score grids and verified finals",
        "verified_overlapping_candidate_pair_rows": global_pairs,
        "candidate_pairs_are_not_independent_trials": True,
        "original_frozen_slate_days": len(per_day),
        "fully_verified_slate_days": sum(x["complete_original_slate"] for x in per_day),
        "initial_complete_day_review_target": MIN_COMPLETE_DAYS,
        "initial_review_ready": sum(x["complete_original_slate"] for x in per_day) >= MIN_COMPLETE_DAYS,
        "unique_verified_physical_fixtures": len(observed_fixtures),
        "repeated_physical_fixture_slots_excluded": duplicated_fixture_slots,
        "original_v8_verified_goal_bands": dict(goal_groups),
        "frozen_ticket_failure_attribution": {p: _counts(c) for p, c in counts.items()},
        "days": per_day,
        "original_v8_unique_fixture_score_diagnostics": baseline,
        "POST_HOC_sensitivity_replays_only": {name: summaries[name] for name in SENSITIVITIES if name != "baseline_v8"},
        "POST_HOC_sensitivity_deltas": differences,
        "score_models_have_identical_verified_fixture_denominator": True,
        "probability_replays_are_not_real_v25_frozen_forecasts": True,
        "no_historical_five_ticket_reselection": True,
        "no_live_model_changes": True,
        "seventy_percent_five_double_hit_rate_unproven": True,
        "original_source_exclusions": v30["invalid_or_tampered_frozen_days_excluded"],
        "verified_result_conflicts_excluded": v30["verified_result_conflicts_excluded"],
    }


def run(*, root=ORIGINAL_ROOT, prematch=PREMATCH, v28_out=V28_OUT, out=OUT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    v30 = verified_v30(root=root, prematch=prematch, v28_out=v28_out, now=clock)
    catalogs, sources = {}, {}
    for day in v30["days"]:
        if day["original_policy"] != "FROZEN":
            continue
        date = day["sydney_day"]
        original, items = verify_decision(
            Path(v28_out) / "decisions" / (date + ".json"),
            root=root, clock=clock,
        )
        if original["candidate_catalog_sha256"] != day["original_candidate_catalog_sha256"]:
            raise ValueError("V28_original_catalog_digest_mismatch")
        catalogs[date] = original["candidate_catalog"]
        sources[date] = {_key(item): item["grid"] for item in items}
    report = summarize(v30, catalogs, sources)
    report["evaluated_at_utc"] = clock.isoformat()
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if not output.exists() or output.read_text(encoding="utf-8") != serialized:
        output.write_text(serialized, encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "unique_physical_fixtures": report["unique_verified_physical_fixtures"],
        "correlated_pair_rows": report["verified_overlapping_candidate_pair_rows"],
        "failure_categories": report["frozen_ticket_failure_attribution"],
        "live_model_changed": False,
    }, sort_keys=True))
