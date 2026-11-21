"""Independent, read-only matched V8/V25 prospective exact-score audit.

Grade ONLY archived V25 shadow sets using their original saved V8 forecast IDs
and independently identity-matched regulation-time finals. The same pair and
the same result grade both policies; post-match reconstructed selections never
enter the denominator. No provider requests, model training or live promotion.

V25's robustness lower envelope is not a normalized probability distribution;
therefore no invented "V25 calibration score" is reported.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from five_double_accuracy import (
    _final, _result_index, _validated_grid, joint_geometry, utc, wilson,
)

ROOT = Path("data/v10_prospective")
PREMATCH = Path("data/prematch")
SETS = Path("data/pro_analyst_shadow/sets")
OUTPUT = Path("data/accuracy_diagnosis/v8_v25_matched.json")
SYDNEY = ZoneInfo("Australia/Sydney")
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
SCORE = re.compile(r"^(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$")
MIN_LEAD = timedelta(minutes=45)
MAX_SOURCE_AGE = timedelta(days=14)
MODEL_VERSION = "pro-analyst-v25-robust-sensitivity-shadow"


def _score(value):
    match = SCORE.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError("invalid_selected_score")
    return int(match.group(1)), int(match.group(2))


def _selection(packet, field):
    entries = packet.get(field)
    if not isinstance(entries, list) or len(entries) != 5:
        raise ValueError("not_five_saved_selections")
    result = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("invalid_selection_entry")
        a, b = _score(item.get("score1")), _score(item.get("score2"))
        result.append((a, b))
    if len(set(result)) != 5:
        raise ValueError("duplicate_saved_selections")
    return result


def _matrix_probability(grid, score):
    h, a = score
    return grid[h][a] if h < len(grid) and a < len(grid[0]) else 0.0


def _selection_mass(picks, first, second):
    return math.fsum(
        _matrix_probability(first, a) * _matrix_probability(second, b)
        for a, b in picks
    )


def _source(root, frozen, selected, day):
    components = [str(frozen.get(k, "")) for k in
                  ("provider", "fixture_id", "forecast_id")]
    if any(not SAFE.fullmatch(x) or x in (".", "..") for x in components):
        raise ValueError("unsafe_source_identity")
    provider, fixture_id, forecast_id = components
    path = Path(root) / "forecasts" / provider / fixture_id / (forecast_id + ".json")
    try:
        original = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("missing_or_unreadable_original_forecast") from None
    if not isinstance(original, dict):
        raise ValueError("source_not_object")
    for key in ("provider", "fixture_id", "forecast_id", "home_team_id",
                "away_team_id", "input_digest", "model_key", "model_version"):
        if str(original.get(key)) != str(frozen.get(key)):
            raise ValueError("archived_source_identity_mismatch")
    if original.get("model_key") not in ("v8", "v8_near"):
        raise ValueError("not_immutable_v8_family")
    if original.get("mode") != "prospective" or original.get("score_basis") != "regulation":
        raise ValueError("not_prospective_regulation_forecast")
    if not original.get("input_digest") or int(original.get("history_matches") or 0) < 45:
        raise ValueError("source_history_or_digest_missing")
    kickoff, capture, evidence = (utc(original[k]) for k in
                                 ("kickoff_utc", "captured_at_utc", "evidence_as_of_utc"))
    if kickoff != utc(frozen["kickoff_utc"]) or capture != utc(frozen["captured_at_utc"]):
        raise ValueError("source_timestamp_mismatch")
    if not (evidence <= capture <= selected <= kickoff - MIN_LEAD):
        raise ValueError("late_or_backdated_source")
    if selected - capture > MAX_SOURCE_AGE:
        raise ValueError("source_older_than_14_days")
    if kickoff.astimezone(SYDNEY).date().isoformat() != day:
        raise ValueError("wrong_sydney_day")
    if original.get("home_team_id") == original.get("away_team_id"):
        raise ValueError("duplicate_teams_in_fixture")
    grid = _validated_grid(original["score_matrix"])
    if min(len(grid), len(grid[0])) < 5 or max(len(grid), len(grid[0])) > 25:
        raise ValueError("v25_ineligible_matrix_shape")
    return original, grid


def _frozen_set(path, root, clock):
    try:
        packet = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("invalid_archived_set_file") from None
    if not isinstance(packet, dict):
        raise ValueError("archived_set_not_object")
    if (packet.get("decision") != "FROZEN_RESEARCH_ONLY"
            or packet.get("model_version") != MODEL_VERSION
            or packet.get("lower_envelope_not_a_calibrated_probability") is not True):
        raise ValueError("not_original_v25_frozen_set")
    day = packet.get("sydney_date")
    if day != path.stem:
        raise ValueError("saved_set_day_or_filename_mismatch")
    selected = utc(packet["selected_at_utc"])
    if selected > clock:
        raise ValueError("selection_after_audit_clock")
    first, left = _source(root, packet["first"], selected, day)
    second, right = _source(root, packet["second"], selected, day)
    if ((first["provider"], first["fixture_id"]) ==
            (second["provider"], second["fixture_id"])):
        raise ValueError("same_fixture_used_twice")
    teams = {(item["provider"], str(item[key])) for item in (first, second)
             for key in ("home_team_id", "away_team_id")}
    if len(teams) != 4:
        raise ValueError("four_distinct_teams_required")
    baseline = _selection(packet, "baseline_five")
    challenger = _selection(packet, "five_doubles")
    optimal = joint_geometry(left, right)["optimal_five"]
    expected = [(_score(item["score1"]), _score(item["score2"]))
                for item in optimal]
    if set(expected) != set(baseline):
        raise ValueError("saved_v8_baseline_not_source_optimal")
    v8_mass = _selection_mass(baseline, left, right)
    v25_using_v8_mass = _selection_mass(challenger, left, right)
    if not math.isclose(v8_mass, float(packet["v8_optimal_five_model_mass"]),
                        abs_tol=1e-8, rel_tol=0):
        raise ValueError("v8_mass_disagrees_with_original_sources")
    if not math.isclose(v25_using_v8_mass,
                        float(packet["robust_five_v8_model_mass"]),
                        abs_tol=1e-8, rel_tol=0):
        raise ValueError("v25_selections_disagree_with_source_v8_mass")
    return packet, (first, second), (left, right), baseline, challenger, (
        v8_mass, v25_using_v8_mass,
    )


def _single_metrics(grid, actual):
    p = _matrix_probability(grid, actual)
    high = max(value for row in grid for value in row)
    top = next((i, j) for i, row in enumerate(grid) for j, value in enumerate(row)
               if value == high)
    hit = top == actual
    return {
        "actual_score": f"{actual[0]}-{actual[1]}",
        "actual_score_probability": p,
        "top_one_exact_score": f"{top[0]}-{top[1]}",
        "top_one_probability": high,
        "top_one_hit": hit,
        "top_one_event_brier": (high - int(hit)) ** 2,
        "full_score_multiclass_brier": (
            1.0 + math.fsum(v * v for row in grid for v in row) - 2 * p
        ),
        "exact_score_log_loss": -math.log(max(p, 1e-15)),
    }


def build_report(root=ROOT, prematch=PREMATCH, sets=SETS, *, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, result_rejections = _result_index(root, prematch)
    exclusions = Counter()
    rows, leg_metrics = [], []
    files = sorted(Path(sets).glob("*.json"))
    for path in files:
        try:
            packet, models, grids, baseline, challenger, masses = _frozen_set(
                path, root, clock
            )
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            exclusions[str(exc) if isinstance(exc, ValueError)
                       else type(exc).__name__] += 1
            continue
        actuals = [_final(model, finals.get(
            (str(model["provider"]), str(model["fixture_id"]))), clock)
                   for model in models]
        row = {
            "sydney_date": packet["sydney_date"],
            "selection_time_utc": utc(packet["selected_at_utc"]).isoformat(),
            "fixture_ids": [str(model["fixture_id"]) for model in models],
            "source_forecast_ids": [str(model["forecast_id"]) for model in models],
            "status": "pending" if any(actual is None for actual in actuals)
                      else "verified",
            "v8_five_model_mass": masses[0],
            "v25_selected_five_mass_under_v8_model": masses[1],
            "v25_lower_envelope_is_not_calibrated_probability": True,
        }
        if row["status"] == "verified":
            v8_hit = any(pick == tuple(actuals) for pick in baseline)
            v25_hit = any(pick == tuple(actuals) for pick in challenger)
            row.update({
                "actual_scores": [f"{a[0]}-{a[1]}" for a in actuals],
                "v8_five_hit": v8_hit, "v25_five_hit": v25_hit,
                "v8_score_probability_diagnostics": [
                    _single_metrics(grid, actual)
                    for grid, actual in zip(grids, actuals)
                ],
            })
            leg_metrics.extend(row["v8_score_probability_diagnostics"])
        rows.append(row)
    finished = [r for r in rows if r["status"] == "verified"]
    n = len(finished)
    b_wins = sum(r["v8_five_hit"] for r in finished)
    c_wins = sum(r["v25_five_hit"] for r in finished)
    challenger_only = sum(r["v25_five_hit"] and not r["v8_five_hit"] for r in finished)
    baseline_only = sum(r["v8_five_hit"] and not r["v25_five_hit"] for r in finished)
    discordant = challenger_only + baseline_only
    pvalue = min(1.0, 2.0 * math.fsum(math.comb(discordant, k)
                   for k in range(min(challenger_only, baseline_only) + 1))
                   / (2 ** discordant)) if discordant else None
    fixture_keys = [(r["fixture_ids"][0], r["fixture_ids"][1]) for r in finished]
    all_ids = [fixture for pair in fixture_keys for fixture in pair]
    mean = lambda key: (math.fsum(x[key] for x in leg_metrics) /
                        len(leg_metrics)) if leg_metrics else None
    return {
        "schema_version": 1,
        "study": "original_v25_robust_selection_vs_original_v8_baseline_same_pairs",
        "evaluated_at_utc": clock.isoformat(),
        "archived_sets_found": len(files),
        "source_valid_sets": len(rows),
        "source_exclusions": dict(exclusions),
        "result_conflicts_excluded": len(conflicts),
        "result_source_rejections": dict(result_rejections),
        "verified_matched_pairs": n,
        "pending_matched_pairs": len(rows) - n,
        "repeated_verified_fixture_ids": len(all_ids) - len(set(all_ids)),
        "v8": {"five_double_hits": b_wins, "five_double_hit_rate": b_wins / n if n else None,
               "wilson_95": wilson(b_wins, n)},
        "v25": {"five_double_hits": c_wins, "five_double_hit_rate": c_wins / n if n else None,
                "wilson_95": wilson(c_wins, n)},
        "same_pair_discordance": {
            "v25_only_hits": challenger_only, "v8_only_hits": baseline_only,
            "exact_two_sided_mcnemar_p": pvalue,
        },
        "v8_exact_score_probability_calibration": {
            "verified_unique_legs": len(set(all_ids)),
            "evaluated_leg_slots": len(leg_metrics),
            "top_one_expected_hits": math.fsum(x["top_one_probability"]
                                               for x in leg_metrics),
            "top_one_observed_hits": sum(x["top_one_hit"] for x in leg_metrics),
            "top_one_brier": mean("top_one_event_brier"),
            "multiclass_score_brier": mean("full_score_multiclass_brier"),
            "exact_score_log_loss": mean("exact_score_log_loss"),
        },
        "v25_probability_calibration": {
            "status": "not_measurable_no_separately_frozen_normalized_v25_grid",
            "top_one_brier": None, "multiclass_score_brier": None,
            "exact_score_log_loss": None,
        },
        "rows": rows,
        "no_provider_calls": True,
        "archived_picks_modified": False,
        "production_model_promoted": False,
        "seventy_percent_accuracy_proven": False,
        "limitations": [
            "The comparison grades two SAVED policies on identical pairs, not automatically selected V8 daily pairs.",
            "V25 lower-envelope scores are not probabilities and cannot be used for log-loss or Brier.",
            "V25 draws its sensitivity models from the same V8 underlying goal rates; evidence is not independent.",
            "V8 mass assigned to V25 selections is a reference only, not calibrated V25 hit probability.",
            "Future full V25 probability calibration requires separately timestamped normalized score matrices.",
            "Archived JSON capture timestamps are checked but are not independent Git commit-time notarization.",
            "Small and overlapping samples cannot establish superiority, reliable accuracy or 70% hit rate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prematch", type=Path, default=PREMATCH)
    parser.add_argument("--sets", type=Path, default=SETS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build_report(args.root, args.prematch, args.sets)
    if args.output.exists():
        try:
            previous = json.loads(args.output.read_text(encoding="utf-8"))
            without_clock = lambda row: {k: v for k, v in row.items()
                                         if k != "evaluated_at_utc"}
            if without_clock(previous) == without_clock(report):
                print("Matched V8/V25 audit unchanged")
                return
        except (ValueError, OSError, TypeError):
            pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"matched_verified": report["verified_matched_pairs"],
                      "pending": report["pending_matched_pairs"],
                      "v8_hits": report["v8"]["five_double_hits"],
                      "v25_hits": report["v25"]["five_double_hits"],
                      "production_promoted": False}, sort_keys=True))


if __name__ == "__main__":
    main()
