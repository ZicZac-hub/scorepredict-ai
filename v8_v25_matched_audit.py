"""Independent, read-only matched V8/V25 prospective exact-score audit.

Grade ONLY archived V25 shadow sets using their original saved V8 forecast IDs
and independently identity-matched regulation-time finals. The same pair and
the same result grade both policies; post-match reconstructed selections never
enter the denominator. No provider requests, model training or live promotion.

Compare probabilities only where a separate V25 distribution was frozen.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
from v21_goal_model_comparison import variants

from five_double_accuracy import (
    _final, _validated_grid, joint_geometry, utc, wilson,
)

from professional_analyst_shadow_v25 import MIXTURE_FIVE_POLICY, PROBABILITY_VERSION
from verified_result_index import index_verified_results
from v25_score_accuracy_research import (
    SENSITIVITIES, describe_score, replay_sensitivities, summarize_scores,
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


def _v25_probability(frozen, source, selected):
    if "v25_probability" not in frozen:
        return None
    record = frozen["v25_probability"]
    if not isinstance(record, dict) or record.get("version") != PROBABILITY_VERSION:
        raise ValueError("unknown_v25_probability_version")
    if (utc(record["frozen_at_utc"]) != selected or
            utc(record["source_evidence_as_of_utc"]) != utc(source["evidence_as_of_utc"])):
        raise ValueError("v25_probability_timestamp_mismatch")
    grid = _validated_grid(record["score_matrix"])
    if (len(grid), len(grid[0])) != (len(source["score_matrix"]), len(source["score_matrix"][0])):
        raise ValueError("v25_probability_support_mismatch")
    # Independently reproduce the preregistered fixed mixture from the
    # ORIGINAL archived V8 matrix. This only checks a previously frozen
    # distribution; it must never fill in missing historical V25 snapshots.
    baseline = np.asarray(source["score_matrix"], dtype=float)
    alternatives = variants(baseline)
    expected = np.zeros_like(baseline)
    for name in ("poisson_marginal", "negative_binomial_r3", "bivariate_poisson_c012"):
        component = alternatives[name]
        expected[:component.shape[0], :component.shape[1]] += component / 3
    expected /= expected.sum()
    if not np.allclose(np.asarray(grid, dtype=float), expected, rtol=0, atol=1e-11):
        raise ValueError("v25_probability_not_reproducible_from_frozen_source")
    return grid


def _mixture_selection(packet, grids):
    if packet.get("mixture_five_policy") != MIXTURE_FIVE_POLICY:
        raise ValueError("unknown_v25_mixture_five_policy")
    saved = _selection(packet, "mixture_five")
    optimal = joint_geometry(*grids)["optimal_five"]
    expected = [(_score(item["score1"]), _score(item["score2"])) for item in optimal]
    if saved != expected:
        raise ValueError("saved_mixture_five_not_source_optimal")
    for saved_item, optimal_item in zip(packet["mixture_five"], optimal):
        if not math.isclose(saved_item["joint_probability"], optimal_item["joint_probability"],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("saved_mixture_five_mass_mismatch")
    return saved


def _set_performance(rows, hit_field):
    wins = sum(row[hit_field] for row in rows)
    return {
        "five_double_hits": wins,
        "five_double_hit_rate": wins / len(rows) if rows else None,
        "wilson_95": wilson(wins, len(rows)),
    }


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


def _calibration(metrics):
    def mean(key):
        return math.fsum(row[key] for row in metrics) / len(metrics) if metrics else None

    return {
        "evaluated_leg_slots": len(metrics),
        "top_one_expected_hits": math.fsum(row["top_one_probability"] for row in metrics),
        "top_one_observed_hits": sum(row["top_one_hit"] for row in metrics),
        "top_one_brier": mean("top_one_event_brier"),
        "multiclass_score_brier": mean("full_score_multiclass_brier"),
        "exact_score_log_loss": mean("exact_score_log_loss"),
    }


def build_report(root=ROOT, prematch=PREMATCH, sets=SETS, *, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, result_rejections = index_verified_results(root, prematch, clock)
    exclusions = Counter()
    rows, leg_metrics = [], []
    matched_v8, matched_v25 = [], []
    shape_v8, shape_v25, shape_matched_v8 = [], [], []
    shape_replay = {name: [] for name in SENSITIVITIES}
    replay_excluded = 0
    probability_coverage = Counter()
    mixture_coverage = Counter()
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
        try:
            v25_grids = [
                _v25_probability(packet[leg], source, utc(packet["selected_at_utc"]))
                for leg, source in zip(("first", "second"), models)
            ]
            if all(grid is not None for grid in v25_grids):
                probability_status = "frozen"
            elif all(grid is None for grid in v25_grids):
                probability_status = "unavailable"
            else:
                probability_status = "incomplete"
        except (ValueError, TypeError, KeyError, OverflowError):
            probability_status = "invalid"
        row["v25_probability_status"] = probability_status
        probability_coverage[probability_status] += 1
        mixture_picks = None
        mixture_status = "unavailable"
        if "mixture_five" in packet or "mixture_five_policy" in packet:
            mixture_status = "invalid"
            if probability_status == "frozen":
                try:
                    mixture_picks = _mixture_selection(packet, v25_grids)
                except (ValueError, TypeError, KeyError, OverflowError):
                    pass
                else:
                    mixture_status = "frozen"
                    row["v25_mixture_five_model_mass"] = _selection_mass(mixture_picks, *v25_grids)
                    row["v25_robust_five_mass_under_mixture"] = _selection_mass(challenger, *v25_grids)
        row["v25_mixture_five_status"] = mixture_status
        mixture_coverage[mixture_status] += 1
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
            for original_grid, actual in zip(grids, actuals):
                shape_v8.append(describe_score(original_grid, actual))
                try:
                    candidate_scores = replay_sensitivities(original_grid, actual)
                except (ValueError, TypeError, OverflowError):
                    # Replay is exploratory, and its absence must never remove
                    # a previously verified five-double result or V8 metric.
                    replay_excluded += 1
                else:
                    for name in SENSITIVITIES:
                        shape_replay[name].append(candidate_scores[name])
            if mixture_picks is not None:
                row["v25_mixture_five_hit"] = tuple(actuals) in mixture_picks
            if probability_status == "frozen":
                row["v25_score_probability_diagnostics"] = [
                    _single_metrics(grid, actual)
                    for grid, actual in zip(v25_grids, actuals)
                ]
                matched_v8.extend(row["v8_score_probability_diagnostics"])
                matched_v25.extend(row["v25_score_probability_diagnostics"])
                shape_matched_v8.extend(
                    describe_score(grid, actual)
                    for grid, actual in zip(grids, actuals)
                )
                shape_v25.extend(
                    describe_score(grid, actual)
                    for grid, actual in zip(v25_grids, actuals)
                )
        rows.append(row)
    finished = [r for r in rows if r["status"] == "verified"]
    n = len(finished)
    mixture_finished = [r for r in finished if "v25_mixture_five_hit" in r]
    challenger_only = sum(r["v25_five_hit"] and not r["v8_five_hit"] for r in finished)
    baseline_only = sum(r["v8_five_hit"] and not r["v25_five_hit"] for r in finished)
    discordant = challenger_only + baseline_only
    pvalue = min(1.0, 2.0 * math.fsum(math.comb(discordant, k)
                   for k in range(min(challenger_only, baseline_only) + 1))
                   / (2 ** discordant)) if discordant else None
    fixture_keys = [(r["fixture_ids"][0], r["fixture_ids"][1]) for r in finished]
    all_ids = [fixture for pair in fixture_keys for fixture in pair]
    return {
        "schema_version": 3,
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
        "v8": _set_performance(finished, "v8_five_hit"),
        "v25": _set_performance(finished, "v25_five_hit"),
        "matched_mixture_five": {
            "policy": MIXTURE_FIVE_POLICY,
            "verified_pairs": len(mixture_finished),
            "pending_pairs": mixture_coverage["frozen"] - len(mixture_finished),
            "selection_coverage_by_pair": dict(mixture_coverage),
            "v8": _set_performance(mixture_finished, "v8_five_hit"),
            "v25_robust": _set_performance(mixture_finished, "v25_five_hit"),
            "v25_mixture": _set_performance(mixture_finished, "v25_mixture_five_hit"),
        },
        "same_pair_discordance": {
            "v25_only_hits": challenger_only, "v8_only_hits": baseline_only,
            "exact_two_sided_mcnemar_p": pvalue,
        },
        "v8_exact_score_probability_calibration": {
            "verified_unique_legs": len(set(all_ids)),
            **_calibration(leg_metrics),
        },
        "v25_probability_calibration": {
            "status": "evaluated" if matched_v25 else "no_verified_pairs_with_frozen_v25_grids",
            "version": PROBABILITY_VERSION,
            **_calibration(matched_v25),
        },
        "matched_probability_calibration": {
            "verified_pairs": len(matched_v25) // 2,
            "pending_pairs": probability_coverage["frozen"] - len(matched_v25) // 2,
            "v25_grid_coverage_by_pair": dict(probability_coverage),
            "v8_on_same_legs": _calibration(matched_v8),
        },
        "v25_score_accuracy_research": {
            "schema_version": 1,
            "same_original_v8_verified_legs": summarize_scores(shape_v8),
            "retrospective_same_source_sensitivity_replays": {
                name: summarize_scores(shape_replay[name])
                for name in SENSITIVITIES
            },
            "retrospective_replay_excluded_legs": replay_excluded,
            "prospective_same_leg_probability_comparison": {
                "verified_leg_slots": len(shape_v25),
                "v8_on_exact_same_legs": summarize_scores(shape_matched_v8),
                "v25_frozen_on_exact_same_legs": summarize_scores(shape_v25),
                "v25_minus_v8_exact_score_log_loss": (
                    summarize_scores(shape_v25)["mean_exact_score_log_loss"]
                    - summarize_scores(shape_matched_v8)["mean_exact_score_log_loss"]
                    if shape_v25 else None
                ),
                "v25_minus_v8_multiclass_score_brier": (
                    summarize_scores(shape_v25)["mean_multiclass_score_brier"]
                    - summarize_scores(shape_matched_v8)["mean_multiclass_score_brier"]
                    if shape_v25 else None
                ),
                "negative_delta_is_better": True,
            },
            "high_score_event_definition": "at least 4 regulation-time total goals",
            "low_score_event_definition": "0 or 1 regulation-time total goals",
            "retrospective_replay_is_not_a_frozen_v25_forecast": True,
            "training_or_weight_optimization_on_verified_results": False,
            "production_model_promoted": False,
            "real_accuracy_improvement_established": False,
        },
        "rows": rows,
        "no_provider_calls": True,
        "archived_picks_modified": False,
        "production_model_promoted": False,
        "seventy_percent_accuracy_proven": False,
        "limitations": [
            "All saved policies are graded on the original V25 fixture pairs.",
            "V25 lower-envelope scores are not probabilities and cannot be used for log-loss or Brier.",
            "V25 draws its sensitivity models from the same V8 underlying goal rates; evidence is not independent.",
            "V8 mass assigned to V25 selections is a reference only, not calibrated V25 hit probability.",
            "Probability comparison uses only verified pairs with both frozen V25 grids, without reconstructing historical probabilities.",
            "Archived JSON capture timestamps are checked but are not independent Git commit-time notarization.",
            "Small and overlapping samples cannot establish superiority, reliable accuracy or 70% hit rate.",
            "Sensitivity replays are post-hoc transformations of archived V8 matrices, not frozen V25 predictions.",
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
