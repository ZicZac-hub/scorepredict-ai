"""Read-only, source-matched exact-score calibration audit.

Each fixture is evaluated once per pre-declared V8 family using its latest
eligible archived forecast at least 30 minutes before kickoff. Results are
identity-checked regulation-time finals. No provider calls, model fitting,
post-match revision of forecasts, or production probability changes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re

from five_double_accuracy import _final, _result_index, _validated_grid, utc

ROOT = Path("data/v10_prospective")
PREMATCH = Path("data/prematch")
OUTPUT = Path("data/accuracy_diagnosis/scoreline_calibration.json")
MODEL_KEYS = ("v8", "v8_near")
MIN_LEAD = timedelta(minutes=30)
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _events(grid, actual):
    h, a = actual
    predicates = {
        "exact_3_0": lambda x, y: x == 3 and y == 0,
        "exact_0_3": lambda x, y: x == 0 and y == 3,
        "at_least_four_total_goals": lambda x, y: x + y >= 4,
        "winning_margin_at_least_three": lambda x, y: abs(x - y) >= 3,
    }
    output = {}
    for name, predicate in predicates.items():
        probability = math.fsum(
            value for x, row in enumerate(grid)
            for y, value in enumerate(row) if predicate(x, y)
        )
        observed = int(predicate(h, a))
        output[name] = {"predicted_probability": probability, "observed": observed}
    actual_p = grid[h][a] if h < len(grid) and a < len(grid[0]) else 0.0
    output["actual_exact_score_probability"] = actual_p
    output["actual_exact_score_rank_min"] = 1 + sum(
        value > actual_p for row in grid for value in row
    )
    return output


def _candidate(record, path, result, now):
    try:
        if (record.get("model_key") not in MODEL_KEYS
                or record.get("mode") != "prospective"
                or record.get("score_basis") != "regulation"
                or record.get("forecast_id") != path.stem
                or not record.get("input_digest")):
            return None
        captured = utc(record["captured_at_utc"])
        evidence = utc(record["evidence_as_of_utc"])
        kickoff = utc(record["kickoff_utc"])
        if not (evidence <= captured <= kickoff - MIN_LEAD and captured <= now):
            return None
        actual = _final(record, result, now)
        if actual is None:
            return None
        matrix = _validated_grid(record["score_matrix"])
        return (captured, record, matrix, actual)
    except (ValueError, TypeError, KeyError, OSError, OverflowError):
        return None


def build_report(root=ROOT, prematch=PREMATCH, *, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, rejected = _result_index(root, prematch)
    chosen = {}
    source_rejections = Counter()
    for (provider, fixture_id), result in sorted(finals.items()):
        if (not SAFE.fullmatch(provider) or not SAFE.fullmatch(fixture_id)
                or provider in (".", "..") or fixture_id in (".", "..")):
            source_rejections["unsafe_result_identity"] += 1
            continue
        folder = Path(root) / "forecasts" / provider / fixture_id
        for path in folder.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    source_rejections["not_a_forecast_object"] += 1
                    continue
                candidate = _candidate(record, path, result, clock)
                if candidate is None:
                    continue
                key = (provider, fixture_id, record["model_key"])
                if key not in chosen or candidate[0] > chosen[key][0]:
                    chosen[key] = candidate
            except (OSError, ValueError, TypeError):
                source_rejections["unreadable_forecast"] += 1
                continue

    details = {key: [] for key in MODEL_KEYS}
    for (provider, fixture_id, family), (captured, record, grid, actual) in sorted(chosen.items()):
        details[family].append({
            "provider": provider,
            "fixture_id": fixture_id,
            "fixture_name": record.get("fixture_name"),
            "forecast_id": record["forecast_id"],
            "captured_at_utc": captured.isoformat(),
            "kickoff_utc": utc(record["kickoff_utc"]).isoformat(),
            "actual_score": str(actual[0]) + "-" + str(actual[1]),
            "metrics": _events(grid, actual),
        })

    aggregates = {}
    for family in MODEL_KEYS:
        items = details[family]
        summary = {}
        for event in ("exact_3_0", "exact_0_3", "at_least_four_total_goals",
                      "winning_margin_at_least_three"):
            expected = math.fsum(item["metrics"][event]["predicted_probability"] for item in items)
            observed = sum(item["metrics"][event]["observed"] for item in items)
            brier = math.fsum(
                (item["metrics"][event]["predicted_probability"]
                 - item["metrics"][event]["observed"]) ** 2 for item in items
            )
            count = len(items)
            summary[event] = {
                "expected_events": expected,
                "observed_events": observed,
                "predicted_rate": expected / count if count else None,
                "observed_rate": observed / count if count else None,
                "rate_gap_observed_minus_predicted": (observed - expected) / count if count else None,
                "brier": brier / count if count else None,
            }
        aggregates[family] = {"verified_fixtures": len(items), "events": summary}

    return {
        "schema_version": 1,
        "study": "read_only_scoreline_calibration_v8_family",
        "result_cutoff_utc": clock.isoformat(),
        "model_selection_rule": "latest_identity_matched_forecast_per_fixture_and_model_family_at_least_30_minutes_before_kickoff",
        "model_keys_separate": list(MODEL_KEYS),
        "result_conflicts_excluded": len(conflicts),
        "result_source_rejections": dict(rejected),
        "source_rejections": dict(source_rejections),
        "aggregates": aggregates,
        "focus": {
            fixture_id: [row for family in MODEL_KEYS for row in details[family]
                         if row["fixture_id"] == fixture_id]
            for fixture_id in ("1550200", "1550202")
        },
        "observations": details,
        "no_provider_calls": True,
        "forecasts_unchanged": True,
        "production_model_promoted": False,
        "improvement_claimed": False,
        "limitations": [
            "The 3-0 score contains three total goals, so a four-plus-goal calibration check alone does not evaluate it.",
            "One missed five-double cannot identify model bias or justify increasing its probability.",
            "V8 and V8-near may contain the same fixtures; their totals are separate and not independent.",
            "This audit does not grade a screenshot-created manual pair as a frozen archived prospective selection.",
            "Estimated probability and independently observed hit rates must remain separate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prematch", type=Path, default=PREMATCH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build_report(args.root, args.prematch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": {key: value["verified_fixtures"]
                                  for key, value in report["aggregates"].items()},
                      "conflicts_excluded": report["result_conflicts_excluded"],
                      "production_promoted": False}, sort_keys=True))


if __name__ == "__main__":
    main()
