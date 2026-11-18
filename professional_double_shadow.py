"""No-API prospective bootstrap for the exact professional auto policy.

The raw-probability policy is frozen separately from premium recommendations,
which still require calibration evidence. Original V8 forecasts and all prior
five-double sets stay unchanged. Hits are recomputed from identity-matched,
verified regulation-time results, never trusted from a saved report.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_five_double_ui import archive_rows
from five_double_accuracy import _final, _result_index, _score, _sources, wilson
from professional_double_optimizer import POLICY, AUTO_SELECTION, auto_shadow, optimal_five
from source_capture_clock import capture_clock
from v10_prospective import ROOT as ORIGINAL_ROOT, utc

ROOT = Path("data/professional_double_shadow")
PREMATCH_ROOT = Path("data/prematch")
SYDNEY = ZoneInfo("Australia/Sydney")
SCHEMA = 1


def _digest(record):
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _names(model):
    return tuple(" ".join(str(model[k]).casefold().split())
                 for k in ("home_team", "away_team"))


def _validated(record, *, source_root, now):
    """Authenticate policy, source IDs/digests and all five probability cells."""
    if (not isinstance(record, dict) or record.get("schema_version") != SCHEMA
            or record.get("policy_id") != POLICY
            or record.get("selection_scope") != AUTO_SELECTION
            or record.get("prospective") is not True
            or record.get("status") != "frozen_pre_kickoff"):
        raise ValueError("not_exact_professional_auto_policy")
    if record.get("record_sha256") != _digest(record):
        raise ValueError("frozen_record_digest_mismatch")
    predicted = utc(record["predicted_at_utc"])
    if predicted > utc(now):
        raise ValueError("record_not_yet_captured")
    report = record["selection"]
    if (report.get("status") != "ready" or report.get("policy_id") != POLICY
            or report.get("selection_scope") != AUTO_SELECTION):
        raise ValueError("selection_policy_mismatch")
    refs = report["source_forecast_refs"]
    if not isinstance(refs, dict) or set(refs) != {"first", "second"}:
        raise ValueError("two_exact_source_forecast_refs_required")
    models = _sources(source_root, refs, predicted)
    if any(not refs[side].get("input_digest") or
           refs[side]["input_digest"] != model["input_digest"]
           for side, model in zip(("first", "second"), models)):
        raise ValueError("source_input_digest_mismatch")
    kickoffs = [utc(model["kickoff_utc"]) for model in models]
    if any((kick-predicted).total_seconds() <= 300 for kick in kickoffs):
        raise ValueError("not_frozen_in_eligible_pre_match_window")
    captured = [utc(model["captured_at_utc"]) for model in models]
    if any(not timedelta(0) <= predicted-at <= timedelta(hours=24) for at in captured):
        raise ValueError("original_source_not_within_24_hours")
    day = record["sydney_date"]
    if (report["sydney_date"] != day or
            {kick.astimezone(SYDNEY).date().isoformat() for kick in kickoffs} != {day}):
        raise ValueError("same_sydney_date_required")
    identifiers = [(model["provider"], str(model[k])) for model in models
                   for k in ("home_team_id", "away_team_id")]
    names = [name for model in models for name in _names(model)]
    if len(set(identifiers)) != 4 or not all(names) or len(set(names)) != 4:
        raise ValueError("four_distinct_known_teams_required")
    for side, model in zip(("first", "second"), models):
        fixture = report[side+"_fixture"]
        if (str(fixture.get("provider") or "football-data.org") != model["provider"]
                or str(fixture["id"]) != str(model["fixture_id"])
                or utc(fixture["utcDate"]) != utc(model["kickoff_utc"])):
            raise ValueError("display_fixture_source_mismatch")
        displayed = []
        for key in ("home", "away"):
            team = fixture[key+"Team"]
            if str(team["id"]) != str(model[key+"_team_id"]):
                raise ValueError("display_team_source_mismatch")
            displayed.append(" ".join(str(team["name"]).casefold().split()))
        if tuple(displayed) != _names(model):
            raise ValueError("display_team_names_source_mismatch")
    versions = [model["model_version"] for model in models]
    if list(report["model_versions"]) != versions:
        raise ValueError("model_version_source_mismatch")
    if [utc(value) for value in report["source_times_utc"]] != captured:
        raise ValueError("source_capture_times_mismatch")
    generated = utc(report["generated_at_utc"])
    if not max(captured) <= generated <= predicted:
        raise ValueError("selection_generation_time_mismatch")
    expected = optimal_five(*(model["score_matrix"] for model in models))
    choices = report["choices"]
    if (not isinstance(choices, list) or len(choices) != 5
            or len({(p["score1"], p["score2"]) for p in choices}) != 5):
        raise ValueError("five_distinct_supported_choices_required")
    if [(p["score1"], p["score2"]) for p in choices] != [
            (p["score1"], p["score2"]) for p in expected]:
        raise ValueError("choices_not_exact_full_grid_probability_optimum")
    for selected, actual in zip(choices, expected):
        _score(selected["score1"])
        _score(selected["score2"])
        for key in ("joint_probability", "first_probability", "second_probability"):
            if key not in selected and key != "joint_probability":
                continue
            probability = selected[key]
            if (type(probability) not in (float, int) or not math.isfinite(probability)
                    or not math.isclose(probability, actual[key], rel_tol=0, abs_tol=1e-12)):
                raise ValueError("choice_probability_disagrees_with_original_grid")
    coverage = math.fsum(p["joint_probability"] for p in expected)
    if (type(report["estimated_coverage"]) not in (float, int)
            or not math.isfinite(report["estimated_coverage"])
            or not math.isclose(report["estimated_coverage"], coverage, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("set_coverage_disagrees_with_original_grids")
    return models, expected, coverage


def freeze(*, root=ROOT, source_root=ORIGINAL_ROOT, now=None, rows=None, clock=None):
    """Write at most one raw-policy shadow per future Sydney date, once only."""
    started, observed = capture_clock(now, clock)
    candidates = archive_rows(now=started, root=source_root) if rows is None else rows
    groups = defaultdict(list)
    rejected = Counter()
    for row in candidates:
        try:
            fixture = row["fixture"]
            kickoff = utc(row.get("kickoff") or fixture["utcDate"])
            groups[kickoff.astimezone(SYDNEY).date().isoformat()].append(row)
        except (ValueError, TypeError, KeyError):
            rejected["invalid_candidate_date"] += 1
    new, preserved = [], []
    for day, group in sorted(groups.items()):
        path = Path(root)/"sets"/(day+".json")
        if path.exists():
            preserved.append(day)
            continue
        result = auto_shadow(group, now=observed(),clock=observed)
        if result.get("status") != "ready":
            rejected.update(result.get("rejections") or {"no_source_qualified_pair": 1})
            continue
        completed = observed()
        record = {"schema_version": SCHEMA, "policy_id": POLICY,
                  "selection_scope": AUTO_SELECTION, "prospective": True,
                  "status": "frozen_pre_kickoff", "sydney_date": day,
                  "predicted_at_utc": completed.isoformat(),
                  "selection": result["selection"],
                  "calibration_bootstrap_only_not_premium_recommendation": True,
                  "target_70_percent_proven": False}
        try:
            record["record_sha256"] = _digest(record)
            _validated(record, source_root=source_root, now=completed)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+"\n")
            new.append(day)
        except FileExistsError:
            preserved.append(day)
        except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
            rejected[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return {"started_at_utc": started.isoformat(), "completed_at_utc": observed().isoformat(),
            "new_sets": new, "existing_sets_preserved": preserved,
            "candidate_dates": len(groups), "rejections": dict(rejected),
            "external_requests": 0, "target_70_percent_proven": False}


def evaluate(*, root=ROOT, source_root=ORIGINAL_ROOT, prematch_root=PREMATCH_ROOT, now=None):
    """Recompute every hit; never trust serialized hit/calibration trial fields."""
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, result_rejections = _result_index(source_root, prematch_root)
    rows, trials, rejected = [], [], Counter()
    paths = sorted((Path(root)/"sets").glob("*.json"))
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            models, choices, coverage = _validated(record, source_root=source_root, now=clock)
            if record["sydney_date"] != path.stem:
                raise ValueError("filename_day_mismatch")
            results = [finals.get((m["provider"], str(m["fixture_id"]))) for m in models]
            actual = [_final(model, result, clock) for model, result in zip(models, results)]
            status, hit = "pending", None
            if all(score is not None for score in actual):
                actual_strings = [f"{h}-{a}" for h, a in actual]
                hit = any(p["score1"] == actual_strings[0] and p["score2"] == actual_strings[1]
                          for p in choices)
                status = "verified"
                trials.append({"policy_id": POLICY, "selection_scope": AUTO_SELECTION,
                               "prospective": True, "verification_status": "verified",
                               "model_versions": [m["model_version"] for m in models],
                               "predicted_at_utc": record["predicted_at_utc"],
                               "verified_at_utc": max(utc(r["verified_at_utc"]) for r in results).isoformat(),
                               "kickoffs_utc": [m["kickoff_utc"] for m in models],
                               "fixture_identities": [[m["provider"], str(m["fixture_id"])] for m in models],
                               "fixture_team_names": [[m["home_team"], m["away_team"]] for m in models],
                               "estimated_coverage": coverage, "hit": hit,
                               "source_record_sha256": record["record_sha256"]})
            rows.append({"sydney_date": record["sydney_date"], "status": status,
                         "hit": hit, "estimated_coverage": coverage,
                         "actual_scores": [list(score) if score is not None else None for score in actual]})
        except (ValueError, TypeError, KeyError, OSError, AttributeError, OverflowError) as exc:
            rejected[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    wins = sum(trial["hit"] for trial in trials)
    return {"schema_version": SCHEMA, "policy_id": POLICY,
            "selection_scope": AUTO_SELECTION, "evaluated_at_utc": clock.isoformat(),
            "frozen_set_files": len(paths), "source_valid_sets": len(rows),
            "verified_sets": len(trials), "hits": wins,
            "pending_sets": sum(row["status"] == "pending" for row in rows),
            "observed_hit_rate": wins/len(trials) if trials else None,
            "wilson_95": wilson(wins, len(trials)),
            "source_rejections": dict(rejected),
            "conflicting_result_identities": len(conflicts),
            "result_rejections": dict(result_rejections), "rows": rows, "trials": trials,
            "external_requests": 0, "target_70_percent_proven": False,
            "premium_release_approved": False,
            "note": "Only raw-policy prospective calibration bootstrap; premium still requires independent calibration evidence."}


def load_verified_trials(*, root=ROOT, source_root=ORIGINAL_ROOT,
                         prematch_root=PREMATCH_ROOT, now=None):
    """Read authenticated original sets and final scores, not performance.json."""
    return evaluate(root=root, source_root=source_root, prematch_root=prematch_root,
                    now=now)["trials"]


def main():
    from persist_prematch import write_json
    status = freeze()
    performance = evaluate()
    write_json(ROOT/"status.json", status)
    write_json(ROOT/"performance.json", performance)
    print(json.dumps({"capture": status, "evaluation": {
        key: value for key, value in performance.items() if key not in ("rows", "trials")}}, indent=2))


if __name__ == "__main__":
    main()
