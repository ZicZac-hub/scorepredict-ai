"""V36 future-only five-double score-tail trial on immutable V28 source pairs.

A new packet is write-once and issued only within V28's original short
pre-kickoff window. Candidate methods have fixed parameters and share the
EXACT same selected pair. This module does not retrospectively issue tickets,
fetch new football data, or promote live models.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from math import fsum
import json
from pathlib import Path

from five_double_accuracy import joint_geometry, utc, wilson
from v25_score_accuracy_research import describe_score
from v26_temporal_score import _write_once
from v28_independent_pair_trial import (
    ORIGINAL_ROOT, OUT as V28_ROOT, _key, verify_decision,
)
from v25_anchored_challenger import _frozen_eligible
from professional_analyst_shadow_v25 import MIN_LEAD
from v30_full_slate_audit import analyze as audit_original_full_slates
from v36_tail_calibration_research import METHODS, transform_score_grid

VERSION = "v36-future-only-original-pair-score-trial-v1"
ROOT = Path("data/v36_prospective_challengers")
MIN_REVIEW_DAYS = 100


def build_packet(original, items, issued):
    issued = utc(issued)
    if original.get("policy") != "FROZEN" or not _frozen_eligible(original, issued):
        raise ValueError("outside_original_pre_kickoff_issue_window")
    selection = original["quality_selection"]
    keys = selection["fixture_keys"]
    if len(keys) != 2 or len({tuple(k) for k in keys}) != 2:
        raise ValueError("invalid_two_original_fixture_keys")
    sources = {_key(item): item for item in items}
    if any(tuple(k) not in sources for k in keys):
        raise ValueError("missing_immutable_original_source")
    legs = [sources[tuple(k)] for k in keys]
    if any(issued > item["kickoff"] - MIN_LEAD for item in legs):
        raise ValueError("too_late_for_a_fixture")
    if len(legs[0]["teams"] | legs[1]["teams"]) != 4:
        raise ValueError("four_independent_teams_required")
    policies = {}
    for method in METHODS:
        grids = [transform_score_grid(item["grid"], method) for item in legs]
        picks = joint_geometry(*grids)["optimal_five"]
        if len(picks) != 5 or len({(p["score1"],p["score2"]) for p in picks}) != 5:
            raise ValueError("not_five_distinct_exact_score_doubles")
        policies[method] = {"score_grids": grids, "five_joint_scores": picks}
    return {
        "version": VERSION,
        "sydney_day": original["sydney_day"],
        "frozen_at_utc": issued.isoformat(),
        "original_v28_issued_at_utc": utc(original["actually_issued_at_utc"]).isoformat(),
        "original_catalog_sha256": original["candidate_catalog_sha256"],
        "original_selected_fixture_keys": keys,
        "original_source_forecast_ids": selection["source_forecast_ids"],
        "predeclared_methods": list(METHODS),
        "policies": policies,
        "research_only": True,
        "same_pre_kickoff_original_pair": True,
        "no_external_provider_calls": True,
        "no_live_model_change": True,
    }


def freeze(*, root=ORIGINAL_ROOT, v28_root=V28_ROOT, out=ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    created, skipped = [], Counter()
    for file in sorted((Path(v28_root)/"decisions").glob("*.json")):
        dest = Path(out)/"sets"/file.name
        if dest.exists():
            skipped["already_frozen"] += 1
            continue
        try:
            original, items = verify_decision(file, root=root, clock=clock)
            if not _frozen_eligible(original, clock):
                skipped["original_v28_window_expired"] += 1
                continue
            issued = utc(now or datetime.now(timezone.utc))
            if not _frozen_eligible(original, issued):
                skipped["issuance_too_late"] += 1
                continue
            packet = build_packet(original, items, issued)
            if _write_once(dest, packet):
                created.append(file.stem)
        except (OSError, ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
            skipped[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return {"created_days": created, "skipped": dict(skipped)}


def verify_packet(file, *, root, v28_root, clock):
    file = Path(file)
    packet = json.loads(file.read_text(encoding="utf-8"))
    if packet.get("version") != VERSION or packet.get("sydney_day") != file.stem:
        raise ValueError("bad_frozen_v36_identity")
    original, items = verify_decision(Path(v28_root)/"decisions"/file.name,
                                      root=root, clock=clock)
    issued = utc(packet["frozen_at_utc"])
    if issued > utc(clock) or not _frozen_eligible(original, issued):
        raise ValueError("late_or_future_v36_issue")
    if build_packet(original, items, issued) != packet:
        raise ValueError("tampered_v36_source_or_ticket")
    return packet


def _hit(picks, scores):
    return any([p["score1"], p["score2"]] == scores for p in picks)


def summarize(rows, *, now):
    verified = [row for row in rows if row["status"] == "verified"]
    # One selected pair per Sydney day; correlated candidate-pair rows are
    # never inserted here and cannot inflate the ticket denominator.
    n = len(verified)
    unique = {}
    fixture_day = {}
    if len({r["sydney_day"] for r in verified}) != n:
        raise ValueError("duplicate_verified_sydney_day")
    for row in verified:
        for i, key in enumerate(row["original_selected_fixture_keys"]):
            k = tuple(key)
            if k in fixture_day and fixture_day[k] != row["sydney_day"]:
                raise ValueError("same_physical_fixture_repeated_across_independent_days")
            fixture_day[k] = row["sydney_day"]
            observation = row["actual_scores"][i]
            if k in unique and unique[k]["score"] != observation:
                raise ValueError("conflicting_physical_fixture_final")
            unique.setdefault(k, {"score": observation,
                                  "proper": {m: row["proper_scores"][m][i]
                                             for m in METHODS}})
    methods = {}
    for method in METHODS:
        hits = sum(int(row["hits"][method]) for row in verified)
        metrics = ("exact_score_log_loss", "multiclass_score_brier",
                   "over_3_5_brier", "under_1_5_brier")
        measured = [item["proper"][method] for item in unique.values()]
        means = {k: fsum(x[k] for x in measured)/len(measured)
                 if measured else None for k in metrics}
        methods[method] = {
            "hits": hits, "verified_independent_sydney_days": n,
            "observed_five_double_hit_rate": hits/n if n else None,
            "wilson95": wilson(hits,n),
            "unique_verified_physical_fixtures": len(unique),
            "out_of_score_grid_finals": sum(
                int(x["outside_score_grid"]) for x in measured),
            "mean_proper_scores": means,
            "research_only_not_live_promoted": True,
        }
    return {
        "version": VERSION, "evaluated_at_utc": utc(now).isoformat(),
        "frozen_prospective_days": len(rows),
        "verified_independent_sydney_days": n,
        "pending_days": len(rows)-n,
        "unique_verified_physical_fixtures": len(unique),
        "minimum_initial_review_days": MIN_REVIEW_DAYS,
        "methods": methods, "days": rows,
        "ready_for_separate_production_review": [],
        "same_pair_comparisons_only": True,
        "no_hindsight_ticket_reselection": True,
        "not_validated_at_seventy_percent": True,
        "no_live_model_change": True,
    }


def _describe(grid, actual):
    report = describe_score(grid, actual)
    report["outside_score_grid"] = (
        actual[0] >= len(grid) or actual[1] >= len(grid[0]))
    return report


def evaluate(*, root=ORIGINAL_ROOT, prematch=Path("data/prematch"),
             v28_root=V28_ROOT, out=ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    audit = audit_original_full_slates(root=root, prematch=prematch,
                                      v28_out=v28_root, now=clock)
    days = {x["sydney_day"]: x for x in audit["days"]
            if x["original_policy"] == "FROZEN"}
    rows, rejected = [], Counter()
    for file in sorted((Path(out)/"sets").glob("*.json")):
        try:
            packet = verify_packet(file, root=root, v28_root=v28_root,
                                   clock=clock)
            day = days.get(packet["sydney_day"])
            if day is None:
                raise ValueError("no_verified_original_v28_day")
            matching = [row for row in day["candidate_rows"]
                        if row["fixture_keys"] == packet["original_selected_fixture_keys"]]
            if len(matching) > 1:
                raise ValueError("duplicate_verified_selected_pair")
            row = {
                "sydney_day": packet["sydney_day"],
                "original_selected_fixture_keys": packet["original_selected_fixture_keys"],
                "frozen_at_utc": packet["frozen_at_utc"],
                "status": "pending",
            }
            if matching:
                actual = matching[0]["actual_scores"]
                goals = [tuple(map(int, s.split("-"))) for s in actual]
                row.update({
                    "status": "verified", "actual_scores": actual,
                    "hits": {method: _hit(packet["policies"][method]["five_joint_scores"],
                                          actual) for method in METHODS},
                    "proper_scores": {method: [
                        _describe(grid, result) for grid, result in zip(
                            packet["policies"][method]["score_grids"], goals)
                    ] for method in METHODS},
                })
            rows.append(row)
        except (OSError, ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
            rejected[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    result = summarize(rows, now=clock)
    result["invalid_or_tampered_v36_sets_excluded"] = dict(rejected)
    result["original_v28_rejected_or_tampered"] = audit["invalid_or_tampered_frozen_days_excluded"]
    result["verified_result_conflicts_excluded"] = audit["verified_result_conflicts_excluded"]
    return result


def run(*, root=ORIGINAL_ROOT, prematch=Path("data/prematch"),
        v28_root=V28_ROOT, out=ROOT, now=None, verify_only=False):
    captured = {"created_days": [], "skipped": {"verify_only": 1}}
    if not verify_only:
        captured = freeze(root=root, v28_root=v28_root, out=out, now=now)
    report = evaluate(root=root, prematch=prematch, v28_root=v28_root,
                      out=out, now=now)
    report["capture"] = captured
    dest = Path(out)/"status.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not dest.exists() or dest.read_text(encoding="utf-8") != content:
        dest.write_text(content, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = run(verify_only=args.verify_only)
    print(json.dumps({"frozen":result["frozen_prospective_days"],
                      "verified":result["verified_independent_sydney_days"],
                      "pending":result["pending_days"],
                      "methods":list(METHODS),
                      "promotion_ready":[]}, sort_keys=True))
