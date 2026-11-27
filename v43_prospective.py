"""Freeze and score V43 against V8 on each day's authenticated original pair."""

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

from five_double_accuracy import SYDNEY, _final, _result_index, wilson
from source_capture_clock import capture_clock
from v10_prospective import utc
import v42_research_trial as sources
from v43_prediction import VERSION as FORECAST_VERSION, combine_grids


ROOT = Path("data/v43_prospective")
VERSION = "v43-original-pair-prospective-v1"
METHODS = ("v8_original", "v43")
TARGET = .70
MINIMUM_REVIEW_PAIRS = 100
PROTOCOL = {
    "version": VERSION,
    "forecast_version": FORECAST_VERSION,
    "original_pair_policy": sources.PROTOCOL["pair_selection"],
    "decision_rule": sources.PROTOCOL["decision_rule"],
    "methods": list(METHODS),
    "target_hit_rate": TARGET,
    "minimum_review_pairs": MINIMUM_REVIEW_PAIRS,
    "review_metric": "wilson_95_lower_bound_first_100_chronologically_issued_nonoverlapping_pairs",
    "review_timing": "once_all_first_100_pairs_have_verified_regulation_results",
    "automatic_model_promotion": False,
}


def _implementations():
    here = Path(__file__).resolve().parent
    hashes = sources._implementation()
    for name in ("v43_prediction.py", "v43_prospective.py", "source_capture_clock.py"):
        hashes[name] = hashlib.sha256((here / name).read_bytes()).hexdigest()
    return hashes


def _digest(record, field):
    return sources._hash({k: value for k, value in record.items() if k != field})


def _protocol(root, now, *, register=False):
    path = root / "protocol.json"
    implementation = _implementations()
    if register and not path.exists():
        record = {"protocol": PROTOCOL, "implementation_sha256": implementation,
                  "registered_at_utc": now.isoformat()}
        record["protocol_sha256"] = _digest(record, "protocol_sha256")
        sources._write_once(path, record)
    record = sources._load(path)
    if (record["protocol"] != PROTOCOL or record["implementation_sha256"] != implementation
            or record["protocol_sha256"] != _digest(record, "protocol_sha256")
            or utc(record["registered_at_utc"]) > now):
        raise ValueError("v43_protocol_or_implementation_mismatch")
    return record


def _arms(pair, models, cutoff, provider_root):
    grids = [model["score_matrix"] for model in models]
    bundles = [sources._components(model, provider_root=provider_root, cutoff=cutoff)
               for model in models]
    original = {
        "status": "issued",
        "full_score_grids": grids,
        "five_doubles": pair["selection"]["choices"],
        "estimated_coverage": pair["selection"]["estimated_coverage"],
    }
    result = combine_grids(grids, bundles)
    combined = {k: result[k] for k in
                ("status", "full_score_grids", "five_doubles", "estimated_coverage", "reason") if k in result}
    combined["evidence_used"] = (result.get("lineage") or {}).get("evidence_used", {})
    return {METHODS[0]: original, METHODS[1]: combined}, sources._references(bundles)


def freeze(*, root=ROOT, pair_root=sources.PAIR_ROOT, source_root=sources.SOURCE_ROOT,
           provider_root=sources.PROVIDER_ROOT, now=None, clock=None):
    """Capture on the first pass inside the declared near-kickoff window."""
    root = Path(root)
    started, observed = capture_clock(now, clock)
    protocol = _protocol(root, started, register=True)
    max_lead = timedelta(seconds=sources.PROTOCOL["maximum_lead_seconds_inclusive"])
    dates = {instant.astimezone(SYDNEY).date().isoformat()
             for instant in (started, started + max_lead)}
    new, preserved, rejections = [], [], Counter()
    for day in sorted(dates):
        destination = root / "trials" / (day + ".json")
        if destination.exists():
            preserved.append(day)
            continue
        candidate = Path(pair_root) / "sets" / destination.name
        if not candidate.is_file():
            rejections["no_original_pair_for_day"] += 1
            continue
        try:
            pair, models = sources._pair(candidate, source_root=source_root, decision=started)
            if not sources._window(models, started):
                rejections["outside_decision_window"] += 1
                continue
            arms, references = _arms(pair, models, started, provider_root)
            predicted = observed()
            if not sources._window(models, predicted):
                raise ValueError("calculation_finished_outside_decision_window")
            record = {
                "version": VERSION, "sydney_date": day, "decision_as_of_utc": started.isoformat(),
                "predicted_at_utc": predicted.isoformat(), "protocol_sha256": protocol["protocol_sha256"],
                "original_pair_sha256": pair["record_sha256"], "component_sources": references,
                "arms": arms,
            }
            record["record_sha256"] = _digest(record, "record_sha256")
            if sources._write_once(destination, record):
                new.append(day)
            else:
                preserved.append(day)
        except (ValueError, KeyError, TypeError, OSError, AttributeError, ArithmeticError) as exc:
            rejections[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return {"version": VERSION, "new_trials": new, "preserved_trials": preserved,
            "rejections": dict(rejections), "external_requests": 0}


def _validated(path, *, protocol, pair_root, source_root, provider_root, now):
    record = sources._load(path)
    cutoff, predicted = utc(record["decision_as_of_utc"]), utc(record["predicted_at_utc"])
    if (record["version"] != VERSION or record["sydney_date"] != path.stem
            or date.fromisoformat(path.stem).isoformat() != path.stem
            or record["protocol_sha256"] != protocol["protocol_sha256"]
            or record["record_sha256"] != _digest(record, "record_sha256")
            or not utc(protocol["registered_at_utc"]) <= cutoff <= predicted <= now):
        raise ValueError("invalid_v43_receipt_or_timestamp")
    candidate = Path(pair_root) / "sets" / path.name
    original, models = sources._pair(candidate, source_root=source_root, decision=cutoff)
    if (original["record_sha256"] != record["original_pair_sha256"]
            or not sources._window(models, cutoff) or not sources._window(models, predicted)):
        raise ValueError("original_pair_mismatch_or_outside_window")
    arms, references = _arms(original, models, cutoff, provider_root)
    if (references != record["component_sources"]
            or not sources._same_calculation(record["arms"], arms)):
        raise ValueError("v43_receipt_does_not_reproduce")
    return record, models


def _fixture_event(model):
    names = tuple(" ".join(model[k].casefold().split()) for k in ("home_team", "away_team"))
    return utc(model["kickoff_utc"]), *names


def _fixture_details(model):
    fields = ("provider", "fixture_id", "home_team", "away_team", "kickoff_utc")
    return {key: model[key] for key in fields}


def _methods(scored):
    output = {}
    for name in METHODS:
        metrics = [row[name] for row in scored]
        count = len(metrics)
        wins = sum(item["hit"] for item in metrics)
        output[name] = {
            "verified_pairs": count, "hits": wins, "hit_rate": wins / count if count else None,
            "wilson_95": wilson(wins, count),
            **{key: math.fsum(item[key] for item in metrics) / count if count else None
               for key in ("score_log_loss", "outcome_brier", "estimated_coverage")},
        }
    return output


def _fixed_review(rows):
    issued = sorted((row for row in rows if row["status"] != "abstained"),
                    key=lambda row: (utc(row["predicted_at_utc"]), row["sydney_date"]))
    cohort = issued[:MINIMUM_REVIEW_PAIRS]
    settled = [row for row in cohort if row["status"] == "verified"]
    complete = len(settled) == MINIMUM_REVIEW_PAIRS
    hits = sum(row["v43_hit"] for row in settled)
    interval = wilson(hits, len(settled)) if complete else None
    met = bool(complete and interval[0] >= TARGET)
    return {"cohort_size": MINIMUM_REVIEW_PAIRS, "issued": len(cohort),
            "verified": len(settled), "hits": hits, "wilson_95": interval,
            "complete": complete, "met": met,
            "status": ("met" if met else "below_target" if complete else
                       "waiting_for_verification" if len(cohort) == MINIMUM_REVIEW_PAIRS else "collecting")}


def _missed_windows(pair_root, existing_days, source_root, registered, now):
    """Count closed windows for real original pairs available before their cutoff."""
    first_day = registered.astimezone(SYDNEY).date().isoformat()
    last_day = now.astimezone(SYDNEY).date().isoformat()
    min_lead = timedelta(seconds=sources.PROTOCOL["minimum_lead_seconds_exclusive"])
    missed = 0
    for path in (Path(pair_root) / "sets").glob("*.json"):
        if path.stem in existing_days or not first_day <= path.stem <= last_day:
            continue
        try:
            pair, models = sources._pair(path, source_root=source_root, decision=now)
            closed = min(utc(model["kickoff_utc"]) for model in models) - min_lead
            eligible_since = max(registered, utc(pair["predicted_at_utc"]))
            missed += eligible_since < closed <= now
        except (ValueError, KeyError, TypeError, OSError, AttributeError, ArithmeticError):
            continue
    return missed


def evaluate(*, root=ROOT, pair_root=sources.PAIR_ROOT, source_root=sources.SOURCE_ROOT,
             provider_root=sources.PROVIDER_ROOT, prematch_root=sources.PREMATCH_ROOT, now=None):
    """Read-only evaluation of original receipts and independently matched finals."""
    now = utc(now or datetime.now(timezone.utc))
    root = Path(root)
    rows, scored, rejections = [], [], Counter()
    seen_ids, seen_events, statuses = set(), set(), Counter()
    paths = sorted((root / "trials").glob("*.json"))
    protocol = None
    conflicts, final_rejections = set(), {}
    if (root / "protocol.json").is_file() or paths:
        try:
            protocol = _protocol(root, now)
        except (ValueError, KeyError, TypeError, OSError, AttributeError, ArithmeticError):
            rejections["unusable_v43_protocol"] = len(paths)
    if protocol and paths:
        finals, conflicts, final_rejections = _result_index(source_root, prematch_root)
        for path in paths:
            try:
                record, models = _validated(path, protocol=protocol, pair_root=pair_root,
                                            source_root=source_root, provider_root=provider_root, now=now)
                ids = {(str(model["provider"]), str(model["fixture_id"])) for model in models}
                events = {_fixture_event(model) for model in models}
                if len(ids) != 2 or len(events) != 2 or ids & seen_ids or events & seen_events:
                    raise ValueError("repeated_fixture_or_cross_provider_event")
                actual = [_final(model, finals.get((str(model["provider"]), str(model["fixture_id"]))), now)
                          for model in models]
                arms = record["arms"]
                combined = arms[METHODS[1]]
                if combined["status"] != "issued":
                    status = "abstained"
                    metrics = {}
                elif all(value is not None for value in actual):
                    status = "verified"
                    metrics = {name: sources._metric(arms[name], actual) for name in METHODS}
                    scored.append(metrics)
                else:
                    status = ("pending_kickoff" if any(utc(m["kickoff_utc"]) > now for m in models)
                              else "pending_verification")
                    metrics = {}
                seen_ids.update(ids)
                seen_events.update(events)
                statuses[status] += 1
                rows.append({
                    "sydney_date": record["sydney_date"], "status": status,
                    "predicted_at_utc": record["predicted_at_utc"],
                    "fixtures": [_fixture_details(model) for model in models],
                    "actual_scores": [f"{score[0]}-{score[1]}" if score is not None else None for score in actual],
                    "v43_choices": combined.get("five_doubles", []),
                    "v8_choices": arms[METHODS[0]]["five_doubles"],
                    "estimated_coverage": combined.get("estimated_coverage"),
                    "evidence_used": combined["evidence_used"],
                    "v43_hit": metrics.get(METHODS[1], {}).get("hit"),
                    "v8_hit": metrics.get(METHODS[0], {}).get("hit"),
                })
            except (ValueError, KeyError, TypeError, OSError, AttributeError, ArithmeticError) as exc:
                rejections[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    methods = _methods(scored)
    observed = methods[METHODS[1]]
    registered = protocol["registered_at_utc"] if protocol else None
    review = _fixed_review(rows)
    return {
        "version": VERSION, "registered_at_utc": registered,
        "status": "ready" if protocol else "invalid_protocol" if rejections else "not_registered",
        "verified_pairs": observed["verified_pairs"],
        "pending_pairs": statuses["pending_kickoff"] + statuses["pending_verification"],
        "pending_kickoff_pairs": statuses["pending_kickoff"],
        "pending_verification_pairs": statuses["pending_verification"],
        "abstained_pairs": statuses["abstained"], "invalid_pairs": sum(rejections.values()),
        "missed_windows": _missed_windows(pair_root, {path.stem for path in paths},
                                          source_root, utc(registered), now) if registered else 0,
        "hits": observed["hits"], "hit_rate": observed["hit_rate"], "wilson_95": observed["wilson_95"],
        "review_gate_met": review["met"], "fixed_review": review,
        "minimum_review_pairs": MINIMUM_REVIEW_PAIRS, "target_hit_rate": TARGET,
        "paired_market_used": sum(row["evidence_used"].get("market", False) for row in rows),
        "paired_xg_used": sum(row["evidence_used"].get("xg", False) for row in rows),
        "methods": methods, "rows": rows, "rejections": dict(rejections),
        "conflicting_result_identities": len(conflicts), "result_rejections": dict(final_rejections),
        "external_requests": 0, "automatic_model_promotion": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Register and capture future eligible trials.")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(), sort_keys=True))
    print(json.dumps(evaluate(), indent=2, sort_keys=True, allow_nan=False))
