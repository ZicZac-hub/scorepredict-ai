"""Freeze a market-only V43 adjustment on the same original five-double pair."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

from five_double_accuracy import SYDNEY, _final, wilson
from persist_prematch import load_json
from source_capture_clock import capture_clock
from v10_prospective import utc
import v42_research_trial as sources
from v43_prediction import VERSION as FORECAST_VERSION, combine_grids
from v43_prospective import _fixture_details, _fixture_event
from v44_early_history import _once, digest
from v45_lineups import _atomic_status
from v45_market_capture import ROOT as MARKET_ROOT, best_market, load_market_reference
from verified_result_index import index_verified_results


ROOT = Path("data/v45_market_trial")
VERSION = "v45-automatic-paired-market-v1"
LEAD_START = timedelta(minutes=20)
LEAD_END = timedelta(minutes=5)
METHODS = ("baseline", "with_market")
REVIEW_PAIRS = 100
PROTOCOL = {
    "version": VERSION,
    "forecast_version": FORECAST_VERSION,
    "pair_policy": sources.PROTOCOL["pair_selection"],
    "decision_window_seconds": {"maximum_inclusive": 1200, "minimum_exclusive": 300},
    "market_policy": "two_qualified_v45_market_receipts_at_shared_cutoff",
    "arms": {"baseline": "v43_with_no_component_adjustment",
             "with_market": "same_v43_original_grids_with_market_only"},
    "minimum_review_pairs": REVIEW_PAIRS,
    "automatic_promotion": False,
}
BAD_RECORD = (ValueError, KeyError, TypeError, OSError, AttributeError, ArithmeticError)


def _hashes():
    here = Path(__file__).resolve().parent
    paths = ("v45_market_trial.py", "v45_market_capture.py", "v43_prediction.py",
             "source_capture_clock.py", "verified_result_index.py", "multi_market_v10.py",
             "v43_prospective.py")
    return {**sources._implementation(), **{
        name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in paths
    }}


def _digest(record, field):
    return digest({key: value for key, value in record.items() if key != field})


def _protocol(root, now, *, register=False):
    path = root / "protocol.json"
    implementation = _hashes()
    if register and not path.exists():
        record = {"protocol": PROTOCOL, "implementation_sha256": implementation,
                  "registered_at_utc": now.isoformat()}
        record["protocol_sha256"] = _digest(record, "protocol_sha256")
        _once(path, record)
    record = sources._load(path)
    if (record["protocol"] != PROTOCOL or record["implementation_sha256"] != implementation
            or record["protocol_sha256"] != _digest(record, "protocol_sha256")
            or utc(record["registered_at_utc"]) > now):
        raise ValueError("invalid_market_trial_protocol")
    return record


def _earliest(models):
    return min(utc(model["kickoff_utc"]) for model in models)


def _within(kickoff, decision):
    return LEAD_END < kickoff - decision <= LEAD_START


def _arms(models, market):
    if any(component is None for component in market):
        return {}, "market_missing_on_one_leg" if any(market) else "market_missing_on_both_legs"
    original = [model["score_matrix"] for model in models]
    control = combine_grids(original, [{}, {}])
    treatment = combine_grids(original, [{"market": component} for component in market])
    if (control["status"] != "issued" or treatment["status"] != "issued"
            or treatment["lineage"]["evidence_used"]["market"] is not True):
        return {}, "market_transform_abstained"
    fields = ("full_score_grids", "five_doubles", "estimated_coverage")
    return {name: {field: forecast[field] for field in fields}
            for name, forecast in zip(METHODS, (control, treatment))}, None


def freeze(*, root=ROOT, pair_root=sources.PAIR_ROOT, source_root=sources.SOURCE_ROOT,
           provider_root=MARKET_ROOT, now=None, clock=None):
    """Freeze or abstain once at the first scheduled decision in the window."""
    root = Path(root)
    started, observed = capture_clock(now, clock)
    protocol = _protocol(root, started, register=True)
    dates = {instant.astimezone(SYDNEY).date().isoformat()
             for instant in (started, started + LEAD_START)}
    frozen, abstained, preserved, rejections = [], [], [], Counter()
    for day in sorted(dates):
        filename = day + ".json"
        destination = root / "trials" / filename
        if destination.exists():
            preserved.append(day)
            continue
        candidate = Path(pair_root) / "sets" / filename
        if not candidate.is_file():
            continue
        try:
            pair, models = sources._pair(candidate, source_root=source_root, decision=started)
            earliest = _earliest(models)
            decision = observed()
            if not _within(earliest, decision):
                continue
            binding = {"provider_root": provider_root, "cutoff": decision,
                       "pair_sha256": pair["record_sha256"], "earliest_kickoff": earliest}
            forecast_evidence = [best_market(model, **binding) for model in models]
            arms, reason = _arms(models, forecast_evidence)
            finished = observed()
            if not _within(earliest, finished) or finished < decision:
                rejections["decision_window_closed_during_calculation"] += 1
                continue
            record = {"version": VERSION, "sydney_date": day,
                      "status": "abstained" if reason else "issued", "reason": reason,
                      "decision_as_of_utc": decision.isoformat(), "predicted_at_utc": finished.isoformat(),
                      "protocol_sha256": protocol["protocol_sha256"],
                      "original_pair_sha256": pair["record_sha256"],
                      "market_sources": [row["reference"] if row else None for row in forecast_evidence],
                      "arms": arms}
            record["record_sha256"] = digest(record)
            if _once(destination, record):
                (abstained if reason else frozen).append(day)
            else:
                preserved.append(day)
        except BAD_RECORD as exc:
            rejections[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return {"issued": frozen, "abstained": abstained, "preserved": preserved,
            "rejections": dict(rejections)}


def _validated(path, *, protocol, pair_root, source_root, provider_root, now):
    record = sources._load(path)
    cutoff, predicted = utc(record["decision_as_of_utc"]), utc(record["predicted_at_utc"])
    if (record["version"] != VERSION or record["sydney_date"] != path.stem
            or record["protocol_sha256"] != protocol["protocol_sha256"]
            or record["record_sha256"] != _digest(record, "record_sha256")
            or not utc(protocol["registered_at_utc"]) <= cutoff <= predicted <= now):
        raise ValueError("invalid_market_trial_receipt")
    pair, models = sources._pair(Path(pair_root) / "sets" / path.name,
                                 source_root=source_root, decision=cutoff)
    earliest = _earliest(models)
    if (record["original_pair_sha256"] != pair["record_sha256"]
            or not _within(earliest, cutoff) or not _within(earliest, predicted)):
        raise ValueError("pair_identity_or_market_window_mismatch")
    references = record["market_sources"]
    if not isinstance(references, list) or len(references) != 2:
        raise ValueError("invalid_market_trial_source_references")
    binding = {"provider_root": provider_root, "cutoff": cutoff,
               "pair_sha256": pair["record_sha256"], "earliest_kickoff": earliest}
    requested = [load_market_reference(model, ref, **binding) if ref is not None else None
                 for model, ref in zip(models, references)]
    arms, reason = _arms(models, requested)
    if (record["reason"] != reason or record["status"] != ("abstained" if reason else "issued")
            or not sources._same_calculation(record["arms"], arms)):
        raise ValueError("market_comparison_does_not_reproduce")
    return record, models


def _missed_windows(pair_root, existing_days, source_root, registered, now):
    start_day = registered.astimezone(SYDNEY).date().isoformat()
    end_day = now.astimezone(SYDNEY).date().isoformat()
    missed = 0
    for path in (Path(pair_root) / "sets").glob("*.json"):
        if path.stem in existing_days or not start_day <= path.stem <= end_day:
            continue
        try:
            pair, models = sources._pair(path, source_root=source_root, decision=now)
            close = _earliest(models) - LEAD_END
            if max(registered, utc(pair["predicted_at_utc"])) < close <= now:
                missed += 1
        except BAD_RECORD:
            continue
    return missed


def _metrics(scored):
    count = len(scored)
    fields = ("score_log_loss", "outcome_brier", "estimated_coverage")
    result = {}
    for name in METHODS:
        results = [row[name] for row in scored]
        hits = sum(row["hit"] for row in results)
        result[name] = {"hits": hits, "verified_pairs": count, "hit_rate": hits / count if count else None,
                        "wilson_95": wilson(hits, count),
                        **{field: math.fsum(row[field] for row in results) / count if count else None
                           for field in fields}}
    result["market_minus_baseline"] = {
        field: sources._mean_interval([row["with_market"][field] - row["baseline"][field]
                                       for row in scored])
        for field in ("score_log_loss", "outcome_brier", "hit")
    }
    result["discordant_hits"] = {
        "market_only": sum(row["with_market"]["hit"] and not row["baseline"]["hit"] for row in scored),
        "baseline_only": sum(row["baseline"]["hit"] and not row["with_market"]["hit"] for row in scored),
    }
    return result


def evaluate(*, root=ROOT, pair_root=sources.PAIR_ROOT, source_root=sources.SOURCE_ROOT,
             provider_root=MARKET_ROOT, prematch_root=sources.PREMATCH_ROOT, now=None):
    """Authenticate exact cited odds and grade both arms against the same finals."""
    now = utc(now or datetime.now(timezone.utc))
    root = Path(root)
    paths = sorted((root / "trials").glob("*.json"))
    rejections, statuses = Counter(), Counter()
    protocol, eligible = None, []
    if (root / "protocol.json").exists() or paths:
        try:
            protocol = _protocol(root, now)
        except BAD_RECORD:
            rejections["invalid_market_trial_protocol"] = max(1, len(paths))

    if protocol is not None:
        for path in paths:
            try:
                record, models = _validated(path, protocol=protocol, pair_root=pair_root,
                                            source_root=source_root, provider_root=provider_root, now=now)
                eligible.append((utc(record["predicted_at_utc"]), path.stem, record, models))
            except BAD_RECORD as exc:
                rejections[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1

    finals, conflicts, final_rejections = index_verified_results(source_root, prematch_root, now)
    seen_ids, seen_events, scored, rows = set(), set(), [], []
    for _, day, record, models in sorted(eligible, key=lambda entry: entry[:2]):
        ids = [(str(model["provider"]), str(model["fixture_id"])) for model in models]
        events = {_fixture_event(model) for model in models}
        if len(set(ids)) != 2 or len(events) != 2 or seen_ids.intersection(ids) or seen_events & events:
            rejections["repeated_market_trial_fixture"] += 1
            continue
        seen_ids.update(ids)
        seen_events.update(events)
        actual = [_final(model, finals.get(identity), now) for model, identity in zip(models, ids)]
        metrics = {}
        if record["status"] == "abstained":
            status = "abstained"
        elif all(score is not None for score in actual):
            status = "verified"
            metrics = {name: sources._metric(record["arms"][name], actual) for name in METHODS}
            scored.append(metrics)
        else:
            status = "pending_kickoff" if any(utc(model["kickoff_utc"]) > now for model in models) else "pending_verification"
        statuses[status] += 1
        rows.append({"sydney_date": day, "status": status, "reason": record["reason"],
                     "predicted_at_utc": record["predicted_at_utc"],
                     "qualified_market_legs": sum(ref is not None for ref in record["market_sources"]),
                     "market_observed_at_utc": [ref["captured_at_utc"] if ref is not None else None
                                                 for ref in record["market_sources"]],
                     "fixtures": [_fixture_details(model) for model in models],
                     "actual_scores": [f"{score[0]}-{score[1]}" if score is not None else None for score in actual],
                     "settlement_conflict": any(identity in conflicts for identity in ids),
                     "methods": metrics})

    registered = protocol["registered_at_utc"] if protocol is not None else None
    missed = (_missed_windows(pair_root, {day for _, day, *_ in eligible},
                              source_root, utc(registered), now) if registered else 0)
    issued = statuses["verified"] + statuses["pending_kickoff"] + statuses["pending_verification"]
    sampled = len(rows) + missed
    source_qualified = sum(row["qualified_market_legs"] == 2 for row in rows)
    skipped = statuses["abstained"] + missed
    return {
        "version": VERSION, "as_of_utc": now.isoformat(), "registered_at_utc": registered,
        "status": ("awaiting_registration" if protocol is None and not rejections else
                   "invalid_protocol" if protocol is None else
                   "ready_for_review" if len(scored) >= REVIEW_PAIRS else "collecting"),
        "matched_verified_pairs": len(scored), "issued_pairs": issued,
        "pending_kickoff_pairs": statuses["pending_kickoff"],
        "pending_verification_pairs": statuses["pending_verification"],
        "abstained_pairs": statuses["abstained"], "missed_windows": missed, "skipped_pairs": skipped,
        "pair_opportunities": sampled, "both_markets_qualified_pairs": source_qualified,
        "paired_market_coverage": source_qualified / sampled if sampled else None,
        "skip_rate": skipped / sampled if sampled else None,
        "minimum_review_pairs": REVIEW_PAIRS, "metrics": _metrics(scored),
        "invalid_receipts": sum(rejections.values()), "receipt_rejections": dict(rejections),
        "conflicting_result_fixtures": len(conflicts), "result_rejections": dict(final_rejections),
        "automatic_promotion": False, "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Capture the current valid decision before reporting.")
    args = parser.parse_args()
    decision = freeze() if args.freeze else None
    report = evaluate()
    if decision is not None:
        report["latest_freeze"] = decision
    status_path = ROOT / "status.json"
    previous = load_json(status_path, {})
    if isinstance(previous, dict):
        previous.pop("as_of_utc", None)
    if previous != {key: value for key, value in report.items() if key != "as_of_utc"}:
        _atomic_status(status_path, report)
    print(json.dumps({k: report[k] for k in ("status", "issued_pairs", "matched_verified_pairs",
                                            "skipped_pairs", "missed_windows", "paired_market_coverage")}))
