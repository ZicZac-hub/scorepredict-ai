"""V27: prospective full-slate, provenance-audited pair selection shadow.

V27 ranks *which two matches* to study, not new underlying score models.
Every decision is write-once before kickoff, on the candidate slate visible
when the ORIGINAL V25 daily set was selected. All qualified pair candidates
are retained for auditing; large slates abstain instead of silently trimming.

A V25-vs-V27 comparison on different pairs is a SAME-DAY policy comparison,
not evidence of improved probability calibration on identical matches.
No changed production picks, new provider requests, or 70%-accuracy claims.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import math

from five_double_accuracy import _final, utc, wilson
from professional_analyst_shadow_v25 import (
    _eligible, _pair, MAX_EARLY_SOURCE_AGE, MIN_LEAD, SYDNEY,
)
from v10_prospective import ROOT as SOURCE_ROOT, read_latest
from v8_v25_matched_audit import _frozen_set, _source, _selection
from verified_result_index import index_verified_results
from v26_temporal_score import _write_once

VERSION = "v27-frozen-full-slate-pair-quality-v1-research-only"
OUT = Path("data/v27_pair_selection")
V25_SETS = Path("data/pro_analyst_shadow/sets")
MAX_SELECTION_AGE = timedelta(minutes=20)
MAX_EVALUATED_PAIRS = 2500
MIN_V8_COVERAGE = 0.055   # research-only gate; NOT an empirical accuracy threshold
MIN_CANDIDATES = 2       # need at least two distinct same-day valid pair options
AGE_FRESH_HOURS = 72.0
AGE_OLD_HOURS = MAX_EARLY_SOURCE_AGE.total_seconds()/3600


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def receipt(item):
    source = item["record"]
    # Use identical raw fields for normalized read_latest records and the
    # original immutable source-file read by the independent audit. The
    # ledger reader may add derived matrix_shape/horizon fields.
    original_fields = {
        key: source[key] for key in (
            "provider", "fixture_id", "forecast_id", "model_key", "model_version",
            "home_team_id", "away_team_id", "input_digest", "kickoff_utc",
            "captured_at_utc", "evidence_as_of_utc", "history_matches",
            "score_matrix")
    }
    # The ledger normalizes timestamps to microsecond Z strings on read.
    # Raw immutable files may instead use +00:00. Hash their UTC instants,
    # not their incidental timestamp serialization.
    for key in ("kickoff_utc", "captured_at_utc", "evidence_as_of_utc"):
        original_fields[key] = utc(original_fields[key]).isoformat()
    for key in ("provider", "fixture_id", "forecast_id", "model_key",
                "model_version", "home_team_id", "away_team_id", "input_digest"):
        original_fields[key] = str(original_fields[key])
    original_fields["history_matches"] = int(original_fields["history_matches"])
    return {
        **{key: str(original_fields[key]) for key in (
            "provider", "fixture_id", "forecast_id", "model_key", "model_version",
            "home_team_id", "away_team_id", "input_digest",
            "kickoff_utc", "captured_at_utc")},
        "source_sha256": digest(original_fields),
    }


def pair_keys(first, second):
    return [
        [str(first["key"][0]), str(first["key"][1])],
        [str(second["key"][0]), str(second["key"][1])],
    ]


def _freshness_multiplier(first, second):
    """Bounded freshness penalty, prespecified before any new results exist."""
    oldest = max(first["source_age_hours"], second["source_age_hours"])
    excess = max(0., oldest - AGE_FRESH_HOURS)
    return 1. - .25 * min(1., excess / (AGE_OLD_HOURS - AGE_FRESH_HOURS))


def candidate_entry(a, b):
    values = _pair(a, b)
    factor = _freshness_multiplier(a, b)
    # V8 is the only reference probability. The V25 robust mass here is
    # evaluated *under V8*, unlike its uncalibrated lower-envelope criterion.
    v8 = values["v8_optimal_five_model_mass"]
    robust = values["robust_five_v8_model_mass"]
    quality = (0.60 * v8 + 0.40 * robust) * factor
    return {
        "fixture_keys": pair_keys(a, b),
        "rank_quality": round(quality, 12),
        "original_v8_top_five_model_mass": round(v8, 12),
        "v25_robust_five_under_v8_mass": round(robust, 12),
        "v25_lower_envelope_is_uncalibrated": True,
        "freshness_multiplier": round(factor, 12),
        "source_age_hours": [a["source_age_hours"], b["source_age_hours"]],
        "five_robust": values["five_doubles"],
        "five_v8_optimal": values["baseline_five"],
    }


def build_slate(original_forecasts, cutoff, day, *, max_pairs=MAX_EVALUATED_PAIRS):
    """Full eligible original pre-match slate, no result data or silent top-N."""
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("invalid_max_pair_budget")
    decision = utc(cutoff)
    by_identity = {}
    excluded = Counter()
    for row in (original_forecasts.values() if isinstance(original_forecasts, dict)
                else original_forecasts):
        item, reason = _eligible(row, decision)
        if item is None:
            excluded[reason] += 1
            continue
        if item["day"] != day:
            continue
        key = item["key"]
        old = by_identity.get(key)
        if old is None or item["captured"] > old["captured"]:
            by_identity[key] = item
    items = [by_identity[k] for k in sorted(by_identity)]
    pairs = [(a, b) for a, b in combinations(items, 2)
             if len(a["teams"] | b["teams"]) == 4]
    sources = [receipt(item) for item in items]
    evidence = {
        "candidate_fixtures": len(items),
        "qualified_four_team_pairs": len(pairs),
        "excluded_original_sources": dict(excluded),
        "eligible_original_source_receipts": sources,
        "source_receipts_sha256": digest(sources),
        "selection_is_research_only": True,
    }
    if len(pairs) > max_pairs:
        return {**evidence, "candidate_catalog": [],
                "candidate_catalog_sha256": digest([]),
                "evaluated_pairs": 0,
                "decision": "V27_ABSTAIN",
                "reason": "too_many_pairs_to_compare_without_truncation"}
    entries = [candidate_entry(a, b) for a, b in pairs]
    entries.sort(key=lambda c: (-c["rank_quality"], c["fixture_keys"]))
    evidence.update({
        "candidate_catalog": entries,
        "candidate_catalog_sha256": digest(entries),
        "evaluated_pairs": len(entries),
    })
    if len(entries) < MIN_CANDIDATES:
        return {**evidence, "decision": "V27_ABSTAIN",
                "reason": "fewer_than_two_distinct_eligible_pairs"}
    winner = entries[0]
    if winner["original_v8_top_five_model_mass"] < MIN_V8_COVERAGE:
        return {**evidence, "decision": "V27_ABSTAIN",
                "reason": "best_pair_below_research_model_coverage_gate"}
    return {
        **evidence,
        "decision": "V27_SELECTED",
        "selected_pair": winner,
        "abstain_when_evidence_insufficient": True,
    }


def _apply_issuance_guard(slate, sources, issued_at):
    if slate["decision"] != "V27_SELECTED":
        return slate
    identifiers = {tuple(x["key"]): x for x in sources}
    for key in slate["selected_pair"]["fixture_keys"]:
        if (tuple(key) not in identifiers or
                issued_at > identifiers[tuple(key)]["kickoff"] - MIN_LEAD):
            return {**{k: v for k, v in slate.items() if k != "selected_pair"},
                    "decision": "V27_ABSTAIN",
                    "reason": "chosen_pair_not_45_minutes_early_at_issue"}
    return slate


def _original_v25(path, root, clock):
    packet, sources, _, _, _, _ = _frozen_set(path, root, clock)
    return packet, sources


def _v25_keys(packet):
    return sorted([[str(packet[side]["provider"]), str(packet[side]["fixture_id"])]
                   for side in ("first", "second")])


def _v27_keys(candidate):
    return sorted(candidate["fixture_keys"])


def freeze(*, root=SOURCE_ROOT, sets=V25_SETS, out=OUT,
           now=None, originals=None):
    clock = utc(now or datetime.now(timezone.utc))
    made, skipped = [], Counter()
    for original in sorted(Path(sets).glob("*.json")):
        try:
            packet, sources = _original_v25(original, root, clock)
            selected = utc(packet["selected_at_utc"])
            if not selected <= clock <= selected + MAX_SELECTION_AGE:
                skipped["not_a_new_v25_freeze"] += 1
                continue
            day = packet["sydney_date"]
            file = Path(out) / "decisions" / (day + ".json")
            if file.exists():
                skipped["existing_immutable_decision"] += 1
                continue
            if any(clock > utc(row["kickoff_utc"]) - MIN_LEAD for row in sources):
                skipped["v25_pair_too_close_to_kickoff"] += 1
                continue
            universe = read_latest(root) if originals is None else originals
            slate = build_slate(universe, selected, day)
            # A future archive update cannot create a new policy decision for
            # an old day. The reference original V25 pair must be present in
            # this exact eligible cutoff universe.
            present = {tuple(map(tuple, entry["fixture_keys"]))
                       for entry in slate["candidate_catalog"]}
            reference = tuple(map(tuple, _v25_keys(packet)))
            if reference not in present and slate["decision"] != "V27_ABSTAIN":
                skipped["v25_reference_not_in_original_candidate_slate"] += 1
                continue
            rows = (universe.values() if isinstance(universe, dict) else universe)
            eligible_sources = [
                item for item in (_eligible(x, selected)[0] for x in rows)
                if item is not None and item["day"] == day
            ]
            # Existing V25 may use a match less than 45 minutes from issuance.
            # V27 must explicitly abstain, not backdate a new prediction.
            slate = _apply_issuance_guard(slate, eligible_sources, clock)
            saved = {
                "schema_version": 1, "version": VERSION,
                "sydney_date": day,
                "v25_decision_at_utc": selected.isoformat(),
                "v27_decision_at_utc": clock.isoformat(),
                "original_v25_packet_sha256": digest(packet),
                "original_v25_pair_keys": _v25_keys(packet),
                "ranking_policy": (
                    "0.60*v8_optimal_mass+0.40*v25_robust_mass_under_v8;"
                    "multiply_by_bounded_72h_freshness;full_slate;"
                    "predeclared_0.055_min_model_mass"),
                **slate,
                "original_forecasts_modified": False,
                "v25_v26_forecasts_modified": False,
                "independent_verified_result_evaluation": True,
                "production_model_promoted": False,
                "70_percent_accuracy_not_proven": True,
            }
            if _write_once(file, saved):
                made.append({"day": day, "decision": saved["decision"]})
        except (ValueError, KeyError, TypeError, OverflowError, OSError):
            skipped["invalid_or_incomplete_source_evidence"] += 1
    return {"new_decisions": made, "not_created": dict(skipped),
            "version": VERSION, "live_predictions_unchanged": True}


def _verify_decision(packet, path, root, sets, clock):
    day = packet.get("sydney_date")
    if (not isinstance(day, str) or day != path.stem or
            packet.get("version") != VERSION or
            packet.get("decision") not in ("V27_SELECTED", "V27_ABSTAIN")):
        raise ValueError("invalid_decision_identity")
    source_file = Path(sets) / (day + ".json")
    original, sources = _original_v25(source_file, root, clock)
    first_time = utc(packet["v25_decision_at_utc"])
    decision_at = utc(packet["v27_decision_at_utc"])
    if (first_time != utc(original["selected_at_utc"]) or
            not first_time <= decision_at <= first_time + MAX_SELECTION_AGE or
            decision_at > clock or
            digest(original) != packet["original_v25_packet_sha256"] or
            _v25_keys(original) != packet["original_v25_pair_keys"]):
        raise ValueError("decision_or_v25_source_changed")
    receipts = packet.get("eligible_original_source_receipts")
    if (not isinstance(receipts, list) or
            len(receipts) != packet.get("candidate_fixtures") or
            digest(receipts) != packet.get("source_receipts_sha256")):
        raise ValueError("original_source_slate_tampered")
    eligible = []
    seen = set()
    for rec in receipts:
        source, _ = _source(root, rec, first_time, day)
        item, reason = _eligible(source, first_time)
        if item is None or receipt(item) != rec or item["key"] in seen:
            raise ValueError("source_slate_unreproducible")
        seen.add(item["key"])
        eligible.append(source)
    recreated = build_slate(eligible, first_time, day)
    eligible_items = [_eligible(x, first_time)[0] for x in eligible]
    recreated = _apply_issuance_guard(recreated, eligible_items, decision_at)
    # Unqualified evidence counters may include other days and change if an
    # archive grows. The enrolled eligible source set, all pairs, their
    # rankings and the selected policy MUST remain reproducible.
    for field in (
        "candidate_fixtures", "qualified_four_team_pairs", "evaluated_pairs",
        "source_receipts_sha256", "candidate_catalog_sha256", "candidate_catalog",
        "decision", "reason", "selected_pair",
    ):
        if recreated.get(field) != packet.get(field):
            raise ValueError("candidate_catalog_or_rank_changed_" + field)
    # A selected V27 fixture must have been selected early enough when frozen.
    if packet["decision"] == "V27_SELECTED":
        # Use independently rechecked source timestamps.
        source_map = {(str(x["provider"]), str(x["fixture_id"])): x for x in eligible}
        for key in packet["selected_pair"]["fixture_keys"]:
            if decision_at > utc(source_map[tuple(key)]["kickoff_utc"]) - MIN_LEAD:
                raise ValueError("v27_selected_after_lead_time")
    return original, sources, recreated


def _grade(picks, actual):
    wanted = tuple(f"{h}-{a}" for h, a in actual)
    return any((x["score1"], x["score2"]) == wanted for x in picks)


def evaluate(*, root=SOURCE_ROOT, prematch=Path("data/prematch"),
             sets=V25_SETS, out=OUT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, rejected = index_verified_results(root, prematch, clock)
    rows, bad = [], Counter()
    for file in sorted((Path(out)/"decisions").glob("*.json")):
        try:
            packet = json.loads(file.read_text(encoding="utf-8"))
            original, v25_sources, _ = _verify_decision(packet, file, root, sets, clock)
            row = {"sydney_date": file.stem, "decision": packet["decision"],
                   "candidate_fixtures": packet["candidate_fixtures"],
                   "eligible_pairs": packet["qualified_four_team_pairs"],
                   "evaluated_pairs": packet["evaluated_pairs"],
                   "model_mass_is_not_measured_accuracy": True}
            if packet["decision"] == "V27_ABSTAIN":
                row.update({"status": "abstained", "reason": packet["reason"]})
                rows.append(row)
                continue
            chosen = packet["selected_pair"]
            selected_sources = []
            for provider, fixture_id in chosen["fixture_keys"]:
                matching = next(r for r in packet["eligible_original_source_receipts"]
                                if r["provider"] == provider and r["fixture_id"] == fixture_id)
                source, _ = _source(root, matching, utc(packet["v25_decision_at_utc"]), file.stem)
                selected_sources.append(source)
            def outcomes(models):
                return [_final(m, finals.get((str(m["provider"]),
                                              str(m["fixture_id"]))), clock)
                        for m in models]
            v27_actual = outcomes(selected_sources)
            v25_actual = outcomes(v25_sources)
            v27_complete = all(x is not None for x in v27_actual)
            v25_complete = all(x is not None for x in v25_actual)
            same_pair = _v27_keys(chosen) == _v25_keys(original)
            row.update({
                "status": ("verified_day_comparison" if v25_complete and v27_complete
                           else "pending"),
                "v27_pair_status": "verified" if v27_complete else "pending",
                "v25_pair_status": "verified" if v25_complete else "pending",
                "same_two_fixtures_as_v25": same_pair,
                "v27_model_rank_score_not_hit_probability": chosen["rank_quality"],
                "v27_frozen_pair_ids": chosen["fixture_keys"],
                "v25_frozen_pair_ids": _v25_keys(original),
            })
            if v27_complete:
                row.update({
                    "v27_verified_scores": [f"{h}-{a}" for h, a in v27_actual],
                    "v27_robust_five_hit": _grade(chosen["five_robust"], v27_actual),
                    "v27_v8_five_hit": _grade(chosen["five_v8_optimal"], v27_actual),
                })
            if v25_complete:
                row.update({
                    "v25_verified_scores": [f"{h}-{a}" for h, a in v25_actual],
                    "v25_original_robust_five_hit": _grade(
                        original["five_doubles"], v25_actual),
                })
            rows.append(row)
        except (ValueError, KeyError, TypeError, IndexError, StopIteration, OSError,
                OverflowError) as exc:
            bad[str(exc) if isinstance(exc, ValueError)
                else type(exc).__name__] += 1
    same_day = [r for r in rows if r["status"] == "verified_day_comparison"]
    v27_finished = [r for r in rows if r.get("v27_pair_status") == "verified"]
    paired_same_fixtures = [r for r in same_day if r["same_two_fixtures_as_v25"]]
    win_count = lambda arm: sum(bool(r[arm]) for r in same_day)
    wins = {
        "v25_original_robust": win_count("v25_original_robust_five_hit"),
        "v27_selected_robust": win_count("v27_robust_five_hit"),
        "v27_selected_v8": win_count("v27_v8_five_hit"),
    }
    report = {
        "schema_version": 1, "version": VERSION,
        "study": "prospective_same_day_pair_selection_policy_vs_original_v25",
        "evaluated_at_utc": clock.isoformat(),
        "frozen_decision_days": len(rows),
        "abstained_days": sum(r["status"] == "abstained" for r in rows),
        "v27_verified_pair_results": len(v27_finished),
        "v27_pending_pair_results": sum(
            r.get("v27_pair_status") == "pending" for r in rows),
        "both_policies_verified_same_days": len(same_day),
        "same_fixture_pair_comparisons": len(paired_same_fixtures),
        "different_fixture_pairs_same_day": len(same_day)-len(paired_same_fixtures),
        "wins_on_same_completed_days": wins,
        "v27_wilson_95_same_days": wilson(wins["v27_selected_robust"], len(same_day)),
        "v25_wilson_95_same_days": wilson(wins["v25_original_robust"], len(same_day)),
        "invalid_or_tampered_decisions_excluded": dict(bad),
        "verified_score_conflicts_excluded": len(conflicts),
        "invalid_verified_results_excluded": dict(rejected),
        "rows": rows,
        "comparability_warning": (
            "The same-day policies may select DIFFERENT match fixtures. "
            "Compare day-level selection success, NOT same-match probability "
            "calibration or a controlled same-fixture prediction experiment."
        ),
        "no_lookahead_or_retrospective_selection": True,
        "no_live_model_promotion": True,
        "70_percent_five_double_accuracy_proven": False,
    }
    return report


def run(*, root=SOURCE_ROOT, prematch=Path("data/prematch"), sets=V25_SETS,
        out=OUT, now=None, verify_only=False, originals=None):
    clock = utc(now or datetime.now(timezone.utc))
    frozen = ({"new_decisions": [], "not_created": {}, "version": VERSION,
               "live_predictions_unchanged": True} if verify_only
              else freeze(root=root, sets=sets, out=out,
                          now=clock, originals=originals))
    report = evaluate(root=root, prematch=prematch, sets=sets, out=out, now=clock)
    payload = {
        "version": VERSION, "evaluated_at_utc": clock.isoformat(),
        "prospective_freeze": frozen, "prospective_comparison": report,
        "original_frozen_predictions_modified": False,
        "no_new_external_requests": True,
        "production_model_promoted": False,
    }
    file = Path(out)/"status.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if not file.exists() or file.read_text(encoding="utf-8") != data:
        file.write_text(data, encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = run(verify_only=args.verify_only)
    summary = output["prospective_comparison"]
    print(json.dumps({
        "new_decisions": output["prospective_freeze"]["new_decisions"],
        "verified_v27_pairs": summary["v27_verified_pair_results"],
        "paired_verified_days": summary["both_policies_verified_same_days"],
        "v27_wins": summary["wins_on_same_completed_days"]["v27_selected_robust"],
        "v25_wins_on_same_days": summary["wins_on_same_completed_days"]["v25_original_robust"],
        "invalid_or_tampered_excluded": summary["invalid_or_tampered_decisions_excluded"],
        "model_promoted": False,
    }, sort_keys=True))
