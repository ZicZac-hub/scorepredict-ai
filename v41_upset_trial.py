"""Separate V41 upset-awareness prospective matched-control trial.

Use identical authenticated original pair and one common evidence cutoff.
The control is the unchanged V41 policy at that cutoff, not a historical
V41 ticket reconstructed after kickoff. No promotion or forced upset picks.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from five_double_accuracy import _final, _result_index, wilson
from professional_double_optimizer import optimal_five
from professional_double_shadow import ROOT as PAIR_ROOT, _validated as validate_pair
from source_capture_clock import capture_clock
from v10_prospective import ROOT as SOURCE_ROOT, utc
from v21_capture_sources import qualified_packet, source_packet_path
from v22_evidence_models import calibrate_markets, combine_goal_evidence, lineup_status, xg_profile
from v22_prospective import _fixture
from v39_poisson_marginal import transform_score_grid
from v41_unified_ensemble import WEIGHTS, core, final_pair
from v41_upset_aware import POLICY as UPSET_POLICY, adjust

ROOT = Path("data/v41_upset_aware")
PROVIDER_ROOT = Path("data/v21_sources")
PREMATCH_ROOT = Path("data/prematch")
VERSION = "v41-fixed-pair-upset-aware-v1"
METHODS = ("unified_final", "upset_aware")
PROTOCOL = {
    "version": VERSION,
    "pair_selection": "existing_authenticated_professional_raw_baseline_set",
    "decision_rule": "first_scheduled_pass_5_to_60_minutes_before_earliest_kickoff",
    "minimum_lead_seconds_exclusive": 300,
    "maximum_lead_seconds_inclusive": 3600,
    "maximum_component_age_seconds": 10800,
    "maximum_market_quote_age_seconds": 43200,
    "component_selection": "newest_usable_same_source_packet_at_common_cutoff_then_snapshot_hash",
    "methods": list(METHODS), "minimum_xg_matches_per_team": 6,
    "minimum_xg_venue_matches_per_team": 3,
    "historical_xg_availability": "all_packet_xg_fixtures_kickoff_plus_3h_before_capture",
    "xg_blend_weight": .25, "market_strength": .35, "market_max_total_variation": .15,
    "fixed_family_weights": WEIGHTS,
    "evidence_policy": "xg_and_market_each_used_only_if_qualified_on_both_legs",
    "comparison_control": "unified_final", "upset_policy": UPSET_POLICY,
    "no_forced_opposite_ticket": True,
    "lineups_and_ai_context_only": True,
    "minimum_review_verified_pairs": 100,
    "numerical_lineup_weight": 0., "numerical_ai_weight": 0.,
    "refit_or_threshold_tuning": False,
    "missing_arms": "abstain_preserve_baseline_opportunity_and_report_matched_overlap",
    "single_score_log_loss_floor": 1e-15,
    "primary_comparison": "paired_same_event_single_exact_score_log_loss",
    "secondary_comparison": "paired_same_event_1x2_brier",
    "set_brier_comparison": "descriptive_different_five_score_events_not_proper_score_superiority",
    "target_70_percent_proven": False, "automatic_model_promotion": False,
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
        return True
    except FileExistsError:
        return False


def _implementation():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here/name).read_bytes()).hexdigest() for name in
            ("v41_upset_trial.py", "v41_upset_aware.py", "v41_unified_ensemble.py", "v35_continuous_double_trials.py",
             "v36_tail_calibration_research.py", "v39_poisson_marginal.py", "v38_margin_calibration.py",
             "v21_goal_model_comparison.py", "professional_double_optimizer.py", "v22_evidence_models.py",
             "v21_provider_evidence.py", "professional_double_shadow.py", "five_double_accuracy.py",
             "forecast_ledger.py", "v21_capture_sources.py", "v22_prospective.py")}


def register_protocol(*, root=ROOT, now=None):
    """Persist fixed settings/code hashes once; publication time is external."""
    path = Path(root)/"protocol.json"
    clock = utc(now or datetime.now(timezone.utc))
    record = {"protocol": PROTOCOL, "implementation_sha256": _implementation(),
              "registered_at_utc": clock.isoformat()}
    record["protocol_sha256"] = _hash(record)
    _write_once(path, record)
    return _read_protocol(root=root, now=clock)


def _read_protocol(*, root, now):
    """Authenticate an existing protocol without writing from evaluation/UI."""
    existing = _load(Path(root)/"protocol.json")
    if (existing.get("protocol") != PROTOCOL
            or existing.get("implementation_sha256") != _implementation()
            or existing.get("protocol_sha256") != _hash({k: v for k, v in existing.items()
                                                         if k != "protocol_sha256"})):
        raise ValueError("fixed_protocol_or_implementation_changed_use_new_study_version")
    if utc(existing["registered_at_utc"]) > utc(now):
        raise ValueError("protocol_not_registered_at_decision")
    return existing


def _markets(packet, cutoff):
    accepted = {}
    captured = utc(packet["captured_at_utc"])
    for name, market in (packet.get("multi_market") or {}).items():
        if name not in ("1x2", "over_2_5", "over_3_5", "btts_yes") or not isinstance(market, dict):
            continue
        try:
            updated = utc(market["last_source_update_utc"])
            if not updated <= captured <= cutoff or cutoff-updated > timedelta(hours=12):
                continue
            probabilities = market["probabilities"]
            required = {"home", "draw", "away"} if name == "1x2" else {"yes", "no"}
            if (set(probabilities) != required or
                    any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
                        for p in probabilities.values()) or
                    abs(math.fsum(probabilities.values())-1) > 1e-6):
                continue
            accepted[name] = market
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    if "1x2" not in accepted or not ({"over_2_5", "over_3_5"} & set(accepted)):
        return {}
    return accepted


def _components(model, *, provider_root, cutoff):
    """Never merge packets or relabel an old observation as fresh evidence."""
    root = Path(provider_root)
    base = source_packet_path(root, model)
    paths = [base, *sorted((root/"versions").glob(base.stem+"_*.json"))]
    found = {kind: [] for kind in ("market", "xg", "lineup")}
    fixture = _fixture(model)
    for path in paths:
        try:
            row = _load(path)
            if not qualified_packet(row, model, now=cutoff):
                continue
            packet = row["provider_snapshot"]
            captured = utc(packet["captured_at_utc"])
            if not timedelta(0) <= cutoff-captured <= timedelta(hours=3):
                continue
            ref = {"path": path.relative_to(root).as_posix(),
                   "snapshot_sha256": packet["snapshot_sha256"],
                   "captured_at_utc": packet["captured_at_utc"]}
            order = (captured, packet["snapshot_sha256"])
            market = _markets(packet, cutoff)
            if market:
                found["market"].append((order, ref, market))
            past_xg = packet.get("verified_historical_fixture_xg") or []
            historical = bool(past_xg) and all(
                utc(row["kickoff_utc"])+timedelta(hours=3) <= captured for row in past_xg)
            xg = xg_profile(fixture, packet, as_of=cutoff, minimum=6) if historical else {}
            if xg.get("ready"):
                found["xg"].append((order, ref, xg))
            lineups = lineup_status(fixture, packet, as_of=cutoff)
            if lineups.get("teams"):
                found["lineup"].append((order, ref, lineups))
        except (ValueError, KeyError, TypeError, OSError, AttributeError):
            continue
    selected = {kind: max(rows, key=lambda r: r[0]) if rows else None for kind, rows in found.items()}
    return {kind: {"reference": value[1], "value": value[2]} if value else None
            for kind, value in selected.items()}


def _arms(models, bundles):
    baseline = [np.asarray(model["score_matrix"], dtype=float) for model in models]
    try:
        matrices, used = final_pair(baseline,bundles)
        adjustments = [adjust(g) for g in matrices]
    except (ValueError,TypeError,OverflowError):
        return {name:{"status":"abstained","reasons":["distribution_transform_unavailable"]}
                for name in METHODS}
    arms={}
    for name, grids in (("unified_final", matrices),
                        ("upset_aware",[g for g,_ in adjustments])):
        choices=optimal_five(*grids)
        if len(choices)!=5 or len({(p["score1"],p["score2"]) for p in choices})!=5:
            raise ValueError("five_distinct_supported_outcomes_required")
        arms[name]={"status":"issued","full_score_grids":[g.tolist() for g in grids],
                    "five_doubles":choices,"estimated_coverage":math.fsum(p["joint_probability"] for p in choices),
                    "paired_xg_used":used["xg"],"paired_market_used":used["market"]}
    arms["upset_aware"]["adjustments"]=[d for _,d in adjustments]
    return arms


def _references(bundles):
    return [{kind: component["reference"] if component else None
             for kind, component in bundle.items()} for bundle in bundles]


def _context(bundles):
    return [{"both_confirmed_xis": bool(bundle["lineup"] and bundle["lineup"]["value"].get("ready")),
             "reported_lineup_teams": len(bundle["lineup"]["value"]["teams"]) if bundle["lineup"] else 0,
             "numerical_lineup_weight": 0., "numerical_ai_weight": 0.,
             "missing_does_not_mean_healthy": True} for bundle in bundles]


def _pair(path, *, source_root, decision):
    record = _load(path)
    models, _, _ = validate_pair(record, source_root=source_root, now=decision)
    if path.stem != record["sydney_date"]:
        raise ValueError("baseline_pair_filename_mismatch")
    return record, models


def _window(models, decision):
    lead = (min(utc(m["kickoff_utc"]) for m in models)-decision).total_seconds()
    return 300 < lead <= 3600


def freeze(*, root=ROOT, pair_root=PAIR_ROOT, source_root=SOURCE_ROOT,
           provider_root=PROVIDER_ROOT, now=None, clock=None):
    started, observed = capture_clock(now, clock)
    protocol = register_protocol(root=root, now=started)
    new, preserved, rejected, outside = [], [], Counter(), 0
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            pair, models = _pair(path, source_root=source_root, decision=started)
            day = pair["sydney_date"]
            destination = Path(root)/"trials"/(day+".json")
            if destination.exists():
                preserved.append(day)
                continue
            if not _window(models, started):
                outside += 1
                continue
            bundles = [_components(m, provider_root=provider_root, cutoff=started) for m in models]
            arms = _arms(models, bundles)
            completed = observed()
            if not _window(models, completed):
                raise ValueError("calculation_finished_outside_pre_match_window")
            record = {"schema_version": 1, "version": VERSION, "status": "frozen_pre_kickoff",
                      "sydney_date": day, "decision_as_of_utc": started.isoformat(),
                      "predicted_at_utc": completed.isoformat(),
                      "protocol_sha256": protocol["protocol_sha256"],
                      "baseline_pair": {"path": path.name, "record_sha256": pair["record_sha256"]},
                      "source_forecast_refs": pair["selection"]["source_forecast_refs"],
                      "model_versions": [m["model_version"] for m in models],
                      "baseline_captured_at_utc": [m["captured_at_utc"] for m in models],
                      "component_sources": _references(bundles), "lineup_context": _context(bundles),
                      "arms": arms, "external_requests": 0, "target_70_percent_proven": False,
                      "model_promoted": False,
                      "timing_note": "unchanged_earlier_baseline_versus_later_observed_evidence_at_one_common_cutoff"}
            record["record_sha256"] = _hash(record)
            if _write_once(destination, record):
                new.append(day)
            else:
                preserved.append(day)
        except (ValueError, KeyError, TypeError, OSError, AttributeError, OverflowError) as exc:
            rejected[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return {"version": VERSION, "started_at_utc": started.isoformat(),
            "completed_at_utc": observed().isoformat(), "new_trials": new,
            "existing_trials_preserved": preserved, "outside_decision_window": outside,
            "rejections": dict(rejected), "external_requests": 0,
            "target_70_percent_proven": False, "model_promoted": False}


def _same_calculation(saved, recomputed):
    """Compare numeric results with machine-portable absolute tolerance only."""
    if type(saved) in (int, float) and type(recomputed) in (int, float):
        return (math.isfinite(saved) and math.isfinite(recomputed)
                and math.isclose(saved, recomputed, rel_tol=0, abs_tol=1e-12))
    if type(saved) is not type(recomputed):
        return False
    if isinstance(saved, dict):
        return saved.keys() == recomputed.keys() and all(
            _same_calculation(saved[key], recomputed[key]) for key in saved)
    if isinstance(saved, list):
        return len(saved) == len(recomputed) and all(
            _same_calculation(a, b) for a, b in zip(saved, recomputed))
    return saved == recomputed


def _validated(record, *, root, pair_root, source_root, provider_root, now):
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or record.get("version") != VERSION or record.get("status") != "frozen_pre_kickoff"
            or record.get("record_sha256") != _hash({k: v for k, v in record.items()
                                                     if k != "record_sha256"})):
        raise ValueError("study_record_digest_or_policy_mismatch")
    predicted, decision = utc(record["predicted_at_utc"]), utc(record["decision_as_of_utc"])
    if not decision <= predicted <= now:
        raise ValueError("study_not_available_as_of")
    protocol = _read_protocol(root=root, now=decision)
    if record["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("study_protocol_mismatch")
    name = record["baseline_pair"]["path"]
    if name != record["sydney_date"]+".json" or Path(name).name != name:
        raise ValueError("unsafe_or_relabelled_baseline_pair_path")
    pair, models = _pair(Path(pair_root)/"sets"/name, source_root=source_root, decision=decision)
    if (pair["record_sha256"] != record["baseline_pair"]["record_sha256"]
            or record["source_forecast_refs"] != pair["selection"]["source_forecast_refs"]
            or record["model_versions"] != [m["model_version"] for m in models]
            or record["baseline_captured_at_utc"] != [m["captured_at_utc"] for m in models]):
        raise ValueError("baseline_pair_or_source_reference_changed")
    if not _window(models, decision) or not _window(models, predicted):
        raise ValueError("study_not_within_predeclared_decision_window")
    bundles = [_components(m, provider_root=provider_root, cutoff=decision) for m in models]
    if record["component_sources"] != _references(bundles):
        raise ValueError("component_sources_do_not_match_predeclared_cutoff")
    if record["lineup_context"] != _context(bundles) or not _same_calculation(record["arms"], _arms(models, bundles)):
        raise ValueError("study_arms_do_not_reproduce_from_original_sources")
    return models


def _metric(arm, actual):
    grid_metrics = []
    for raw, (h, a) in zip(arm["full_score_grids"], actual):
        grid = np.asarray(raw, dtype=float)
        probability = float(grid[h, a]) if h < grid.shape[0] and a < grid.shape[1] else 0.
        ih, ia = np.indices(grid.shape)
        outcome = [float(grid[ih > ia].sum()), float(grid[ih == ia].sum()), float(grid[ih < ia].sum())]
        observed = 0 if h > a else 1 if h == a else 2
        grid_metrics.append({"score_log_loss": -math.log(max(1e-15, probability)),
                             "outcome_brier": math.fsum((p-float(i == observed))**2
                                                       for i, p in enumerate(outcome))})
    strings = [f"{h}-{a}" for h, a in actual]
    hit = any(p["score1"] == strings[0] and p["score2"] == strings[1]
              for p in arm["five_doubles"])
    p = arm["estimated_coverage"]
    return {"hit": hit, "estimated_coverage": p, "set_brier_descriptive": (p-float(hit))**2,
            "score_log_loss": math.fsum(r["score_log_loss"] for r in grid_metrics)/2,
            "outcome_brier": math.fsum(r["outcome_brier"] for r in grid_metrics)/2}


def _mean_interval(values):
    if not values:
        return {"mean": None, "normal_95_interval": None, "sample_pairs": 0}
    mean, n = math.fsum(values)/len(values), len(values)
    interval = None
    if n > 1:
        sd = math.sqrt(math.fsum((x-mean)**2 for x in values)/(n-1))
        half = 1.96*sd/math.sqrt(n)
        interval = [mean-half, mean+half]
    return {"mean": mean, "normal_95_interval": interval, "sample_pairs": n}


def evaluate(*, root=ROOT, pair_root=PAIR_ROOT, source_root=SOURCE_ROOT,
             provider_root=PROVIDER_ROOT, prematch_root=PREMATCH_ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, result_rejections = _result_index(source_root, prematch_root)
    rejected, rows, seen_ids, seen_events = Counter(), [], set(), set()
    paths = sorted((Path(root)/"trials").glob("*.json"))
    for path in paths:
        try:
            record = _load(path)
            models = _validated(record, root=root, pair_root=pair_root, source_root=source_root,
                                provider_root=provider_root, now=clock)
            if path.stem != record["sydney_date"]:
                raise ValueError("study_filename_day_mismatch")
            ids = {(m["provider"], str(m["fixture_id"])) for m in models}
            events = {(utc(m["kickoff_utc"]).isoformat(),
                       " ".join(m["home_team"].casefold().split()),
                       " ".join(m["away_team"].casefold().split())) for m in models}
            if ids & seen_ids or events & seen_events:
                raise ValueError("repeated_fixture_or_cross_provider_event")
            seen_ids.update(ids)
            seen_events.update(events)
            actual = [_final(m, finals.get((m["provider"], str(m["fixture_id"]))), clock) for m in models]
            settled = all(score is not None for score in actual)
            methods = {}
            for name, arm in record["arms"].items():
                methods[name] = {"status": "abstained" if arm["status"] != "issued" else
                                 "verified" if settled else "pending", "metrics":
                                 _metric(arm, actual) if settled and arm["status"] == "issued" else None}
            rows.append({"sydney_date": record["sydney_date"], "status": "verified" if settled else "pending",
                         "actual_scores": [list(s) if s is not None else None for s in actual],
                         "methods": methods, "lineup_context": record["lineup_context"],
                         "upset_adjustments": record["arms"]["upset_aware"].get("adjustments", []),
                         "unified_final_evidence": {k:record["arms"]["unified_final"].get(k)
                             for k in ("paired_xg_used","paired_market_used")}})
        except (ValueError, KeyError, TypeError, OSError, AttributeError, OverflowError) as exc:
            rejected[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    summaries, paired = {}, {}
    for name in METHODS:
        verified = [row["methods"][name]["metrics"] for row in rows
                    if row["methods"][name]["status"] == "verified"]
        wins = sum(r["hit"] for r in verified)
        summaries[name] = {"opportunities": len(paths), "invalid_opportunities": len(paths)-len(rows), "issued": sum(row["methods"][name]["status"] != "abstained"
                                                                     for row in rows),
                           "abstained": sum(row["methods"][name]["status"] == "abstained" for row in rows),
                           "verified": len(verified), "hits": wins,
                           "pending": sum(row["methods"][name]["status"] == "pending" for row in rows),
                           "hit_rate": wins/len(verified) if verified else None,
                           "wilson_95": wilson(wins, len(verified)), "single_fixtures_evaluated": 2*len(verified),
                           **{metric: math.fsum(r[metric] for r in verified)/len(verified) if verified else None
                              for metric in ("score_log_loss", "outcome_brier", "set_brier_descriptive")}}
        summaries[name]["single_score_log_loss"] = summaries[name].pop("score_log_loss")
        if name == "unified_final":
            continue
        overlap = [row for row in rows if all(row["methods"][m]["status"] == "verified"
                                              for m in ("unified_final", name))]
        common = {}
        for method in ("unified_final", name):
            matched = [row["methods"][method]["metrics"] for row in overlap]
            hits = sum(row["hit"] for row in matched)
            common[method] = {"verified_sets": len(matched), "hits": hits,
                              "hit_rate": hits/len(matched) if matched else None,
                              "wilson_95": wilson(hits, len(matched)),
                              "single_score_log_loss": math.fsum(row["score_log_loss"] for row in matched)/len(matched) if matched else None,
                              "outcome_brier": math.fsum(row["outcome_brier"] for row in matched)/len(matched) if matched else None}
        paired[name] = {"verified_common_pairs": len(overlap), "same_pair_metrics": common,
                        "baseline_only_set_wins": sum(row["methods"]["unified_final"]["metrics"]["hit"] and
                                                      not row["methods"][name]["metrics"]["hit"] for row in overlap),
                        "challenger_only_set_wins": sum(row["methods"][name]["metrics"]["hit"] and
                                                        not row["methods"]["unified_final"]["metrics"]["hit"] for row in overlap),
                        "proper_same_event_score_log_loss_advantage": _mean_interval([
                            row["methods"]["unified_final"]["metrics"]["score_log_loss"]-
                            row["methods"][name]["metrics"]["score_log_loss"] for row in overlap]),
                        "proper_same_event_outcome_brier_advantage": _mean_interval([
                            row["methods"]["unified_final"]["metrics"]["outcome_brier"]-
                            row["methods"][name]["metrics"]["outcome_brier"] for row in overlap]),
                        "positive_advantage_means_lower_challenger_loss": True}
    return {"version": VERSION, "evaluated_at_utc": clock.isoformat(),
            "comparison_control": "unified_final", "minimum_review_verified_pairs": 100,
            "research_only": True, "no_forced_opposite_ticket": True,
            "no_automatic_promotion": True,
            "opportunities": len(paths), "invalid_opportunities": len(paths)-len(rows),
            "valid_opportunities": len(rows), "verified_opportunities": sum(row["status"] == "verified" for row in rows),
            "pending_opportunities": sum(row["status"] == "pending" for row in rows),
            "methods": summaries, "paired_comparisons": paired, "rows": rows,
            "lineup_coverage": {
                "both_matches_confirmed_at_decision": sum(all(leg["both_confirmed_xis"] for leg in row["lineup_context"]) for row in rows),
                "some_reported_lineup_at_decision": sum(any(leg["reported_lineup_teams"] > 0 for leg in row["lineup_context"]) for row in rows),
                "numerical_lineup_weight": 0.},
            "source_rejections": dict(rejected), "conflicting_result_identities": len(conflicts),
            "result_rejections": dict(result_rejections), "external_requests": 0,
            "target_70_percent_proven": False, "model_promoted": False, "realized_roi": None,
            "limitations": ["Control and challenger use identical V41 probability grids and evidence cutoff; only the prespecified upset adjustment differs.",
                            "Set Brier scores concern different five-score events and are descriptive; proper paired comparison uses the same actual score/outcome.",
                            "Wilson and normal intervals assume suitable independence; serial teams/leagues and repeated scheduled environments can violate it.",
                            "Missing arms remain abstentions; compare models only on explicitly reported common verified pairs.",
                            "Lineup context has zero numerical weight; no player-impact improvement or betting ROI is established."]}


def main(*, root=ROOT, pair_root=PAIR_ROOT, source_root=SOURCE_ROOT,
         provider_root=PROVIDER_ROOT, prematch_root=PREMATCH_ROOT, now=None):
    from persist_prematch import write_json
    status = freeze(root=root, pair_root=pair_root, source_root=source_root,
                    provider_root=provider_root, now=now)
    performance = evaluate(root=root, pair_root=pair_root, source_root=source_root,
                           provider_root=provider_root, prematch_root=prematch_root, now=now)
    write_json(Path(root)/"status.json", status)
    write_json(Path(root)/"performance.json", performance)
    return {"capture": status, "evaluation": {k: v for k, v in performance.items() if k != "rows"}}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
