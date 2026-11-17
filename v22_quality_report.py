"""V22 accuracy and source completeness board, strict no-hindsight status.

This report is regenerated from already frozen forecasts, original verified
scores and genuinely captured AI/provider packets. Never infers provider
success from API keys, pending match losses, or model code test success.
"""
import json
from datetime import datetime,timezone
from pathlib import Path

from persist_prematch import load_json,write_json
from v10_prospective import ROOT as V8_ROOT,read_latest
from v22_ai_gate import main as ai_gate
from v22_calibration import chronological_evaluation

ROOT=Path("data/v22_prospective")
SOURCE_ROOT=Path("data/v21_sources")


def report(*,original_root=V8_ROOT,root=ROOT,source_root=SOURCE_ROOT):
    originals=list(read_latest(original_root).values())
    verified=load_json(Path(original_root)/"verified_results.json",{})
    calibration=chronological_evaluation(originals,verified)
    ai_results_path=Path(root)/"ai_status.json"
    providers=load_json(ai_results_path,{})
    prospective=load_json(Path(root)/"performance.json",{})
    v23_report=load_json(Path("data/v23_prospective")/"performance.json",{})
    v23_status=load_json(Path("data/v23_prospective")/"status.json",{})
    source_files=(list(Path(source_root).glob("API-Football_*.json"))+
                  list((Path(source_root)/"versions").glob("API-Football_*.json")))
    status=load_json(Path(source_root)/"status.json",{})
    early=load_json(Path(source_root)/"market_early_status.json",{})
    wide=load_json(Path("data/v24_market_catalog")/"status.json",{})
    extracted={"packets_checked":len(source_files),
               "with_historical_xg":0,"with_6_per_team":0,
               "with_verified_historical_match_statistics":0,
               "with_stats_but_no_genuine_xg":0,
               "with_independent_provider_prediction":0,
               "with_reported_injuries":0,
               "with_2_market_types":0,"with_both_starting_XIs":0}
    for path in source_files:
        row=load_json(path,{})
        snap=row.get("provider_snapshot") or {}
        xg=snap.get("verified_historical_fixture_xg") or []
        if xg:
            extracted["with_historical_xg"]+=1
        raw_stats=snap.get("verified_historical_match_statistics") or []
        if raw_stats:
            extracted["with_verified_historical_match_statistics"]+=1
            if not xg:
                extracted["with_stats_but_no_genuine_xg"]+=1
        if snap.get("provider_pre_match_opinion"):
            extracted["with_independent_provider_prediction"]+=1
        if snap.get("reported_injuries"):
            extracted["with_reported_injuries"]+=1
        ids=[str((row.get("source_forecast") or {}).get(k))
             for k in ("home_team_id","away_team_id")]
        count={tid:len({r.get("fixture_id") for r in xg if
             tid in (str(r.get("home_team_id")),str(r.get("away_team_id")))})
             for tid in ids}
        if ids[0] and ids[1] and min(count.values() or [0])>=6:
            extracted["with_6_per_team"]+=1
        if len(snap.get("multi_market") or {})>=2:
            extracted["with_2_market_types"]+=1
        if len(snap.get("confirmed_starting_xi") or [])==2:
            extracted["with_both_starting_XIs"]+=1
    stats=(prospective.get("methods") or {})
    return {
        "schema":1,"checked_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_completeness":extracted,
        "source_capture_enabled":status.get("enabled") if status else None,
        "source_capture_last_run":status.get("at_utc") if status else None,
        "source_capture_reason":status.get("reason") if status else None,
        "source_capture_eligible_legs":status.get("eligible_legs") if status else None,
        "source_capture_attempted_legs":status.get("attempted_legs") if status else None,
        "source_capture_saved_packets":len(status.get("new_packets") or []),
        "source_capture_source_queries":status.get("calls") if status else None,
        "source_capture_skipped_reasons":status.get("skipped") or [],
        "source_capture_daily_attempts_used":status.get("daily_attempts_used") if status else None,
        "source_capture_daily_attempts_limit":status.get("daily_attempts_limit") if status else None,
        "v23_early_market_capture_enabled":early.get("enabled") if early else None,
        "v23_early_market_last_run":early.get("at_utc") if early else None,
        "v23_early_market_new_packets":len(early.get("new_packets") or []),
        "v24_broad_market_collection":{
            "enabled":wide.get("enabled") if wide else None,
            "last_run_utc":wide.get("at_utc") if wide else None,
            "queries_last_run":wide.get("queries",0),
            "requests_used_today":wide.get("requests_used_today",0),
            "max_requests_per_day":wide.get("daily_limit"),
            "new_fixture_markets_last_run":len(wide.get("new_catalog_fixtures") or []),
            "new_v8_source_bindings_last_run":len(wide.get("new_source_bindings") or []),
            "not_all_external_web_sources":True},
        "frozen_v22_vs_v8":stats,
        "v23_market_form_vs_v8":v23_report.get("methods") or {},
        "v23_source_qualified_pairs":v23_status.get("qualifying_pairs",0),
        "v23_not_observed_success_without_verified_pairs":not bool(
            ((v23_report.get("methods") or {}).get("v23_market_form") or {})
            .get("verified")),
        "calibration_prequential_by_league":calibration["leagues"],
        "matched_ai_provider_results":providers.get("provider_metrics") or {},
        "model_promoted":False,"ai_weight_applied":0.0,
        "required_for_promotion":{
            "same_paired_forward_doubles_minimum":100,
            "same_fixtures_robust_score_log_loss":True,
            "positive_calibration_out_of_sample":True,
            "source_availability_and_bias_review":True,
            "owner_approval_and_independent_holdout":True},
        "missing_records_are_not_wins_or_losses":True,
        "observed_v22_accuracy_unavailable_if_no_verified_pairs":(
            not bool((stats.get("v22_research") or {}).get("verified"))),
        "note":"Training/retrospective figures do not prove five-double hit improvement."}


def main(*,root=ROOT):
    ai_gate(out=Path(root)/"ai_status.json")
    status=report(root=root)
    write_json(Path(root)/"quality_report.json",status)
    return {"source_packets":status["source_completeness"]["packets_checked"],
            "v22_verified":(status["frozen_v22_vs_v8"].get("v22_research") or {}).get("verified",0),
            "ai_provider_count":len(status["matched_ai_provider_results"]),
            "production_promotion":False}


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
