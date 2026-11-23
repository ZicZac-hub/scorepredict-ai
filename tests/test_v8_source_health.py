"""Read-only, time-credible source capture diagnostics.

These are collector coverage counts, never an evaluation of finished tickets.
"""
from datetime import datetime, timedelta, timezone

from continuous_accuracy_bot import assemble, source_capture_health


NOW = datetime(2035, 3, 1, 12, tzinfo=timezone.utc)


def coverage(started=None):
    result = {
        "processed_fixtures":23,
        "v10_shadow_v8_saved":1,
        "v10_shadow_skipped":{"team_history_below_five":12,
                              "league_history_below_thirty_five":10},
        "v10_shadow_skipped_by_league":{
            "PL":{"team_history_below_five":6},
            "PD":{"team_history_below_five":6,
                  "league_history_below_thirty_five":10},
        },
        "v10_shadow_v8_saved_by_league":{"PL":1},
        "v10_shadow_errors":[],
    }
    if started is not None:
        result["collector_started_at_utc"]=started
    return result


def test_missing_legacy_timestamp_does_not_claim_fresh_collection():
    health=source_capture_health(coverage(),now=NOW)
    assert health["collector_freshness"]=="unknown_missing_capture_timestamp"
    assert health["collector_started_at_utc"] is None
    assert health["age_minutes"] is None
    assert health["processed_fixtures"]==23
    assert health["v8_saved"]==1
    assert health["v8_abstentions"]==22
    assert sum(sum(x.values()) for x in
               health["v8_abstention_reasons_by_league"].values())==22
    assert health["not_a_prediction_accuracy_measure"]


def test_fresh_stale_and_future_collector_timestamp_are_distinguished():
    fresh=source_capture_health(
        coverage((NOW-timedelta(minutes=22)).isoformat()),now=NOW)
    assert fresh["collector_freshness"]=="recent"
    assert fresh["age_minutes"]==22.0
    stale=source_capture_health(
        coverage((NOW-timedelta(hours=4)).isoformat()),now=NOW)
    assert stale["collector_freshness"]=="stale"
    future=source_capture_health(
        coverage((NOW+timedelta(minutes=40)).isoformat()),now=NOW)
    assert future["collector_freshness"]=="invalid_future_capture_timestamp"
    assert future["age_minutes"] is None
    naive=source_capture_health(coverage("2035-03-01T11:30:00"),now=NOW)
    assert naive["collector_freshness"]=="invalid_capture_timestamp"


def test_no_capture_file_does_not_invent_successes():
    health=source_capture_health(None,now=NOW)
    assert health["v8_saved"]==0
    assert health["v8_abstentions"]==0
    assert health["collector_freshness"]=="unknown_missing_capture_timestamp"


def test_readiness_counts_cannot_change_five_double_denominator():
    status=assemble(None,None,None,None,None,now=NOW,coverage=coverage())
    assert status["source_capture_health"]["v8_abstentions"]==22
    assert status["original_v8_daily_five_doubles"]["verified_sets"]==0
    assert status["goal_verified"] is False
    assert status["production_model_changed_by_bot"] is False


def test_capture_instrumentation_is_separate_from_provider_budget():
    from pathlib import Path
    src=Path("capture_prematch.py").read_text(encoding="utf-8")
    assert "'collector_started_at_utc': started.isoformat()" in src
    assert "'v10_shadow_skipped_by_league':{}" in src
    assert "league_reasons[reason]=league_reasons.get(reason,0)+1" in src
    assert "max_api_calls=_positive_limit" in src
    assert "v10_shadow_v8_saved_by_league" in src


def test_research_priority_tracks_source_readiness_not_model_hit_rate():
    unknown=assemble(None,None,None,None,None,now=NOW,coverage=coverage())
    assert "collector" in unknown["next_priority"]
    fresh=assemble(
        None,None,None,None,None,now=NOW,
        coverage=coverage((NOW-timedelta(minutes=20)).isoformat()))
    assert "league-specific" in fresh["next_priority"]
    assert not fresh["goal_verified"]


def test_source_health_reports_verified_team_history_readiness_without_hits():
    payload=coverage((NOW-timedelta(minutes=10)).isoformat())
    payload["v8_min_team_history_bins_by_league"]={
        "CL":{"0":4,"1-2":3,"3-4":3},
        "EL":{"0":2,"1-2":8,"3-4":2},
    }
    payload["v8_qualified_home_history_bins_by_league"]={
        "CL":{"3-4":10},"EL":{"5+":12},
    }
    payload["v8_qualified_away_history_bins_by_league"]={
        "CL":{"0":4,"1-2":6},"EL":{"0":2,"3-4":10},
    }
    summary=source_capture_health(payload,now=NOW)
    assert summary["collector_freshness"]=="recent"
    assert summary["v8_min_team_history_bins_by_league"]["CL"]["3-4"]==3
    assert sum(sum(x.values()) for x in
        summary["v8_min_team_history_bins_by_league"].values())==22
    assert summary["v8_home_history_bins_by_league"]["EL"]["5+"]==12
    assert summary["v8_away_history_bins_by_league"]["CL"]["0"]==4
    bot=assemble(None,None,None,None,None,now=NOW,coverage=payload)
    assert bot["source_capture_health"]["v8_abstentions"]==22
    assert bot["original_v8_daily_five_doubles"]["verified_sets"]==0
    assert bot["goal_verified"] is False


def test_source_readiness_bins_do_not_change_provider_request_budget():
    from pathlib import Path
    collector=Path("capture_prematch.py").read_text(encoding="utf-8")
    assert "team_history_band(readiness.get(field))" in collector
    assert "'v8_min_team_history_bins_by_league':{}" in collector
    assert "max_api_calls=_positive_limit" in collector
    assert "result.get('team_history_readiness')" in collector
    assert "provider_adapter_calls" in collector
