"""Strict historical Sportmonks xG provenance and time-leakage regressions."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from sportmonks_xg_validation import validate_packet, verify_packet_collection


NOW = datetime(2032, 8, 16, 10, tzinfo=timezone.utc)
KICK = NOW + timedelta(hours=20)


def frozen():
    return {
        "provider":"API-Football", "model_key":"v8",
        "fixture_id":"321", "forecast_id":"frozen-abc",
        "input_digest":"immutable-evidence-123",
        "home_team":"Home FC", "away_team":"Away United",
        "captured_at_utc":(NOW-timedelta(hours=1)).isoformat(),
        "kickoff_utc":KICK.isoformat(),
    }


def packet():
    hist = NOW-timedelta(days=4)
    return {
        "provider":"Sportmonks", "forecast_provider":"API-Football",
        "research_only":True, "not_automatically_promoted":True,
        "source_forecast_id":"frozen-abc",
        "source_input_digest":"immutable-evidence-123",
        "api_football_fixture_id":"321",
        "sportmonks_fixture_id":"54321",
        "sportmonks_home_team_id":"31",
        "sportmonks_away_team_id":"32",
        "home_team":"Home FC", "away_team":"Away United",
        "kickoff_utc":KICK.isoformat(),
        "captured_at_utc":NOW.isoformat(),
        "xg_type_id":5304,
        "historical_xg":[
            {"sportmonks_fixture_id":"900", "kickoff_utc":hist.isoformat(),
             "source":"Sportmonks", "type_id":5304,
             "home_team_id":"31","away_team_id":"80",
             "home_xg":1.4, "away_xg":0.8,
             "available_before_target_kickoff":True},
            {"sportmonks_fixture_id":"901",
             "kickoff_utc":(hist+timedelta(hours=3)).isoformat(),
             "source":"Sportmonks","type_id":5304,
             "home_team_id":"70","away_team_id":"32",
             "home_xg":0.6, "away_xg":1.9,
             "available_before_target_kickoff":True},
        ],
        "historical_xg_home_count":1,
        "historical_xg_away_count":1,
        "both_teams_six_xg":False,
    }


def test_valid_genuine_history_requires_both_teams_from_same_original():
    check=validate_packet(packet(),[frozen()])
    assert check["valid"]
    assert check["home_history"]==1
    assert check["away_history"]==1
    assert check["both_teams_have_xg"] is True
    audit=verify_packet_collection([packet()],[frozen()])
    assert audit["source_validated"]==1
    assert audit["with_both_teams"]==1
    assert audit["with_six_per_team"]==0
    assert audit["no_proven_incremental_score_accuracy"]


def test_wrong_forecast_digest_and_provider_ids_cannot_fake_source():
    bad=packet()
    bad["source_input_digest"]="wrong-digest"
    assert validate_packet(bad,[frozen()])["reason"]=="original_identity_not_verified"
    bad=packet()
    bad["sportmonks_away_team_id"]="31"
    assert validate_packet(bad,[frozen()])["reason"]=="invalid_source_ids"
    bad=packet()
    bad["api_football_fixture_id"]="322"
    assert not validate_packet(bad,[frozen()])["valid"]


def test_xg_type_and_values_must_be_exact_and_finite():
    for value in (-1, float("nan"), float("inf"), True, "1.7", 16):
        bad=packet()
        bad["historical_xg"][0]["home_xg"]=value
        assert validate_packet(bad,[frozen()])["reason"]=="invalid_historical_xg_values"
    bad=packet()
    bad["historical_xg"][1]["type_id"]=5305
    assert validate_packet(bad,[frozen()])["reason"]=="unverified_historical_xg_source"


def test_no_late_xg_data_can_enter_pre_match_research():
    bad=packet()
    bad["historical_xg"][0]["kickoff_utc"]=(NOW-timedelta(hours=1)).isoformat()
    assert validate_packet(bad,[frozen()])["reason"]=="historical_match_not_finished_by_capture"
    bad=packet()
    bad["captured_at_utc"]=(KICK+timedelta(seconds=1)).isoformat()
    assert validate_packet(bad,[frozen()])["reason"]=="invalid_pre_match_timing"
    bad=packet()
    bad["historical_xg"][0]["available_before_target_kickoff"]=False
    assert validate_packet(bad,[frozen()])["reason"]=="unverified_historical_xg_source"


def test_duplicate_and_unrelated_fixture_data_fail_closed():
    bad=packet()
    bad["historical_xg"][1]["sportmonks_fixture_id"]="900"
    assert validate_packet(bad,[frozen()])["reason"]=="invalid_historical_ids"
    bad=packet()
    for item in bad["historical_xg"]:
        item["home_team_id"]="75"
        item["away_team_id"]="80"
    assert validate_packet(bad,[frozen()])["reason"]=="history_unrelated_to_selected_teams"


def test_partial_history_is_not_counted_as_both_team_xg():
    partial=packet()
    partial["historical_xg"]=partial["historical_xg"][:1]
    partial["historical_xg_away_count"]=0
    check=validate_packet(partial,[frozen()])
    assert check["valid"] is True
    assert check["home_history"]==1
    assert check["away_history"]==0
    assert check["both_teams_have_xg"] is False
    incorrect=deepcopy(partial)
    incorrect["historical_xg_away_count"]=1
    assert validate_packet(incorrect,[frozen()])["reason"]=="history_counts_disagree"


def test_invalid_packet_is_excluded_from_aggregate_coverage():
    bad=packet()
    bad["source_forecast_id"]="post-hoc-reassigned"
    audit=verify_packet_collection([packet(),bad], [frozen()])
    assert audit["checked"]==2
    assert audit["source_validated"]==1
    assert audit["with_both_teams"]==1
    assert audit["rejected_by_reason"]["original_identity_not_verified"]==1
