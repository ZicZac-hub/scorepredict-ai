"""Professional dossier uses genuine as-of facts without inventing features."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from professional_match_evidence import build_dossier, feature_rows, validated_history

NOW = datetime(2031, 11, 20, 8, tzinfo=timezone.utc)
FIXTURE = {"provider": "API-Football", "id": 900, "utcDate": (NOW+timedelta(hours=4)).isoformat(),
           "competition": {"id": 71, "code": "BSA"},
           "homeTeam": {"id": 11, "name": "Home"}, "awayTeam": {"id": 12, "name": "Away"}}


def match(days, home, away, hg=2, ag=1):
    return {"id": days*100+home, "provider": "API-Football", "status": "FINISHED",
            "utcDate": (NOW-timedelta(days=days)).isoformat(), "competition": {"id": 71, "code": "BSA"},
            "homeTeam": {"id": home, "name": str(home)}, "awayTeam": {"id": away, "name": str(away)},
            "score": {"duration": "REGULAR", "fullTime": {"home": hg, "away": ag}}}


def seal(packet):
    packet = deepcopy(packet)
    packet.pop("snapshot_sha256", None)
    packet["snapshot_sha256"] = hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return packet


def packet():
    prior = match(1, 11, 33)
    return seal({"status": "captured_pre_kickoff", "provider": "API-Football",
                 "fixture_id": "900", "home_team_id": "11", "away_team_id": "12",
                 "kickoff_utc": FIXTURE["utcDate"], "captured_at_utc": (NOW-timedelta(minutes=1)).isoformat(),
                 "verified_historical_fixture_xg": [{"fixture_id": str(prior["id"]), "kickoff_utc": prior["utcDate"],
                    "home_team_id": "11", "away_team_id": "33", "home_xg": 1.8, "away_xg": .5}],
                 "confirmed_starting_xi": [{"team_id": "11", "formation": "4-3-3", "starting_xi": [str(i) for i in range(11)]}],
                 "reported_injuries": [{"team_id": "12", "player": "Player A", "reason": "Suspended"}],
                 "multi_market": {"1x2": {"last_source_update_utc": (NOW-timedelta(minutes=10)).isoformat(),
                                             "probabilities": {"home": .5, "draw": .3, "away": .2}}}})


def test_5_10_20_windows_are_actual_counts_and_provider_scoped():
    history = [match(n, 11, 30+n, 3, 1) for n in range(1, 21)]
    history += [match(n, 60+n, 12, 0, 2) for n in range(1, 8)]
    wrong = match(1, 11, 999, 9, 0); wrong["provider"] = "other"
    future = match(1, 11, 888, 9, 0); future["utcDate"] = (NOW+timedelta(hours=1)).isoformat()
    same_day = match(1, 11, 777, 9, 0); same_day["utcDate"] = (NOW-timedelta(hours=1)).isoformat()
    history += [wrong, future, same_day]
    dossier = build_dossier(FIXTURE, history, as_of=NOW, statistical={"home_xg": 8.0})
    form = dossier["features"]["goal_form"]["value"]
    assert [form["home"]["last"+str(n)]["matches"] for n in (5, 10, 20)] == [5, 10, 20]
    assert form["home"]["last20"]["goals_for_per_match"] == 3
    assert form["away"]["last20"]["matches"] == 7
    assert not form["away"]["last20"]["complete_window"]
    assert dossier["features"]["shot_xg_xga"]["status"] == "missing"
    assert dossier["model_expected_goals_not_observed_shot_xg"]["home"] == 8.0
    assert dossier["features"]["goal_form"]["source_observed_at_utc"] is None
    assert dossier["history_exclusions"]["result_not_available_as_of"] == 1
    assert dossier["external_calls_made"] == 0
    assert dossier["ai_numeric_weight"] == 0


def test_conflicting_results_excluded_and_late_fetch_does_not_backdate():
    one = match(2, 11, 44); conflict = deepcopy(one)
    conflict["score"]["fullTime"]["home"] = 8
    late = match(3, 11, 55); late["fetched_at_utc"] = (NOW+timedelta(seconds=1)).isoformat()
    rows, reasons = validated_history(FIXTURE, [one, conflict, late], as_of=NOW)
    assert rows == []
    assert reasons["conflicting_duplicate"] == 2
    assert reasons["source_observed_after_cutoff"] == 1


def test_regulation_time_and_timestamp_basis_are_explicit():
    row = match(1, 11, 44)
    row["score"] = {"duration": "EXTRA_TIME", "regularTime": {"home": 1, "away": 1},
                    "fullTime": {"home": 3, "away": 2}}
    rows, _ = validated_history(FIXTURE, [row], as_of=NOW)
    assert rows[0]["regulation_score"] == [1, 1]
    assert rows[0]["result_availability_basis"] == "conservative_kickoff_plus_3h"
    row["score"].pop("regularTime")
    assert validated_history(FIXTURE, [row], as_of=NOW)[0] == []


def test_packet_features_available_only_with_identity_hash_and_asof_validation():
    p = packet()
    dossier = build_dossier(FIXTURE, [match(1, 11, 33)], as_of=NOW, provider_packet=p)
    features = dossier["features"]
    assert features["shot_xg_xga"]["value"]["home"]["last5"]["xg_for_per_match"] == 1.8
    assert features["starting_lineups"]["status"] == "partial"
    assert features["formations"]["value"][0]["formation"] == "4-3-3"
    assert features["suspensions"]["value"][0]["player"] == "Player A"
    assert features["markets_1x2"]["available"]
    assert features["goalkeeper"]["status"] == "missing"
    assert features["weather"]["status"] == "missing"
    assert features["travel"]["status"] == "missing"
    assert features["starting_lineups"]["source_observed_at_utc"] == p["captured_at_utc"]
    tampered = deepcopy(p); tampered["home_team_id"] = "99"
    assert not build_dossier(FIXTURE, [], as_of=NOW, provider_packet=tampered)["features"]["starting_lineups"]["available"]
    later = seal({**p, "captured_at_utc": (NOW+timedelta(minutes=1)).isoformat()})
    assert not build_dossier(FIXTURE, [], as_of=NOW, provider_packet=later)["features"]["shot_xg_xga"]["available"]


def test_existing_statistics_packets_supply_shots_without_new_provider_requests():
    raw=packet()
    prior=match(1,11,33)
    raw["verified_historical_match_statistics"]=[{
        "fixture_id":str(prior["id"]),"kickoff_utc":prior["utcDate"],
        "home_team_id":"11","away_team_id":"33",
        "home":{"total_shots":12,"shots_on_target":5},
        "away":{"total_shots":4,"shots_on_target":1}}]
    dossier=build_dossier(FIXTURE,[prior],as_of=NOW,provider_packet=seal(raw))
    assert dossier["features"]["shots"]["value"]["home"]["last5"]["shots_for_per_match"]==12
    assert dossier["features"]["shots_on_target"]["value"]["home"]["last5"]["shots_on_target_for_per_match"]==5
    assert dossier["external_calls_made"]==0


def test_source_xg_does_not_require_fabricated_result_history_and_rejects_future_xg():
    p = packet()
    assert build_dossier(FIXTURE, [], as_of=NOW, provider_packet=p)["features"]["shot_xg_xga"]["available"]
    p["verified_historical_fixture_xg"][0]["kickoff_utc"] = (NOW-timedelta(hours=1)).isoformat()
    p = seal(p)
    assert not build_dossier(FIXTURE, [], as_of=NOW, provider_packet=p)["features"]["shot_xg_xga"]["available"]


def test_unverified_raw_context_never_turns_into_facts():
    dossier = build_dossier(FIXTURE, [], as_of=NOW, context={"injuries": ["Fake injury"], "weather": "Storm"})
    assert dossier["features"]["injuries_absences"]["value"] is None
    assert "Fake injury" not in json.dumps(dossier)
    assert all(r["Observed UTC"] == "Unknown" for r in feature_rows(dossier))
    with pytest.raises(ValueError):
        build_dossier(FIXTURE, [], as_of=FIXTURE["utcDate"])


def test_shots_from_existing_source_response_without_xg_or_fake_zero():
    p = packet(); p["verified_historical_fixture_xg"] = []
    p["verified_historical_fixture_shots"] = [{"fixture_id": "111", "kickoff_utc": match(1, 11, 33)["utcDate"],
        "home_team_id": "11", "away_team_id": "33", "home_shots": 14, "away_shots": 8,
        "home_shots_on_target": 5, "away_shots_on_target": 2}]
    dossier = build_dossier(FIXTURE, [], as_of=NOW, provider_packet=seal(p))
    assert dossier["features"]["shots"]["value"]["home"]["last5"]["shots_for_per_match"] == 14
    assert dossier["features"]["shots_on_target"]["value"]["home"]["last5"]["shots_on_target_for_per_match"] == 5
    assert not dossier["features"]["shot_xg_xga"]["available"]
    p["verified_historical_fixture_shots"][0]["home_shots_on_target"] = None
    dossier = build_dossier(FIXTURE, [], as_of=NOW, provider_packet=seal(p))
    assert dossier["features"]["shots"]["available"]
    assert not dossier["features"]["shots_on_target"]["available"]


def test_prompt_uses_same_dossier_and_rejects_raw_news_without_source():
    from on_demand_ai import briefing
    from llm_research_panel import briefing as scheduled_briefing
    prompt = briefing(FIXTURE, [match(1, 11, 33)], {}, now=NOW,
                      team_context={"injuries": ["UNVERIFIED PLAYER NEWS"], "v21_provider_snapshot": packet()})
    assert "UNVERIFIED PLAYER NEWS" not in prompt
    assert "professional_match_evidence" in prompt
    assert '"last20"' in prompt
    assert '"ai_numeric_weight": 0.0' in prompt
    row = {"provider": "API-Football", "fixture_id": "900", "kickoff_utc": FIXTURE["utcDate"],
           "home_team_id": "11", "away_team_id": "12", "home_team": "Home", "away_team": "Away",
           "captured_at_utc": (NOW-timedelta(hours=1)).isoformat()}
    scheduled = scheduled_briefing(row, provider_packet=packet(), as_of=NOW)
    assert '"shot_xg_xga"' in scheduled
    assert '"source_observed_at_utc"' in scheduled
    assert '"starting_xi"' in scheduled


def test_stale_or_future_market_quotes_are_missing_and_not_retimed():
    p = packet()
    for at in (NOW-timedelta(hours=13), NOW+timedelta(minutes=1)):
        p["multi_market"]["1x2"]["last_source_update_utc"] = at.isoformat()
        d = build_dossier(FIXTURE, [], as_of=NOW, provider_packet=seal(p))
        assert not d["features"]["markets_1x2"]["available"]


def test_actual_fetch_timestamp_is_distinct_from_estimated_final_availability():
    row = match(1, 11, 33)
    row["fetched_at_utc"] = (NOW-timedelta(hours=2)).isoformat()
    d = build_dossier(FIXTURE, [row], as_of=NOW)
    assert d["features"]["goal_form"]["source_observed_at_utc"] == row["fetched_at_utc"]
    assert d["historical_results"][0]["result_available_at_utc"] != row["fetched_at_utc"]
    assert d["historical_results"][0]["result_availability_basis"] == "conservative_kickoff_plus_3h"
