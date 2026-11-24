"""Regression tests: V25 never invents missing source evidence."""
from datetime import datetime, timedelta, timezone
import hashlib
import json

from v25_evidence_audit import _matching_source, trace_leg, trace_pair

NOW = datetime(2031, 11, 18, tzinfo=timezone.utc)
KICK = (NOW + timedelta(hours=5)).isoformat()


def row():
    source = {"provider":"API-Football", "fixture_id":"101",
              "forecast_id":"original123", "input_digest":"digest123",
              "home_team_id":"201", "away_team_id":"202",
              "kickoff_utc":KICK}
    fixture = {"provider":"API-Football", "id":"101", "utcDate":KICK,
               "homeTeam":{"id":"201","name":"Home"},
               "awayTeam":{"id":"202","name":"Away"}}
    return {"row":{"source_forecast":source, "fixture":fixture}}


def validated_record():
    source = row()["row"]["source_forecast"]
    packet = {"status":"captured_pre_kickoff", "provider":"API-Football",
              "fixture_id":"101", "home_team_id":"201", "away_team_id":"202",
              "kickoff_utc":KICK, "captured_at_utc":NOW.isoformat(),
              "verified_historical_fixture_xg":[
                  {"fixture_id":"12","home_team_id":"201","away_team_id":"301",
                   "type":"api_football_reported_expected_goals"},
                  {"fixture_id":"13","home_team_id":"202","away_team_id":"302",
                   "type":"api_football_reported_expected_goals"}],
              "confirmed_starting_xi":[
                  {"team_id":"201","provider_reported":True,"starting_xi":[f"player {i}" for i in range(11)]},
                  {"team_id":"202","provider_reported":True,"starting_xi":[f"player {i}" for i in range(11)]}],
              "reported_injuries":[],
              "multi_market":{
                  "1x2":{"last_source_update_utc":(NOW-timedelta(hours=1)).isoformat()},
                  "over_2_5":{"last_source_update_utc":(NOW-timedelta(hours=1)).isoformat()}
              }}
    encoded=json.dumps(packet,sort_keys=True,separators=(",",":"),allow_nan=False)
    packet["snapshot_sha256"]=hashlib.sha256(encoded.encode()).hexdigest()
    return {"source_forecast":source.copy(), "provider_snapshot":packet}


def test_unmatched_identity_never_calls_provider_packet_loader():
    candidate=row()
    del candidate["row"]["source_forecast"]
    result=trace_leg(candidate,NOW,packet_loader=lambda *a,**k: (_ for _ in ()).throw(AssertionError()))
    assert result["matching_provider_packet"] is False
    assert result["verified_historical_xg_both_teams"] is False
    assert result["independent_context_applied_to_v25_model"] is False


def test_mismatched_team_identity_blocks_evidence():
    candidate=row()
    candidate["row"]["source_forecast"]["home_team_id"]="999"
    assert _matching_source(candidate["row"]) is None
    assert trace_leg(candidate,NOW)["reason"]=="missing_original_source_identity"


def test_matching_genuine_timestamped_snapshot_is_described_not_blended():
    result=trace_leg(row(),NOW,packet_loader=lambda *a,**k: validated_record())
    assert result["matching_provider_packet"] is True
    assert result["verified_pre_match_market"] is True
    assert result["verified_historical_xg_both_teams"] is True
    assert result["confirmed_starting_xi_both_teams"] is True
    assert result["reported_injury_entries"] == 0
    assert result["independent_context_applied_to_v25_model"] is False


def test_late_market_not_presented_as_fresh():
    record=validated_record()
    record["provider_snapshot"]["multi_market"]["1x2"]["last_source_update_utc"] = (NOW-timedelta(hours=15)).isoformat()
    packet=record["provider_snapshot"]
    body={k:v for k,v in packet.items() if k!="snapshot_sha256"}
    packet["snapshot_sha256"]=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    result=trace_leg(row(),NOW,packet_loader=lambda *a,**k: record)
    assert result["matching_provider_packet"] is True
    assert result["verified_pre_match_market"] is False


def test_no_fabricated_injuries_and_originals_remain_unchanged():
    candidate=row()
    prior=candidate["row"]["source_forecast"].copy()
    info=trace_pair(candidate,candidate,NOW,packet_loader=lambda *a,**k: None)
    assert all(x["reported_injury_entries"] is None for x in info)
    assert candidate["row"]["source_forecast"] == prior


def reseal(record):
    packet = record["provider_snapshot"]
    body = {k:v for k,v in packet.items() if k != "snapshot_sha256"}
    packet["snapshot_sha256"] = hashlib.sha256(json.dumps(
        body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return record


def test_history_rich_packet_does_not_hide_later_confirmed_lineups(tmp_path):
    from v21_capture_sources import best_packet, source_packet_path
    from copy import deepcopy
    original = validated_record()
    later = deepcopy(original)
    original["provider_snapshot"]["confirmed_starting_xi"] = []
    original["provider_snapshot"]["captured_at_utc"] = (NOW-timedelta(minutes=10)).isoformat()
    later["provider_snapshot"]["verified_historical_fixture_xg"] = []
    reseal(original)
    reseal(later)
    source = original["source_forecast"]
    path = source_packet_path(tmp_path, source)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(original))
    versions = tmp_path/"versions"
    versions.mkdir()
    version = versions/(path.stem+"_late.json")
    version.write_text(json.dumps(later))
    before = (path.read_bytes(), version.read_bytes())
    assert best_packet(tmp_path,source,now=NOW)["provider_snapshot"] == original["provider_snapshot"]
    assert best_packet(tmp_path,source,now=NOW,priority="lineup")["provider_snapshot"] == later["provider_snapshot"]
    result = trace_leg(row(),NOW,root=tmp_path)
    assert result["verified_historical_xg_both_teams"]
    assert result["confirmed_starting_xi_both_teams"]
    assert result["lineup_snapshot_at_utc"] == NOW.isoformat()
    assert not result["independent_context_applied_to_v25_model"]
    assert (path.read_bytes(), version.read_bytes()) == before


def test_duplicate_or_wrong_identity_lineup_not_counted():
    from copy import deepcopy
    record = validated_record()
    record["provider_snapshot"]["confirmed_starting_xi"][0]["starting_xi"] = ["duplicate"]*11
    reseal(record)
    other = deepcopy(validated_record())
    other["source_forecast"]["forecast_id"] = "other"
    result = trace_leg(row(),NOW,packet_loader=lambda *a,priority,**k:
                       other if priority=="lineup" else record)
    assert not result["confirmed_starting_xi_both_teams"]
    assert "lineup_snapshot_at_utc" not in result


def test_conflicting_provider_lineups_not_counted():
    from copy import deepcopy
    record = validated_record()
    changed = deepcopy(record["provider_snapshot"]["confirmed_starting_xi"][0])
    changed["starting_xi"][0] = "different player"
    record["provider_snapshot"]["confirmed_starting_xi"].append(changed)
    reseal(record)
    assert not trace_leg(row(),NOW,packet_loader=lambda *a,**k:record)[
        "confirmed_starting_xi_both_teams"]
