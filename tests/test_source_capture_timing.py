"""Slow responses cannot become backdated pre-match source evidence."""
from datetime import datetime, timedelta, timezone
import hashlib
import json

from source_capture_clock import capture_clock
from v21_provider_evidence import capture, valid_snapshot, _fetch
import v23_early_odds as early
import v24_bulk_market_capture as bulk

NOW = datetime(2032, 7, 3, 3, tzinfo=timezone.utc)


def fixture(hours=1):
    return {"provider":"API-Football", "id":5001,
            "utcDate":(NOW+timedelta(hours=hours)).isoformat(),
            "homeTeam":{"id":101}, "awayTeam":{"id":102}}


def odds(target, updated):
    return {"fixture":{"id":target["id"], "date":target["utcDate"]},
            "update":updated.isoformat(),
            "bookmakers":[{"name":"Test bookmaker", "bets":[
                {"name":"Match Winner", "values":[
                    {"value":"Home", "odd":"2.25"},
                    {"value":"Draw", "odd":"3.30"},
                    {"value":"Away", "odd":"3.50"}]},
                {"name":"Goals Over/Under", "values":[
                    {"value":"Over 2.5", "odd":"1.85"},
                    {"value":"Under 2.5", "odd":"2.07"}]}]}]}


def test_capture_rejects_response_after_kickoff_and_stops_requests():
    target=fixture()
    at=[NOW]
    calls=[]
    def delayed(*args):
        calls.append(args[1])
        at[0]=NOW+timedelta(hours=1,seconds=1)
        return []
    packet=capture(target,[],"test",now=NOW,clock=lambda:at[0],request=delayed)
    assert packet["status"]=="unavailable"
    assert packet["reason"]=="capture_completed_after_kickoff"
    assert calls==["/fixtures/lineups"]


def test_receipt_timestamp_not_request_start_and_wrong_provider_rejected():
    target=fixture()
    at=[NOW]
    def delayed(key,endpoint,params):
        at[0]+=timedelta(seconds=10)
        return [odds(target,at[0])] if endpoint=="/odds" else []
    packet=capture(target,[],"test",now=NOW,clock=lambda:at[0],request=delayed)
    assert packet["captured_at_utc"]==(NOW+timedelta(seconds=30)).isoformat()
    assert packet["multi_market"]["1x2"]
    assert valid_snapshot(packet,target,now=at[0])
    assert not valid_snapshot(packet,target,now=NOW)
    wrong={k:v for k,v in packet.items() if k!="snapshot_sha256"}
    wrong["provider"]="another-provider"
    wrong["snapshot_sha256"]=hashlib.sha256(json.dumps(
        wrong,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    assert not valid_snapshot(wrong,target,now=at[0])


def test_early_odds_rejects_late_response_and_preserves_attempt(monkeypatch,tmp_path):
    target=fixture(hours=8)
    leg={"provider":"API-Football", "fixture_id":"5001", "forecast_id":"original",
         "input_digest":"unchanged", "home_team_id":"101", "away_team_id":"102",
         "kickoff_utc":target["utcDate"]}
    monkeypatch.setattr(early,"eligible_legs",lambda root,now:[leg])
    at=[NOW]
    def delayed(*args):
        at[0]=NOW+timedelta(hours=8,seconds=1)
        return [odds(target,NOW)]
    result=early.capture(now=NOW,enabled=True,api_key="test",request=delayed,
                         source_root=tmp_path,root=tmp_path,clock=lambda:at[0])
    assert result["queries"]==1
    assert result["new_packets"]==[]
    assert result["skipped"]==["5001:response_after_kickoff"]
    assert not list((tmp_path/"versions").glob("*.json"))
    assert len(list((tmp_path/"early_attempts").glob("*.json")))==1


def test_bulk_odds_rejects_fixture_started_during_request(monkeypatch,tmp_path):
    target=fixture()
    monkeypatch.setattr(bulk,"_day_priority",lambda clock:[NOW.date().isoformat()])
    at=[NOW]
    def delayed(*args):
        at[0]=NOW+timedelta(hours=1,seconds=1)
        return {"metadata":{"fetched_at_utc":at[0].isoformat(),"stale":False},
                "data":[odds(target,NOW)]}
    result=bulk.capture(now=NOW,enabled=True,api_key="test",request=delayed,
                        root=tmp_path/"catalog",source_root=tmp_path/"sources",
                        original_root=tmp_path/"original",originals=[],
                        max_per_run=1,daily_limit=1,clock=lambda:at[0])
    assert result["queries"]==1
    assert result["new_catalog_fixtures"]==[]
    assert not list((tmp_path/"catalog"/"fixtures").glob("*/*.json"))


def test_actual_fetch_after_run_start_is_accepted_without_fake_quote_time(monkeypatch,tmp_path):
    target=fixture()
    monkeypatch.setattr(bulk,"_day_priority",lambda clock:[NOW.date().isoformat()])
    at=[NOW]
    def delayed(*args):
        at[0]+=timedelta(seconds=20)
        item=odds(target,NOW)
        item.pop("update")
        return {"metadata":{"fetched_at_utc":at[0].isoformat(),"stale":False},
                "data":[item]}
    result=bulk.capture(now=NOW,enabled=True,api_key="test",request=delayed,
                        root=tmp_path/"catalog",source_root=tmp_path/"sources",
                        original_root=tmp_path/"original",originals=[],
                        max_per_run=1,daily_limit=1,clock=lambda:at[0])
    assert result["new_catalog_fixtures"]==["5001"]
    row=json.loads(next((tmp_path/"catalog"/"fixtures"/"5001").glob("*.json")).read_text())
    quote=row["market_types"]["1x2"]["quotes"][0]
    assert quote["timestamp_basis"]=="api_response_fetched_at"
    assert row["captured_at_utc"]==at[0].isoformat()


def test_malformed_response_counts_one_request():
    diag={"queries":0,"unavailable":{}}
    assert _fetch(lambda *args:"malformed","key","/odds",{},diag) is None
    assert diag["queries"]==1


def test_explicit_simulation_time_advances_by_monotonic_elapsed(monkeypatch):
    elapsed=[10.0]
    monkeypatch.setattr("source_capture_clock.time.monotonic",lambda:elapsed[0])
    start,current=capture_clock(NOW)
    elapsed[0]=31.5
    assert start==NOW
    assert current()==NOW+timedelta(seconds=21.5)
