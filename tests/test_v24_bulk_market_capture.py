"""V24 source-wide market capture: real identity, quotas, page handling, no hindsight."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path

import pytest

import v24_bulk_market_capture as bulk
import v23_early_odds as early
from v10_prospective import utc

NOW=datetime(2032,7,3,3,tzinfo=timezone.utc)


def setup_fixture():
    date=bulk._day_priority(NOW)[0]
    kickoff=datetime.fromisoformat(date).replace(tzinfo=timezone.utc)+timedelta(hours=15)
    source={"provider":"API-Football","fixture_id":"5001",
            "forecast_id":"v8_original_5001","input_digest":"real_saved_digest_5001",
            "model_key":"v8","home_team_id":"101","away_team_id":"102",
            "home_team":"Home","away_team":"Away","competition":"PL",
            "kickoff_utc":kickoff.isoformat(),
            "captured_at_utc":(NOW-timedelta(hours=2)).isoformat()}
    return source


def market_item(fid, kickoff):
    return {"fixture":{"id":fid,"date":utc(kickoff).isoformat()},
            "league":{"id":39},
            "bookmakers":[{"name":"Verified Book","bets":[
                {"name":"Match Winner","values":[
                    {"value":"Home","odd":"2.25"},
                    {"value":"Draw","odd":"3.30"},
                    {"value":"Away","odd":"3.50"}]},
                {"name":"Goals Over/Under","values":[
                    {"value":"Over 2.5","odd":"1.85"},
                    {"value":"Under 2.5","odd":"2.07"}]},
                {"name":"Both Teams Score","values":[
                    {"value":"Yes","odd":"1.78"},
                    {"value":"No","odd":"2.20"}]}]}]}


def provider_response(items):
    return {"metadata":{"fetched_at_utc":
                            (NOW-timedelta(minutes=2)).isoformat(),
                         "stale":False,"cache_status":"network"},
            "data":items}


def test_daily_budget_broad_capture_and_write_once_source_binding(monkeypatch,tmp_path):
    source=setup_fixture()
    monkeypatch.setattr(bulk,"original_model",lambda root,leg:source)
    dates=[]
    def fake(key,endpoint,params):
        assert endpoint=="/odds"
        dates.append(params["date"])
        first=bulk._day_priority(NOW)[0]
        return provider_response(
            [market_item(5001,source["kickoff_utc"])]
            if params["date"]==first else [])
    summary=bulk.capture(now=NOW,enabled=True,api_key="fake-test-key",
                         root=tmp_path/"market",source_root=tmp_path/"sources",
                         original_root=tmp_path/"original",
                         request=fake,originals=[source],
                         max_per_run=2,daily_limit=2)
    assert summary["queries"]==2
    assert summary["new_catalog_fixtures"]==["5001"]
    assert summary["new_source_bindings"]==["5001"]
    assert summary["binding"]["v8_checked"]==1
    packet_file=next((tmp_path/"sources"/"versions").glob("*.json"))
    before=packet_file.read_text()
    record=json.loads(before)
    assert record["source_forecast"]["forecast_id"]==source["forecast_id"]
    assert "1x2" in record["provider_snapshot"]["multi_market"]
    assert "over_2_5" in record["provider_snapshot"]["multi_market"]
    assert "btts_yes" in record["provider_snapshot"]["multi_market"]
    assert record["provider_snapshot"]["captured_at_utc"]<source["kickoff_utc"]
    repeated=bulk.capture(now=NOW,enabled=True,api_key="fake-test-key",
                          root=tmp_path/"market",source_root=tmp_path/"sources",
                          original_root=tmp_path/"original",
                          request=fake,originals=[source],
                          max_per_run=2,daily_limit=2)
    assert repeated["queries"]==0
    assert packet_file.read_text()==before


def test_catalog_collects_before_any_original_model_exists(tmp_path):
    date=bulk._day_priority(NOW)[0]
    kickoff=datetime.fromisoformat(date).replace(tzinfo=timezone.utc)+timedelta(hours=15)
    calls=[]
    def fake(key,endpoint,params):
        calls.append(params)
        if params["date"]==date:
            return provider_response([market_item(6001,kickoff)])
        return provider_response([])
    report=bulk.capture(now=NOW,enabled=True,api_key="fake",
                        root=tmp_path/"market",source_root=tmp_path/"sources",
                        original_root=tmp_path/"original",request=fake,
                        originals=[],max_per_run=1,daily_limit=1)
    assert report["queries"]==1
    assert report["new_catalog_fixtures"]==["6001"]
    assert report["binding"]["v8_checked"]==0
    assert not list((tmp_path/"sources").rglob("*.json"))
    assert list((tmp_path/"market"/"fixtures"/"6001").glob("*.json"))


def test_rejects_wrong_teams_kickoff_league_tamper_and_post_kickoff(monkeypatch,tmp_path):
    source=setup_fixture()
    item=market_item(5001,source["kickoff_utc"])
    fid=bulk._save_fixture_market(item,clock=NOW,
                fetched=(NOW-timedelta(minutes=2)).isoformat(),
                root=tmp_path,query_date=bulk._day_priority(NOW)[0])
    assert fid=="5001"
    path=next((tmp_path/"fixtures"/"5001").glob("*.json"))
    market=json.loads(path.read_text())
    assert bulk._valid_catalog(market,source,NOW)
    wrong_league={**source,"competition":"SA"}
    assert not bulk._valid_catalog(market,wrong_league,NOW)
    wrong_date={**source,"kickoff_utc":
                (utc(source["kickoff_utc"])+timedelta(days=1)).isoformat()}
    assert not bulk._valid_catalog(market,wrong_date,NOW)
    after={**source,"kickoff_utc":(NOW-timedelta(hours=1)).isoformat()}
    assert not bulk._valid_catalog(market,after,NOW)
    tampered={**market,"fixture_id":"9999"}
    assert not bulk._valid_catalog(tampered,source,NOW)


def test_multi_page_and_request_budget(monkeypatch,tmp_path):
    first=bulk._day_priority(NOW)[0]
    kickoff=datetime.fromisoformat(first).replace(tzinfo=timezone.utc)+timedelta(hours=15)
    calls=[]
    def fake(key,endpoint,params):
        calls.append((params["date"],params["page"]))
        if params["date"]!=first:
            return provider_response([])
        if params["page"]==1:
            return provider_response([market_item(i,kickoff) for i in range(100,110)])
        if params["page"]==2:
            return provider_response([market_item(i,kickoff) for i in range(110,112)])
        raise AssertionError("Must not request page 3 after fewer than 10 results")
    report=bulk.capture(now=NOW,enabled=True,api_key="test",
                        root=tmp_path/"market",source_root=tmp_path/"sources",
                        original_root=tmp_path/"original",request=fake,
                        originals=[],max_per_run=2,daily_limit=2)
    assert calls==[(first,1),(first,2)]
    assert report["queries"]==2
    assert len(report["new_catalog_fixtures"])==12
    assert len(list((tmp_path/"market"/"fixtures").glob("*/*.json")))==12


def test_opt_out_does_not_call_provider(tmp_path):
    def forbidden(*args,**kwargs):
        raise AssertionError("No API request without opt-in")
    result=bulk.capture(now=NOW,enabled=False,api_key="test",
                        root=tmp_path/"market",source_root=tmp_path/"sources",
                        original_root=tmp_path/"original",
                        request=forbidden,originals=[])
    assert result["queries"]==0
    assert result["reason"]=="not_enabled_or_api_key_missing"


def test_individual_v8_forecasts_no_longer_require_an_existing_double(monkeypatch,tmp_path):
    source=setup_fixture()
    source["kickoff_utc"]=(NOW+timedelta(hours=8)).isoformat()
    monkeypatch.setattr("v10_prospective.read_latest",
                        lambda root={}: {("API-Football","5001","v8"):source})
    monkeypatch.setattr(early,"original_model",lambda root,leg:source)
    found=early.eligible_legs(tmp_path,NOW)
    assert [x["fixture_id"] for x in found]==["5001"]
