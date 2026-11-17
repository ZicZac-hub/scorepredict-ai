"""Sportmonks genuine xG: cross-provider identity, no leakage, no fake coverage."""
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path

import requests

from sportmonks_xg_research import (
    _xg_pair, _matched_future, eligible_originals, parse_history, capture,
)

NOW=datetime(2032,8,16,10,tzinfo=timezone.utc)
KICK=NOW+timedelta(hours=20)


def original():
    return {
        "provider":"API-Football", "fixture_id":"321",
        "forecast_id":"original-xyz", "input_digest":"preserved-digest",
        "model_key":"v8", "home_team":"Home FC", "away_team":"Away United",
        "home_team_id":"11","away_team_id":"22",
        "captured_at_utc":(NOW-timedelta(hours=1)).isoformat(),
        "kickoff_utc":KICK.isoformat(),
    }


def next_match():
    return {
        "id":54321,"starting_at_timestamp":int(KICK.timestamp()),
        "participants":[
            {"id":31,"name":"Home FC","meta":{"location":"home"}},
            {"id":32,"name":"Away United","meta":{"location":"away"}},
        ],
    }


def historical(fid,at,home,away,xhome=1.4,xaway=0.8,state_id=5,
               xg_type=5304):
    return {
        "id":fid,"starting_at_timestamp":int(at.timestamp()),
        "state_id":state_id,
        "participants":[
            {"id":home,"name":"Alpha","meta":{"location":"home"}},
            {"id":away,"name":"Beta","meta":{"location":"away"}},
        ],
        "xgfixture":[
            {"fixture_id":fid,"type_id":xg_type,"participant_id":home,
             "location":"home","data":{"value":xhome}},
            {"fixture_id":fid,"type_id":xg_type,"participant_id":away,
             "location":"away","data":{"value":xaway}},
        ],
    }


def test_type_5304_only_with_both_verified_participants():
    item=historical(101,NOW-timedelta(days=3),31,80)
    assert _xg_pair(item)["home_xg"]==1.4
    assert _xg_pair(historical(101,NOW-timedelta(days=3),31,80,
                               xg_type=5305)) is None
    item["xgfixture"][0]["participant_id"]=999
    assert _xg_pair(item) is None


def test_cross_provider_match_rejects_wrong_teams_kickoff_and_ambiguity():
    right=next_match()
    assert _matched_future(original(),[right])=={
        "fixture_id":54321,"home_team_id":31,"away_team_id":32}
    wrong=next_match()
    wrong["participants"][1]["name"]="Different Club"
    assert _matched_future(original(),[wrong]) is None
    wrong_date=next_match()
    wrong_date["starting_at_timestamp"]+=1200
    assert _matched_future(original(),[wrong_date]) is None
    second=next_match()
    second["id"]=54322
    assert _matched_future(original(),[right,second]) is None


def test_only_finished_past_xg_without_future_leak():
    old=historical(100,NOW-timedelta(days=5),31,80)
    new=historical(101,NOW-timedelta(minutes=30),31,81)
    extra=historical(102,NOW-timedelta(days=6),31,82,state_id=7)
    wrong=historical(103,NOW-timedelta(days=7),66,67)
    rows=parse_history([old,new,extra,wrong],31,NOW,KICK)
    assert len(rows)==1
    assert rows[0]["sportmonks_fixture_id"]=="100"
    assert rows[0]["home_xg"]==1.4


def test_collect_source_matched_real_historical_xg_with_request_limit(tmp_path):
    calls=[]
    home=historical(900,NOW-timedelta(days=5),31,80)
    away=historical(901,NOW-timedelta(days=4),70,32)
    fixture=next_match()
    class Response:
        status_code=200
        def __init__(self,body):self.body=body
        def raise_for_status(self):pass
        def json(self):return self.body
    def get(url,**kwargs):
        calls.append((url,kwargs["params"]))
        if "/fixtures/date/" in url:
            return Response({"data":[fixture],"pagination":{"has_more":False}})
        if url.endswith("/31"):
            return Response({"data":[home]})
        if url.endswith("/32"):
            return Response({"data":[away]})
        raise AssertionError(url)
    env={"SPORTMONKS_XG_ENABLED":"true",
         "THIRD_FOOTBALL_API_KEY":"secret-value"}
    rows={("API-Football","321","v8"):original()}
    s=capture(now=NOW,root=tmp_path,originals=rows,
              environ=env,request=get,clock=lambda:NOW)
    assert s["reason"]=="genuine_sportmonks_xg_captured"
    assert s["captured"]==1
    assert s["requests_this_run"]==3
    assert s["daily_requests_used"]==3
    files=list((tmp_path/"forecasts").glob("*.json"))
    assert len(files)==1
    data=json.loads(files[0].read_text())
    assert data["source_input_digest"]=="preserved-digest"
    assert data["api_football_fixture_id"]=="321"
    assert data["sportmonks_fixture_id"]=="54321"
    assert data["historical_xg_home_count"]==1
    assert data["historical_xg_away_count"]==1
    assert {x["sportmonks_fixture_id"] for x in data["historical_xg"]}=={"900","901"}
    assert data["not_automatically_promoted"]
    assert "secret-value" not in json.dumps(s)
    assert "secret-value" not in files[0].read_text()
    assert "/fixtures/900" not in str(calls)
    # A scheduled retry cannot overwrite or requery the same forecast.
    second=capture(now=NOW+timedelta(minutes=15),root=tmp_path,
                   originals=rows,environ=env,request=get,
                   clock=lambda:NOW+timedelta(minutes=15))
    assert second["reason"]=="eligible_forecasts_already_checked"
    assert len(calls)==3


def test_missing_token_or_disabled_do_not_attempt_any_calls(tmp_path):
    def denied(*args,**kwargs):
        raise AssertionError("No request without token")
    env={"SPORTMONKS_XG_ENABLED":"true"}
    s=capture(now=NOW,root=tmp_path,originals=[original()],
              environ=env,request=denied)
    assert s["reason"]=="sportmonks_token_missing"
    assert s["token_present"] is False
    assert s["requests_this_run"]==0


def test_unsupported_addon_reports_uncertainty_and_never_fakes_xg(tmp_path):
    class Denied:
        status_code=403
        def raise_for_status(self):
            raise requests.HTTPError("403 subscription")
        def json(self):
            raise AssertionError("Should not inspect unapproved data")
    env={"SPORTMONKS_XG_ENABLED":"true","SPORTMONKS_API_TOKEN":"secret"}
    calls=[]
    def denied(*args,**kwargs):
        calls.append(True)
        return Denied()
    s=capture(now=NOW,root=tmp_path,originals=[original()],
              environ=env,request=denied)
    assert s["reason"]=="provider_access_subscription_or_network_error"
    assert not list((tmp_path/"forecasts").glob("*.json"))
    assert s["subscription_confirmed"] is False
    assert s["requests_this_run"]==1
    assert len(calls)==1
    assert "secret" not in json.dumps(s)


def test_quota_zero_makes_no_paid_calls(tmp_path):
    env={"SPORTMONKS_XG_ENABLED":"true","SPORTMONKS_API_TOKEN":"secret"}
    def no_call(*args,**kwargs):
        raise AssertionError("Quota exhausted")
    s=capture(now=NOW,root=tmp_path,originals=[original()],
              environ=env,request=no_call,daily_limit=0)
    assert s["reason"]=="daily_request_budget_exhausted"
    assert s["requests_this_run"]==0


def test_wrong_match_names_cannot_be_relabelled_as_xg(tmp_path):
    class Response:
        status_code=200
        def raise_for_status(self):pass
        def json(self):
            wrong=next_match()
            wrong["participants"][0]["name"]="Unrelated"
            return {"data":[wrong],"pagination":{"has_more":False}}
    calls=[]
    def fake(*args,**kwargs):
        calls.append(args[0])
        return Response()
    env={"SPORTMONKS_XG_ENABLED":"true","SPORTMONKS_API_TOKEN":"secret"}
    s=capture(now=NOW,root=tmp_path,originals=[original()],
              environ=env,request=fake,clock=lambda:NOW)
    assert s["reason"]=="no_unique_cross_provider_fixture_match"
    assert s["captured"]==0
    assert len(calls)==2  # Targeted search then fallback date scan.


def test_reviewed_aliases_match_actual_clubs_but_never_juniors():
    from sportmonks_xg_research import _team_key
    forecast=dict(original(),home_team="Coritiba",away_team="Santos")
    fixture=next_match()
    fixture["participants"][0]["name"]="Coritiba Foot Ball Club"
    fixture["participants"][1]["name"]="Santos FC"
    assert _matched_future(forecast,[fixture])=={
        "fixture_id":54321,"home_team_id":31,"away_team_id":32}
    fixture["participants"][1]["name"]="Santos U20"
    assert _matched_future(forecast,[fixture]) is None
    assert _team_key("Santos Laguna") != _team_key("Santos")


def test_filtered_pagination_finds_second_page_but_only_after_complete_scan(tmp_path):
    from sportmonks_xg_research import _query_pages
    fixture=next_match()
    calls=[]
    def call(url,params):
        calls.append(dict(params))
        if params["page"]==1:
            return {"data":[],"pagination":{"has_more":True}}
        return {"data":[fixture],"pagination":{"has_more":False}}
    found,reason,meta=_query_pages(original(),NOW,tmp_path,call)
    assert reason=="source_identity_matched"
    assert found["fixture_id"]==54321
    assert [x["page"] for x in calls]==[1,2]
    assert all(x["per_page"]==50 for x in calls)
    # Subsequent pass reuses persisted, timestamped pages instead of re-billing.
    def forbidden(*args,**kwargs):
        raise AssertionError("Replayed discovery should come from cache")
    repeat,why,rec=_query_pages(original(),NOW+timedelta(minutes=15),
                                tmp_path,forbidden)
    assert repeat==found
    assert why=="source_identity_matched"
    assert rec["cached_pages_used"]==2


def test_ambiguous_second_page_is_not_accepted_as_unique(tmp_path):
    from sportmonks_xg_research import _query_pages
    first=next_match()
    second=next_match()
    second["id"]=54322
    def call(url,params):
        return {"data":[first] if params["page"]==1 else [second],
                "pagination":{"has_more":params["page"]==1}}
    match,reason,_=_query_pages(original(),NOW,tmp_path,call)
    assert match is None
    assert reason=="ambiguous_cross_provider_fixture"


def test_nonexhaustive_paginated_scan_cannot_confirm_identity(tmp_path):
    from sportmonks_xg_research import _query_pages
    calls=[]
    def call(url,params):
        calls.append(params["page"])
        return {"data":[next_match()] if params["page"]==1 else [],
                "pagination":{"has_more":True}}
    found,reason,_=_query_pages(original(),NOW,tmp_path,call)
    assert found is None
    assert reason=="discovery_pagination_incomplete"
    assert len(calls)==8


def test_legacy_unmatched_marker_retries_after_cooldown_then_stops_at_three(tmp_path):
    env={"SPORTMONKS_XG_ENABLED":"true","SPORTMONKS_API_TOKEN":"secret"}
    forecast=original()
    marker=tmp_path/"attempts"/"API-Football_321_original-xyz.json"
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({"forecast_id":"original-xyz",
        "api_football_fixture_id":"321",
        "attempted_at_utc":(NOW-timedelta(hours=1)).isoformat()}))
    calls=[]
    class Response:
        status_code=200
        def raise_for_status(self): pass
        def json(self):
            return {"data":[],"pagination":{"has_more":False}}
    def empty(*args,**kwargs):
        calls.append(True)
        return Response()
    status=capture(now=NOW,root=tmp_path,originals=[forecast],
                   environ=env,request=empty,clock=lambda:NOW)
    assert status["attempt_number"]==2
    assert status["reason"]=="no_unique_cross_provider_fixture_match"
    assert len(calls)==2
    # Early return on the same frozen fixture: no extra API calls.
    again=capture(now=NOW+timedelta(minutes=10),root=tmp_path,
                  originals=[forecast],environ=env,request=empty,
                  clock=lambda:NOW+timedelta(minutes=10))
    assert again["reason"]=="eligible_forecasts_already_checked"
    assert len(calls)==2
    later=NOW+timedelta(minutes=50)
    third=capture(now=later,root=tmp_path,originals=[forecast],
                  environ=env,request=empty,clock=lambda:later)
    assert third["attempt_number"]==3
    assert third["reason"]=="no_unique_cross_provider_fixture_match"
    assert json.loads(marker.read_text())["terminal"]
    final=capture(now=later+timedelta(hours=2),root=tmp_path,
                  originals=[forecast],environ=env,request=empty,
                  clock=lambda:later+timedelta(hours=2))
    assert final["reason"]=="eligible_forecasts_already_checked"


def test_matching_other_forecast_never_reuses_wrong_team_alias(tmp_path):
    fake=next_match()
    fake["participants"][0]["name"]="Coritiba U20"
    fake["participants"][1]["name"]="Santos FC"
    forecast=dict(original(),home_team="Coritiba",away_team="Santos")
    assert _matched_future(forecast,[fake]) is None
