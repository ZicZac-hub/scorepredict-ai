"""Focused Today/Tomorrow model discovery must respect time and quota."""
from datetime import datetime,timezone,date

from ai_five_live_finder import research_day


NOW=datetime(2026,11,15,0,0,tzinfo=timezone.utc)
DAY=date(2026,11,15)


def fixture(fid,kickoff, *, provider="API-Football", home_id=None):
    home_id=fid*2 if home_id is None else home_id
    data={
        "provider":provider,"id":fid,"utcDate":kickoff,
        "homeTeam":{"id":home_id,"name":"Home "+str(fid)},
        "awayTeam":{"id":fid*2+1,"name":"Away "+str(fid)},
    }
    return {"source":"API-Football","fixture":data}


def modeled(event):
    item=event["fixture"]
    return {
        "fixture":item,"kickoff":item["utcDate"],"past":[],
        "league":"test", "result":{
            "scores":[[1,1,.15],[1,0,.13],[2,1,.12],
                      [0,0,.11],[2,0,.09],[0,1,.08]],
            "model_version":"verified-v8-test",
        }
    }


def test_date_scoped_search_only_models_upcoming_fixtures_four_at_a_time():
    events=[fixture(i,"2026-11-15T08:00:00Z") for i in range(10,16)]
    events.append(fixture(16,"2026-11-14T00:00:00Z"))
    calls=[]
    def discover(key,day,days):
        assert key=="test-key"
        assert day==DAY and days==1
        return events, []
    def model(key,event,model_variant=None):
        calls.append(event["fixture"]["id"])
        assert model_variant=="research_v8"
        return modeled(event),None
    result=research_day("test-key",DAY,limit=20,discover=discover,
                        model=model,now=NOW)
    assert result["discovered"]==6
    assert len(result["fixtures"])==6
    assert {x["fixture_id"] for x in result["fixtures"]}=={
        "10","11","12","13","14","15"}
    assert all(x["source_status"].startswith("provider-listed")
               for x in result["fixtures"])
    assert result["checked"]==4
    assert len(result["modeled"])==4
    assert result["remaining"]==2
    assert calls==[10,11,12,13]
    second=research_day("test-key",DAY,known_ids=result["checked_ids"],
                        discover=discover,model=model,now=NOW)
    assert second["checked"]==2
    assert len(second["modeled"])==2
    assert second["remaining"]==0
    assert calls[-2:]==[14,15]


def test_no_key_never_calls_network():
    def forbid(*args,**kwargs):
        raise AssertionError("No data requests should happen without a valid key")
    report=research_day("",DAY,discover=forbid,model=forbid,now=NOW)
    assert report["checked"]==0
    assert report["modeled"]==[]
    assert "API_FOOTBALL_KEY" in report["errors"][0]


def test_today_search_rejects_started_invalid_and_unmodelled_fixtures():
    events=[fixture(1,"2026-11-15T00:02:00Z"),
            fixture(2,"2026-11-15T04:00:00Z",provider="other"),
            fixture(3,"2026-11-15T05:00:00Z",home_id=None),
            fixture(4,"2026-11-15T07:00:00Z")]
    events[2]["fixture"]["homeTeam"]["id"]=None
    attempts=[]
    def model(_,event,model_variant=None):
        attempts.append(event["fixture"]["id"])
        return None,"Not enough verified team histories"
    report=research_day("key",DAY,
                        discover=lambda *a:(events,[]),
                        model=model,now=NOW)
    assert report["discovered"]==1
    assert report["fixtures"][0]["fixture_id"]=="4"
    assert report["checked"]==1
    assert report["modeled"]==[]
    assert report["unmodeled"]==1
    assert attempts==[4]
    assert "Not enough verified" in report["errors"][0]


def test_does_not_bypass_pre_kickoff_quality_controls():
    report=research_day("key",DAY,discover=lambda *a:([
        fixture(20,"2026-11-15T08:00:00Z")],[]),
        model=lambda key,event,model_variant=None: (
            {**modeled(event),"result":{"scores":[[1,1,.5]]}},None),
        now=NOW)
    assert report["checked"]==1
    assert report["modeled"]==[]
    assert report["unmodeled"]==1


def test_quota_limited_discovery_reports_blocked_not_empty_football_day():
    def no_model(*args,**kwargs):
        raise AssertionError("No modelling permitted after source quota failure")
    report=research_day(
        "test-key",DAY,now=NOW,
        discover=lambda *a: ([],["2026-11-15: API-Football request quota or rate limit"]),
        model=no_model)
    assert report["quota_limited"] is True
    assert report["discovered"]==0
    assert report["checked"]==0
    assert report["modeled"]==[]
    assert report["errors"]


def test_quota_error_stops_other_fixture_model_calls_in_same_search():
    events=[fixture(i,"2026-11-15T08:00:00Z") for i in range(10,15)]
    attempts=[]
    def failed_model(key,event,model_variant=None):
        attempts.append(event["fixture"]["id"])
        return None,"Provider history quota or rate limit"
    result=research_day(
        "test-key",DAY,now=NOW,
        discover=lambda *a:(events,[]),model=failed_model)
    assert result["quota_limited"] is True
    assert result["checked"] == 1
    assert result["remaining"] == 4
    assert attempts == [10]


def test_user_selected_twelve_fixture_batch_is_bounded_and_default_remains_four():
    events=[fixture(i,"2026-11-15T08:00:00Z") for i in range(100,120)]
    calls=[]
    def model(key,event,model_variant=None):
        calls.append(event["fixture"]["id"])
        return modeled(event),None
    first=research_day("key",DAY,discover=lambda *a:(events,[]),
                       model=model,now=NOW)
    assert first["checked"]==4
    second=research_day("key",DAY,known_ids=first["checked_ids"],limit=12,
                        discover=lambda *a:(events,[]),model=model,now=NOW)
    assert second["checked"]==12
    assert second["remaining"]==4
    assert len(set(first["checked_ids"]+second["checked_ids"]))==16
    assert len(calls)==16
    # Oversized caller requests are still hard-capped by policy.
    third=research_day("key",DAY,limit=1000,discover=lambda *a:(events,[]),
                       model=model,now=NOW)
    assert third["checked"]==12
