"""No-network regression checks for the experimental v8 challenger."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import engine
from research_ensemble_v8 import predict, _asof_history
from smart_double_v8 import rank_research_pairs


PAIRS=[(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),
       (1,3),(2,4),(3,5),(4,6),(5,1),(6,2)]


def _fixture(now,home=1,away=2):
    return {"id":999,"utcDate":(now+timedelta(days=3)).isoformat(),
            "provider":"API-Football","status":"SCHEDULED",
            "competition":{"id":44,"code":"TST"},
            "homeTeam":{"id":home,"name":"Team "+str(home)},
            "awayTeam":{"id":away,"name":"Team "+str(away)}}


def _past(now):
    rows=[]
    for n in range(96):
        h,a=PAIRS[n%len(PAIRS)]
        gh=(n//3+2)%5
        ga=(n//5+1)%3
        rows.append({"id":n+1,"provider":"API-Football","competition":{"id":44,"code":"TST"},
                     "utcDate":(now-timedelta(days=180-n)).isoformat(),
                     "status":"FINISHED",
                     "homeTeam":{"id":h,"name":"Team "+str(h)},
                     "awayTeam":{"id":a,"name":"Team "+str(a)},
                     "score":{"fullTime":{"home":gh,"away":ga}}})
    return rows


def test_v8_preserves_baseline_and_normalizes():
    now=datetime.now(timezone.utc).replace(microsecond=0)
    fixture=_fixture(now)
    games=_past(now)
    original=engine.predict(fixture,games)
    assert original is not None
    unchanged=list(original["scores"])
    candidate=predict(fixture,games,league_history=games,as_of=now)
    assert candidate is not None
    assert len(candidate["component_weights"])>=2
    assert abs(float(candidate["matrix"].sum())-1.0)<1e-10
    assert np.all(candidate["matrix"]>=0)
    assert sum(candidate["component_weights"].values())==1.0
    assert candidate["original_baseline_score"]==tuple(original["scores"][0][:2])
    assert original["scores"]==unchanged
    assert candidate["llm_provider_calls"]==0
    assert len(candidate["scores"])==10


def test_v8_cannot_leak_future_or_not_available_results():
    now=datetime.now(timezone.utc).replace(microsecond=0)
    fixture=_fixture(now)
    games=_past(now)
    base=predict(fixture,games,league_history=games,as_of=now)
    extra={**games[-1],"id":987654,
           "utcDate":(now+timedelta(hours=2)).isoformat(),
           "score":{"fullTime":{"home":9,"away":0}}}
    later=games+[extra]
    candidate=predict(fixture,later,league_history=later,as_of=now)
    assert np.allclose(base["matrix"],candidate["matrix"])
    assert len(_asof_history(fixture,later,now))==len(games)


def test_v8_rejects_predictions_after_kickoff():
    now=datetime.now(timezone.utc).replace(microsecond=0)
    fixture=_fixture(now)
    assert predict(fixture,_past(now),as_of=now+timedelta(days=4)) is None


def test_same_sydney_day_four_team_double_ranking():
    tz=ZoneInfo("Australia/Sydney")
    start=datetime(2027,1,11,17,0,tzinfo=tz)
    def item(home,away,dt,fid):
        fixture={"id":fid,"provider":"API-Football",
                 "homeTeam":{"id":fid*2,"name":home},
                 "awayTeam":{"id":fid*2+1,"name":away}}
        return {"fixture":fixture,"kickoff":dt.astimezone(timezone.utc).isoformat(),
                "result":{"scores":[(1,0,.25),(2,0,.20),(1,1,.15)]}}
    a=item("Red","Blue",start,1)
    b=item("Green","Yellow",start+timedelta(hours=3),2)
    c=item("Red","White",start+timedelta(hours=2),3)
    d=item("Black","Silver",start+timedelta(days=1),4)
    ranked=rank_research_pairs([a,b,c,d])
    assert len(ranked)==2
    assert ranked[0]["first"]["fixture"]["id"]==1
    assert ranked[0]["second"]["fixture"]["id"]==2
    assert len(ranked[0]["doubles"])==5
    assert 0<ranked[0]["coverage"]<1


def test_forecast_router_exposes_v8_only_when_opted_in(monkeypatch):
    import forecast_model
    import research_ensemble_v8
    token = {"model_version":"test-v8"}
    calls = []
    def candidate(fixture,history,league_history=None,max_goals=10):
        calls.append((fixture,history,league_history,max_goals))
        return token
    monkeypatch.setattr(research_ensemble_v8,'predict',candidate)
    fixture = {"id":123}
    assert forecast_model.predict(fixture,[],variant='research_v8') is token
    assert len(calls)==1
    assert calls[0][0] is fixture
