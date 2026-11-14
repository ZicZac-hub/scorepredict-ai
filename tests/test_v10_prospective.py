"""V10 prospective score, market, double and verification integrity tests."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from forecast_ledger import append_file_record, read_file_records
import v10_prospective as shadow
import v10_evaluate as grader
from reconcile_capture_conflicts import merge_verified


def when():
    return datetime.now(timezone.utc).replace(microsecond=0)


def fixture(fid,now,home,away,delta_hours=25):
    return {"id":fid,"provider":"API-Football",
            "utcDate":(now+timedelta(hours=delta_hours)).isoformat(),
            "status":"SCHEDULED","provider_status":"NS",
            "homeTeam":{"id":fid*2,"name":home},
            "awayTeam":{"id":fid*2+1,"name":away},
            "competition":{"code":"PL","id":39}}


def candidate():
    grid=np.array([[.11,.06,.02,.01],[.20,.14,.07,.02],
                   [.14,.10,.06,.02],[.02,.015,.01,.005]],dtype=float)
    grid/=grid.sum()
    rank=np.argsort(-grid.ravel(),kind="stable")[:10]
    scores=[(int(i//4),int(i%4),float(grid.ravel()[i])) for i in rank]
    return {"model_version":"ziczac-evidence-ensemble-v8-experimental",
            "matrix":grid,"scores":scores,
            "home_xg":float((grid*np.arange(4)[:,None]).sum()),
            "away_xg":float((grid*np.arange(4)[None,:]).sum()),
            "component_weights":{"preserved_v6":.55,"opponent_dixon_coles_v7":.45}}


def evidence(now):
    return {"as_of_utc":now.isoformat(),"input_digest":"abc123",
            "policy_version":"two-season-asof-v1",
            "history":[{"x":1}]*35,"league_history":[{"x":1}]*40}


def save(root,game,now,model=None):
    estimate=model or candidate()
    record=shadow._record(game,evidence(now),estimate,"v8",now)
    assert append_file_record(record,root=root,kind="forecasts")
    return record


def verified(row,gh,ga,valid=True):
    kickoff=shadow.utc(row["kickoff_utc"])
    return {"provider":row["provider"],"fixture_id":int(row["fixture_id"]),
            "kickoff_utc":row["kickoff_utc"],"home_team_id":int(row["home_team_id"]),
            "away_team_id":int(row["away_team_id"]),
            "home_team":row["home_team"],"away_team":row["away_team"],
            "score_basis":"regulation","evidence_version":2,
            "home_goals":gh,"away_goals":ga,
            "status":"FT" if valid else "NS",
            "verified_at_utc":(kickoff+timedelta(hours=3)).isoformat()}


def test_capture_original_v8_even_without_market(monkeypatch,tmp_path):
    monkeypatch.setattr(shadow,"research_predict",lambda *a,**k:candidate())
    now=when()
    game=fixture(81001,now,"Club A","Club B")
    got=shadow.capture(game,evidence(now),now,api_key=None,
                       root=tmp_path,market_calls_remaining=0)
    assert got["v8_recorded"] and not got["v9_recorded"]
    saved=read_file_records(root=tmp_path,kind="forecasts")
    assert len(saved)==1 and saved[0]["model_key"]=="v8"
    assert saved[0]["score_basis"]=="regulation"
    assert saved[0]["captured_at_utc"]<saved[0]["kickoff_utc"]
    assert "api_key" not in json.dumps(saved[0]).lower()


def test_verified_market_creates_distinct_v9_without_replacing_v8(monkeypatch,tmp_path):
    monkeypatch.setattr(shadow,"research_predict",lambda *a,**k:candidate())
    now=when()
    game=fixture(81002,now,"Club A","Club B")
    def request(key,endpoint,params,with_metadata=False):
        assert key=="api-test" and endpoint=="/odds" and with_metadata
        return {"metadata":{"freshness_known":True,"stale":False,
                            "fetched_at_utc":now.isoformat(),
                            "retrieved_at_utc":now.isoformat()},
                "data":[{"fixture":{"id":81002},"update":now.isoformat(),
                         "bookmakers":[{"name":"Pinnacle","bets":[
                             {"name":"Match Winner","values":[
                                {"value":"Home","odd":"2.05"},
                                {"value":"Draw","odd":"3.50"},
                                {"value":"Away","odd":"3.95"}]}]}]}]}
    def patched(*args,**kwargs):
        data=dict(candidate())
        data["model_version"]="market-informed-v9-experimental"
        data["market_used"]=True
        return data
    monkeypatch.setattr(shadow,"market_predict",patched)
    got=shadow.capture(game,evidence(now),now,api_key="api-test",
                       request=request,root=tmp_path,market_calls_remaining=1)
    assert got["v8_recorded"] and got["v9_recorded"]
    records=read_file_records(root=tmp_path,kind="forecasts")
    assert {r["model_key"] for r in records}=={"v8","v9"}
    assert sum(bool(r["market_used"]) for r in records)==1
    assert "api-test" not in json.dumps(records)


def test_no_v9_record_for_stale_or_missing_market(monkeypatch,tmp_path):
    monkeypatch.setattr(shadow,"research_predict",lambda *a,**k:candidate())
    now=when()
    game=fixture(81003,now,"Club A","Club B")
    def missing(*args,**kwargs):
        return {"metadata":{"freshness_known":False,"stale":True,
                            "fetched_at_utc":now.isoformat(),
                            "retrieved_at_utc":now.isoformat()},"data":[]}
    got=shadow.capture(game,evidence(now),now,api_key="test",
                       request=missing,root=tmp_path,market_calls_remaining=1)
    assert got["v8_recorded"] and not got["v9_recorded"]
    assert len(read_file_records(root=tmp_path,kind="forecasts"))==1


def test_freeze_five_doubles_is_same_day_and_write_once(tmp_path):
    now=when()
    a=fixture(82101,now,"A","B",delta_hours=26)
    b=fixture(82102,now,"C","D",delta_hours=27)
    records=[save(tmp_path,game,now) for game in (a,b)]
    frozen=grader.freeze_doubles(root=tmp_path,now=now)
    assert len(frozen["days_saved"])==1
    day=frozen["days_saved"][0]
    path=tmp_path/"doubles"/(day+".json")
    original=path.read_bytes()
    packet=json.loads(original)
    assert packet["mode"]=="prospective"
    assert packet["captured_at_utc"]<min(packet["first"]["kickoff_utc"],packet["second"]["kickoff_utc"])
    assert len(packet["five_doubles"])==5
    assert packet["estimated_coverage"]>=grader.MIN_COVERAGE
    assert grader.freeze_doubles(root=tmp_path,now=now)["days_saved"]==[]
    assert original==path.read_bytes()


def test_a_real_verified_hit_and_pending_are_separate(tmp_path):
    now=when()
    a=fixture(82201,now,"Alfa","Beta",26)
    b=fixture(82202,now,"Gamma","Delta",27)
    ra,rb=[save(tmp_path,game,now) for game in (a,b)]
    grader.freeze_doubles(root=tmp_path,now=now)
    complete={str(a["id"]):verified(ra,1,0),str(b["id"]):verified(rb,1,0)}
    report=grader.evaluate(root=tmp_path,results=complete)
    assert report["models"]["v8"]["verified_fixtures"]==2
    assert report["models"]["v8"]["exact_score_hits"]==2
    assert report["models"]["v9"]["frozen_fixtures"]==0
    assert report["five_doubles"]["verified_pairs"]==1
    assert report["five_doubles"]["wins"]==1
    # A not-yet-final result is NOT a verified loss.
    complete[str(b["id"])]["status"]="NS"
    unverified=grader.evaluate(root=tmp_path,results=complete)
    assert unverified["models"]["v8"]["verified_fixtures"]==1
    assert unverified["five_doubles"]["verified_pairs"]==0
    assert unverified["five_doubles"]["awaiting_both_results"]==1


def test_conflicting_final_goals_can_never_be_silently_merged():
    one={"fixture_id":100,"provider":"API-Football","evidence_version":2,
         "score_basis":"regulation","home_goals":1,"away_goals":0}
    two={**one,"home_goals":2}
    with pytest.raises(ValueError):
        merge_verified({"100":one},{"100":two})
    merged=merge_verified({"100":one},{"100":one,"101":{**one,"fixture_id":101}})
    assert set(merged)=={"100","101"}
    assert merged["100"]==one
