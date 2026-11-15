"""V12 frozen shadow-forecast regression checks without network or paid APIs."""
from datetime import datetime, timezone, timedelta
import numpy as np
import engine
import v10_prospective as shadow
import v10_evaluate as evaluate
from forecast_ledger import read_file_records
from score_calibration_v12 import VERSION


NOW=datetime(2030,6,1,12,tzinfo=timezone.utc)


def upcoming():
    return {
        "provider":"API-Football","id":67001,"status":"SCHEDULED",
        "utcDate":(NOW+timedelta(days=1)).isoformat(),
        "homeTeam":{"id":2,"name":"Home"},
        "awayTeam":{"id":3,"name":"Away"},
        "competition":{"id":39,"code":"PL"},
    }


def past():
    games=[]
    for i in range(115):
        game={
            "id":i+1,"provider":"API-Football",
            "utcDate":(NOW-timedelta(days=170-i)).isoformat(),
            "status":"FINISHED","competition":{"id":39,"code":"PL"},
            "homeTeam":{"id":(i%6)+1,"name":"Home league"},
            "awayTeam":{"id":((i+2)%6)+1,"name":"Away league"},
            "score":{"fullTime":{"home":i%4,"away":(i//3)%4}}
        }
        games.append(game)
    return games


def baseline():
    matrix=engine._grid(1.40,1.12,max_goals=10)
    rank=np.argsort(-matrix.ravel(),kind="stable")[:10]
    scores=[(int(i//matrix.shape[1]),int(i%matrix.shape[1]),
             float(matrix.ravel()[i])) for i in rank]
    return {
        "model_version":"ziczac-evidence-ensemble-v8-experimental",
        "matrix":matrix,"scores":scores,
        "home_xg":float((matrix*np.arange(matrix.shape[0])[:,None]).sum()),
        "away_xg":float((matrix*np.arange(matrix.shape[1])[None,:]).sum()),
        "component_weights":{"preserved_v6":0.55,"opponent_dixon_coles_v7":0.45},
    }


def evidence():
    games=past()
    return {"as_of_utc":NOW.isoformat(),"input_digest":"v12-test-same-evidence",
            "policy_version":"two-season-asof-v1","history":games,
            "league_history":games}


def settled_record(original,home=1,away=0):
    return {
        "provider":original["provider"],"fixture_id":int(original["fixture_id"]),
        "kickoff_utc":original["kickoff_utc"],
        "home_team_id":int(original["home_team_id"]),
        "away_team_id":int(original["away_team_id"]),
        "home_team":original["home_team"],"away_team":original["away_team"],
        "score_basis":"regulation","evidence_version":2,"home_goals":home,
        "away_goals":away,"status":"FT",
        "verified_at_utc":(shadow.utc(original["kickoff_utc"])+timedelta(hours=4)).isoformat()
    }


def test_v12_shadow_saves_separately_from_v8_and_grades_fairly(tmp_path,monkeypatch):
    monkeypatch.setattr(shadow,"research_predict",lambda *args,**kwargs:baseline())
    record=upcoming()
    data=evidence()
    result=shadow.capture(record,data,NOW,root=tmp_path)
    assert result["v8_recorded"] is True
    assert result["v12_recorded"] is True
    saved=shadow.read_latest(tmp_path)
    v8=saved[("API-Football",str(record["id"]),"v8")]
    v12=saved[("API-Football",str(record["id"]),"v12")]
    assert v8["input_digest"]==v12["input_digest"]
    assert v8["score_matrix"]==baseline()["matrix"].tolist()
    assert v12["model_version"]==VERSION
    assert v12["release_status"]=="experimental_shadow_only_not_promoted"
    assert v8["captured_at_utc"]<=v12["captured_at_utc"]<v8["kickoff_utc"]
    assert len(read_file_records(root=tmp_path,kind="forecasts"))==2
    # The write-once ledger rejects another snapshot with same model identity.
    assert shadow.capture(record,data,NOW,root=tmp_path)["v8_recorded"] is False
    report=evaluate.evaluate(
        root=tmp_path,results={str(record["id"]):settled_record(v8)})
    assert report["models"]["v12"]["verified_fixtures"]==1
    assert report["paired_v8_v12"]["verified_same_fixture_count"]==1
    assert report["paired_v8_v12"]["insufficient_to_claim_improvement"] is True


def test_insufficient_league_history_preserves_original_only(tmp_path,monkeypatch):
    monkeypatch.setattr(shadow,"research_predict",lambda *args,**kwargs:baseline())
    data=evidence()
    data["league_history"]=data["league_history"][:18]
    result=shadow.capture(upcoming(),data,NOW,root=tmp_path)
    assert result["v8_recorded"] is True
    assert result["v12_recorded"] is False
    assert set(r["model_key"] for r in read_file_records(root=tmp_path))=={"v8"}
