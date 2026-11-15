"""V13 write-once prospective single and five-double verification tests."""
from datetime import datetime,timedelta,timezone
import numpy as np
import engine
import v10_prospective as capture
import v10_evaluate as verify
from forecast_ledger import read_file_records

NOW=datetime(2030,6,1,12,tzinfo=timezone.utc)


def fixture(fid,home,away,hours):
    return {"provider":"API-Football","id":fid,"status":"SCHEDULED",
            "utcDate":(NOW+timedelta(hours=hours)).isoformat(),
            "homeTeam":{"id":home,"name":"Team "+str(home)},
            "awayTeam":{"id":away,"name":"Team "+str(away)},
            "competition":{"id":44,"code":"TST"}}


def past():
    pairs=[(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),
           (1,3),(2,4),(3,5),(4,6),(5,1),(6,2)]
    games=[]
    for i in range(110):
        home,away=pairs[i%len(pairs)]
        games.append({"id":i+1,"provider":"API-Football",
            "utcDate":(NOW-timedelta(days=150-i)).isoformat(),
            "status":"FINISHED","competition":{"id":44,"code":"TST"},
            "homeTeam":{"id":home,"name":"Team "+str(home)},
            "awayTeam":{"id":away,"name":"Team "+str(away)},
            "score":{"fullTime":{"home":i%4,"away":(i//3)%4}}})
    return games


def v8():
    grid=engine._grid(1.4,1.2)
    indices=np.argsort(-grid.ravel(),kind="stable")[:10]
    return {"model_version":"v8-test-research","matrix":grid,
            "scores":[(int(i//grid.shape[1]),int(i%grid.shape[1]),
                       float(grid.ravel()[i])) for i in indices],
            "home_xg":float((grid*np.arange(grid.shape[0])[:,None]).sum()),
            "away_xg":float((grid*np.arange(grid.shape[1])[None,:]).sum()),
            "component_weights":{"preserved_v6":.55,"opponent_dixon_coles_v7":.45}}


def evidence(f):
    return {"as_of_utc":NOW.isoformat(),
            "input_digest":"different-fixture-"+str(f["id"]),
            "policy_version":"two-season-asof-v1",
            "history":past(),"league_history":past()}


def actual(row,home,away):
    return {
        "provider":row["provider"],"fixture_id":int(row["fixture_id"]),
        "kickoff_utc":row["kickoff_utc"],
        "home_team_id":int(row["home_team_id"]),
        "away_team_id":int(row["away_team_id"]),
        "home_team":row["home_team"],"away_team":row["away_team"],
        "score_basis":"regulation","evidence_version":2,
        "home_goals":home,"away_goals":away,"status":"FT",
        "verified_at_utc":(capture.utc(row["kickoff_utc"])+timedelta(hours=4)).isoformat()}


def test_v13_shadow_and_alternates_are_frozen_before_kickoff(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,"research_predict",lambda *args,**kwargs:v8())
    f1=fixture(71001,1,2,20)
    f2=fixture(71002,3,4,21)
    for f in (f1,f2):
        report=capture.capture(f,evidence(f),NOW,root=tmp_path)
        assert report["v8_recorded"] is True
        assert report["v13_recorded"] is True
        assert not report["v9_recorded"]
    before=capture.read_latest(tmp_path)
    for f in (f1,f2):
        identity=("API-Football",str(f["id"]))
        saved8=before[(*identity,"v8")]
        saved13=before[(*identity,"v13")]
        assert saved13["input_digest"]==saved8["input_digest"]
        assert saved13["model_key"]=="v13"
        assert saved13["captured_at_utc"]<=saved13["kickoff_utc"]
        assert saved13["release_status"]=="experimental_shadow_only_not_promoted"
        assert saved8["score_matrix"]==v8()["matrix"].tolist()
    frozen=verify.freeze_doubles(root=tmp_path,now=NOW)
    assert len(frozen["days_saved"])==1
    import json
    saved_day=frozen["days_saved"][0]
    packet=json.loads((tmp_path/"doubles"/(saved_day+".json")).read_text())
    assert set(["v13","v13_balanced","v13_diversified"]).issubset(
        packet["same_pair_challengers"])
    for name in ("v13","v13_balanced","v13_diversified"):
        choices=packet["same_pair_challengers"][name]
        assert len(choices)==5
        assert len({(x["score1"],x["score2"]) for x in choices})==5
    frozen_bytes=(tmp_path/"doubles"/(saved_day+".json")).read_bytes()
    assert verify.freeze_doubles(root=tmp_path,now=NOW)["days_saved"]==[]
    assert frozen_bytes==(tmp_path/"doubles"/(saved_day+".json")).read_bytes()
    verified={str(f["id"]):actual(before[("API-Football",str(f["id"]),"v8")],
                                 1,0) for f in (f1,f2)}
    scored=verify.evaluate(root=tmp_path,results=verified)
    assert scored["models"]["v13"]["verified_fixtures"]==2
    assert scored["paired_v8_v13"]["verified_same_fixture_count"]==2
    for name in ("v13","v13_balanced","v13_diversified"):
        assert scored["five_doubles"]["same_pair_challengers"][name]["same_pair_verified_days"]==1
    assert scored["promotion_gate"].startswith("Never promote")


def test_v13_absent_when_no_genuine_population(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,"research_predict",lambda *a,**k:v8())
    f=fixture(72001,1,2,20)
    packet=evidence(f)
    packet["league_history"]=packet["league_history"][:12]
    result=capture.capture(f,packet,NOW,root=tmp_path)
    assert result["v8_recorded"] is True
    assert result["v13_recorded"] is False
    assert "v13" not in {row["model_key"] for row in
                         read_file_records(root=tmp_path,kind="forecasts")}
