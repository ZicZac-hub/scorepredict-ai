"""V17 tests: confidence estimates never masquerade as observed accuracy."""
from datetime import datetime,timedelta,timezone
import json

import pytest
from v17_confidence import (probabilities,candidates,select,freeze,evaluate,
                            wilson,evidence_dashboard,VERSION)

NOW = datetime(2035,6,1,12,tzinfo=timezone.utc)


def forecast(fid,home,away,minutes=120, matrix=None):
    grid=matrix or [[0.06,0.04],[0.04,0.86]]
    row={"model_key":"v8","provider":"API-Football",
         "fixture_id":str(fid),"forecast_id":"source"+str(fid),
         "input_digest":"digest"+str(fid),
         "home_team_id":str(home),"away_team_id":str(away),
         "home_team":"Team "+str(home),"away_team":"Team "+str(away),
         "captured_at_utc":(NOW-timedelta(hours=2)).isoformat(),
         "evidence_as_of_utc":(NOW-timedelta(hours=2,minutes=1)).isoformat(),
         "kickoff_utc":(NOW+timedelta(minutes=minutes)).isoformat(),
         "score_matrix":grid,
         "outcome_probabilities":{"home":sum(v for i,a in enumerate(grid) for j,v in enumerate(a) if i>j),
                                  "draw":sum(v for i,a in enumerate(grid) for j,v in enumerate(a) if i==j),
                                  "away":sum(v for i,a in enumerate(grid) for j,v in enumerate(a) if i<j)},
         "history_matches":120,"league_history_matches":120,
         "component_weights":{"v6":0.5,"v7":0.5}}
    return row


def source_record(root,model):
    folder=root/"forecasts"/model["provider"]/model["fixture_id"]
    folder.mkdir(parents=True,exist_ok=True)
    (folder/(model["forecast_id"]+".json")).write_text(json.dumps(model))


def finished(row,home=0,away=0):
    return {"provider":row["provider"],"fixture_id":int(row["fixture_id"]),
            "home_team_id":row["home_team_id"],
            "away_team_id":row["away_team_id"],
            "kickoff_utc":row["kickoff_utc"],
            "home_goals":home,"away_goals":away,
            "status":"FT","score_basis":"regulation",
            "evidence_version":2,
            "verified_at_utc":(NOW+timedelta(hours=5)).isoformat()}


def test_model_masses_and_probability_validation():
    row=forecast(1,11,12)
    p=probabilities(row)
    assert p["1"]==pytest.approx(0.04)
    assert p["X"]==pytest.approx(0.92)
    assert p["2"]==pytest.approx(0.04)
    assert p["1X"]==pytest.approx(0.96)
    row["outcome_probabilities"]["draw"]=0.5
    with pytest.raises(ValueError,match="mismatch"):
        probabilities(row)


def test_rejects_missing_digest_late_timestamp_and_tampered_grid():
    a=forecast(1,11,12)
    a["input_digest"]=""
    with pytest.raises(ValueError):
        probabilities(a)
    a=forecast(1,11,12)
    a["captured_at_utc"]=(NOW+timedelta(hours=3)).isoformat()
    with pytest.raises(ValueError):
        probabilities(a)
    a=forecast(1,11,12)
    a["score_matrix"][0][1]=float("nan")
    with pytest.raises(ValueError):
        probabilities(a)


def test_separate_single_and_pair_and_four_team_requirement():
    a,b=forecast(1,11,12),forecast(2,13,14)
    selected,fail=select([a,b],NOW)
    assert not fail
    row=list(selected.values())[0]
    assert row["single"]["selection"] in ("1X","X2","12")
    assert row["pair"]["joint_model_estimate"]==pytest.approx(0.96**2)
    assert row["pair"]["independence_assumption"]
    # A duplicated team is not a valid two-match pair.
    c=forecast(3,11,15)
    selected,_=select([a,c],NOW)
    assert list(selected.values())[0]["pair"] is None


def test_abstains_when_estimated_mass_is_below_target():
    flat=[[0.25,0.25],[0.25,0.25]]
    a,b=forecast(1,11,12,matrix=flat),forecast(2,13,14,matrix=flat)
    selected,_=select([a,b],NOW)
    day=list(selected.values())[0]
    assert day["single"] is not None  # 75% estimated single chance
    assert day["pair"] is None  # 56.25% estimated chance for both
    assert day["pair_abstention"]


def test_freeze_write_once_and_verified_both_leg_settlement(tmp_path):
    source=tmp_path/"source"
    dest=tmp_path/"confidence"
    a,b=forecast(1,11,12),forecast(2,13,14)
    for row in (a,b):
        source_record(source,row)
    first=freeze(root=dest,source=source,now=NOW,forecasts=[a,b])
    assert len(first["frozen_dates"])==1
    path=list((dest/"forecasts").glob("*.json"))[0]
    original=path.read_text()
    assert not freeze(root=dest,source=source,now=NOW,forecasts=[a,b])["frozen_dates"]
    assert path.read_text()==original
    verified={a["fixture_id"]:finished(a,0,0)}
    pending=evaluate(root=dest,source=source,verified=verified)
    assert pending["single"]["verified"]==1
    assert pending["two_match_double_chance"]["verified"]==0
    assert pending["two_match_double_chance"]["pending"]==1
    verified[b["fixture_id"]]=finished(b,0,0)
    result=evaluate(root=dest,source=source,verified=verified)
    assert result["two_match_double_chance"]["verified"]==1
    assert result["two_match_double_chance"]["wins"]==1
    assert result["meets_70_percent_claim_gate"] is False
    assert result["two_match_double_chance"]["wilson_95"][0]<0.7


def test_wrong_team_result_and_source_digest_never_grade_as_win(tmp_path):
    source=tmp_path/"source"
    dest=tmp_path/"confidence"
    a,b=forecast(1,11,12),forecast(2,13,14)
    for row in (a,b):
        source_record(source,row)
    freeze(root=dest,source=source,now=NOW,forecasts=[a,b])
    wrong=finished(b,0,0)
    wrong["home_team_id"]="wrong"
    result=evaluate(root=dest,source=source,
                    verified={a["fixture_id"]:finished(a,0,0),
                              b["fixture_id"]:wrong})
    assert result["two_match_double_chance"]["verified"]==0
    path=source/"forecasts"/a["provider"]/a["fixture_id"]/(a["forecast_id"]+".json")
    altered=json.loads(path.read_text()); altered["input_digest"]="changed"
    path.write_text(json.dumps(altered))
    result=evaluate(root=dest,source=source,
                    verified={a["fixture_id"]:finished(a,0,0),
                              b["fixture_id"]:finished(b,0,0)})
    assert result["single"]["verified"]==0
    assert result["two_match_double_chance"]["verified"]==0


def test_no_retroactive_freezing_for_completed_fixtures(tmp_path):
    source=tmp_path/"source"; dest=tmp_path/"dest"
    a=forecast(1,11,12,minutes=-1)
    source_record(source,a)
    result=freeze(root=dest,source=source,now=NOW,forecasts=[a])
    assert not result["frozen_dates"]
    assert not list((dest/"forecasts").glob("*.json"))


def test_wilson_and_evidence_status_are_honest(tmp_path):
    assert wilson(0,0) is None
    assert wilson(1,1)[0]<0.7
    assert wilson(100,100)[0]>0.7
    assert evidence_dashboard(source=tmp_path)["real_xg_feed"] is False
