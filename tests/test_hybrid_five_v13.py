"""V13 independent statistical hybrid and five-double regression suite."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import numpy as np
import pytest
import engine
from hybrid_five_v13 import (
    predict, pair, best_pairs, valid_pair, VERSION, WEIGHTS)
from hybrid_v13_ui import research

NOW=datetime(2030,6,1,12,tzinfo=timezone.utc)
PAIRS=((1,2),(2,3),(3,4),(4,5),(5,6),(6,1),
       (1,3),(2,4),(3,5),(4,6),(5,1),(6,2))


def fixture(fid=7001,home=1,away=2,hours=20):
    return {
        "provider":"API-Football","id":fid,
        "utcDate":(NOW+timedelta(hours=hours)).isoformat(),
        "status":"SCHEDULED","competition":{"id":44,"code":"TST"},
        "homeTeam":{"id":home,"name":"Team "+str(home)},
        "awayTeam":{"id":away,"name":"Team "+str(away)},
    }


def history(n=105):
    rows=[]
    for i in range(n):
        h,a=PAIRS[i%len(PAIRS)]
        rows.append({
            "id":i+1,"provider":"API-Football",
            "utcDate":(NOW-timedelta(days=155-i)).isoformat(),
            "status":"FINISHED","competition":{"id":44,"code":"TST"},
            "homeTeam":{"id":h,"name":"Team "+str(h)},
            "awayTeam":{"id":a,"name":"Team "+str(a)},
            "score":{"fullTime":{"home":(i*7)%5,"away":(i*3+1)%4}}
        })
    return rows


def test_v13_combines_genuine_model_grids_without_touching_inputs():
    f=fixture()
    games=history()
    saved=deepcopy(games)
    v13=predict(f,games,league_history=games,as_of=NOW)
    assert v13 is not None
    assert v13["model_version"]==VERSION
    assert v13["release_status"]=="experimental_not_production"
    assert set(v13["component_weights"])==set(WEIGHTS)
    assert sum(v13["component_weights"].values())==pytest.approx(1.0)
    assert np.isclose(v13["matrix"].sum(),1.0)
    assert np.all(v13["matrix"]>=0)
    assert len(v13["scores"])==10
    assert v13["llm_provider_calls"]==0
    assert v13["independent_prospective_win_rate"] is None
    assert games==saved
    modal=np.unravel_index(int(v13["matrix"].argmax()),v13["matrix"].shape)
    assert tuple(modal)==v13["scores"][0][:2]


def test_v13_abstains_if_research_unavailable_or_kickoff_passed():
    f=fixture()
    assert predict(f,history(12),as_of=NOW) is None
    assert predict(f,history(),as_of=NOW+timedelta(hours=21)) is None


def test_v13_ignore_unavailable_future_results_and_other_leagues():
    f=fixture()
    before=history()
    first=predict(f,before,league_history=before,as_of=NOW)
    assert first is not None
    future={**before[-1],"id":9921,
        "utcDate":(NOW+timedelta(hours=2)).isoformat(),
        "score":{"fullTime":{"home":10,"away":0}}}
    past_other={**before[-1],"id":9922,
        "competition":{"id":999,"code":"OTHER"}}
    later=before+[future,past_other]
    updated=predict(f,later,league_history=later,as_of=NOW)
    assert np.allclose(first["matrix"],updated["matrix"])


def test_v13_renormalizes_when_v12_optional_missing():
    f=fixture()
    games=history(78)
    v13=predict(f,games,league_history=games,as_of=NOW)
    assert v13 is not None
    assert "v12" not in v13["component_weights"]
    assert sum(v13["component_weights"].values())==pytest.approx(1.0)
    assert v13["missing_optional_models"]==["v12"]


def _row(f,forecast):
    return {"fixture":f,"kickoff":f["utcDate"],"result":forecast}


def test_v13_five_double_method_and_pair_constraints():
    games=history()
    a=fixture(7001,1,2)
    b=fixture(7002,3,4,24)
    ra=_row(a,predict(a,games,as_of=NOW))
    rb=_row(b,predict(b,games,as_of=NOW))
    assert valid_pair(ra,rb,now=NOW)
    top=pair(ra,rb,mode="top5",now=NOW)
    balanced=pair(ra,rb,mode="balanced",now=NOW)
    diverse=pair(ra,rb,mode="diversified",now=NOW)
    assert all(p and len(p["doubles"])==5 for p in (top,balanced,diverse))
    assert top["estimated_joint_coverage"]>=balanced["estimated_joint_coverage"]-1e-12
    assert top["estimated_joint_coverage"]>=diverse["estimated_joint_coverage"]-1e-12
    assert top["archived_originals_modified"] is False
    for result in (top,balanced,diverse):
        assert len({(r["Score 1"],r["Score 2"]) for r in result["doubles"]})==5
        assert all(0<r["Joint"]<1 for r in result["doubles"])
    best=best_pairs([ra,rb],mode="top5",now=NOW)
    assert len(best)==1
    assert pair(ra,ra,now=NOW) is None
    assert pair(ra,rb,now=NOW+timedelta(days=3)) is None
    with pytest.raises(ValueError):
        pair(ra,rb,mode="guaranteed",now=NOW)


def test_v13_researcher_never_uses_unvalidated_history_from_other_league():
    f=fixture()
    row={"fixture":f,"kickoff":f["utcDate"],"league":"TST",
         "result":{"scores":[(1,1,.1)]}}
    result,_=research([row],{"TST":history()},now=NOW)
    assert len(result)==1
    assert result[0]["result"]["model_version"]==VERSION
    assert len(result[0]["evidence_digest"])==64
    missing,_=research([row],{"TST":[{**x,"competition":{"id":9,"code":"NO"}}
                                   for x in history()]},now=NOW)
    assert not missing
