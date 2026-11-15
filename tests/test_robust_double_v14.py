"""V14 full-grid robust double tests: no invented scores, no archive mutation."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import numpy as np
import engine
from smart_double_ai import five_doubles
from robust_double_v14 import grids, robust_pair, rank_robust_pairs, VERSION

NOW=datetime(2030,6,1,12,tzinfo=timezone.utc)


def sample(fid,home,away,hours,mu7=(1.4,1.0),mu8=(1.3,1.2)):
    g7=engine._grid(*mu7,max_goals=10)
    g8=engine._grid(*mu8,max_goals=10)
    return {"fixture":{"id":fid,"provider":"API-Football",
                       "utcDate":(NOW+timedelta(hours=hours)).isoformat(),
                       "homeTeam":{"name":home},"awayTeam":{"name":away}},
            "kickoff":(NOW+timedelta(hours=hours)).isoformat(),
            "result":{"model_version":"ziczac-hybrid-v13-prospective-challenger",
                      "component_score_matrices":{"v7":g7,"v8":g8},
                      "component_versions":{"v7":"v7-full","v8":"v8-full"}}}


def pair():
    return sample(10,"Lions","Tigers",15),sample(11,"Bears","Hawks",16)


def test_v14_correct_joint_probability_minimum_and_five_distinct_outcomes():
    a,b=pair()
    before=(deepcopy(a),deepcopy(b))
    chosen=robust_pair(a,b,now=NOW)
    assert chosen is not None
    assert chosen["model_version"]==VERSION
    assert len(chosen["doubles"])==5
    assert chosen["robust_lower_envelope"] is True
    assert chosen["conservative_not_a_calibrated_probability"] is True
    assert chosen["original_forecasts_changed"] is False
    observed={(x["Score 1"],x["Score 2"]) for x in chosen["doubles"]}
    assert len(observed)==5
    for option in chosen["doubles"]:
        assert option["Joint"]==min(option["V7 joint"],option["V8 joint"])
    v7=np.outer(a["result"]["component_score_matrices"]["v7"].ravel(),
                b["result"]["component_score_matrices"]["v7"].ravel())
    v8=np.outer(a["result"]["component_score_matrices"]["v8"].ravel(),
                b["result"]["component_score_matrices"]["v8"].ravel())
    expected=float(np.sort(np.minimum(v7,v8).ravel())[-5:].sum())
    assert np.isclose(chosen["estimated_joint_coverage"],expected)
    for original,copy in zip((a,b),before):
        assert original["fixture"]==copy["fixture"]
        assert all(np.array_equal(original["result"]["component_score_matrices"][k],
                                  copy["result"]["component_score_matrices"][k])
                   for k in ("v7","v8"))


def test_v14_when_models_identical_is_top_v8_joint_probability_ranking():
    a,b=pair()
    for row in (a,b):
        grid=row["result"]["component_score_matrices"]["v8"]
        row["result"]["component_score_matrices"]["v7"]=grid.copy()
        indexes=np.argsort(-grid.ravel(),kind="stable")[:10]
        row["result"]["scores"]=[(int(i//grid.shape[1]),
                                int(i%grid.shape[1]),float(grid.ravel()[i]))
                                for i in indexes]
    chosen=robust_pair(a,b,now=NOW)
    expected=five_doubles(a,b)
    assert [(x["Score 1"],x["Score 2"]) for x in chosen["doubles"]]==[
        (x["Score 1"],x["Score 2"]) for x in expected]
    assert chosen["estimated_joint_coverage"]==__import__("pytest").approx(
        sum(x["Joint"] for x in expected))


def test_v14_refuses_missing_or_fake_grids_and_invalid_teams_dates():
    a,b=pair()
    no_v7=deepcopy(a)
    del no_v7["result"]["component_score_matrices"]["v7"]
    assert grids(no_v7) is None
    assert robust_pair(no_v7,b,now=NOW) is None
    bad=deepcopy(a)
    bad["result"]["component_score_matrices"]["v7"]*=2
    assert robust_pair(bad,b,now=NOW) is None
    assert robust_pair(a,a,now=NOW) is None
    assert robust_pair(a,b,now=NOW+timedelta(days=3)) is None
    shared=deepcopy(b)
    shared["fixture"]["homeTeam"]["name"]="Lions"
    assert robust_pair(a,shared,now=NOW) is None
    another_day=deepcopy(b)
    another_day["kickoff"]=(NOW+timedelta(days=2)).isoformat()
    assert robust_pair(a,another_day,now=NOW) is None


def test_v14_valid_pair_ranking_and_abstain_threshold():
    a,b=pair()
    c=sample(12,"Eagles","Wolves",17,mu7=(2.7,0.6),mu8=(2.5,0.7))
    ranked=rank_robust_pairs([a,b,c],now=NOW,limit=3)
    assert len(ranked)==3
    coverages=[x["estimated_joint_coverage"] for x in ranked]
    assert coverages==sorted(coverages,reverse=True)
    assert rank_robust_pairs([a,b],now=NOW,min_coverage=0.99)==[]
    assert rank_robust_pairs([a,b],now=NOW,limit=0)==[]
