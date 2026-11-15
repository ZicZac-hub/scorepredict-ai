"""V21 source-gated AI Auto Select: compare all pairs, abstain without data."""
from datetime import datetime,timedelta,timezone
import numpy as np
from v21_pair_selector import candidates
from v21_goal_model_comparison import _poisson,top_scores

NOW=datetime(2031,7,1,7,tzinfo=timezone.utc)
LEAGUE={"id":100,"code":"PL"}
MATRIX=np.outer(_poisson(1.9,11),_poisson(1.2,11))


def row(fid,home,away,hour=5):
    kick=(NOW+timedelta(hours=hour)).isoformat()
    fixture={"provider":"API-Football","id":fid,"utcDate":kick,
             "homeTeam":{"id":home,"name":str(home)},
             "awayTeam":{"id":away,"name":str(away)},
             "competition":LEAGUE}
    hist=[]
    for i in range(1,11):
        dt=(NOW-timedelta(days=i)).isoformat()
        hist.append({"id":fid*100+i,"provider":"API-Football",
                    "competition":LEAGUE,"status":"FINISHED",
                    "utcDate":dt,"homeTeam":{"id":home},
                    "awayTeam":{"id":fid*1000+i},
                    "score":{"fullTime":{"home":3,"away":1}}})
        hist.append({"id":fid*100+50+i,"provider":"API-Football",
                    "competition":LEAGUE,"status":"FINISHED",
                    "utcDate":dt,"homeTeam":{"id":fid*2000+i},
                    "awayTeam":{"id":away},
                    "score":{"fullTime":{"home":1,"away":2}}})
    return {"fixture":fixture,"kickoff":kick,"league":"PL",
            "past":hist,"result":{"matrix":MATRIX,"scores":top_scores(MATRIX)}}


def test_all_matching_research_pairs_ranked():
    rows=[row(1,11,12),row(2,21,22),row(3,31,32)]
    result=candidates(rows,now=NOW,min_coverage=.001)
    assert result["eligible_fixtures"]==3
    assert result["pairs_above_threshold"]==3
    assert len(result["pairs"])==3
    for candidate in result["pairs"]:
        assert candidate["conservative_coverage"]>0
        assert candidate["not_observed_win_rate"]


def test_no_data_abstains_no_fake_1_1():
    rows=[row(1,11,12),row(2,21,22)]
    for r in rows:
        r["past"]=[]
    result=candidates(rows,now=NOW,min_coverage=0)
    assert result["abstained"]
    assert result["reasons"]["insufficient_dated_goal_evidence"]==2


def test_extreme_coverage_gate_abstains():
    rows=[row(1,11,12),row(2,21,22)]
    result=candidates(rows,now=NOW,min_coverage=1.0)
    assert result["abstained"]
    assert result["reasons"]["below_coverage"]==1
