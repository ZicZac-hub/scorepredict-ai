"""V16 frozen pair prioritisation: immutable evidence, no AI billing."""
import json
from datetime import datetime,timezone

from v16_pair_research import frozen_pair_priority
import llm_research_panel as ai

NOW=datetime(2031,11,19,22,tzinfo=timezone.utc)
# 03:00 UTC Nov 20 is 14:00 Sydney Nov 20.
DAY="2031-11-20"


def _write(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc))


def _record(fid,home,away,hid,aid,hour=3):
    return {"provider":"API-Football","fixture_id":str(fid),
            "home_team":home,"away_team":away,
            "home_team_id":str(hid),"away_team_id":str(aid),
            "model_key":"v8","forecast_id":"original-"+str(fid),
            "input_digest":"f"*63+str(fid)[-1],
            "captured_at_utc":"2031-11-19T18:00:00Z",
            "kickoff_utc":f"2031-11-20T0{hour}:00:00Z",
            "expected_goals":{"home":1.2,"away":1.1},
            "top10_scores":[[1,1,.12],[1,0,.10],[0,1,.08],
                            [0,0,.08],[2,1,.07]]}


def _setup(root,*,bad=False):
    first=_record(101,"A","B",1,2,3)
    second=_record(102,"C","D",3,4,4)
    for model in (first,second):
        _write(root/"forecasts"/"API-Football"/model["fixture_id"]/
               (model["forecast_id"]+".json"),model)
    pairs=[{"score1":"1-1","score2":"1-1","joint_probability":.02},
           {"score1":"1-0","score2":"1-1","joint_probability":.017},
           {"score1":"0-1","score2":"0-0","joint_probability":.013},
           {"score1":"2-1","score2":"0-1","joint_probability":.011},
           {"score1":"0-0","score2":"1-0","joint_probability":.009}]
    original={"mode":"prospective","status":"frozen_pre_kickoff",
              "captured_at_utc":"2031-11-19T18:00:00Z",
              "sydney_date":DAY,"first":first,"second":second,
              "five_doubles":pairs}
    if bad:
        original["second"]["forecast_id"]="different-forecast-id"
    _write(root/"doubles"/(DAY+".json"),original)
    return first,second


def test_exact_original_pair_is_prioritized_even_if_newer_generic_fixtures_exist(tmp_path):
    a,b=_setup(tmp_path)
    chosen,day=frozen_pair_priority(tmp_path,NOW,set(),ai.eligible)
    assert day==DAY
    assert [r["forecast_id"] for r in chosen]==[
        a["forecast_id"],b["forecast_id"]]
    assert all(ai.eligible(r,NOW) for r in chosen)


def test_missing_original_digest_or_mismatched_forecast_abstains(tmp_path):
    _setup(tmp_path,bad=True)
    selected,day=frozen_pair_priority(tmp_path,NOW,set(),ai.eligible)
    assert selected==[] and day is None


def test_no_repeated_cost_when_any_original_leg_already_researched(tmp_path):
    first,_=_setup(tmp_path)
    selected,day=frozen_pair_priority(
        tmp_path,NOW,{("API-Football",first["fixture_id"])},ai.eligible)
    assert selected==[] and day is None


def test_no_after_kickoff_or_backdated_original_pair(tmp_path):
    _setup(tmp_path)
    after=datetime(2031,11,20,5,tzinfo=timezone.utc)
    selected,day=frozen_pair_priority(tmp_path,after,set(),ai.eligible)
    assert selected==[] and day is None


def test_no_two_match_mode_exceeds_owner_budget(tmp_path):
    records=[_record(101,"A","B",1,2),
             _record(102,"C","D",3,4,4)]
    result,paired=ai.paired_candidates(records,limit=1)
    assert len(result)==1 and paired is False
    assert ai.MAX_MATCHES_PER_RUN==1


def test_different_sydney_day_is_never_relabelled_a_valid_pair(tmp_path):
    _setup(tmp_path)
    p=tmp_path/"doubles"/(DAY+".json")
    data=json.loads(p.read_text())
    data["sydney_date"]="2031-11-21"
    p.write_text(json.dumps(data))
    result,day=frozen_pair_priority(tmp_path,NOW,set(),ai.eligible)
    assert result==[] and day is None
