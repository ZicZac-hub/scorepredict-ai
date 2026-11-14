"""Freeze conflict rules: keep FIRST valid published pick, never rewrite."""
from datetime import datetime,timedelta,timezone
import copy
from reconcile_capture_conflicts import valid_freeze


def test_daily_pick_first_publication_must_be_pre_kickoff():
    now=datetime.now(timezone.utc)
    path="data/daily_picks/"+now.date().isoformat()+".json"
    row={"prediction_date":now.date().isoformat(),
         "status":"frozen_in_capture_workflow",
         "captured_at_utc":now.isoformat(),
         "kickoff_utc":(now+timedelta(hours=2)).isoformat(),
         "exact_score":[1,0]}
    assert valid_freeze(path,row)
    bad={**row,"captured_at_utc":(now+timedelta(hours=3)).isoformat()}
    assert not valid_freeze(path,bad)
    assert not valid_freeze(path,{**row,"exact_score":["1","0"]})
    assert not valid_freeze("data/daily_picks/2000-01-01.json",row)


def test_five_double_original_is_pre_match_and_unique():
    now=datetime.now(timezone.utc)
    path="data/v10_prospective/doubles/"+now.date().isoformat()+".json"
    packet={"sydney_date":now.date().isoformat(),
            "status":"frozen_pre_kickoff","captured_at_utc":now.isoformat(),
            "first":{"fixture_id":10,
                     "kickoff_utc":(now+timedelta(hours=2)).isoformat()},
            "second":{"fixture_id":11,
                      "kickoff_utc":(now+timedelta(hours=4)).isoformat()},
            "five_doubles":[{"score1":str(x)+"-0","score2":"1-1"}
                            for x in range(5)]}
    assert valid_freeze(path,packet)
    assert not valid_freeze(path,{**packet,"sydney_date":"1900-01-01"})
    assert not valid_freeze(path,{**packet,"five_doubles":packet["five_doubles"][:4]})
    assert not valid_freeze(path,{**packet,
         "first":{**packet["first"],"fixture_id":11}})
    assert not valid_freeze(path,{**packet,
         "captured_at_utc":(now+timedelta(hours=3)).isoformat()})
