"""Research-pool and automatic scheduled five-double display regression tests."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json

from zoneinfo import ZoneInfo
from smart_double_v8 import select_auto_research_pool
from automatic_doubles_ui import read_upcoming,result_status


SYD=ZoneInfo("Australia/Sydney")


def _row(fid,kickoff,form=8):
    return {
      "kickoff":kickoff.astimezone(timezone.utc).isoformat(),
      "fixture":{"id":fid,"provider":"API-Football",
                 "homeTeam":{"id":fid*2,"name":"Home "+str(fid)},
                 "awayTeam":{"id":fid*2+1,"name":"Away "+str(fid)}},
      "result":{"scores":[(1,0,.12),(1,1,.10)],
                "home_last10":[1]*form,"away_last10":[1]*form,
                "home_home10":[1]*5,"away_away10":[1]*5}
    }


def test_research_pool_reserves_same_day_pairs_and_quality():
    now=datetime(2027,1,11,8,tzinfo=SYD)
    first=now+timedelta(hours=4)
    rows=[]
    for day in range(4):
        for n in range(10):
            rows.append(_row(day*100+n,first+timedelta(days=day,minutes=30*n),
                             form=8 if n%2 else 4))
    chosen=select_auto_research_pool(rows,max_fixtures=18,max_days=3,as_of=now)
    assert len(chosen)==18
    ids={r["fixture"]["id"] for r in chosen}
    assert len(ids)==18
    days={}
    for row in chosen:
        day=datetime.fromisoformat(row["kickoff"]).astimezone(SYD).date().isoformat()
        days[day]=days.get(day,0)+1
    assert len(days)==3
    assert min(days.values())>=2
    assert (first+timedelta(days=3)).date().isoformat() not in days
    # When there is a poor and strong source on each day, research at least
    # two per date rather than accidentally using only earliest fixtures.
    for day in range(3):
        assert any(r["fixture"]["id"]//100==day and
                   len(r["result"]["home_last10"])==8 for r in chosen)


def test_read_only_auto_doubles_does_not_show_unfrozen_or_stale_day(tmp_path):
    now=datetime(2027,1,11,10,tzinfo=SYD).astimezone(timezone.utc)
    first=now+timedelta(hours=5)
    second=now+timedelta(hours=6)
    date=first.astimezone(SYD).date().isoformat()
    path=tmp_path/"doubles"/(date+".json")
    path.parent.mkdir(parents=True)
    packet={"mode":"prospective","status":"frozen_pre_kickoff",
            "sydney_date":date,"captured_at_utc":now.isoformat(),
            "first":{"fixture_id":"1","kickoff_utc":first.isoformat(),
                     "home_team":"North","away_team":"South","model_key":"v8_near"},
            "second":{"fixture_id":"2","kickoff_utc":second.isoformat(),
                      "home_team":"East","away_team":"West","model_key":"v8_near"},
            "estimated_coverage":.054,
            "five_doubles":[{"score1":f"{i}-1","score2":"1-0",
                             "joint_probability":.0108}
                             for i in range(5)]}
    path.write_text(json.dumps(packet))
    selected=read_upcoming(tmp_path,now)
    assert selected["first"]["model_key"]=="v8_near"
    assert len(selected["five_doubles"])==5
    assert read_upcoming(tmp_path,now+timedelta(hours=10)) is None
    packet["captured_at_utc"]=(second+timedelta(minutes=2)).isoformat()
    path.write_text(json.dumps(packet))
    assert read_upcoming(tmp_path,now) is None


def test_automatic_doubles_outcome_remains_pending_until_verified(tmp_path):
    packet={"sydney_date":"2027-01-11"}
    file=tmp_path/"performance.json"
    file.write_text(json.dumps({"five_doubles":{"rows":[]}}))
    assert result_status(packet,file)["status"]=="awaiting_verified_results"
    file.write_text(json.dumps({"five_doubles":{"rows":[{"sydney_date":"2027-01-11",
                            "status":"verified","hit":True,"actual_scores":["1-0","1-1"]}]}}))
    assert result_status(packet,file)["hit"] is True
