"""Two-match AI research budget must be explicit and matchday-aware."""
from datetime import datetime,timezone
import llm_research_panel as panel

UTC=datetime(2031,11,20,0,tzinfo=timezone.utc)


def row(fid,home,away,kickoff):
    return {"provider":"API-Football","fixture_id":str(fid),
            "kickoff_utc":kickoff,"home_team":home,"away_team":away}


def test_one_match_is_default_and_has_no_implicit_api_budget_increase():
    records=[row(1,"A","B","2031-11-20T03:00:00Z"),
             row(2,"C","D","2031-11-20T04:00:00Z")]
    chosen,paired=panel.paired_candidates(records,limit=1)
    assert len(chosen)==1 and paired is False
    assert panel.MAX_MATCHES_PER_RUN==1


def test_two_match_option_prioritizes_valid_same_sydney_day():
    records=[row(1,"A","B","2031-11-20T03:00:00Z"),
             row(2,"A","C","2031-11-20T04:00:00Z"),
             row(3,"D","E","2031-11-20T05:00:00Z"),
             row(4,"F","G","2031-11-21T05:00:00Z")]
    chosen,paired=panel.paired_candidates(records,limit=2)
    assert paired
    assert [x["fixture_id"] for x in chosen]==["1","3"]
    assert len(set(x for row in chosen for x in
                   (row["home_team"],row["away_team"])))==4


def test_two_match_option_can_abstain_and_fall_back_to_one():
    records=[row(1,"A","B","2031-11-20T03:00:00Z"),
             row(2,"B","C","2031-11-20T04:00:00Z")]
    chosen,paired=panel.paired_candidates(records,limit=2)
    assert len(chosen)==1 and paired is False
    assert panel.paired_candidates(records,limit=0)==([],False)
