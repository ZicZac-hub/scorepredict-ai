"""V25 must show actual Sydney TODAY onward, even with zero score grids."""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from v25_date_discovery import (
    visible_days,date_label,merge_modelled,fixture_table,
)
from ai_five_double_engine import identity

SYDNEY=ZoneInfo("Australia/Sydney")
NOW=datetime(2026,11,16,21,52,tzinfo=timezone.utc)
# 17 November 2026, 08:52 Sydney.


def sample_model(fid,kick):
    return {
        "fixture":{
            "id":fid,"provider":"API-Football","utcDate":kick,
            "homeTeam":{"id":fid*2,"name":"Model Home "+str(fid)},
            "awayTeam":{"id":fid*2+1,"name":"Model Away "+str(fid)}
        },
        "kickoff":kick,
        "result":{"scores":[[0,0,.14],[1,1,.12],[1,0,.11],
                            [2,1,.10],[0,1,.07]]},
    }


def test_calendar_starts_today_and_tomorrow_even_without_models():
    days=visible_days(NOW,{"2026-11-19":[1],"2026-12-10":[2]})
    assert days[0]=="2026-11-17"
    assert days[1]=="2026-11-18"
    assert days[2]=="2026-11-19"
    assert days[-1]=="2026-12-10"
    assert len(days)==15  # default 14-day horizon plus 10 December.
    assert "Today" in date_label(days[0],NOW,eligible_count=0)
    assert "Tomorrow" in date_label(days[1],NOW,eligible_count=0)
    assert "search fixtures" in date_label(days[0],NOW)
    assert "2 fixtures, no score model" in date_label(days[0],NOW,discovered_count=2)
    assert "3 ready" in date_label(days[2],NOW,eligible_count=3)


def test_sydney_date_boundary_does_not_follow_utc_day():
    # UTC November 16 remains Sydney November 17, not 16.
    previous=datetime(2026,11,16,13,20,tzinfo=timezone.utc)
    later=datetime(2026,11,16,21,52,tzinfo=timezone.utc)
    assert visible_days(previous)[0]=="2026-11-17"
    assert visible_days(later)[0]=="2026-11-17"
    midnight=datetime(2026,11,16,12,59,tzinfo=timezone.utc)
    assert visible_days(midnight)[0]=="2026-11-16"


def test_fixture_list_includes_real_provider_games_without_inventing_scores():
    real=[{
        "provider":"API-Football","fixture_id":"7777",
        "kickoff_utc":"2026-11-17T03:00:00+00:00",
        "home_team":"Real Alpha","away_team":"Real Beta",
        "competition":"Test","source_status":"provider-listed"
    },{
        "provider":"API-Football","fixture_id":"8888",
        "kickoff_utc":"2026-11-18T03:00:00+00:00",
        "home_team":"Tomorrow A","away_team":"Tomorrow B"
    }]
    listed=fixture_table("2026-11-17",[],real)
    assert listed==[{
        "Kickoff (Sydney)":"2:00 PM",
        "Match":"Real Alpha vs Real Beta",
        "Score model":"Fixture found · not yet qualified"
    }]
    # When one fixture earns a genuine full score model, it is flagged ready;
    # this has nothing to do with inventing an exact-score forecast.
    model=sample_model(7777,"2026-11-17T03:00:00+00:00")
    valid=[{
        "row":model,"title":"Real Alpha vs Real Beta",
        "kickoff":datetime.fromisoformat(model["kickoff"])
    }]
    rows=fixture_table("2026-11-17",valid,real)
    assert len(rows)==1
    assert rows[0]["Score model"]=="V25 ready · full pre-match grid"


def test_session_modelled_rows_deduped_without_mutating_frozen_sources():
    first=sample_model(1234,"2026-11-17T15:00:00+00:00")
    second=sample_model(5678,"2026-11-17T17:00:00+00:00")
    orig_first=first.copy()
    combined=merge_modelled([first],[first,second])
    assert len(combined)==2
    assert identity(combined[0])==identity(first)
    assert combined[0] is first
    assert first==orig_first


def test_invalid_horizon_and_old_days_not_present():
    for h in [0,1,32,1.2]:
        try:
            visible_days(NOW,horizon=h)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid date horizon must fail")
    assert "2026-11-16" not in visible_days(NOW,{"2026-11-16":[1]})

def test_cloud_entrypoint_exposes_today_forward_release_marker():
    """Distinguish a deployed new V25 app from a stale Streamlit process."""
    from pathlib import Path
    app=Path("app.py").read_text(encoding="utf-8")
    smoke=Path(".github/workflows/streamlit-live-smoke.yml").read_text(
        encoding="utf-8")
    assert "V25 CALENDAR · 17 NOV 2026 UPDATE · TODAY & TOMORROW ENABLED" in app
    assert app.index("V25 CALENDAR · 17 NOV 2026 UPDATE") < app.index(
        'render_ai_five_double_studio(st, [], {})')
    for path in ("v25_upcoming_ui.py","v25_date_discovery.py",
                 "ai_five_live_finder.py"):
        assert path in smoke
