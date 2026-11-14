"""Safe time-zone/date selection and frozen daily pick regression tests."""
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import daily_pick_ui as daily

SYDNEY=ZoneInfo("Australia/Sydney")


def row(name,utc,prob=.1,home_games=10,venue_games=7,ident=1):
    return {"league":"Premier League",
            "fixture":{"provider":"API-Football","id":ident,
                       "utcDate":utc.isoformat(),
                       "homeTeam":{"id":ident*10+1,"name":name+" Home"},
                       "awayTeam":{"id":ident*10+2,"name":name+" Away"}},
            "kickoff":utc.isoformat(),
            "past":[],
            "result":{"model_version":daily.MODEL,
                      "scores":[(2,1,prob),(1,0,prob*.8),(1,1,prob*.7)],
                      "home_win":.6,"draw":.25,"away_win":.15,
                      "home_xg":1.6,"away_xg":1.1,
                      "home_last10":[1]*home_games,
                      "away_last10":[1]*home_games,
                      "home_home10":[1]*venue_games,
                      "away_away10":[1]*venue_games,
                      "h2h_last10":[],
                      "component_weights":{"preserved_v6":.45,"elo_goal_contrast":.2,"opponent_dixon_coles_v7":.35},
                      "component_count":3,"component_top_score_agreement":2}}


def test_chooses_today_over_higher_probability_tomorrow():
    now=datetime(2026,11,14,10,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    today=row("Today",now+timedelta(hours=2),.08,ident=1)
    tomorrow=row("Tomorrow",now+timedelta(hours=26),.3,ident=2)
    assert daily.daily_candidates([tomorrow,today],now)==[today]


def test_chooses_best_supported_daily_history_not_a_random_score():
    now=datetime(2026,11,14,11,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    low=row("Weak",now+timedelta(hours=4),.25,home_games=4,venue_games=2,ident=1)
    strong=row("Researched",now+timedelta(hours=5),.08,home_games=9,venue_games=7,ident=2)
    assert daily.daily_candidates([low,strong],now)[0] is strong


def test_forecast_frozen_before_kickoff_and_stays_after_kickoff(monkeypatch):
    import research_ensemble_v8
    now=datetime(2026,11,14,11,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    first=row("First",now+timedelta(hours=3),.13,ident=12)
    replacement=row("Second",now+timedelta(hours=8),.32,ident=17)
    calls=[]
    def prediction(fixture,history,league_history,as_of):
        calls.append(fixture["id"])
        return first["result"] if fixture["id"]==12 else replacement["result"]
    monkeypatch.setattr(research_ensemble_v8,"predict",prediction)
    session={}
    original=daily.choose_daily_snapshot([first],session,now)
    assert original["exact_score"]==[2,1]
    assert original["model"]==daily.MODEL
    assert original["captured_at_utc"]<original["kickoff_utc"]
    assert daily.choose_daily_snapshot([replacement],session,now+timedelta(hours=4)) is original
    assert calls==[12]


def test_no_postkickoff_new_forecast_and_no_invented_fixture(monkeypatch):
    import research_ensemble_v8
    now=datetime(2026,11,14,11,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    complete=row("Started",now-timedelta(minutes=10),.2,ident=3)
    assert daily.daily_candidates([complete],now)==[]
    assert daily.build_snapshot(complete,now,complete["result"]) is None
    assert daily.choose_daily_snapshot([complete],{},now) is None


def test_next_day_is_explicit_and_previous_session_day_expires(monkeypatch):
    import research_ensemble_v8
    now=datetime(2026,11,14,11,0,tzinfo=SYDNEY).astimezone(timezone.utc)
    nextday=row("Tomorrow",now+timedelta(hours=26),.15,ident=33)
    monkeypatch.setattr(research_ensemble_v8,"predict",lambda *args,**kwargs:nextday["result"])
    session={}
    snapshot=daily.choose_daily_snapshot([nextday],session,now)
    assert snapshot["prediction_date"]=="2026-11-15"
    assert daily._label(snapshot,now).startswith("NEXT")
    assert daily.choose_daily_snapshot([],session,now+timedelta(hours=20)) is snapshot
    assert daily.choose_daily_snapshot([],session,now+timedelta(hours=45)) is None
