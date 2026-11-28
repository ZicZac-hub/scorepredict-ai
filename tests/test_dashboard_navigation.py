"""Exercise the deployed Streamlit routing with offline provider adapters."""
from pathlib import Path

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest


@pytest.fixture
def offline_app(monkeypatch):
    import data
    import worldwide
    import featured_match
    import history_store
    import revision_jobs
    import revision_ui
    import requests

    monkeypatch.setenv('FOOTBALL_DATA_API_KEY', 'offline-navigation-test')
    for key in ('API_FOOTBALL_KEY', 'THIRD_FOOTBALL_API_KEY', 'DATABASE_URL', 'OPENAI_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(data, 'competitions', lambda *a, **k: [
        {'name': 'Premier League', 'code': 'PL', 'area': {'name': 'England'}}])
    monkeypatch.setattr(data, 'fixtures', lambda *a, **k: [])
    monkeypatch.setattr(data, 'history', lambda *a, **k: [])
    monkeypatch.setattr(worldwide, 'worldwide', lambda *a, **k: ([], []))
    monkeypatch.setattr(featured_match, 'choose_feature', lambda *a, **k: (None, None))
    monkeypatch.setattr(history_store, 'connect', lambda *a, **k: None)
    monkeypatch.setattr(revision_jobs, 'get_revision_status', lambda: {'state': 'idle'})
    monkeypatch.setattr(revision_ui, 'render_revision_panel', lambda *a, **k: None)

    def no_network(*args, **kwargs):
        raise AssertionError('Navigation must not require a real provider request in this test.')
    monkeypatch.setattr(requests.sessions.Session, 'request', no_network)
    return AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'), default_timeout=30)


def _param(app, name):
    value = app.query_params.get(name)
    return value[-1] if isinstance(value, list) else value


def _buttons(app):
    return {button.key for button in app.button}


def _assert_page(app, page, title):
    assert not app.exception
    assert app.session_state['ziczac_active_page'] == page
    assert any(f'<h1>{title}</h1>' in item.value for item in app.markdown)
    assert any('STUDIO 2.1' in item.value for item in app.markdown)


def test_navigation_opens_selected_content_without_home_cards(offline_app):
    app = offline_app.run()
    _assert_page(app, 'home', 'Matchday Studio')
    assert {'studio_auto', 'studio_manual', 'feature_research'} <= _buttons(app)

    for route, title in [('predict', 'Daily Predictions'), ('deep', 'Deep Research AI'),
                         ('ai_five', 'AI Five-Double Studio'),
                         ('archive', 'Prediction History'), ('finished', 'Real Performance'),
                         ('private', 'Private Access'), ('more', 'Settings &amp; Methods')]:
        app.button(key='zz_route_' + route).click().run()
        _assert_page(app, route, title)
        assert _param(app, 'view') == route
        assert not {'studio_auto', 'studio_manual', 'feature_research'} & _buttons(app)

    app.button(key='zz_route_home').click().run()
    _assert_page(app, 'home', 'Matchday Studio')
    assert 'studio_auto' in _buttons(app)


def test_home_auto_and_manual_open_the_existing_mode_selector(offline_app):
    app = offline_app.run()
    app.button(key='studio_auto').click().run()
    _assert_page(app, 'doubles', 'Five Score Doubles')
    assert app.radio(key='smart_double_mode').value == 'AI Auto Select'
    assert _param(app, 'mode') == 'auto'
    assert 'studio_auto' not in _buttons(app)

    app.radio(key='smart_double_mode').set_value('Manual Select').run()
    assert not app.exception
    assert _param(app, 'mode') == 'manual'
    assert app.radio(key='smart_double_mode').value == 'Manual Select'

    app.button(key='zz_route_home').click().run()
    app.button(key='studio_manual').click().run()
    _assert_page(app, 'doubles', 'Five Score Doubles')
    assert app.radio(key='smart_double_mode').value == 'Manual Select'
    assert _param(app, 'mode') == 'manual'


@pytest.mark.parametrize('mode,label', [('auto', 'AI Auto Select'), ('manual', 'Manual Select')])
def test_pwa_deep_link_selects_doubles_mode(offline_app, mode, label):
    offline_app.query_params['view'] = ['doubles']
    offline_app.query_params['mode'] = [mode]
    app = offline_app.run()
    _assert_page(app, 'doubles', 'Five Score Doubles')
    assert app.radio(key='smart_double_mode').value == label
    assert 'studio_auto' not in _buttons(app)


def test_ai_five_button_opens_fast_studio_and_generates_five_no_cost_doubles(offline_app,monkeypatch):
    """A nav tap must not wait for primary, worldwide, or premium API calls."""
    import ai_five_double_ui
    from datetime import datetime,timedelta,timezone
    from zoneinfo import ZoneInfo

    day=datetime.now(ZoneInfo("Australia/Sydney")).date()+timedelta(days=1)
    def row(fid, home, away, hour):
        kick=datetime(day.year,day.month,day.day,hour,
                      tzinfo=ZoneInfo("Australia/Sydney")).astimezone(timezone.utc).isoformat()
        return {
            "fixture":{"provider":"API-Football","id":fid,"utcDate":kick,
                       "homeTeam":{"id":fid*2,"name":home},
                       "awayTeam":{"id":fid*2+1,"name":away}},
            "kickoff":kick,"league":"PL","past":[],
            "result":{"model_version":"offline-v8",
                      "scores":[[1,1,.13],[1,0,.115],[2,1,.104],
                                [0,0,.092],[2,0,.089],[0,1,.073]],
                      "home_xg":1.2,"away_xg":1.0}
        }

    monkeypatch.setattr(ai_five_double_ui,"_minute_archived_candidates",
                        lambda minute: [row(101,"Alpha","Beta",12),
                                        row(102,"Gamma","Delta",14)])
    monkeypatch.setattr(ai_five_double_ui,"frozen_for_fixture",lambda fixture:None)
    app=offline_app.run()
    app.button(key="zz_route_ai_five").click().run()
    _assert_page(app,"ai_five","AI Five-Double Studio")
    assert _param(app,"view")=="ai_five"
    assert "ziczac_ai_five_run" in _buttons(app)
    assert "zz_ai_five_live_feed" in {x.key for x in app.toggle}
    assert not {'studio_auto','studio_manual'} & _buttons(app)

    app.button(key="ziczac_ai_five_run").click().run()
    assert not app.exception
    assert any("Final five exact-score doubles" in x.value for x in app.markdown)


def test_pro_provider_works_without_primary_key_or_catalog(offline_app, monkeypatch):
    import data
    import api_football
    import global_fixtures

    monkeypatch.delenv('FOOTBALL_DATA_API_KEY', raising=False)
    monkeypatch.setenv('API_FOOTBALL_KEY', 'offline-pro-navigation')
    def no_primary(*args, **kwargs):
        raise AssertionError('A missing primary key must not make primary requests')
    monkeypatch.setattr(data, 'competitions', no_primary)
    monkeypatch.setattr(data, 'fixtures', no_primary)
    monkeypatch.setattr(data, 'history', no_primary)
    pro_codes=[]
    def pro_fixtures(_key, code, *args):
        pro_codes.append(code)
        return []
    monkeypatch.setattr(api_football, 'get_fixtures', pro_fixtures)
    monkeypatch.setattr(api_football, 'get_history', lambda *a, **k: [])
    monkeypatch.setattr(global_fixtures, 'global_fixtures', lambda *a, **k: ([], []))
    app=offline_app.run()
    _assert_page(app, 'home', 'Matchday Studio')
    assert {'PL', 'PD', 'CL'} <= set(pro_codes)
    assert {'studio_auto', 'studio_manual', 'feature_history'} <= _buttons(app)


def test_primary_catalog_outage_does_not_block_history_and_methods(offline_app, monkeypatch):
    import data
    def unavailable(*args, **kwargs):
        raise RuntimeError('catalog offline')
    monkeypatch.setattr(data, 'competitions', unavailable)
    app=offline_app.run()
    _assert_page(app, 'home', 'Matchday Studio')
    app.button(key='feature_history').click().run()
    _assert_page(app, 'archive', 'Prediction History')
    assert not any('Unable to retrieve competitions' in x.value for x in app.error)
    app.button(key='zz_route_more').click().run()
    _assert_page(app, 'more', 'Settings &amp; Methods')
    assert any('Mathematical model:' in x.value for x in app.markdown)


def test_no_provider_keys_keeps_private_settings_and_navigation_available(offline_app, monkeypatch):
    monkeypatch.delenv('FOOTBALL_DATA_API_KEY', raising=False)
    app=offline_app.run()
    _assert_page(app, 'home', 'Matchday Studio')
    app.button(key='zz_route_private').click().run()
    _assert_page(app, 'private', 'Private Access')
    assert any(x.value == '🔒 Private Access' for x in app.subheader)


def test_other_pages_do_not_refresh_hidden_home_feature(offline_app, monkeypatch):
    import featured_match
    checks=[]
    def feature(*args, **kwargs):
        checks.append(True)
        return None, None
    monkeypatch.setattr(featured_match, 'choose_feature', feature)
    app=offline_app.run()
    assert len(checks) == 1
    app.button(key='zz_route_archive').click().run()
    _assert_page(app, 'archive', 'Prediction History')
    app.button(key='zz_route_more').click().run()
    _assert_page(app, 'more', 'Settings &amp; Methods')
    assert len(checks) == 1
    app.button(key='zz_route_home').click().run()
    assert len(checks) == 2

def test_v25_live_date_selector_starts_today_without_saved_forecasts(offline_app,monkeypatch):
    """On iPhone, V25 must not hide today simply for missing score grids.

    With API credentials intentionally absent, clicking discovery must not
    attempt a network request or display fabricated football fixtures.
    """
    from datetime import datetime,timedelta
    from zoneinfo import ZoneInfo
    import ai_five_double_ui
    monkeypatch.setattr(ai_five_double_ui,"_minute_archived_candidates",
                        lambda minute: [])
    app=offline_app.run()
    app.button(key="zz_route_ai_five").click().run()
    app.radio(key="zz_ai_five_engine_choice").set_value(
        "🧮 V25 Professional Analyst · all upcoming matches").run()
    assert not app.exception
    current=datetime.now(ZoneInfo("Australia/Sydney")).date()
    picker=app.selectbox(key="zz_v25_matchday")
    assert picker.value==current.isoformat()
    assert "Today" in picker.options[0]
    assert "Tomorrow" in picker.options[1]
    keys=_buttons(app)
    assert {"zz_v25_go_today","zz_v25_go_tomorrow",
            "zz_v25_discover_day"} <= keys
    assert "zz_v25_manual_run" not in keys
    app.button(key="zz_v25_go_tomorrow").click().run()
    assert not app.exception
    assert app.selectbox(key="zz_v25_matchday").value==(
        current+timedelta(days=1)).isoformat()
    app.button(key="zz_v25_go_today").click().run()
    assert app.selectbox(key="zz_v25_matchday").value==current.isoformat()
    app.button(key="zz_v25_discover_day").click().run()
    assert not app.exception
    assert any("API_FOOTBALL_KEY" in e.value for e in app.error)


def test_v43_route_generates_a_pair_and_passes_valid_scores_to_research(offline_app, monkeypatch):
    from datetime import datetime, timedelta, timezone

    import v43_ui
    from ai_five_double_engine import SYDNEY

    day = datetime.now(SYDNEY).date() + timedelta(days=1)

    def candidate(ident, home, away, hour):
        kickoff = datetime(day.year, day.month, day.day, hour, tzinfo=SYDNEY)
        row = {"fixture": {"id": ident, "provider": "API-Football",
                           "utcDate": kickoff.isoformat(),
                           "homeTeam": {"id": ident * 2, "name": home},
                           "awayTeam": {"id": ident * 2 + 1, "name": away}}}
        return {"key": ("API-Football", str(ident)), "title": f"{home} vs {away}",
                "row": row, "kickoff": kickoff, "captured_at_utc": datetime.now(timezone.utc).isoformat()}

    matches = [candidate(1, "Alpha", "Beta", 12), candidate(2, "Gamma", "Delta", 15)]
    monkeypatch.setattr(v43_ui, "_upcoming_archive", lambda minute: [])
    monkeypatch.setattr(v43_ui, "available", lambda rows, now: ({day.isoformat(): matches}, {}))
    monkeypatch.setattr(v43_ui, "pair_catalog", lambda selected, now: [tuple(selected)])

    def model(item):
        fixture = item["row"]["fixture"]
        return {"home_team": fixture["homeTeam"]["name"], "away_team": fixture["awayTeam"]["name"],
                "kickoff_utc": fixture["utcDate"], "top_scores": [{"score": "10-2", "probability": .05}],
                "model_probabilities": {"home": .5, "draw": .25, "away": .25},
                "expected_goals": {"home": 1.5, "away": .8}}

    calls = []
    monkeypatch.setattr(v43_ui, "forecast_pair", lambda first, second: {
        "status": "issued", "fixture_models": [model(item) for item in matches],
        "five_doubles": [{"score1": "10-2", "score2": "0-1", "joint_probability": .01}],
        "estimated_coverage": .01, "lineage": {},
        "decision_as_of_utc": datetime.now(timezone.utc).isoformat(),
    })
    monkeypatch.setattr(v43_ui, "render_v43_research",
                        lambda st, fixture, history, statistical, **kwargs: calls.append((fixture["id"], statistical)))

    offline_app.query_params["view"] = ["v43"]
    app = offline_app.run()
    _assert_page(app, "v43", "V43 Integrated Forecast")
    app.selectbox(key="v43_second").set_value(matches[1]["key"]).run()
    app.button(key="v43_generate").click().run()
    assert not app.exception
    assert any(item.value == "Combined five exact-score doubles" for item in app.subheader)
    assert calls[-1][1]["scores"] == [[10, 2, .05]]

    app.radio(key="v43_detail_match").set_value(1).run()
    assert not app.exception
    assert calls[-1][0] == 2
    app.selectbox(key="v43_second").set_value(None).run()
    assert not any(item.value == "Combined five exact-score doubles" for item in app.subheader)


def test_v43_recheck_final_scores_loads_read_only_target_panel(offline_app):
    offline_app.query_params["view"] = ["v43"]
    app = offline_app.run()
    app.radio(key="v43_pane").set_value("Research and validation").run()
    _assert_page(app, "v43", "V43 Integrated Forecast")
    assert any(item.value == "70% five-double target" for item in app.subheader)
    assert any("at least 79" in item.value for item in app.caption)

    app.button(key="v43_recheck_finals").click().run()
    assert not app.exception
    assert any(item.label == "Final scores and reasons for misses" for item in app.expander)


def test_v45_data_route_renders_archived_quality_without_network(offline_app):
    offline_app.query_params['view'] = ['v45']
    app = offline_app.run()
    _assert_page(app, 'v45', 'V45 Data Intelligence')
    assert any(item.value == 'Match information quality' for item in app.subheader)
    app.button(key='v45_audit').click().run()
    assert not app.exception
    assert any(item.label == 'Recovered historical results' for item in app.metric)
