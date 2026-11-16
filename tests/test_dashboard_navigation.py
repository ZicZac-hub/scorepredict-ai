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
    assert any("API_FOOTBALL_KEY" in e.message for e in app.error)
