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
    assert any('VERSION 2.0' in item.value for item in app.markdown)


def test_navigation_opens_selected_content_without_home_cards(offline_app):
    app = offline_app.run()
    _assert_page(app, 'home', 'Matchday Studio')
    assert {'studio_auto', 'studio_manual', 'feature_research'} <= _buttons(app)

    for route, title in [('predict', 'Daily Predictions'), ('deep', 'Deep Research AI'),
                         ('archive', 'Prediction History'), ('finished', 'Real Performance'),
                         ('more', 'Settings &amp; Methods')]:
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
