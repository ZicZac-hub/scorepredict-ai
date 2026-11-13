from unittest.mock import Mock

from result_verification import fetch_saved_result, primary_result, verification_batch


def saved(provider='football-data.org'):
    return {'provider': provider, 'fixture_id': 12, 'home_id': '1', 'away_id': '2',
            'home': 'A', 'away': 'B', 'kickoff': '2026-11-01T12:00:00Z'}


def primary():
    return {'id': 12, 'utcDate': '2026-11-01T12:00:00Z', 'status': 'FINISHED',
            'homeTeam': {'id': 1, 'name': 'A'}, 'awayTeam': {'id': 2, 'name': 'B'},
            'score': {'duration': 'REGULAR', 'fullTime': {'home': 2, 'away': 0}}}


def secondary(home='A'):
    return {'fixture': {'id': 12, 'date': '2026-11-01T12:00:00Z',
                        'status': {'short': 'FT'}},
            'teams': {'home': {'id': 1, 'name': home}, 'away': {'id': 2, 'name': 'B'}},
            'goals': {'home': 2, 'away': 0}, 'score': {'fulltime': {'home': 2, 'away': 0}}}


def test_primary_forecast_never_queries_secondary():
    p, s = Mock(return_value=primary()), Mock()
    result, errors = fetch_saved_result(saved(), p, s)
    assert result['provider'] == 'football-data.org'
    assert not errors
    p.assert_called_once_with('/matches/12')
    s.assert_not_called()


def test_secondary_forecast_never_queries_primary():
    p, s = Mock(), Mock(return_value=[secondary()])
    result, errors = fetch_saved_result(saved('API-Football'), p, s)
    assert result['provider'] == 'API-Football'
    p.assert_not_called()
    s.assert_called_once_with('/fixtures', {'id': 12})


def test_legacy_collision_checks_other_provider_after_wrong_teams():
    p, s = Mock(return_value=primary()), Mock(return_value=[secondary('Different team')])
    result, errors = fetch_saved_result(saved('legacy-unknown'), p, s)
    assert result['provider'] == 'football-data.org'
    p.assert_called_once()


def test_wrong_kickoff_never_returns_match():
    match = primary()
    match['utcDate'] = '2026-11-02T12:00:00Z'
    assert fetch_saved_result(saved(), Mock(return_value=match))[0] is None


def test_extra_time_uses_explicit_regulation_score():
    match = primary()
    match['score'] = {'duration': 'EXTRA_TIME', 'fullTime': {'home': 3, 'away': 1},
                      'regularTime': {'home': 1, 'away': 1}}
    result = primary_result(match)
    assert result['score']['fullTime'] == {'home': 1, 'away': 1}
    assert result['score']['duration'] == 'REGULAR'
    assert match['score']['fullTime'] == {'home': 3, 'away': 1}


def test_missing_regulation_score_remains_unknown():
    match = primary()
    match['score']['duration'] = 'EXTRA_TIME'
    assert primary_result(match)['score']['fullTime'] == {'home': None, 'away': None}


def test_unfinished_legacy_candidate_does_not_block_other_provider():
    raw = secondary()
    raw['fixture']['status']['short'] = 'NS'
    result, errors = fetch_saved_result(saved('legacy-unknown'), Mock(return_value=primary()),
                                        Mock(return_value=[raw]))
    assert result['provider'] == 'football-data.org'


def test_rotating_checks_reach_old_rows_when_recent_results_never_resolve():
    rows, cursor, seen = list(range(23)), 0, set()
    for _ in range(3):
        batch, cursor = verification_batch(rows, cursor, limit=10)
        seen.update(batch)
    assert seen == set(rows)
    assert verification_batch([], 42) == ([], 0)
