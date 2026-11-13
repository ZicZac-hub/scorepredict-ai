"""Regression checks for competition provenance and supported strong-team rates."""
import numpy as np
from engine import predict, _grid, mix_score_matrices


def match(idx, home=1, away=2, goals=(2, 1), competition=39,
          provider='API-Football', date=None):
    return {'id': idx, 'provider': provider,
            'competition': {'id': competition, 'code': 'PL'},
            'utcDate': date or f'2026-09-{idx % 28 + 1:02d}T12:00:00Z',
            'status': 'FINISHED',
            'homeTeam': {'id': home, 'name': str(home)},
            'awayTeam': {'id': away, 'name': str(away)},
            'score': {'fullTime': {'home': goals[0], 'away': goals[1]}}}


def fixture():
    return {'provider': 'API-Football',
            'competition': {'id': 39, 'code': 'PL'},
            'utcDate': '2026-11-01T12:00:00Z',
            'homeTeam': {'id': 1}, 'awayTeam': {'id': 2}}


def test_provider_and_competition_id_collisions_cannot_change_prediction():
    history = [match(i) for i in range(1, 21)]
    expected = predict(fixture(), history)
    wrong = [match(100+i, goals=(5, 0), competition=78) for i in range(20)]
    wrong += [match(200+i, goals=(0, 5), provider='football-data.org')
              for i in range(20)]
    actual = predict(fixture(), history+wrong)
    assert actual['home_xg'] == expected['home_xg']
    assert actual['away_xg'] == expected['away_xg']
    assert actual['history_count'] == len(history)


def test_full_league_baseline_includes_other_teams_but_excludes_future():
    form = [match(i) for i in range(1, 13)]
    league = [match(100+i, home=3, away=4, goals=(4, 2)) for i in range(20)]
    expected = predict(fixture(), form, league_history=league)
    assert expected['league_history_count'] == 20
    assert expected['league_baseline_source'] == 'competition_history'
    assert expected['home_xg'] != predict(fixture(), form)['home_xg']
    future = match(999, home=3, away=4, goals=(20, 20),
                   date=fixture()['utcDate'])
    actual = predict(fixture(), form, league_history=league+[future])
    assert actual['home_xg'] == expected['home_xg']
    assert actual['league_history_count'] == 20


def test_missing_explicit_league_history_uses_generic_prior():
    form = [match(i, goals=(4, 2)) for i in range(1, 21)]
    actual = predict(fixture(), form, league_history=[])
    assert actual['league_history_count'] == 0
    assert actual['league_baseline_source'] == 'generic_prior'
    assert actual['home_xg'] != predict(fixture(), form)['home_xg']


def test_insufficient_same_competition_history_never_uses_cup_results():
    cups = [match(i, competition=2) for i in range(1, 21)]
    assert predict(fixture(), cups) is None


def test_four_plus_modal_score_supported_by_history():
    history = [match(i, goals=(5, 0)) for i in range(1, 21)]
    result = predict(fixture(), history)
    assert result['home_xg'] >= 4
    assert result['scores'][0][0] >= 4
    assert np.isclose(result['matrix'].sum(), 1.0)


def test_poisson_grid_extends_tail_instead_of_clipping_strong_favourite():
    matrix = _grid(8.0, 0.5, max_goals=3)
    assert matrix.shape[0] > 10
    assert abs(float(matrix.sum())-1.0) < 1e-12
    home_mean = float((np.arange(matrix.shape[0])[:, None]*matrix).sum())
    assert abs(home_mean-8.0) < 1e-7


def test_unlabelled_legacy_fixture_is_still_supported():
    legacy = fixture()
    legacy.pop('competition')
    legacy.pop('provider')
    history = [match(i, provider='football-data.org') for i in range(1, 13)]
    assert predict(legacy, history) is not None


def test_mixture_preserves_scores_across_different_tail_sizes():
    small = np.array([[0.5,0.0],[0.0,0.5]])
    large = np.zeros((4,4))
    large[3,0] = 1.0
    mixed = mix_score_matrices([small,large],[0.25,0.75])
    assert mixed.shape == (4,4)
    assert mixed[3,0] == 0.75
    assert mixed[1,1] == 0.125
    assert mixed.sum() == 1.0


def test_calibration_baseline_does_not_restore_removed_goal_ceiling():
    from home_away_calibration import evaluate_rows
    result = evaluate_rows([(4.6, 0.5, 4, 0)], 1.0, 1.0)
    assert result['exact_hits'] == 1
