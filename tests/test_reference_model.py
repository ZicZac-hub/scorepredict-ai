"""Numerical and provenance checks for the reference-informed live model."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import numpy as np
import engine
from forecast_model import predict, LIVE_MODEL_VERSION
from reference_model import fit, score_matrix


def full_history():
    start = datetime(2025, 10, 1, tzinfo=timezone.utc)
    history = []
    for cycle in range(2):
        for h in range(1, 7):
            for a in range(1, 7):
                if h == a:
                    continue
                index = len(history)
                gh, ga = ((5, 0) if h==1 else (0, 4) if a==1 else (2, 1))
                history.append({'id': index+1, 'provider': 'API-Football',
                    'competition': {'id': 39, 'code': 'PL'}, 'utcDate': (start+timedelta(days=index)).isoformat(),
                    'status': 'FINISHED', 'homeTeam': {'id': h, 'name': str(h)},
                    'awayTeam': {'id': a, 'name': str(a)},
                    'score': {'duration': 'REGULAR', 'fullTime': {'home': gh, 'away': ga}}})
    return history


def fixture():
    return {'id': 1000, 'provider': 'API-Football', 'competition': {'id': 39, 'code': 'PL'},
        'utcDate': '2026-01-15T12:00:00Z', 'status': 'SCHEDULED',
        'homeTeam': {'id': 1, 'name': '1'}, 'awayTeam': {'id': 2, 'name': '2'}}


def test_live_wrapper_uses_fitted_prior_only_model_and_preserves_baseline():
    result = predict(fixture(), full_history(), variant='reference_v7', league_history=full_history())
    assert result['model_version'] == LIVE_MODEL_VERSION
    assert result['baseline_reference']['model_version'] == engine.LIVE_MODEL_VERSION
    assert 'fitted' in result['model_sources'][-1]
    assert result['model_weights']['fitted_opponent_dixon_coles'] > 0
    assert result['llm_provider_calls'] == 0
    assert all('reference' in value for value in result['reference_providers'].values())


def test_fitted_distribution_is_positive_complete_and_normalized():
    result = predict(fixture(), full_history(), variant='reference_v7', league_history=full_history())
    assert np.all(result['matrix'] >= 0)
    assert np.isclose(result['matrix'].sum(), 1)
    assert np.isclose(result['home_win']+result['draw']+result['away_win'], 1)
    assert result['scores'][0][2] == float(result['matrix'].max())
    assert result['home_xg'] > result['away_xg']


def test_four_goal_strong_team_forecast_is_possible_without_manual_boost():
    estimate = fit(fixture(), full_history(), regularization=2)
    assert estimate['home_xg'] > 3.8
    matrix = score_matrix(estimate)
    assert np.unravel_index(matrix.argmax(), matrix.shape)[0] >= 4


def test_evaluated_fixture_actual_score_never_changes_forecast():
    original = predict(fixture(), full_history(), variant='reference_v7', league_history=full_history())
    completed = deepcopy(fixture())
    completed.update(status='FINISHED', score={'fullTime': {'home': 0, 'away': 9}})
    revised = predict(completed, full_history(), variant='reference_v7', league_history=full_history())
    assert np.array_equal(original['matrix'], revised['matrix'])


def test_future_results_cannot_change_fit_or_rho():
    history = full_history()
    original = fit(fixture(), history)
    future = deepcopy(history[0])
    future.update(id=9999, utcDate=fixture()['utcDate'])
    future['score']['fullTime'] = {'home': 20, 'away': 0}
    assert fit(fixture(), history+[future]) == original


def test_other_providers_and_competitions_cannot_change_fit():
    history = full_history()
    other = deepcopy(history)
    for match in other:
        match['provider'] = 'football-data.org'
        match['score']['fullTime'] = {'home': 20, 'away': 0}
    assert fit(fixture(), history+other) == fit(fixture(), history)
    for match in other:
        match['provider'] = 'API-Football'
        match['competition']['id'] = 78
    assert fit(fixture(), history+other) == fit(fixture(), history)


def test_missing_explicit_league_population_is_disclosed_fallback():
    result = predict(fixture(), full_history(), variant='reference_v7', league_history=[])
    assert result['model_version'] == LIVE_MODEL_VERSION+'-fallback-v6'
    assert result['fallback_reason']
    assert result['model_weights']['fitted_opponent_dixon_coles'] == 0


def test_team_only_history_is_not_mislabeled_full_league_population():
    history = [m for m in full_history() if {1,2}.intersection((m['homeTeam']['id'], m['awayTeam']['id']))]
    assert len(history) >= 30
    result = predict(fixture(), history, variant='reference_v7', league_history=history)
    assert result['model_version'].endswith('fallback-v6')
    assert result['evidence_dossier']['independent_league_matches'] == 0


def test_explicit_experimental_variants_remain_legacy_comparable():
    actual = predict(fixture(), full_history(), variant='regularized')
    expected = engine.predict(fixture(), full_history(), variant='regularized')
    assert actual['model_version'] == expected['model_version']
    assert np.array_equal(actual['matrix'], expected['matrix'])


def test_favoured_outcome_alternative_is_separate_and_probability_preserved():
    result = predict(fixture(), full_history(), variant='reference_v7', league_history=full_history())
    alternative = result['favoured_outcome_alternative']
    h, a = alternative['score']
    assert alternative['unconditional_score_probability'] == result['matrix'][h,a]
    assert alternative['outcome'] == 'home'
    assert h > a
    assert result['scores'][0][2] == result['matrix'].max()


def test_rho_is_fitted_to_training_outcomes_and_positive_for_new_matchup():
    estimate = fit(fixture(), full_history())
    h, a, rho = estimate['home_xg'], estimate['away_xg'], estimate['rho']
    assert min(1-h*a*rho, 1+h*rho, 1+a*rho, 1-rho) > 0
    assert estimate['last_training_kickoff_utc'] < estimate['fit_cutoff_utc']


def test_no_forecast_is_fabricated_for_missing_team_history():
    unknown = fixture()
    unknown['homeTeam']['id'] = 999
    assert predict(unknown, full_history(), league_history=full_history()) is None


def test_default_preserves_better_validated_v6_scores_and_shows_v7_comparison():
    history = full_history()
    actual = predict(fixture(), history, league_history=history)
    baseline = engine.predict(fixture(), history, league_history=history)
    assert actual['model_version'] == LIVE_MODEL_VERSION+'-v6-retained'
    assert actual['active_model_version'] == engine.LIVE_MODEL_VERSION
    assert np.array_equal(actual['matrix'], baseline['matrix'])
    assert actual['scores'] == baseline['scores']
    assert actual['candidate_summary']['model_version'] == LIVE_MODEL_VERSION
    assert actual['model_weights'] == {'legacy_v6': 1.0, 'fitted_opponent_dixon_coles': 0.0}
    assert all(isinstance(v, tuple) for v in actual['methods'].values())
