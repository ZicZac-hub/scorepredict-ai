"""Reference-informed live forecasting with preserved v6 evidence and fallback.

Model providers named in research are references, never fabricated API votes.
The numerical contributors here are local statistical models.
"""
from copy import deepcopy
import numpy as np
import engine as legacy
from reference_model import fit, score_matrix

LIVE_MODEL_VERSION = 'reference-informed-v7'
MODEL_VERSION = LIVE_MODEL_VERSION
# Selected using 2023/24 validation only; selection evidence is documented in
# REFERENCE_MODEL_VALIDATION.md. Never refit these choices on the holdout.
PARAMETERS = {'regularization': 2.0, 'half_life_days': 180.0, 'fitted_weight': 1.0}
PROMOTION_STATUS = 'v6_retained_after_exact_score_validation'
PROMOTION_EVIDENCE = {'paired_matches': 1385, 'v6_exact_hits': 163, 'v7_exact_hits': 157,
                      'sample': 'Independent five-league 2026 provider replay; separate from live archive',
                      'evidence_protocol': 'legacy kickoff-only historical replay, before shared-evidence-v1'}


def _summary(result):
    return {'model_version': result['model_version'],
            'predicted_score': [int(v) for v in result['scores'][0][:2]],
            'score_probability': float(result['scores'][0][2]),
            'expected_goals': {'home': float(result['home_xg']), 'away': float(result['away_xg'])},
            'outcome_probabilities': {'home': float(result['home_win']), 'draw': float(result['draw']), 'away': float(result['away_win'])}}


def predict(fixture, history, max_goals=10, variant='baseline', league_history=None):
    """Retain validated v6 live scores; expose fitted v7 as an explicit comparison.

    A full league population is needed for fitting opponent ratings. Team-only
    histories retain a clearly disclosed v6 fallback rather than manufactured
    league evidence. Original saved predictions are never modified here.
    """
    if variant == 'research_v8':
        # Explicit experimental opt-in. The original default and immutable
        # pre-kickoff forecasts remain unchanged; no vendor AI is called.
        from research_ensemble_v8 import predict as predict_research_v8
        return predict_research_v8(fixture, history, league_history=league_history,
                                   max_goals=max_goals)
    if variant not in ('baseline', 'reference_v7'):
        return legacy.predict(fixture, history, max_goals=max_goals,
                              variant=variant, league_history=league_history)
    result = legacy.predict(fixture, history, max_goals=max_goals, league_history=league_history)
    if result is None:
        return None
    baseline = _summary(result)
    population = legacy._competition_history(fixture, history if league_history is None else league_history)
    targets = {fixture['homeTeam']['id'], fixture['awayTeam']['id']}
    unrelated = sum(not targets.intersection((m['homeTeam']['id'], m['awayTeam']['id'])) for m in population)
    is_population = len(population) >= 30 and unrelated >= max(3, int(.1*len(population)))
    estimate = (fit(fixture, population, regularization=PARAMETERS['regularization'],
                    half_life_days=PARAMETERS['half_life_days']) if is_population else None)
    result['baseline_reference'] = baseline
    result['reference_providers'] = {'ScoreGPT': 'public_methodology_reference',
        'ChatGPT/OpenAI': 'public_documentation_reference', 'Claude': 'public_documentation_reference',
        'DeepSeek': 'public_documentation_reference', 'Gemini': 'public_documentation_reference'}
    result['llm_provider_calls'] = 0
    result['model_sources'] = ['Preserved v6 venue-form statistical baseline']
    result['evidence_dossier'] = {'provider': fixture.get('provider', 'football-data.org'),
        'competition': fixture.get('competition') or {},
        'fixture_kickoff_utc': fixture.get('utcDate'),
        'league_history_matches': len(population), 'independent_league_matches': unrelated,
        'observed_features': ['regulation goals', 'match dates', 'team IDs', 'home/away venue'],
        'unavailable_numerical_features': ['verified chance-quality xG', 'player strength adjustments', 'bookmaker odds'],
        'research_context_changes_probabilities': False}
    result['release_version'] = LIVE_MODEL_VERSION
    result['baseline_reference_version'] = legacy.LIVE_MODEL_VERSION
    result['promotion_status'] = PROMOTION_STATUS
    result['promotion_evidence'] = dict(PROMOTION_EVIDENCE)
    retained = deepcopy(result)
    if estimate is None:
        result['model_version'] = LIVE_MODEL_VERSION+'-fallback-v6'
        result['fallback_reason'] = 'Insufficient full-league or target-team history for fitted opponent ratings.'
        result['model_weights'] = {'legacy_v6': 1.0, 'fitted_opponent_dixon_coles': 0.0}
        result['uncertainty_note'] = 'V6 fallback: full-league evidence is insufficient for the fitted v7 contributor. Exact scores remain uncertain.'
        return result
    fitted = score_matrix(estimate, max_goals)
    weight = PARAMETERS['fitted_weight']
    matrix = legacy.mix_score_matrices([result['matrix'], fitted], [1-weight, weight])
    ranked = np.argsort(-matrix.ravel(), kind='stable')[:10]
    scores = [(int(i//matrix.shape[1]), int(i%matrix.shape[1]), float(matrix.ravel()[i])) for i in ranked]
    result.update({'matrix': matrix, 'scores': scores,
        'home_xg': float((matrix*np.arange(matrix.shape[0])[:, None]).sum()),
        'away_xg': float((matrix*np.arange(matrix.shape[1])[None, :]).sum()),
        'home_win': float(np.tril(matrix, -1).sum()), 'draw': float(np.trace(matrix)),
        'away_win': float(np.triu(matrix, 1).sum()), 'model_version': LIVE_MODEL_VERSION,
        'model_weights': {'legacy_v6': 1-weight, 'fitted_opponent_dixon_coles': weight},
        'fitted_model': estimate,
        'fallback_reason': None,
        'uncertainty_note': 'A local statistical forecast fitted only to prior results, with the preserved v6 comparison. Public AI/site methodologies are references; no external AI votes are claimed. No exact score is guaranteed.'})
    result['model_sources'] = (['Preserved v6 venue-form statistical baseline'] if weight < 1 else [])
    result['model_sources'].append('Time-decayed fitted opponent attack/defence with fitted Dixon-Coles dependence')
    result['methods']['fitted_opponent'] = (round(estimate['home_xg'], 3), round(estimate['away_xg'], 3))
    result['evidence_dossier']['fitted_parameters'] = {'regularization': estimate['regularization'],
        'half_life_days': estimate['half_life_days'], 'rho': estimate['rho'],
        'home_advantage': estimate['fitted_home_advantage']}
    fitted_top = tuple(int(v) for v in np.unravel_index(fitted.argmax(), fitted.shape))
    result['model_disagreement'] = {'different_top_scores': tuple(baseline['predicted_score']) != fitted_top,
        'v6_top_score': baseline['predicted_score'], 'fitted_top_score': list(fitted_top),
        'home_goal_rate_difference': float(estimate['home_xg']-baseline['expected_goals']['home']),
        'away_goal_rate_difference': float(estimate['away_xg']-baseline['expected_goals']['away'])}
    # An outcome-consistent alternative is separately labelled; it never
    # replaces the unconditional most probable exact score.
    outcome = max(('home', 'draw', 'away'), key=lambda key: result[{'home':'home_win','draw':'draw','away':'away_win'}[key]])
    cells = [(i,j,float(matrix[i,j])) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])
             if (i>j if outcome=='home' else i==j if outcome=='draw' else i<j)]
    alternative = max(cells, key=lambda cell: cell[2])
    result['favoured_outcome_alternative'] = {'outcome': outcome, 'score': list(alternative[:2]),
        'unconditional_score_probability': alternative[2],
        'label': 'Highest-probability score within the favoured match outcome; separate from top exact score'}
    result['selection_policy'] = 'explicit_v7_research_candidate'
    if variant == 'reference_v7':
        return result
    # The newly fitted model failed to improve the primary exact-score metric
    # on the independent provider replay. Do not silently promote it merely
    # because its Brier score improved or it produces fewer 1-1 forecasts.
    retained.update({'model_version': LIVE_MODEL_VERSION+'-v6-retained',
        'active_model_version': legacy.LIVE_MODEL_VERSION,
        'model_weights': {'legacy_v6': 1.0, 'fitted_opponent_dixon_coles': 0.0},
        'candidate_summary': _summary(result), 'candidate_fitted_model': estimate,
        'candidate_disagreement': result['model_disagreement'],
        'selection_policy': 'v6_retained_after_independent_exact_score_gate',
        'uncertainty_note': 'The v7 fitted candidate is available for comparison. V6 remains active because it scored more exact hits on the independent historical replay; no future accuracy is guaranteed.'})
    return retained
