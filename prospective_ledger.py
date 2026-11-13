"""Contemporaneous same-evidence forecasts and independently verified comparisons.

This ledger starts when both numerical models are actually run before kickoff.
It never promotes a historical reconstruction into prospective accuracy.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import random
import re

import numpy as np

import engine
import forecast_model
from history_pipeline import build_evidence

SCHEMA_VERSION = 1
EPS = 1e-12
MODELS = ('v6', 'v7', 'league_poisson', 'always_1_1')
MAX_CAPTURE_LAG = timedelta(minutes=15)


def _utcnow():
    return datetime.now(timezone.utc)


def _date(value):
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def horizon_for(kickoff, captured_at):
    """Classify the real completion time, allowing a fifteen-minute collection cadence."""
    minutes = (_date(kickoff)-_date(captured_at)).total_seconds()/60
    if minutes <= 0:
        raise ValueError('Prospective forecasts must finish before kickoff')
    if 23.75*60 <= minutes <= 24.25*60:
        return 'T24h'
    if 45 <= minutes <= 75:
        return 'T60m'
    if 15 <= minutes < 45:
        return 'T30m'
    return 'research'


def _matrix(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or min(result.shape) < 2 or result.size > 1_000_000:
        raise ValueError('A complete two-dimensional score grid is required')
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('Score-grid probabilities must be finite and non-negative')
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError('Score-grid probabilities must sum to one')
    return result


def _rank(matrix, limit=10):
    ranked = np.argsort(-matrix.ravel(), kind='stable')
    return [[int(i//matrix.shape[1]), int(i % matrix.shape[1]), float(matrix.ravel()[i])]
            for i in ranked[:limit] if matrix.ravel()[i] > 0]


def _source_metadata(value):
    allowed = {'provider', 'endpoint', 'season', 'fixture_id', 'fetched_at_utc', 'retrieved_at_utc',
               'age_seconds', 'cache_status', 'stale', 'ttl_seconds', 'freshness_known', 'kind'}
    if isinstance(value, dict):
        return {key: deepcopy(item) for key, item in value.items() if key in allowed
                and (item is None or type(item) in (str, int, float, bool))}
    if isinstance(value, list):
        return [_source_metadata(item) for item in value if isinstance(item, dict)]
    return {}


def _identity(fixture):
    home, away = fixture.get('homeTeam') or {}, fixture.get('awayTeam') or {}
    if fixture.get('id') is None or home.get('id') is None or away.get('id') is None or str(home['id']) == str(away['id']):
        raise ValueError('A provider fixture and two distinct team IDs are required')
    if any(type(value) is bool for value in (fixture['id'], home['id'], away['id'])):
        raise ValueError('Boolean fixture or team IDs are invalid')
    return {'provider': fixture.get('provider') or 'football-data.org',
            'fixture_id': fixture['id'], 'home_team_id': home['id'], 'away_team_id': away['id'],
            'home_team': home.get('name'), 'away_team': away.get('name'),
            'kickoff_utc': _date(fixture['utcDate']).isoformat(),
            'competition': deepcopy(fixture.get('competition') or {})}


def pair_slot_key(fixture, horizon, policy_version, model_versions=None, release_version=None):
    """Stable deduplication slot, independent of refreshed fetch timestamps."""
    identity = _identity(fixture)
    return _hash({'provider': identity['provider'], 'fixture_id': str(identity['fixture_id']),
                  'kickoff_utc': identity['kickoff_utc'], 'horizon': horizon,
                  'policy_version': policy_version,
                  'release_version': release_version or forecast_model.LIVE_MODEL_VERSION,
                  'model_versions': model_versions or {'v6': engine.LIVE_MODEL_VERSION,
                                                       'v7': forecast_model.LIVE_MODEL_VERSION},
                  'v7_parameters': forecast_model.PARAMETERS})


def capture_pair(fixture, evidence, observed_at=None, horizon=None):
    """Compute and freeze all contributors from one verified evidence object.

    ``observed_at`` is the caller's collection time, never a historical replay
    date. The actual completion clock determines the forecast horizon. Callers
    should persist the returned plain JSON records atomically/write-once.
    """
    started = _utcnow()
    observation = _date(observed_at or started)
    if observation > started+timedelta(seconds=5) or started-observation > MAX_CAPTURE_LAG:
        raise ValueError('Historical or future observation times cannot create prospective forecasts')
    identity = _identity(fixture)
    kickoff = _date(identity['kickoff_utc'])
    if kickoff <= started or fixture.get('status') not in ('SCHEDULED', 'TIMED'):
        raise ValueError('Only genuinely upcoming fixtures can enter this ledger')
    if fixture.get('provider_status') not in (None, '', 'NS', 'TBD'):
        raise ValueError('The provider does not confirm an unstarted fixture')
    full = (fixture.get('score') or {}).get('fullTime') or {}
    if any(full.get(side) is not None for side in ('home', 'away')):
        raise ValueError('A prospective target must not contain its eventual score')
    cutoff = _date(evidence['as_of_utc'])
    if cutoff > observation or cutoff >= kickoff or started-cutoff > MAX_CAPTURE_LAG:
        raise ValueError('Evidence must be contemporaneous and available before capture')
    population = evidence.get('league_history', evidence.get('history', []))
    checked = build_evidence(fixture, population, cutoff, source_metadata=evidence.get('provenance'))
    if checked['input_digest'] != evidence.get('input_digest'):
        raise ValueError('Evidence digest does not match its actual model inputs')
    # One canonical league population drives both form and league parameters.
    # Reject, rather than silently relabel, caller histories with other inputs.
    form_checked = build_evidence(fixture, evidence.get('history', []), cutoff)
    if form_checked['input_digest'] != checked['input_digest']:
        raise ValueError('Form and league histories must follow the shared policy')
    target = deepcopy(checked['fixture'])
    target['score'] = {'fullTime': {'home': None, 'away': None}, 'duration': 'REGULAR'}
    history = checked['history']
    v6 = engine.predict(deepcopy(target), deepcopy(history), league_history=deepcopy(history))
    v7 = forecast_model.predict(deepcopy(target), deepcopy(history), variant='reference_v7', league_history=deepcopy(history))
    if v6 is None or v7 is None:
        raise ValueError('Insufficient shared history for prospective forecasts')
    base_h, base_a = engine.league_baselines(history)
    baseline = {'matrix': engine._grid(base_h, base_a), 'model_version': 'prior-league-poisson-v1',
                'parameters': {'home_rate': base_h, 'away_rate': base_a}}
    point = np.zeros((2, 2), dtype=float)
    point[1, 1] = 1.0
    fixed = {'matrix': point, 'model_version': 'fixed-always-1-1-v1'}
    completed = _utcnow()
    if completed >= kickoff:
        raise ValueError('Model computation finished after kickoff')
    classified = horizon_for(kickoff, completed)
    requested = 'research' if horizon == 'baseline' else horizon
    if requested is not None and requested != classified:
        raise ValueError('Requested horizon does not match the actual capture time')
    common = dict(identity, schema_version=SCHEMA_VERSION, mode='prospective',
                  score_basis='regulation', captured_at_utc=completed.isoformat(),
                  evidence_as_of_utc=cutoff.isoformat(), horizon=classified,
                  lead_minutes=(kickoff-completed).total_seconds()/60,
                  policy_version=checked['policy_version'], input_digest=checked['input_digest'],
                  history_window_start_utc=checked['history_window_start_utc'],
                  provenance=deepcopy(checked['provenance']), history_matches=len(history))
    # These observations explain source freshness. They remain context, not
    # numerical features, and cannot carry API keys or arbitrary provider text.
    for name in ('fixture_source', 'news_sources'):
        if name in (evidence.get('provenance') or {}):
            common['provenance'][name] = _source_metadata(evidence['provenance'][name])
    common['slot_key'] = pair_slot_key(fixture, classified, checked['policy_version'])
    common['pair_id'] = _hash({key: common[key] for key in ('provider', 'fixture_id', 'kickoff_utc',
        'captured_at_utc', 'evidence_as_of_utc', 'horizon', 'policy_version', 'input_digest')})
    output = []
    for key, forecast in zip(MODELS, (v6, v7, baseline, fixed)):
        matrix = _matrix(forecast['matrix'])
        ranked = _rank(matrix)
        fallback = forecast.get('fallback_reason') if key == 'v7' else None
        record = dict(common, model_key=key, model_version=forecast['model_version'],
            numerical_model_version=(engine.LIVE_MODEL_VERSION if fallback else
                forecast.get('active_model_version', forecast['model_version'])),
            parameters=deepcopy(forecast.get('parameters', forecast_model.PARAMETERS if key == 'v7' else {})),
            eligible_for_comparison=not bool(fallback), fallback_reason=fallback,
            probability_metrics_eligible=key != 'always_1_1',
            score_matrix=matrix.tolist(), matrix_shape=list(matrix.shape), matrix_mass=float(matrix.sum()),
            matrix_support={'home_max': matrix.shape[0]-1, 'away_max': matrix.shape[1]-1},
            tail_policy='Out-of-grid actuals use epsilon=1e-12 and are counted separately.',
            top10_scores=ranked, predicted_score=ranked[0][:2],
            predicted_score_probability=ranked[0][2],
            outcome_probabilities={'home': float(np.tril(matrix, -1).sum()),
                                   'draw': float(np.trace(matrix)), 'away': float(np.triu(matrix, 1).sum())},
            expected_goals={'home': float((matrix*np.arange(matrix.shape[0])[:, None]).sum()),
                            'away': float((matrix*np.arange(matrix.shape[1])[None, :]).sum())})
        # The storage layer independently checks the same identity fields.
        from forecast_ledger import forecast_id
        record['forecast_id'] = forecast_id(record)
        output.append(record)
    return output


def _valid_record(record):
    if record.get('schema_version') != SCHEMA_VERSION or record.get('mode') != 'prospective' or record.get('score_basis') != 'regulation':
        raise ValueError('Not a prospective regulation-time ledger record')
    if record.get('model_key') not in MODELS:
        raise ValueError('Unknown model contributor')
    captured, cutoff, kickoff = (_date(record[key]) for key in ('captured_at_utc', 'evidence_as_of_utc', 'kickoff_utc'))
    if cutoff > captured or captured >= kickoff or captured-cutoff > MAX_CAPTURE_LAG:
        raise ValueError('Invalid prospective chronology')
    if horizon_for(kickoff, captured) != record.get('horizon'):
        raise ValueError('Invalid capture horizon')
    if not record.get('input_digest') or not record.get('pair_id') or not record.get('policy_version'):
        raise ValueError('Missing same-evidence provenance')
    if not record.get('model_version') or not record.get('numerical_model_version'):
        raise ValueError('Missing numerical model identity')
    if any(record.get(key) is None for key in ('provider', 'fixture_id', 'home_team_id', 'away_team_id')):
        raise ValueError('Missing provider-qualified fixture identity')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', str(record['provider'])):
        raise ValueError('Invalid provider identity')
    for key in ('fixture_id', 'home_team_id', 'away_team_id'):
        if isinstance(record[key], bool) or not str(record[key]).isdigit() or int(record[key]) <= 0:
            raise ValueError('Provider fixture and team IDs must be positive integers')
    if str(record['home_team_id']) == str(record['away_team_id']):
        raise ValueError('Ordered teams must be distinct')
    matrix = _matrix(record.get('score_matrix'))
    if list(matrix.shape) != record.get('matrix_shape') or not math.isclose(float(matrix.sum()), float(record.get('matrix_mass', -1)), abs_tol=1e-6):
        raise ValueError('Stored score-grid metadata is inconsistent')
    ranked = _rank(matrix)
    if ranked[0][:2] != record.get('predicted_score') or not math.isclose(ranked[0][2], float(record.get('predicted_score_probability', -1)), abs_tol=1e-9):
        raise ValueError('Stored score selection does not match the full grid')
    return matrix


def _canonical_record(raw):
    record = deepcopy(raw)
    _valid_record(record)
    for key in ('fixture_id', 'home_team_id', 'away_team_id'):
        record[key] = str(int(record[key]))
    for key in ('kickoff_utc', 'captured_at_utc', 'evidence_as_of_utc'):
        record[key] = _date(record[key]).isoformat()
    # Storage uses accurate summation and can differ by the last floating-point
    # bit from a NumPy reduction. Equivalent immutable grids remain duplicates.
    record['matrix_mass'] = math.fsum(value for row in record['score_matrix'] for value in row)
    return record


def _same_context(left, right):
    same_fields = all(str(left.get(field)) == str(right.get(field)) for field in
               ('provider', 'fixture_id', 'home_team_id', 'away_team_id', 'kickoff_utc',
                'captured_at_utc', 'evidence_as_of_utc', 'input_digest', 'policy_version', 'horizon'))
    def competition(row):
        value = row.get('competition') or {}
        return tuple(str(value.get(field, '')) for field in ('id', 'code', 'season'))
    return same_fields and competition(left) == competition(right)


def _eligible(record):
    if not record or record.get('eligible_for_comparison') is not True:
        return False
    if record.get('model_key') == 'v7' and (record.get('fallback_reason') or
            record.get('numerical_model_version') == engine.LIVE_MODEL_VERSION):
        return False
    return True


def _settlement_matches(record, result):
    try:
        if not (result.get('verified') is True or result.get('verification_status') == 'verified'
                or result.get('evidence_version') == 2):
            return False
        if result.get('score_basis') != 'regulation' or result.get('status') not in ('FINISHED', 'FT', 'AET', 'PEN'):
            return False
        if str(record['provider']) != str(result.get('provider')):
            return False
        for key in ('fixture_id', 'home_team_id', 'away_team_id'):
            if isinstance(result.get(key), bool) or str(int(record[key])) != str(int(result.get(key))):
                return False
        if _date(record['kickoff_utc']) != _date(result.get('kickoff_utc')):
            return False
        if _date(result.get('verified_at_utc')) <= max(_date(record['captured_at_utc']), _date(record['kickoff_utc'])):
            return False
        return all(type(result.get(side+'_goals')) is int and result[side+'_goals'] >= 0 for side in ('home', 'away'))
    except (ValueError, TypeError, KeyError):
        return False


def _metrics(rows):
    n = len(rows)
    if not n:
        return {'evaluated': 0, 'exact_hits': 0, 'exact_hit_rate': None,
                'top3_hit_rate': None, 'score_log_loss': None, 'outcome_brier': None,
                'expected_exact_hits': None, 'calibration': []}
    probabilistic = [row for row in rows if row['probabilistic']]
    bins = []
    for lower, upper in ((0, .05), (.05, .10), (.10, .15), (.15, .20), (.20, 1.000001)):
        values = [row for row in probabilistic if lower <= row['probability'] < upper]
        if values:
            bins.append({'lower': lower, 'upper': min(upper, 1.0), 'matches': len(values),
                         'mean_probability': sum(row['probability'] for row in values)/len(values),
                         'observed_rate': sum(row['hit'] for row in values)/len(values)})
    hits = sum(row['hit'] for row in rows)
    repeated = len({row['fixture_identity'] for row in rows}) != n
    z = 1.959963984540054
    center = (hits/n+z*z/(2*n))/(1+z*z/n)
    width = z*math.sqrt((hits/n)*(1-hits/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'evaluated': n, 'exact_hits': hits, 'exact_hit_rate': hits/n,
            'exact_rate_wilson95': None if repeated else [max(0, center-width), min(1, center+width)],
            'wilson_interval_status': ('suppressed_repeated_fixture_horizons' if repeated else
                                      'one_slot_per_physical_fixture'),
            'probability_evaluated': len(probabilistic),
            'top3_hit_rate': sum(r['top3'] for r in probabilistic)/len(probabilistic) if probabilistic else None,
            'score_log_loss': sum(r['loss'] for r in probabilistic)/len(probabilistic) if probabilistic else None,
            'outcome_brier': sum(r['brier'] for r in probabilistic)/len(probabilistic) if probabilistic else None,
            'outcome_accuracy': sum(r['outcome_hit'] for r in probabilistic)/len(probabilistic) if probabilistic else None,
            'expected_exact_hits': sum(r['probability'] for r in probabilistic) if probabilistic else None,
            'out_of_grid_actuals': sum(r['out_of_grid'] for r in probabilistic),
            'calibration': bins}


def _paired(rows):
    n = len(rows)
    physical = {(str(row['provider']), str(row['fixture_id']), row['kickoff_utc']) for row in rows}
    repeated = len(physical) != n
    a = sum(row['v6_hit'] and not row['v7_hit'] for row in rows)
    b = sum(row['v7_hit'] and not row['v6_hit'] for row in rows)
    discordant = a+b
    if discordant:
        logs = [math.lgamma(discordant+1)-math.lgamma(k+1)-math.lgamma(discordant-k+1)-discordant*math.log(2)
                for k in range(min(a, b)+1)]
        p = min(1.0, 2*sum(math.exp(value) for value in logs))
    else:
        p = 1.0
    blocks = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row['competition'], row['kickoff_utc'][:10])
        blocks[key][0] += int(row['v7_hit'])-int(row['v6_hit'])
        blocks[key][1] += 1
    interval = None
    if len(blocks) >= 2:
        rng, values, draws = random.Random(0), list(blocks.values()), []
        for _ in range(1000):
            chosen = rng.choices(values, k=len(values))
            draws.append(sum(value[0] for value in chosen)/sum(value[1] for value in chosen))
        interval = [float(v) for v in np.percentile(draws, [2.5, 97.5])]
    return {'matches': n, 'v6_only_hits': a, 'v7_only_hits': b,
            'both_hits': sum(row['v6_hit'] and row['v7_hit'] for row in rows),
            'neither_hits': sum(not row['v6_hit'] and not row['v7_hit'] for row in rows),
            'v7_minus_v6_exact_rate': (b-a)/n if n else None,
            'mcnemar_exact_p': p if n and not repeated else None,
            'mcnemar_status': ('suppressed_repeated_fixture_horizons' if repeated else
                               'one_slot_per_physical_fixture' if n else 'no_verified_pairs'),
            'unique_fixtures': len(physical), 'repeated_fixture_slots': n-len(physical),
            'league_date_block_bootstrap95': interval,
            'uncertainty_note': 'Diagnostic intervals; shared teams, repeated horizons and evolving histories limit independence assumptions. No automatic model promotion.'}


def evaluate(forecasts, results):
    """Grade the earliest complete common snapshot per fixture and horizon.

    Verification is joined by provider, fixture, ordered team IDs and kickoff.
    Existing original accuracy files and retrospective revisions are untouched.
    """
    if isinstance(results, dict):
        results = results.values()
    results = list(results)
    exclusions, groups = Counter(), defaultdict(dict)
    for raw in forecasts:
        try:
            record = _canonical_record(raw)
            group_key = (str(record['provider']), str(record['fixture_id']), record['kickoff_utc'], record['horizon'])
            capture_key = (record['pair_id'], record['captured_at_utc'], record['evidence_as_of_utc'], record['input_digest'], record['policy_version'])
            candidates = groups[group_key].setdefault(capture_key, {})
            model = record['model_key']
            if model in candidates and _json(candidates[model]) != _json(record):
                candidates[model] = None
                exclusions['conflicting_forecast'] += 1
            elif model not in candidates:
                candidates[model] = record
        except (ValueError, TypeError, KeyError, OverflowError):
            exclusions['invalid_forecast'] += 1
    selected = []
    for captures in groups.values():
        paired = [models for models in captures.values() if models.get('v6') and models.get('v7')]
        if not paired:
            exclusions['no_common_v6_v7_snapshot'] += 1
            continue
        # All comparisons use the same cohort, including the two deliberately
        # simple baselines. An interrupted early write must not reserve this
        # fixture forever or create a more favourable baseline subset.
        complete = [models for models in paired if all(models.get(key) for key in MODELS)]
        if not complete:
            exclusions['no_complete_four_model_snapshot'] += 1
            continue
        complete.sort(key=lambda models: _date(models['v6']['captured_at_utc']))
        eligible = [models for models in complete if all(_eligible(models[key]) for key in MODELS)
                    and all(_same_context(models['v6'], models[key]) for key in MODELS)]
        if not eligible:
            exclusions['v7_fallback_or_ineligible'] += 1
            continue
        selected.append(eligible[0])
    model_rows, evaluated, pending = defaultdict(list), [], 0
    for models in selected:
        record = models['v6']
        matching = [r for r in results if _settlement_matches(record, r)]
        scores = {(r['home_goals'], r['away_goals']) for r in matching}
        if not matching:
            pending += 1
            exclusions['awaiting_verified_result'] += 1
            continue
        if len(scores) != 1:
            exclusions['conflicting_verified_results'] += 1
            continue
        hg, ag = next(iter(scores))
        detail = {'provider': record['provider'], 'fixture_id': record['fixture_id'],
            'kickoff_utc': record['kickoff_utc'], 'captured_at_utc': record['captured_at_utc'],
            'horizon': record['horizon'], 'input_digest': record['input_digest'],
            'pair_id': record['pair_id'], 'competition': (record.get('competition') or {}).get('code', 'unknown'),
            'actual_score': [hg, ag]}
        for key, forecast in models.items():
            if forecast is None or not _eligible(forecast):
                continue
            # Contributor IDs, ordered teams and times must remain identical,
            # even if a manually imported record reused an otherwise valid pair ID.
            if not _same_context(forecast, record):
                exclusions['mismatched_contributor_identity'] += 1
                continue
            matrix = _matrix(forecast['score_matrix'])
            ranked = _rank(matrix)
            hit = ranked[0][:2] == [hg, ag]
            out_of_grid = hg >= matrix.shape[0] or ag >= matrix.shape[1]
            probability = 0.0 if out_of_grid else float(matrix[hg, ag])
            outcome = 0 if hg > ag else 1 if hg == ag else 2
            probs = [float(np.tril(matrix, -1).sum()), float(np.trace(matrix)), float(np.triu(matrix, 1).sum())]
            row = {'hit': hit, 'probability': ranked[0][2],
                   'top3': [hg, ag] in [cell[:2] for cell in ranked[:3]],
                   'loss': -math.log(max(EPS, probability)),
                   'brier': sum((p-int(i == outcome))**2 for i, p in enumerate(probs)),
                   'outcome_hit': int(max(range(3), key=lambda i: probs[i]) == outcome),
                   'out_of_grid': out_of_grid,
                   'probabilistic': forecast.get('probability_metrics_eligible', True),
                   'competition': detail['competition'], 'horizon': detail['horizon'],
                   'fixture_identity': (detail['provider'], str(detail['fixture_id']), detail['kickoff_utc'])}
            model_rows[key].append(row)
            detail[key+'_hit'] = hit
            detail[key+'_score'] = ranked[0][:2]
        if 'v6_hit' in detail and 'v7_hit' in detail:
            evaluated.append(detail)
    segments = {}
    for dimension in ('competition', 'horizon'):
        values = sorted({row[dimension] for row in evaluated})
        segments[dimension] = {value: {key: _metrics([row for row in model_rows[key] if row[dimension] == value]) for key in MODELS}
                               for value in values}
    unique_selected = {(str(models['v6']['provider']), str(models['v6']['fixture_id']), models['v6']['kickoff_utc']) for models in selected}
    unique_verified = {(str(row['provider']), str(row['fixture_id']), row['kickoff_utc']) for row in evaluated}
    return {'schema_version': SCHEMA_VERSION, 'comparison_kind': 'genuine_prospective_same_evidence',
            'count_unit': 'provider fixture and forecast horizon',
            'common_frozen_slots': len(selected), 'common_unique_fixtures': len(unique_selected),
            'paired_verified_slots': len(evaluated), 'unique_verified_fixtures': len(unique_verified),
            'common_frozen_fixtures': len(selected), 'paired_verified_matches': len(evaluated),
            'pending_common_forecasts': pending, 'models': {key: _metrics(model_rows[key]) for key in MODELS},
            'paired': _paired(evaluated),
            'paired_by_horizon': {horizon: _paired([row for row in evaluated if row['horizon'] == horizon])
                                  for horizon in sorted({row['horizon'] for row in evaluated})},
            'by_league': segments['competition'],
            'by_horizon': segments['horizon'], 'exclusions': dict(exclusions), 'rows': evaluated,
            'note': 'Earliest complete common pre-kickoff four-contributor snapshot per provider fixture and horizon. Later revisions and old originals are not backfilled. Repeated horizons are separate strata, not independent matches; pooled McNemar inference is suppressed when they repeat a physical fixture.',
            'tail_policy': 'Full stored grids are scored; out-of-grid actual probability uses epsilon=1e-12 and is counted.',
            'fixed_baseline_note': 'Always 1-1 is a point-forecast benchmark. Its artificial one-hot grid is excluded from probability, top-three and calibration metrics.'}
