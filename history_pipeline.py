"""One deterministic, provider-qualified history policy for every forecast path.

The same fixture, as-of time and source rows produce the same evidence digest.
Provider retrieval metadata is recorded separately and never changes that digest.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json

POLICY_VERSION = 'two-season-asof-v1'
HISTORY_DAYS = 730
RESULT_DELAY_HOURS = 3


def _date(value):
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return (result.replace(tzinfo=timezone.utc) if result.tzinfo is None
            else result.astimezone(timezone.utc))


def _id(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return value


def _provider(match):
    return str(match.get('provider') or 'football-data.org')


def season_year(fixture, at=None):
    """Use supplied provider season metadata before the conventional July boundary."""
    competition = fixture.get('competition') or {}
    value = competition.get('season') or fixture.get('season')
    if isinstance(value, dict):
        value = value.get('year') or str(value.get('startDate') or '')[:4]
    try:
        return int(value)
    except (TypeError, ValueError):
        when = _date(at or fixture.get('utcDate'))
        return when.year if competition.get('code') == 'BSA' or when.month >= 7 else when.year-1


def _same_competition(fixture, match):
    expected = fixture.get('competition') or {}
    actual = match.get('competition') or {}
    if expected.get('id') is not None:
        return actual.get('id') is not None and str(actual['id']) == str(expected['id'])
    return bool(expected.get('code') and actual.get('code') == expected['code'])


def _normalise(match):
    result = deepcopy(match)
    result['provider'] = _provider(match)
    result['utcDate'] = _date(match.get('utcDate')).isoformat()
    result['id'] = _id(result.get('id'))
    for side in ('homeTeam', 'awayTeam'):
        result[side] = dict(result.get(side) or {})
        result[side]['id'] = _id(result[side].get('id'))
    result['competition'] = dict(result.get('competition') or {})
    if result['competition'].get('id') is not None:
        result['competition']['id'] = _id(result['competition']['id'])
    return result


def _canonical(match, available):
    competition = match.get('competition') or {}
    goals = match['score']['fullTime']
    return {'provider': _provider(match), 'fixture_id': match.get('id'),
            'kickoff_utc': match['utcDate'], 'result_available_at_utc': available.isoformat(),
            'competition_id': competition.get('id'), 'competition_code': competition.get('code'),
            'season': season_year(match),
            'home_id': match['homeTeam']['id'], 'away_id': match['awayTeam']['id'],
            'home_goals': goals['home'], 'away_goals': goals['away']}


def _safe_metadata(source_metadata):
    values = source_metadata or []
    if isinstance(values, dict):
        values = values.get('sources', [values])
    allowed = {'provider', 'endpoint', 'season', 'fetched_at_utc', 'retrieved_at_utc',
               'age_seconds', 'cache_status', 'stale', 'ttl_seconds', 'freshness_known',
               'source_season_mismatches'}
    return [{key: row[key] for key in sorted(allowed) if key in row}
            for row in values if isinstance(row, dict)]


def build_evidence(fixture, matches, as_of, source_metadata=None):
    """Filter source results using the identical league/as-of policy everywhere.

    Explicit final-whistle timestamps are preferred. When unavailable, three
    hours after kickoff is a conservative estimate of result availability.
    Conflicting duplicate results are excluded rather than selected by order.
    """
    fixture = _normalise(fixture)
    requested = _date(as_of)
    cutoff = min(requested, _date(fixture['utcDate']))
    start = cutoff-timedelta(days=HISTORY_DAYS)
    seasons = {season_year(fixture), season_year(fixture)-1}
    exclusions = Counter()
    warnings = []
    if requested > cutoff:
        warnings.append('History cutoff was limited to the target kickoff.')
    groups = {}
    for raw in matches:
        try:
            if not isinstance(raw, dict):
                exclusions['invalid_row'] += 1
                continue
            if _provider(raw) != _provider(fixture):
                exclusions['different_provider'] += 1
                continue
            if not _same_competition(fixture, raw):
                exclusions['different_competition'] += 1
                continue
            match = _normalise(raw)
            if fixture.get('id') is not None and str(match.get('id')) == str(fixture['id']):
                exclusions['target_fixture'] += 1
                continue
            if match.get('status') != 'FINISHED':
                exclusions['unfinished'] += 1
                continue
            score = match.get('score') or {}
            if score.get('duration') not in (None, '', 'REGULAR') or match.get('score_basis', 'regulation') != 'regulation':
                exclusions['non_regulation'] += 1
                continue
            full = score.get('fullTime') or {}
            if any(type(full.get(side)) is not int or full[side] < 0 for side in ('home', 'away')):
                exclusions['invalid_score'] += 1
                continue
            home, away = match['homeTeam']['id'], match['awayTeam']['id']
            if home is None or away is None or home == away or type(home) is bool or type(away) is bool:
                exclusions['invalid_teams'] += 1
                continue
            when = _date(match['utcDate'])
            if when < start:
                exclusions['outside_window'] += 1
                continue
            if season_year(match) not in seasons:
                exclusions['outside_seasons'] += 1
                continue
            available = _date(match['finishedAt']) if match.get('finishedAt') else when+timedelta(hours=RESULT_DELAY_HOURS)
            if available < when:
                exclusions['invalid_result_timestamp'] += 1
                continue
            if when >= cutoff or available > cutoff:
                exclusions['not_available_as_of'] += 1
                continue
            canonical = _canonical(match, available)
            key = (match.get('id') if match.get('id') is not None else
                   (match['utcDate'], str(home), str(away)))
            groups.setdefault(str(key), []).append((match, canonical))
        except (KeyError, ValueError, TypeError, OverflowError):
            exclusions['invalid_row'] += 1
    selected = []
    for group in groups.values():
        distinct = {json.dumps(value, sort_keys=True, separators=(',', ':')) for _, value in group}
        if len(distinct) != 1:
            exclusions['conflicting_duplicate'] += len(group)
            continue
        exclusions['duplicate'] += max(0, len(group)-1)
        # Extra descriptive fields are not numerical inputs. Choose a stable
        # representative so all consumers receive equal rows as well as digests.
        match, canonical = min(group, key=lambda pair: json.dumps(pair[0], sort_keys=True, default=str))
        selected.append((match, canonical))
    selected.sort(key=lambda pair: (_date(pair[0]['utcDate']), str(pair[0].get('id'))), reverse=True)
    history = [match for match, _ in selected]
    competition = fixture.get('competition') or {}
    identity = {'provider': _provider(fixture), 'fixture_id': fixture.get('id'),
                'kickoff_utc': fixture['utcDate'], 'competition_id': competition.get('id'),
                'competition_code': competition.get('code'), 'season': season_year(fixture),
                'home_id': (fixture.get('homeTeam') or {}).get('id'),
                'away_id': (fixture.get('awayTeam') or {}).get('id')}
    canonical_inputs = {'policy_version': POLICY_VERSION, 'fixture': identity,
                        'as_of_utc': cutoff.isoformat(), 'history_window_start_utc': start.isoformat(),
                        'results': [value for _, value in selected]}
    digest = hashlib.sha256(json.dumps(canonical_inputs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    sources = _safe_metadata(source_metadata)
    if any(row.get('stale') for row in sources):
        warnings.append('At least one history response is a labelled stale fallback.')
    if not sources or any(not row.get('freshness_known', row.get('fetched_at_utc') is not None) for row in sources):
        warnings.append('Provider fetch freshness is unknown for at least one history source.')
    return {'fixture': fixture, 'history': history, 'league_history': history,
            'as_of_utc': cutoff.isoformat(), 'history_window_start_utc': start.isoformat(),
            'policy_version': POLICY_VERSION, 'input_digest': digest,
            'provenance': {'provider': _provider(fixture), 'seasons': sorted(seasons, reverse=True),
                           'history_days': HISTORY_DAYS, 'requested_as_of_utc': requested.isoformat(),
                           'result_availability_policy': 'finishedAt, otherwise kickoff plus three hours',
                           'sources': sources},
            'exclusions': {key: value for key, value in sorted(exclusions.items()) if value},
            'warnings': warnings}


def _with_metadata(callback, *args):
    try:
        parameters = inspect.signature(callback).parameters.values()
        accepts = any(p.name == 'with_metadata' or p.kind == p.VAR_KEYWORD for p in parameters)
    except (ValueError, TypeError):
        accepts = False
    value = callback(*args, with_metadata=True) if accepts else callback(*args)
    if isinstance(value, dict) and 'data' in value and isinstance(value.get('metadata'), dict):
        return value['data'], dict(value['metadata'])
    return value, {'freshness_known': False, 'cache_status': 'unreported', 'stale': None}


def load_evidence(fixture, as_of=None, api_key='', primary_get=None, api_request=None, cache=None):
    """Fetch both seasons once per run and apply the shared pure evidence policy.

    Callbacks may accept ``with_metadata=True`` and return a data/metadata
    envelope; old callbacks returning only their raw payload remain compatible.
    A caller-supplied cache dictionary shares league requests across fixtures.
    """
    fixture = deepcopy(fixture)
    as_of = as_of or datetime.now(timezone.utc)
    provider = _provider(fixture)
    competition = fixture.get('competition') or {}
    code = competition.get('code') or ''
    current = season_year(fixture)
    cache = {} if cache is None else cache
    matches, sources, errors = [], [], []
    for season in (current, current-1):
        cache_key = (provider, str(competition.get('id') or code), season)
        if cache_key not in cache:
            try:
                if provider == 'API-Football':
                    from api_football import _request, _convert, LEAGUES
                    callback = api_request or _request
                    league = competition.get('id') or LEAGUES.get(code)
                    if league is None:
                        raise ValueError('League identifier is unavailable')
                    if api_request is None and not api_key:
                        raise ValueError('API-Football credentials are unavailable')
                    raw, metadata = _with_metadata(callback, api_key, '/fixtures', {'league': league, 'season': season})
                    if isinstance(raw, dict):
                        raw = raw.get('response', [])
                    rows = [_convert(item, code) for item in (raw or [])]
                elif provider == 'football-data.org':
                    if primary_get is None:
                        raise ValueError('Primary-provider history callback is unavailable')
                    from result_verification import primary_result
                    raw, metadata = _with_metadata(primary_get, f'/competitions/{code}/matches?season={season}&status=FINISHED')
                    rows = [primary_result(item) for item in (raw or {}).get('matches', [])]
                else:
                    raise ValueError('Unsupported history provider')
                wrong_season=sum(season_year(row) != season for row in rows)
                rows=[row for row in rows if season_year(row) == season]
                metadata = dict(metadata, provider=provider, season=season,
                                source_season_mismatches=wrong_season,
                                endpoint='/fixtures' if provider == 'API-Football' else f'/competitions/{code}/matches')
                cache[cache_key] = (rows, metadata, None)
            except Exception as exc:
                # Never persist callback exceptions or provider response text:
                # a custom adapter could include a credential in its message.
                cache[cache_key] = ([], {'provider': provider, 'season': season, 'freshness_known': False}, type(exc).__name__)
        rows, metadata, error = cache[cache_key]
        matches.extend(rows)
        sources.append(metadata)
        if error:
            errors.append(f'{provider} season {season} history unavailable ({error}).')
        if metadata.get('source_season_mismatches'):
            errors.append(f"{metadata['source_season_mismatches']} source rows outside requested season {season} were excluded.")
    result = build_evidence(fixture, matches, as_of, sources)
    result['warnings'].extend(errors)
    return result
