"""Capture immutable pre-kickoff forecasts from the shared evidence policy.

Every supported future fixture is eligible. A configurable run budget is
reported explicitly, and known fixture/horizon captures are skipped so the
15-minute collector prioritises new and near-kickoff evidence.
"""
import json
import inspect
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from api_football import LEAGUES, _convert, _request
from forecast_model import predict
from history_pipeline import load_evidence

CODES = ("PL", "PD", "BL1", "SA", "FL1", "CL", "EL", "DED", "PPL", "BSA", "ELC")
LOOKAHEAD_DAYS = 14
OUTPUT = Path("prematch_snapshots")
LEDGER_ROOT = Path("data/prospective_ledger")

DISCOVERY_TTL_SECONDS = 21600


class CaptureBudgetExceeded(RuntimeError):
    pass


def _unpack(value):
    if isinstance(value, dict) and 'data' in value:
        allowed = ('fetched_at_utc','retrieved_at_utc','age_seconds','cache_status','stale','ttl_seconds','freshness_known')
        return value['data'], {name:value.get('metadata', {}).get(name) for name in allowed if name in value.get('metadata', {})}
    return value, {'freshness_known': False, 'cache_status': 'unreported'}


def _write_discovery(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def _discovery(request, key, code, season, started, ledger_root):
    """Reuse durable schedule discovery; urgent due fixtures are checked by ID."""
    path = Path(ledger_root)/'discovery'/(code+'_'+str(season)+'.json')
    params = {'league': LEAGUES[code], 'season': season,
              'from': started.date().isoformat(),
              'to': (started+timedelta(days=LOOKAHEAD_DAYS)).date().isoformat()}
    try:
        saved = json.loads(path.read_text(encoding='utf-8'))
        age = (started-_time(saved['observed_at_utc'])).total_seconds()
        if saved['params'] == params and 0 <= age < DISCOVERY_TTL_SECONDS:
            metadata = dict(saved.get('metadata') or {})
            metadata.update(cache_status='committed_discovery_cache', discovery_age_seconds=age)
            return saved['fixtures'], metadata, True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    fixtures, metadata = _unpack(request(key, '/fixtures', params, with_metadata=True))
    if not isinstance(fixtures, list):
        raise ValueError('Fixture discovery response is not a list')
    _write_discovery(path, {'observed_at_utc': started.isoformat(), 'params': params,
                           'fixtures': fixtures, 'metadata': metadata})
    return fixtures, metadata, False


def _news(request, key, fixture, now):
    """Capture actual confirmed-XI and injury payloads under the same API budget."""
    report = {'checked_at': now.isoformat(), 'lineups': [], 'injuries': [], 'warnings': [], 'provenance': []}
    for endpoint in ('/fixtures/lineups', '/injuries'):
        try:
            data, metadata = _unpack(request(key, endpoint, {'fixture': fixture['id']}, with_metadata=True))
            report['provenance'].append(dict(metadata, endpoint=endpoint))
            if endpoint.endswith('lineups'):
                for row in data or []:
                    players = [entry.get('player', {}).get('name') for entry in row.get('startXI', [])]
                    players = [name for name in players if name]
                    if players:
                        report['lineups'].append({'team': (row.get('team') or {}).get('name'),
                            'formation': row.get('formation'), 'starters': players, 'confirmed': len(players)==11})
            else:
                for row in (data or [])[:30]:
                    player = row.get('player') or {}
                    if player.get('name'):
                        report['injuries'].append({'team': (row.get('team') or {}).get('name'),
                            'player': player['name'], 'reason': player.get('reason') or 'Not provided'})
        except CaptureBudgetExceeded:
            raise
        except Exception as exc:
            report['warnings'].append(endpoint+' unavailable ('+type(exc).__name__+').')
    return report


def request_fixtures(key, params):
    """Retain the old public helper while respecting provider cache metadata."""
    return _request(key, "/fixtures", params)


def _time(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def _positive_limit(value, default):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def known_captures(records):
    """Only one coherent, eligible four-contributor package completes a slot.

    A process may stop between immutable file writes. Combining its v6 row
    with a later v7 row would manufacture a comparison that was never made;
    incomplete or fallback packages must remain retryable.
    """
    from history_pipeline import POLICY_VERSION
    from engine import LIVE_MODEL_VERSION as V6_VERSION
    from forecast_model import LIVE_MODEL_VERSION as V7_VERSION
    expected = {'v6': V6_VERSION, 'v7': V7_VERSION,
                'league_poisson': 'prior-league-poisson-v1',
                'always_1_1': 'fixed-always-1-1-v1'}
    groups = {}
    for row in records:
        if (row.get('mode') != 'prospective' or row.get('policy_version') != POLICY_VERSION
                or row.get('eligible_for_comparison') is not True):
            continue
        key = row.get('model_key')
        if key not in expected or row.get('model_version') != expected[key]:
            continue
        try:
            required = ('slot_key','pair_id','input_digest','provider','fixture_id','horizon',
                        'home_team_id','away_team_id')
            if any(row.get(field) is None or str(row[field]) == '' for field in required):
                continue
            context = tuple(str(row[field]) for field in required) + tuple(
                _time(row[field]).isoformat() for field in
                ('evidence_as_of_utc','captured_at_utc','kickoff_utc'))
        except (KeyError, ValueError, TypeError):
            continue
        groups.setdefault(context,set()).add(key)
    return {context[0] for context, keys in groups.items() if keys == set(expected)}


def _legacy_record(fixture, forecast, code, evidence, captured_at, report):
    """Keep the original snapshot format for existing accuracy and doubles readers."""
    return {
        'captured_at_utc': captured_at, 'fixture_id': fixture['id'], 'competition': code,
        'provider': fixture.get('provider', 'API-Football'),
        'home_team_id': fixture['homeTeam']['id'], 'away_team_id': fixture['awayTeam']['id'],
        'score_basis': 'regulation', 'kickoff_utc': fixture['utcDate'],
        'home_team': fixture['homeTeam']['name'], 'away_team': fixture['awayTeam']['name'],
        'forecast_model': forecast.get('model_version', 'baseline'),
        'model_sources': forecast.get('model_sources', []),
        'reference_model_comparison': forecast.get('candidate_summary'),
        # The old experimental comparators remain separate; their absence is
        # explicit rather than invented from the new pair's probabilities.
        'prospective_challenger': {'available': False, 'reason': 'Use the versioned prospective v6/v7 ledger for this capture.'},
        'five_model_consensus': {'available': False, 'reason': 'Use the versioned prospective v6/v7 ledger for this capture.'},
        'predicted_score': list(forecast['scores'][0][:2]),
        'double_top10_scores': [[int(h), int(a), float(p)] for h, a, p in forecast['scores'][:10]],
        'predicted_score_probability': float(forecast['scores'][0][2]),
        'predicted_outcome_probabilities': {'home': float(forecast['home_win']), 'draw': float(forecast['draw']), 'away': float(forecast['away_win'])},
        'expected_goals': {'home': float(forecast['home_xg']), 'away': float(forecast['away_xg'])},
        'lineups': report.get('lineups', []), 'injuries': report.get('injuries', []),
        'provider_warnings': report.get('warnings', []) + evidence.get('warnings', []),
        'news_provenance': report.get('provenance', []),
        'policy_version': evidence['policy_version'], 'input_digest': evidence['input_digest'],
        'evidence_as_of_utc': evidence['as_of_utc'], 'provenance': evidence.get('provenance', []),
    }


def collect(key, now=None, *, codes=CODES, max_fixtures=0, max_seconds=600,
            ledger_root=LEDGER_ROOT, database_url=None, max_api_calls=120, api_request=None,
            evidence_loader=None, pair_builder=None, clock=None,
            daily_recorder=None, research_recorder=None,
            research_backfill_limit=0):
    """Return old-format snapshots and full-grid ledger coverage without fake history.

    Optional callbacks make data policy, timing and budget tests independent of
    network access. Zero fixture/time limits mean no artificial run limit.
    """
    from prospective_ledger import capture_pair, horizon_for, pair_slot_key
    from history_pipeline import POLICY_VERSION
    from forecast_ledger import append_file_record, read_file_records, open_ledger, sync_file_ledger
    callback = api_request or _request
    provider_calls = 0
    try:
        parameters = inspect.signature(callback).parameters.values()
        accepts_metadata = any(p.name=='with_metadata' or p.kind==inspect.Parameter.VAR_KEYWORD for p in parameters)
    except (ValueError, TypeError):
        accepts_metadata = False
    def request(*args, **kwargs):
        nonlocal provider_calls
        if max_api_calls and provider_calls >= max_api_calls:
            raise CaptureBudgetExceeded('Configured provider-call budget exhausted')
        provider_calls += 1
        if not accepts_metadata:
            kwargs.pop('with_metadata', None)
        return callback(*args, **kwargs)
    evidence_loader = evidence_loader or load_evidence
    pair_builder = pair_builder or capture_pair
    fixed_now = _time(now) if now is not None else None
    current_time = clock or ((lambda: fixed_now) if fixed_now is not None else (lambda: datetime.now(timezone.utc)))
    started = _time(current_time())
    monotonic_start = time.monotonic()
    diagnostics = {'leagues_checked': 0, 'upcoming_fixtures': 0, 'eligible_horizon_fixtures': 0,
        'already_captured': 0, 'processed_fixtures': 0, 'captured_fixtures': 0, 'ledger_records_written': 0,
        'prediction_unavailable': 0, 'prediction_errors': [], 'league_errors': [], 'news_errors': [], 'history_source_errors': [],
        'budget_skipped': [], 'provider_fixture_counts': {}, 'next_kickoff_by_league': {},
        'lookahead_days': LOOKAHEAD_DAYS, 'fixture_limit': max_fixtures, 'time_limit_seconds': max_seconds,
        'database_status': 'configured' if database_url else 'not_configured', 'database_records_written': 0,
        'database_recovered_forecasts': 0, 'database_recovered_settlements': 0,
        'discovery_cache_hits': 0, 'api_call_limit': max_api_calls, 'provider_adapter_calls': 0, 'discovery_budget_skipped': [],
        'v10_shadow_v8_saved':0,'v10_shadow_v9_saved':0,'v10_market_calls':0,
        'v10_shadow_skipped':{},'v10_shadow_errors':[]}
    # Recover the immutable outbox before deciding that every horizon is done.
    # A database outage must not strand old files simply because no new pair is due.
    store = None
    if database_url:
        try:
            store = open_ledger(database_url=database_url)
            recovered = sync_file_ledger(store, root=ledger_root)
            diagnostics['database_recovered_forecasts'] = recovered.get('forecasts',0)
            diagnostics['database_recovered_settlements'] = recovered.get('settlements',0)
            diagnostics['database_status'] = 'connected'
        except Exception as exc:
            diagnostics['database_status'] = 'unavailable'
            diagnostics['database_error_type'] = type(exc).__name__
            if store is not None:
                store.close()
                store = None
    old_forecasts=read_file_records(root=ledger_root,kind='forecasts')
    known = known_captures(old_forecasts)
    fixtures, fixture_sources = {}, {}
    for code in codes:
        diagnostics['leagues_checked'] += 1
        season = started.year if code == 'BSA' or started.month >= 7 else started.year-1
        try:
            raw, metadata, cached = _discovery(request, key, code, season, started, ledger_root)
            diagnostics['discovery_cache_hits'] += int(cached)
        except CaptureBudgetExceeded:
            diagnostics['discovery_budget_skipped'].append({'competition':code,'reason':'configured_api_call_budget'})
            continue
        except Exception as exc:
            diagnostics['league_errors'].append({'league': code, 'stage': 'fixtures', 'error_type': type(exc).__name__})
            continue
        diagnostics['provider_fixture_counts'][code] = len(raw)
        for item in raw:
            fixture = _convert(item, code)
            try:
                kickoff = _time(fixture['utcDate'])
            except (ValueError, TypeError):
                continue
            if not started < kickoff <= started+timedelta(days=LOOKAHEAD_DAYS) or fixture['status'] not in ('SCHEDULED', 'TIMED'):
                continue
            if fixture.get('provider_status') not in (None, '', 'NS', 'TBD'):
                continue
            identity = (fixture.get('provider'), str(fixture['id']))
            fixtures[identity] = (kickoff, fixture, code)
            fixture_sources[identity] = metadata
        candidates = [value[0] for value in fixtures.values() if value[2] == code]
        if candidates:
            diagnostics['next_kickoff_by_league'][code] = min(candidates).isoformat()
    diagnostics['upcoming_fixtures'] = len(fixtures)
    priority = {'T30m': 0, 'T60m': 1, 'T24h': 2, 'research': 3, 'baseline': 3}
    due = []
    for kickoff, fixture, code in fixtures.values():
        horizon = horizon_for(fixture['utcDate'], started)
        identity = pair_slot_key(fixture,horizon,POLICY_VERSION)
        if identity in known:
            diagnostics['already_captured'] += 1
            continue
        due.append((priority.get(horizon, 4), kickoff, fixture, code, horizon))
    due.sort(key=lambda item: (item[0], item[1], str(item[2]['id'])))
    diagnostics['eligible_horizon_fixtures'] = len(due)
    records, ledger_records, history_cache = [], [], {}
    try:
        for _, kickoff, fixture, code, horizon in due:
            if ((max_fixtures and diagnostics['processed_fixtures'] >= max_fixtures)
                    or (max_seconds and time.monotonic()-monotonic_start >= max_seconds)):
                diagnostics['budget_skipped'].append({'fixture_id': fixture['id'], 'provider': fixture.get('provider'), 'horizon': horizon, 'reason': 'configured_run_budget'})
                continue
            diagnostics['processed_fixtures'] += 1
            try:
                as_of = _time(current_time())
                if as_of >= kickoff:
                    raise ValueError('fixture started while collecting evidence')
                if kickoff-as_of <= timedelta(minutes=90):
                    fresh, fresh_metadata = _unpack(request(key, '/fixtures', {'id':str(fixture['id'])}, with_metadata=True))
                    refreshed = next((_convert(item,code) for item in fresh
                                      if str((item.get('fixture') or {}).get('id')) == str(fixture['id'])), None)
                    if (not refreshed or _time(refreshed['utcDate']) != kickoff
                            or refreshed.get('provider_status') not in ('NS','TBD')):
                        raise ValueError('near-kickoff fixture is no longer confirmed at the captured kickoff')
                    fixture = refreshed
                    fixture_sources[(fixture.get('provider'),str(fixture['id']))] = fresh_metadata
                evidence = evidence_loader(fixture, as_of=as_of, api_key=key, api_request=request, cache=history_cache)
                evidence.setdefault('provenance', {})['fixture_source'] = fixture_sources.get((fixture.get('provider'),str(fixture['id'])), {})
                if any('CaptureBudgetExceeded' in warning for warning in evidence.get('warnings', [])):
                    raise CaptureBudgetExceeded('History lookup reached the configured provider-call budget')
                if any('history unavailable' in warning for warning in evidence.get('warnings', [])):
                    diagnostics['history_source_errors'].append({'fixture_id':fixture['id'],'provider':fixture.get('provider'),'reason':'one_or_more_history_sources_unavailable'})
                # Save the new independent research forecast BEFORE the old
                # baseline gate. A sparse v6/v7 forecast must not erase a
                # separately valid v8 candidate, and vice versa.
                observed_research = _time(current_time())
                if observed_research < kickoff:
                    if research_recorder is not None:
                        try:
                            result = research_recorder(
                                fixture, evidence, observed_research, code, request,
                                max(0, 4-diagnostics['v10_market_calls']))
                            diagnostics['v10_shadow_v8_saved'] += int(result.get('v8_recorded',False))
                            diagnostics['v10_shadow_v9_saved'] += int(result.get('v9_recorded',False))
                            diagnostics['v10_market_calls'] += int(result.get('market_calls',0))
                            reason = result.get('reason') or 'unspecified'
                            if not result.get('v8_recorded'):
                                skipped=diagnostics['v10_shadow_skipped']
                                skipped[reason]=skipped.get(reason,0)+1
                        except Exception as exc:
                            diagnostics['v10_shadow_errors'].append({
                                'fixture_id':fixture['id'],'error_type':type(exc).__name__})
                    if daily_recorder is not None:
                        try:
                            daily_recorder(fixture,evidence,observed_research,code)
                        except Exception as exc:
                            diagnostics.setdefault('daily_pick_errors',[]).append({
                                'fixture_id':fixture['id'],'error_type':type(exc).__name__})
                forecast = predict(fixture, evidence['history'], league_history=evidence['league_history'])
                if not forecast:
                    diagnostics['prediction_unavailable'] += 1
                    continue
                # Re-evaluate horizon after network work; a late collection may
                # enter another real window but must never claim the old one.
                observed = _time(current_time())
                horizon = horizon_for(fixture['utcDate'], observed)
                identity = pair_slot_key(fixture,horizon,POLICY_VERSION)
                if identity in known:
                    diagnostics['already_captured'] += 1
                    continue
                # News is useful close to kickoff; broad early captures need
                # not spend two extra calls on lineups that do not exist yet.
                report = {'lineups': [], 'injuries': [], 'warnings': []}
                if kickoff-observed <= timedelta(minutes=90):
                    try:
                        report = _news(request, key, fixture, observed)
                    except CaptureBudgetExceeded:
                        report['warnings'].append('Near-kickoff news was not fetched because the provider-call budget is exhausted.')
                        diagnostics['news_errors'].append({'fixture_id':fixture['id'],'reason':'configured_api_call_budget'})
                    except Exception as exc:
                        diagnostics['news_errors'].append({'fixture_id': fixture['id'], 'error_type': type(exc).__name__})
                observed = _time(current_time())
                if observed >= kickoff:
                    raise ValueError('fixture started before evidence collection completed')
                horizon = horizon_for(fixture['utcDate'], observed)
                evidence['provenance']['news_sources'] = report.get('provenance', [])
                pair = pair_builder(fixture, evidence, observed_at=observed, horizon=horizon)
                if not pair:
                    diagnostics['prediction_unavailable'] += 1
                    continue
                for entry in pair:
                    written = append_file_record(entry, root=ledger_root, kind='forecasts')
                    diagnostics['ledger_records_written'] += int(bool(written))
                    ledger_records.append(entry)
                if store is not None:
                    try:
                        diagnostics['database_records_written'] += store.put_forecasts(pair)
                        diagnostics['database_status'] = 'connected'
                    except Exception as exc:
                        diagnostics['database_status'] = 'unavailable'
                        diagnostics['database_error_type'] = type(exc).__name__
                        store.close()
                        store = None
                known.add(pair[0].get('slot_key') or pair_slot_key(fixture,horizon,POLICY_VERSION))
                captured_at = pair[0].get('captured_at_utc') or observed.isoformat()
                records.append(_legacy_record(fixture, forecast, code, evidence, captured_at, report))
                diagnostics['captured_fixtures'] += 1
            except CaptureBudgetExceeded:
                diagnostics['budget_skipped'].append({'fixture_id':fixture['id'],'provider':fixture.get('provider'),'horizon':horizon,'reason':'configured_api_call_budget'})
            except Exception as exc:
                diagnostics['prediction_errors'].append({'fixture_id': fixture['id'], 'provider': fixture.get('provider'), 'error_type': type(exc).__name__})
    finally:
        if store is not None:
            store.close()
    # Existing v6/v7 snapshots are NOT recreated or modified. When a fixture
    # is still upcoming but its original research horizon was captured days
    # earlier, record an independent *new* v10 as-of snapshot today.
    # This is deliberately bounded and only enabled on heavy runs.
    diagnostics['v10_research_backfill_checked']=0
    diagnostics['v10_research_backfill_saved']=0
    diagnostics['v10_research_backfill_errors']=[]
    if research_recorder is not None and research_backfill_limit>0:
        try:
            from forecast_ledger import read_file_records as _read_shadow
            old_ids={(row.get('provider'),str(row.get('fixture_id')))
                     for row in old_forecasts if row.get('model_key') in ('v6','v7')}
            already={(row.get('provider'),str(row.get('fixture_id')))
                     for row in _read_shadow(root='data/v10_prospective',kind='forecasts')
                     if row.get('model_key')=='v8'}
            backfill=[(kickoff,fixture,code) for kickoff,fixture,code in fixtures.values()
                      if ((fixture.get('provider'),str(fixture['id'])) in old_ids
                          and (fixture.get('provider'),str(fixture['id'])) not in already
                          and kickoff>_time(current_time()))]
            backfill.sort(key=lambda x:(x[0],str(x[1]['id'])))
            for kickoff,fixture,code in backfill[:research_backfill_limit]:
                if max_seconds and time.monotonic()-monotonic_start>=max_seconds:
                    break
                diagnostics['v10_research_backfill_checked']+=1
                try:
                    now=_time(current_time())
                    if now>=kickoff:
                        continue
                    evidence=evidence_loader(fixture,as_of=now,api_key=key,
                                             api_request=request,cache=history_cache)
                    result=research_recorder(fixture,evidence,_time(current_time()),code,
                                             request,max(0,4-diagnostics['v10_market_calls']))
                    diagnostics['v10_market_calls']+=int(result.get('market_calls',0))
                    saved=int(bool(result.get('v8_recorded')))
                    diagnostics['v10_shadow_v8_saved']+=saved
                    diagnostics['v10_shadow_v9_saved']+=int(bool(result.get('v9_recorded')))
                    diagnostics['v10_research_backfill_saved']+=saved
                    if saved and daily_recorder is not None:
                        daily_recorder(fixture,evidence,_time(current_time()),code)
                except CaptureBudgetExceeded:
                    diagnostics['v10_research_backfill_errors'].append({
                        'fixture_id':fixture['id'],'error_type':'provider_call_budget'})
                    break
                except Exception as exc:
                    diagnostics['v10_research_backfill_errors'].append({
                        'fixture_id':fixture['id'],'error_type':type(exc).__name__})
        except Exception as exc:
            diagnostics['v10_research_backfill_errors'].append({
                'error_type':type(exc).__name__,'stage':'backfill_discovery'})
    diagnostics['provider_adapter_calls'] = provider_calls
    diagnostics['news_coverage_complete'] = not diagnostics['news_errors']
    diagnostics['coverage_complete'] = not (diagnostics['discovery_budget_skipped'] or diagnostics['budget_skipped'] or diagnostics['league_errors'] or diagnostics['prediction_errors'] or diagnostics['prediction_unavailable'] or diagnostics['history_source_errors'])
    return {'captured_at_utc': started.isoformat(), 'records': records, 'ledger_records': ledger_records,
            'diagnostics': diagnostics, 'note': 'Immutable original forecasts and full-grid same-evidence model pairs; no post-kickoff observations are prospective.'}


def main():
    key = os.environ.get('API_FOOTBALL_KEY', '')
    if not key:
        raise RuntimeError('API_FOOTBALL_KEY secret required')
    from daily_pick_capture import candidate as daily_candidate, save_for_dates
    from v10_prospective import capture as capture_v10
    _daily_candidates=[]
    def _record_v10(fixture,evidence,observed,code,request,remaining):
        return capture_v10(fixture,evidence,observed,api_key=key,
                           request=request,market_calls_remaining=remaining)
    def _record_daily(fixture, evidence, observed, code):
        result=daily_candidate(fixture,evidence,observed,code)
        if result:
            _daily_candidates.append(result)
    payload = collect(key, max_fixtures=_positive_limit(os.environ.get('CAPTURE_MAX_FIXTURES'), 0),
        max_seconds=_positive_limit(os.environ.get('CAPTURE_MAX_SECONDS'), 600),
        database_url=os.environ.get('DATABASE_URL') or None,
        max_api_calls=_positive_limit(os.environ.get('CAPTURE_MAX_API_CALLS'),120),
        daily_recorder=_record_daily,research_recorder=_record_v10,
        research_backfill_limit=_positive_limit(os.environ.get('V10_BACKFILL_LIMIT'),0))
    saved_days=save_for_dates(_daily_candidates)
    payload['diagnostics']['daily_research_candidates']=len(_daily_candidates)
    payload['diagnostics']['daily_pick_days_written']=saved_days
    OUTPUT.mkdir(parents=True, exist_ok=True)
    timestamp = _time(payload['captured_at_utc']).strftime('%Y%m%dT%H%M%S%fZ')
    path = OUTPUT / ('prematch_'+timestamp+'.json')
    # Full matrices already live in the separate ledger; do not duplicate
    # every grid in legacy snapshot artifacts.
    legacy_payload = {key: value for key, value in payload.items() if key != 'ledger_records'}
    path.write_text(json.dumps(legacy_payload, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    coverage = LEDGER_ROOT / 'capture_coverage.json'
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text(json.dumps(payload['diagnostics'], indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'saved_records': len(payload['records']), 'diagnostics': payload['diagnostics']}, indent=2))


if __name__ == '__main__':
    main()
