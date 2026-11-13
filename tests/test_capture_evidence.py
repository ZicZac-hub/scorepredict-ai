"""Collection parity, quota and fixed-horizon regression coverage."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np

from capture_prematch import collect
from history_pipeline import build_evidence
from prospective_ledger import capture_pair
from forecast_ledger import read_file_records
from reference_provider_benchmark import evaluate


def raw_fixture(fid, kickoff, status='NS'):
    season = kickoff.year if kickoff.month >= 7 else kickoff.year-1
    return {'fixture': {'id':fid, 'date':kickoff.isoformat(), 'status':{'short':status}},
            'league': {'id':39, 'code':'PL', 'season':season},
            'teams': {'home':{'id':1,'name':'A'}, 'away':{'id':2,'name':'B'}},
            'goals': {'home':None,'away':None}, 'score':{'fulltime':{'home':None,'away':None}}}


def history(now):
    rows = []
    for h in range(1,7):
        for a in range(1,7):
            if h == a:
                continue
            date = now-timedelta(days=50-len(rows))
            rows.append({'id':1000+len(rows), 'provider':'API-Football',
                'competition':{'id':39,'code':'PL','season':date.year if date.month>=7 else date.year-1},
                'utcDate':date.isoformat(),'status':'FINISHED',
                'homeTeam':{'id':h,'name':str(h)},'awayTeam':{'id':a,'name':str(a)},
                'score':{'duration':'REGULAR','fullTime':{'home':2,'away':1}}})
    return rows


def loader(now):
    rows = history(now)
    def load(fixture, as_of, **kwargs):
        return build_evidence(fixture, rows, as_of)
    return load


def forecast(fixture, rows, **kwargs):
    return {'scores':[(1,1,.12),(1,0,.11),(2,1,.1)], 'model_version':'test',
            'model_sources':[], 'home_xg':1.5,'away_xg':1.1,
            'home_win':.45,'draw':.3,'away_win':.25}


def test_capture_no_longer_truncates_at_thirty_and_reuses_discovery(tmp_path):
    now = datetime.now(timezone.utc)
    upcoming = [raw_fixture(i,now+timedelta(days=2,minutes=i)) for i in range(1,33)]
    request = Mock(return_value=upcoming)
    with patch('capture_prematch.predict',side_effect=forecast):
        report = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
                         api_request=request,evidence_loader=loader(now),max_seconds=0)
    assert report['diagnostics']['captured_fixtures'] == 32
    assert report['diagnostics']['coverage_complete'] is True
    assert len(read_file_records(root=tmp_path)) == 32*4
    request.assert_called_once()
    repeat = Mock(side_effect=AssertionError('Discovery must use persisted cache'))
    with patch('capture_prematch.predict',side_effect=AssertionError('Known slot must not recompute')):
        second = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
                         api_request=repeat,evidence_loader=loader(now),max_seconds=0)
    assert second['diagnostics']['already_captured'] == 32
    assert second['diagnostics']['discovery_cache_hits'] == 1
    assert second['diagnostics']['provider_adapter_calls'] == 0
    assert not second['records']


def test_configured_fixture_budget_reports_every_omitted_fixture(tmp_path):
    now = datetime.now(timezone.utc)
    items = [raw_fixture(i,now+timedelta(days=2,minutes=i)) for i in range(1,5)]
    with patch('capture_prematch.predict',side_effect=forecast):
        report = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            api_request=Mock(return_value=items),evidence_loader=loader(now),
            max_fixtures=2,max_seconds=0)
    assert report['diagnostics']['captured_fixtures'] == 2
    assert len(report['diagnostics']['budget_skipped']) == 2
    assert not report['diagnostics']['coverage_complete']


def test_horizon_capture_refreshes_fixture_and_records_news_before_freeze(tmp_path):
    now = datetime.now(timezone.utc)
    target = raw_fixture(1,now+timedelta(minutes=30))
    def request(key, endpoint, params, with_metadata=False):
        if endpoint == '/fixtures':return [target]
        if endpoint == '/fixtures/lineups':
            return [{'team':{'name':'A'},'formation':'4-3-3',
                     'startXI':[{'player':{'name':str(i)}} for i in range(11)]}]
        return []
    calls = Mock(side_effect=request)
    with patch('capture_prematch.predict',side_effect=forecast):
        report = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            api_request=calls,evidence_loader=loader(now),max_seconds=0)
    assert report['diagnostics']['captured_fixtures'] == 1
    assert any(call.args[2] == {'id':'1'} for call in calls.call_args_list)
    assert report['records'][0]['lineups'][0]['confirmed'] is True
    assert all(row['horizon']=='T30m' for row in report['ledger_records'])


def test_api_budget_never_becomes_silent_missing_coverage(tmp_path):
    now = datetime.now(timezone.utc)
    target = raw_fixture(1,now+timedelta(minutes=30))
    with patch('capture_prematch.predict',side_effect=forecast):
        report = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            api_request=Mock(return_value=[target]),evidence_loader=loader(now),
            max_api_calls=1,max_seconds=0)
    assert report['diagnostics']['provider_adapter_calls'] == 1
    assert report['diagnostics']['budget_skipped'][0]['reason']=='configured_api_call_budget'
    assert not report['diagnostics']['coverage_complete']
    assert not read_file_records(root=tmp_path)


def test_postponed_near_kickoff_fixture_never_freezes_old_discovery(tmp_path):
    now = datetime.now(timezone.utc)
    target = raw_fixture(1,now+timedelta(minutes=30))
    postponed = deepcopy(target)
    postponed['fixture']['status']['short']='PST'
    request = Mock(side_effect=[[target],[postponed]])
    with patch('capture_prematch.predict',side_effect=forecast):
        report = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            api_request=request,evidence_loader=loader(now),max_seconds=0)
    assert not report['records']
    assert report['diagnostics']['prediction_errors']
    assert not read_file_records(root=tmp_path)


def test_failed_central_database_does_not_drop_immutable_file_capture(tmp_path):
    now = datetime.now(timezone.utc)
    target = raw_fixture(1,now+timedelta(days=2))
    with patch('capture_prematch.predict',side_effect=forecast), \
         patch('forecast_ledger.open_ledger',side_effect=RuntimeError('Database unavailable')):
        report = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            database_url='postgresql://test-only',api_request=Mock(return_value=[target]),
            evidence_loader=loader(now),max_seconds=0)
    assert report['diagnostics']['database_status']=='unavailable'
    assert len(read_file_records(root=tmp_path)) == 4
    assert report['diagnostics']['captured_fixtures']==1


def test_capture_and_rescore_share_identical_inputs_and_digest():
    from rescore_predictions import rescore_records
    now = datetime.now(timezone.utc)
    target = {'id':1,'provider':'API-Football','competition':{'id':39,'code':'PL'},
              'utcDate':(now+timedelta(days=2)).isoformat(),'status':'SCHEDULED',
              'homeTeam':{'id':1,'name':'A'},'awayTeam':{'id':2,'name':'B'},
              'score':{'fullTime':{'home':None,'away':None}}}
    evidence = build_evidence(target,history(now),now)
    original = {'provider':'API-Football','fixture_id':1,'competition':'PL',
        'kickoff_utc':target['utcDate'],'captured_at_utc':(now-timedelta(days=1)).isoformat(),
        'home_team':'A','away_team':'B','home_team_id':1,'away_team_id':2,
        'predicted_score':[1,1],'forecast_model':'v6'}
    predictor = Mock(side_effect=forecast)
    report = rescore_records([original],lambda *args:history(now),lambda r:target,now,predictor)
    assert report['revisions'][0]['input_digest']==evidence['input_digest']
    assert predictor.call_args.args[1]==evidence['history']


def test_benchmark_uses_forecast_lead_and_result_availability():
    kickoff = datetime(2026,5,10,18,tzinfo=timezone.utc)
    target={'id':90,'provider':'API-Football','competition':{'id':39,'code':'PL','season':2025},
        'utcDate':kickoff.isoformat(),'status':'FINISHED','homeTeam':{'id':1},'awayTeam':{'id':2},
        'score':{'fullTime':{'home':1,'away':1}}}
    recent = deepcopy(target);recent['id']=89;recent['utcDate']=(kickoff-timedelta(hours=2)).isoformat()
    prediction={'matrix':np.array([[.1,.2],[.3,.4]])}
    with patch('reference_provider_benchmark.engine.predict',return_value=prediction) as old, \
         patch('reference_provider_benchmark.predict',return_value=prediction):
        report=evaluate([recent,target],forecast_lead_minutes=60)
    assert old.call_args.args[1]==[]
    assert report['forecast_lead_minutes']==60
    assert report['fixed_baselines']['always_1_1_exact_hits']==2
    assert report['paired']['league_poisson']['matches']==2


def test_rescheduled_fixture_gets_a_new_deduplication_slot(tmp_path):
    from api_football import _convert
    from capture_prematch import known_captures
    from history_pipeline import POLICY_VERSION
    from prospective_ledger import pair_slot_key
    now = datetime.now(timezone.utc)
    raw = raw_fixture(1,now+timedelta(days=2))
    with patch('capture_prematch.predict',side_effect=forecast):
        report = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            api_request=Mock(return_value=[raw]),evidence_loader=loader(now),max_seconds=0)
    slots = known_captures(report['ledger_records'])
    changed = deepcopy(raw)
    changed['fixture']['date'] = (now+timedelta(days=3)).isoformat()
    assert pair_slot_key(_convert(changed,'PL'),'research',POLICY_VERSION) not in slots


def test_api_budget_exhaustion_in_shared_history_is_reported_as_budget_skip(tmp_path):
    now = datetime.now(timezone.utc)
    target = raw_fixture(1,now+timedelta(days=2))
    with patch('capture_prematch.predict',side_effect=AssertionError('No fitting after budget exhaustion')):
        report = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            api_request=Mock(return_value=[target]),max_api_calls=1,max_seconds=0)
    assert report['diagnostics']['provider_adapter_calls']==1
    assert report['diagnostics']['budget_skipped'][0]['reason']=='configured_api_call_budget'
    assert not read_file_records(root=tmp_path)


def _captured_package(now):
    from api_football import _convert
    fixture = _convert(raw_fixture(1,now+timedelta(days=2)),'PL')
    evidence = build_evidence(fixture,history(now),now)
    return capture_pair(fixture,evidence,observed_at=now,horizon='research')


def test_slot_requires_all_four_eligible_models_from_one_capture_context():
    from capture_prematch import known_captures
    now = datetime.now(timezone.utc)
    records = _captured_package(now)
    assert known_captures(records) == {records[0]['slot_key']}
    assert not known_captures(records[:2])
    assert not known_captures([row for row in records if row['model_key']!='league_poisson'])
    mixed = deepcopy(records)
    v7 = next(row for row in mixed if row['model_key']=='v7')
    v7['pair_id']='another-capture'
    v7['input_digest']='f'*64
    assert not known_captures(mixed)
    fallback = deepcopy(records)
    next(row for row in fallback if row['model_key']=='v7')['eligible_for_comparison']=False
    assert not known_captures(fallback)
    other_time = deepcopy(records)
    v7 = next(row for row in other_time if row['model_key']=='v7')
    v7['captured_at_utc']=(now+timedelta(seconds=1)).isoformat()
    assert not known_captures(other_time)


def test_interrupted_file_package_is_retryable_without_overwriting_originals(tmp_path):
    from forecast_ledger import append_file_record
    from capture_prematch import known_captures
    now = datetime.now(timezone.utc)
    target = raw_fixture(1,now+timedelta(days=2))
    calls = 0
    def interrupted(record,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError('Simulated interrupted package')
        return append_file_record(record,**kwargs)
    with patch('capture_prematch.predict',side_effect=forecast), \
         patch('forecast_ledger.append_file_record',side_effect=interrupted):
        first = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            api_request=Mock(return_value=[target]),evidence_loader=loader(now),max_seconds=0)
    original_files = {path:path.read_bytes() for path in (tmp_path/'forecasts').rglob('*.json')}
    assert len(original_files)==2 and first['diagnostics']['prediction_errors']
    assert not known_captures(read_file_records(root=tmp_path))
    with patch('capture_prematch.predict',side_effect=forecast):
        second = collect('test-only',codes=('PL',),ledger_root=tmp_path,
            api_request=Mock(side_effect=AssertionError('Committed discovery must be reused')),
            evidence_loader=loader(now),max_seconds=0)
    assert second['diagnostics']['captured_fixtures']==1
    assert len(read_file_records(root=tmp_path))==6
    assert len(known_captures(read_file_records(root=tmp_path)))==1
    assert all(path.read_bytes()==value for path,value in original_files.items())


def test_database_recovery_syncs_outbox_even_when_all_slots_are_already_captured(tmp_path):
    from forecast_ledger import open_ledger
    now = datetime.now(timezone.utc)
    target = raw_fixture(1,now+timedelta(days=2))
    with patch('capture_prematch.predict',side_effect=forecast):
        first = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            api_request=Mock(return_value=[target]),evidence_loader=loader(now),max_seconds=0)
    assert first['diagnostics']['captured_fixtures']==1
    central_path=tmp_path/'central.sqlite3'
    with patch('forecast_ledger.open_ledger',side_effect=lambda **kwargs:open_ledger(path=central_path)), \
         patch('capture_prematch.predict',side_effect=AssertionError('Complete slot must not refit')):
        recovered = collect('test-only',now,codes=('PL',),ledger_root=tmp_path,
            database_url='postgresql://test-only',
            api_request=Mock(side_effect=AssertionError('Discovery must be cached')),
            evidence_loader=loader(now),max_seconds=0)
    assert recovered['diagnostics']['already_captured']==1
    assert recovered['diagnostics']['captured_fixtures']==0
    assert recovered['diagnostics']['database_status']=='connected'
    assert recovered['diagnostics']['database_recovered_forecasts']==4
    with open_ledger(path=central_path) as central:
        assert len(central.iter_forecasts())==4
