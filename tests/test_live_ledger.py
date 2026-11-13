from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import forecast_model
import history_store
import live_ledger
from forecast_ledger import open_ledger
from history_pipeline import build_evidence


def inputs():
    now = datetime.now(timezone.utc)
    fixture = {'id': 90001, 'provider': 'API-Football', 'utcDate': (now+timedelta(days=2)).isoformat(),
               'status': 'SCHEDULED', 'competition': {'id': 39, 'code': 'PL', 'season': 2026},
               'homeTeam': {'id': 1, 'name': 'A'}, 'awayTeam': {'id': 2, 'name': 'B'},
               'score': {'fullTime': {'home': None, 'away': None}}}
    history=[]
    for index in range(60):
        home,away = ((1,3),(4,2),(5,6),(2,4),(3,1),(6,5))[index % 6]
        history.append({'id':index+1,'provider':'API-Football',
            'competition':fixture['competition'], 'utcDate':(now-timedelta(days=index+2)).isoformat(),
            'status':'FINISHED','homeTeam':{'id':home},'awayTeam':{'id':away},
            'score':{'duration':'REGULAR','fullTime':{'home':2,'away':1}}})
    evidence=build_evidence(fixture,history,now)
    result=forecast_model.predict(fixture,evidence['history'],league_history=evidence['league_history'])
    rows=[{'fixture':fixture,'result':result,'evidence':evidence,'kickoff':fixture['utcDate'],'league':'PL'}]
    return rows, {'PL':history}


def test_first_capture_and_repeated_widget_render_preserve_one_complete_slot(tmp_path):
    conn=history_store.connect(':memory:')
    rows,histories=inputs()
    kwargs={'ledger_path':tmp_path/'ledger.sqlite3','file_root':tmp_path/'files'}
    status,report=live_ledger.maintain_archive(conn,rows,histories,**kwargs)
    assert status['captured_slots']==1
    assert report['common_frozen_slots']==1
    saved=deepcopy(history_store.load(conn))
    status,report=live_ledger.maintain_archive(conn,rows,histories,**kwargs)
    assert status['captured_slots']==0
    assert history_store.load(conn)==saved
    with open_ledger(database_url='',path=kwargs['ledger_path']) as store:
        assert len(store.iter_forecasts())==4
        assert all(row.get('slot_key') for row in store.iter_forecast_metadata())


def test_partial_or_ineligible_capture_does_not_block_a_complete_retry(tmp_path):
    rows,histories=inputs()
    records=live_ledger.capture_pair(rows[0]['fixture'],rows[0]['evidence'])
    with open_ledger(database_url='',path=tmp_path/'ledger.sqlite3') as store:
        store.put_forecasts(records[:1])
        assert live_ledger._slots(store.iter_forecast_metadata())==set()
        store.put_forecasts(records[1:])
        metadata=store.iter_forecast_metadata()
        assert len(live_ledger._slots(metadata))==1
        metadata[1]['eligible_for_comparison']=False
        assert live_ledger._slots(metadata)==set()


def test_failed_central_restore_defers_new_original_saves(tmp_path):
    conn=history_store.connect(':memory:')
    rows,histories=inputs()
    real=open_ledger(database_url='',path=tmp_path/'ledger.sqlite3')
    class FailedRestore:
        def __enter__(self):return self
        def __exit__(self,*args):real.close()
        def status(self):return {'backend':'postgresql','durable':True}
        def iter_archive_records(self,kind='single'):raise RuntimeError('do-not-expose-database-secret')
        def __getattr__(self,name):return getattr(real,name)
    with patch('live_ledger.open_ledger',return_value=FailedRestore()):
        status,_=live_ledger.maintain_archive(conn,rows,histories,database_url='configured',file_root=tmp_path/'files')
    assert history_store.load(conn)==[]
    assert status['deferred_original_saves']==1
    assert status['state']=='partial'
    assert 'do-not-expose' not in str(status)


def test_remote_original_is_restored_before_newer_card_is_saved(tmp_path):
    conn=history_store.connect(':memory:')
    rows,histories=inputs()
    original={'provider':'API-Football','fixture_id':90001,
              'saved_at':(datetime.now(timezone.utc)-timedelta(days=1)).isoformat(),
              'kickoff':rows[0]['kickoff'],'league':'PL','home':'A','away':'B',
              'home_id':'1','away_id':'2','predicted_score':'3-0','home_prob':.6,'draw_prob':.2,
              'away_prob':.2,'scores_json':'[[3,0,0.12]]','verification_status':'pending',
              'model_version':'original-model'}
    ledger_path=tmp_path/'ledger.sqlite3'
    with open_ledger(database_url='',path=ledger_path) as store:
        store.put_archive_record(original)
    status,_=live_ledger.maintain_archive(conn,rows,histories,ledger_path=ledger_path,file_root=tmp_path/'files')
    restored=history_store.load(conn)[0]
    assert restored['predicted_score']=='3-0'
    assert restored['saved_at']==original['saved_at']
    assert status['restored']['restored_singles']==1


def test_new_double_is_backed_up_without_another_app_render(tmp_path):
    from double_tracker import save
    from smart_double_ai import five_doubles
    conn=history_store.connect(':memory:')
    rows,_=inputs()
    first=rows[0]
    second=deepcopy(first)
    second['fixture'].update(id=90002,homeTeam={'id':3,'name':'C'},awayTeam={'id':4,'name':'D'})
    assert save(conn,first,second,five_doubles(first,second))
    path=tmp_path/'ledger.sqlite3'
    status=live_ledger.backup_after_changes(conn,ledger_path=path)
    assert status['uploaded_doubles']==1
    with open_ledger(database_url='',path=path) as store:
        assert len(store.iter_archive_records(kind='double'))==1
