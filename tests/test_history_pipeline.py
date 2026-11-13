"""Shared history policy, provider freshness, and cross-path provenance checks."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import api_football
from history_pipeline import build_evidence, load_evidence, POLICY_VERSION

AS_OF = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)


def fixture(provider='API-Football'):
    return {'id':900,'provider':provider,'utcDate':'2026-11-15T12:00:00Z',
            'competition':{'id':39 if provider=='API-Football' else 2021,'code':'PL','season':2026},
            'homeTeam':{'id':1,'name':'A'},'awayTeam':{'id':2,'name':'B'}}


def match(idx=1, when='2026-11-01T12:00:00Z', provider='API-Football', season=2026):
    return {'id':idx,'provider':provider,'utcDate':when,'status':'FINISHED',
            'competition':{'id':39 if provider=='API-Football' else 2021,'code':'PL','season':season},
            'homeTeam':{'id':1,'name':'A'},'awayTeam':{'id':3,'name':'C'},
            'score':{'duration':'REGULAR','fullTime':{'home':2,'away':1}}}


def raw_api(m):
    return {'fixture':{'id':m['id'],'date':m['utcDate'],'status':{'short':'FT'}},
            'league':{'id':m['competition']['id'],'season':m['competition']['season']},
            'teams':{'home':m['homeTeam'],'away':m['awayTeam']},
            'goals':m['score']['fullTime'],'score':{'fulltime':m['score']['fullTime']}}


def test_input_digest_and_rows_are_order_independent_and_numerical_only():
    rows=[match(1),match(2,'2026-11-02T12:00:00Z')]
    first=build_evidence(fixture(), rows, AS_OF)
    richer=deepcopy(rows);richer[0]['homeTeam']['name']='Renamed display label'
    second=build_evidence(fixture(), list(reversed(richer)), AS_OF,
                          {'fetched_at_utc':AS_OF.isoformat(),'cache_status':'network','api_key':'never-store'})
    assert first['input_digest']==second['input_digest']
    assert first['policy_version']==POLICY_VERSION
    assert 'never-store' not in json.dumps(second)
    assert [m['id'] for m in first['history']]==[2,1]


def test_target_actual_score_never_changes_input_digest():
    target=fixture();target['score']={'fullTime':{'home':9,'away':0}}
    first=build_evidence(target,[match(),dict(target,status='FINISHED')],AS_OF)
    target['score']['fullTime']={'home':0,'away':9}
    second=build_evidence(target,[match(),dict(target,status='FINISHED')],AS_OF)
    assert first['input_digest']==second['input_digest']
    assert first['exclusions']['target_fixture']==1


def test_results_require_final_whistle_availability_not_only_prior_kickoff():
    recent=match(1,(AS_OF-timedelta(hours=2)).isoformat())
    explicit=match(2,(AS_OF-timedelta(hours=2)).isoformat())
    explicit['finishedAt']=(AS_OF-timedelta(minutes=10)).isoformat()
    later=match(3,(AS_OF-timedelta(hours=4)).isoformat())
    later['finishedAt']=(AS_OF+timedelta(minutes=1)).isoformat()
    report=build_evidence(fixture(),[recent,explicit,later],AS_OF)
    assert [m['id'] for m in report['history']]==[2]
    assert report['exclusions']['not_available_as_of']==2


def test_current_and_previous_seasons_only_with_730_day_limit():
    current=match(1)
    previous=match(2,'2025-10-01T12:00:00Z',season=2025)
    old_season=match(3,'2024-12-01T12:00:00Z',season=2024)
    too_old=match(4,'2024-10-01T12:00:00Z',season=2025)
    result=build_evidence(fixture(),[current,previous,old_season,too_old],AS_OF)
    assert {m['id'] for m in result['history']}=={1,2}
    assert result['provenance']['seasons']==[2026,2025]
    assert result['history_window_start_utc']==(AS_OF-timedelta(days=730)).isoformat()
    assert result['exclusions']['outside_seasons']==1
    assert result['exclusions']['outside_window']==1


def test_provider_competition_and_regulation_scores_are_separate():
    wrong_provider=match(2,provider='football-data.org')
    wrong_comp=match(3);wrong_comp['competition']['id']=78
    extra=match(4);extra['score']['duration']='EXTRA_TIME'
    missing=match(5);missing['score']['fullTime']['home']=None
    boolean=match(6);boolean['score']['fullTime']['home']=True
    scheduled=match(7);scheduled['status']='SCHEDULED'
    result=build_evidence(fixture(),[match(),wrong_provider,wrong_comp,extra,missing,boolean,scheduled],AS_OF)
    assert [m['id'] for m in result['history']]==[1]
    assert result['exclusions']['different_provider']==1
    assert result['exclusions']['different_competition']==1
    assert result['exclusions']['non_regulation']==1
    assert result['exclusions']['invalid_score']==2
    assert result['exclusions']['unfinished']==1


def test_conflicting_duplicate_results_are_excluded_without_order_dependence():
    first=match();conflict=deepcopy(first);conflict['score']['fullTime']['home']=3
    a=build_evidence(fixture(),[first,conflict,match(2)],AS_OF)
    b=build_evidence(fixture(),[match(2),conflict,first],AS_OF)
    assert a['input_digest']==b['input_digest']
    assert [m['id'] for m in a['history']]==[2]
    assert a['exclusions']['conflicting_duplicate']==2


def test_same_result_duplicates_are_deduplicated_and_ids_normalized():
    m=match();duplicate=deepcopy(m);duplicate['id']='1';duplicate['homeTeam']['id']='1'
    result=build_evidence(fixture(),[m,duplicate],AS_OF)
    assert len(result['history'])==1
    assert result['exclusions']['duplicate']==1


def test_load_fetches_two_complete_seasons_and_shared_cache_reuses_requests():
    calls=[]
    def request(key,endpoint,params,with_metadata=False):
        calls.append((endpoint,dict(params)))
        rows=[raw_api(match())] if params['season']==2026 else [raw_api(match(2,'2025-10-01T12:00:00Z',season=2025))]
        return {'data':rows,'metadata':{'fetched_at_utc':AS_OF.isoformat(),'age_seconds':0,
                                      'cache_status':'network','freshness_known':True,'stale':False}}
    cache={}
    a=load_evidence(fixture(),AS_OF,api_request=request,cache=cache)
    b=load_evidence(fixture(),AS_OF,api_request=request,cache=cache)
    assert calls==[('/fixtures',{'league':39,'season':2026}),('/fixtures',{'league':39,'season':2025})]
    assert a['input_digest']==b['input_digest']
    assert {m['id'] for m in a['history']}=={1,2}
    assert len(a['provenance']['sources'])==2
    assert not a['warnings']


def test_legacy_callback_payloads_remain_supported_with_unknown_freshness():
    request=lambda key,endpoint,params:[raw_api(match())] if params['season']==2026 else []
    result=load_evidence(fixture(),AS_OF,api_request=request)
    assert len(result['history'])==1
    assert result['provenance']['sources'][0]['freshness_known'] is False
    assert any('freshness is unknown' in warning for warning in result['warnings'])


def test_primary_and_api_loaders_apply_the_same_policy():
    primary=fixture('football-data.org')
    source=[match(provider='football-data.org'),match(2,'2025-10-01T12:00:00Z',provider='football-data.org',season=2025)]
    calls=[]
    def get(path,with_metadata=False):
        calls.append(path)
        year=int(path.split('season=')[1].split('&')[0])
        return {'data':{'matches':[m for m in source if m['competition']['season']==year]},
                'metadata':{'fetched_at_utc':AS_OF.isoformat(),'stale':False,'freshness_known':True}}
    loaded=load_evidence(primary,AS_OF,primary_get=get)
    expected=build_evidence(primary,source,AS_OF)
    assert loaded['input_digest']==expected['input_digest']
    assert calls==['/competitions/PL/matches?season=2026&status=FINISHED','/competitions/PL/matches?season=2025&status=FINISHED']


def test_provider_errors_are_reported_without_credentials():
    def bad(*args,**kwargs):raise RuntimeError('secret-db-and-api-key')
    result=load_evidence(fixture(),AS_OF,api_request=bad,api_key='secret-db-and-api-key')
    assert not result['history']
    assert 'secret-db-and-api-key' not in json.dumps(result)
    assert sum('history unavailable' in s for s in result['warnings'])==2


def test_stale_source_is_labelled_without_affecting_feature_digest():
    raw={'sources':[{'stale':True,'fetched_at_utc':'2026-11-12T12:00:00Z','age_seconds':86400,
                     'cache_status':'stale_fallback','freshness_known':True,'key':'secret'}]}
    stale=build_evidence(fixture(),[match()],AS_OF,raw)
    fresh=build_evidence(fixture(),[match()],AS_OF)
    assert stale['input_digest']==fresh['input_digest']
    assert any('stale fallback' in warning for warning in stale['warnings'])
    assert 'secret' not in json.dumps(stale)


def test_endpoint_ttls_follow_provider_update_lifecycles():
    assert api_football.cache_ttl('/fixtures',{'id':1})==60
    assert api_football.cache_ttl('/fixtures/lineups',{'fixture':1})==900
    assert api_football.cache_ttl('/odds',{})==10800
    assert api_football.cache_ttl('/fixtures',{'league':39,'season':2026})==900


def test_api_cache_exposes_original_fetch_time_and_age(tmp_path):
    response=Mock(status_code=200);response.json.return_value={'response':[{'value':1}]}
    with patch.object(api_football,'CACHE',tmp_path), patch.object(api_football.requests,'get',return_value=response) as get, patch.object(api_football.time,'time',return_value=1000):
        first=api_football._request('key','/fixtures',{'id':1},with_metadata=True)
    with patch.object(api_football,'CACHE',tmp_path), patch.object(api_football.requests,'get') as get, patch.object(api_football.time,'time',return_value=1010):
        cached=api_football._request('key','/fixtures',{'id':1},with_metadata=True)
        get.assert_not_called()
    assert first['metadata']['cache_status']=='network'
    assert cached['metadata']['cache_status']=='fresh_cache'
    assert cached['metadata']['age_seconds']==10
    assert cached['metadata']['fetched_at_utc']==first['metadata']['fetched_at_utc']


def test_api_stale_fallback_is_bounded_and_labelled(tmp_path):
    response=Mock(status_code=200);response.json.return_value={'response':[{'value':1}]}
    with patch.object(api_football,'CACHE',tmp_path), patch.object(api_football.requests,'get',return_value=response), patch.object(api_football.time,'time',return_value=1000):
        api_football._request('key','/fixtures',{'id':1})
    with patch.object(api_football,'CACHE',tmp_path), patch.object(api_football.requests,'get',side_effect=api_football.requests.ConnectionError('key')), patch.object(api_football.time,'time',return_value=1120):
        stale=api_football._request('key','/fixtures',{'id':1},with_metadata=True)
    assert stale['metadata']['stale'] is True
    assert stale['metadata']['age_seconds']==120
    with patch.object(api_football,'CACHE',tmp_path), patch.object(api_football.requests,'get',side_effect=api_football.requests.ConnectionError('key')), patch.object(api_football.time,'time',return_value=1400):
        with pytest.raises(RuntimeError,match='ConnectionError'):
            api_football._request('key','/fixtures',{'id':1},with_metadata=True)
