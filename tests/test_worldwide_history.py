"""Worldwide modelling uses the same complete evidence policy as other paths."""
from datetime import datetime, timezone
from unittest.mock import patch

from api_football import _convert
from history_pipeline import build_evidence, POLICY_VERSION
from worldwide_model import model_worldwide


class ObservedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 10, tzinfo=timezone.utc)


def raw_match(identifier, home=1, away=3, *, league=39, season=2025,
              day='2025-12-01T12:00:00Z', status='FT'):
    return {'fixture':{'id':identifier,'date':day,'status':{'short':status}},
            'teams':{'home':{'id':home,'name':f'Team {home}'},
                     'away':{'id':away,'name':f'Team {away}'}},
            'league':{'id':league,'season':season,'name':'Test league'},
            'goals':{'home':2,'away':1},'score':{'fulltime':{'home':2,'away':1}}}


def event():
    return {'league':'Premier League','league_id':39,'season':2025,
            'fixture':{'id':1000,'utcDate':'2026-01-20T12:00:00Z',
                       'provider':'API-Football','competition':{'id':39,'code':'PL'},
                       'homeTeam':{'id':1,'name':'Team 1'},
                       'awayTeam':{'id':2,'name':'Team 2'}}}


def population(season=2025):
    return ([raw_match(i,1,3,season=season) for i in range(1,11)]
            +[raw_match(i,4,2,season=season) for i in range(11,21)]
            +[raw_match(21,5,6,season=season)])


def run_model(request, match_event=None, variant='baseline'):
    with patch('worldwide_model.datetime',ObservedDatetime), \
         patch('worldwide_model._request',side_effect=request) as requests, \
         patch('worldwide_model.predict',return_value={'model_version':'test'}) as prediction:
        row,reason=model_worldwide('test-key',match_event or event(),model_variant=variant)
    return row,reason,prediction,requests


def test_complete_population_and_two_seasons_are_always_requested():
    all_rows=population()+[
        raw_match(30,5,6,league=140),raw_match(31,5,6,season=2024),
        raw_match(32,5,6,day='2026-01-15T12:00:00Z'),
        raw_match(33,5,6,day='2026-01-20T12:00:00Z'),raw_match(34,5,6,status='NS')]
    def request(_key,endpoint,params,with_metadata=False):
        return all_rows if endpoint=='/fixtures' and params=={'league':39,'season':2025} else []
    row,reason,prediction,requests=run_model(request)
    assert reason is None
    league=prediction.call_args.kwargs['league_history']
    assert {m['id'] for m in league}==set(range(1,22))
    assert any(m['homeTeam']['id']==5 and m['awayTeam']['id']==6 for m in league)
    assert row['history_counts']=={'home':10,'away':10,'league':21}
    assert [c.args[2] for c in requests.call_args_list if c.args[1]=='/fixtures']==[
        {'league':39,'season':2025},{'league':39,'season':2024}]


def test_sparse_current_season_includes_previous_complete_season_without_team_fallback():
    def request(_key,endpoint,params,with_metadata=False):
        if endpoint!='/fixtures':return []
        if params=={'league':39,'season':2025}:return [raw_match(100,1,2)]
        if params=={'league':39,'season':2024}:return population(2024)
        raise AssertionError('No team-only history endpoint may be used')
    row,reason,prediction,requests=run_model(request)
    assert reason is None
    assert len(prediction.call_args.kwargs['league_history'])==22
    assert row['history_counts']['home']==11


def test_unavailable_league_history_abstains_instead_of_borrowing_selected_team_games():
    def request(_key,endpoint,params,with_metadata=False):
        if endpoint=='/fixtures' and 'league' in params:raise RuntimeError('test-key')
        raise AssertionError('No fallback call should be attempted')
    row,reason,prediction,requests=run_model(request)
    assert row is None
    assert 'home team 0, away team 0' in reason
    assert 'test-key' not in reason
    prediction.assert_not_called()
    assert len(requests.call_args_list)==2


def test_cup_only_history_cannot_support_a_league_forecast():
    def request(_key,endpoint,params,with_metadata=False):
        return [raw_match(i,1 if i<6 else 2,8,league=2) for i in range(1,11)]
    row,reason,prediction,_=run_model(request)
    assert row is None
    assert 'home team 0, away team 0' in reason
    prediction.assert_not_called()


def test_fixture_metadata_can_supply_league_and_season_without_event_duplicates():
    value=event();del value['league_id'];del value['season']
    value['fixture']['competition']['season']=2025
    def request(_key,endpoint,params,with_metadata=False):
        return population() if endpoint=='/fixtures' and params=={'league':39,'season':2025} else []
    row,reason,prediction,_=run_model(request,value)
    assert reason is None
    assert row['fixture']['competition']['season']==2025
    assert len(prediction.call_args.kwargs['league_history'])==21


def test_real_engine_receives_complete_population_and_returns_shared_evidence():
    def request(_key,endpoint,params,with_metadata=False):
        return population() if endpoint=='/fixtures' and params=={'league':39,'season':2025} else []
    with patch('worldwide_model.datetime',ObservedDatetime),patch('worldwide_model._request',side_effect=request):
        row,reason=model_worldwide('test-key',event())
    assert reason is None
    assert row['result']['league_history_count']==21
    assert row['result']['history_count']==21
    assert abs(float(row['result']['matrix'].sum())-1)<1e-10
    assert row['evidence']['policy_version']==POLICY_VERSION
    expected=build_evidence(row['fixture'],[_convert(raw,'PL') for raw in population()],ObservedDatetime.now())
    assert row['evidence']['input_digest']==expected['input_digest']


def test_results_with_prior_kickoff_but_unavailable_final_score_are_excluded():
    late=raw_match(100,5,6,day='2026-01-09T22:30:00Z')
    explicit=raw_match(101,5,6,day='2026-01-09T22:30:00Z')
    explicit['finishedAt']='2026-01-09T23:59:00Z'
    def request(_key,endpoint,params,with_metadata=False):
        return population()+[late,explicit] if endpoint=='/fixtures' and params=={'league':39,'season':2025} else []
    row,reason,prediction,_=run_model(request)
    assert reason is None
    ids={m['id'] for m in prediction.call_args.kwargs['league_history']}
    assert 100 not in ids
    assert 101 in ids


def test_selected_model_variant_is_forwarded_on_identical_evidence():
    def request(_key,endpoint,params,with_metadata=False):
        return population() if endpoint=='/fixtures' and params=={'league':39,'season':2025} else []
    baseline,_,first,_=run_model(request)
    research,_,second,_=run_model(request,variant='reference_v7')
    assert first.call_args.kwargs['variant']=='baseline'
    assert second.call_args.kwargs['variant']=='reference_v7'
    assert baseline['evidence']['input_digest']==research['evidence']['input_digest']
