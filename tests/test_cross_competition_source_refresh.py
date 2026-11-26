from datetime import datetime,timedelta,timezone
import pytest
from cross_competition_source_refresh import refresh, DOMESTIC

NOW=datetime(2031,10,1,10,tzinfo=timezone.utc)

def fixture(fid=101,code="CL",minutes=120):
    return {"id":fid,"provider":"API-Football","provider_status":"NS",
            "utcDate":(NOW+timedelta(minutes=minutes)).isoformat(),
            "competition":{"code":code,"id":2 if code=="CL" else 3,"season":2031},
            "homeTeam":{"id":1},"awayTeam":{"id":2}}

def harness(tmp_path,fixtures,*,limit=20,remaining=120,cache=None,available=True):
    cache={} if cache is None else cache
    calls=[]; frozen=[]
    def request(key,endpoint,params,**kwargs):
        calls.append((params["league"],params["season"]))
        return []
    def loader(fixture,as_of,api_key,api_request,cache):
        league=fixture["competition"]["id"]
        for year in (2031,2030):
            k=("API-Football",str(league),year)
            if k not in cache:
                api_request(api_key,"/fixtures",{"league":league,"season":year})
                cache[k]=([],{"freshness_known":True,"stale":False,
                              "fetched_at_utc":NOW.isoformat()},None)
        return {"history":[],"as_of_utc":as_of.isoformat()}
    def recorder(fixture,evidence,cache,**kwargs):
        frozen.append((fixture["id"],len(cache)))
        return "frozen"
    report=refresh(fixtures,cache,key="synthetic",request=request,clock=lambda:NOW,
                   remaining_calls=lambda:remaining-len(calls),
                   time_available=lambda:available,extra_limit=limit,root=tmp_path,
                   loader=loader,recorder=recorder,sparse_predict=lambda *a,**k:True)
    return report,calls,frozen,cache

def test_two_competitions_share_league_season_requests_and_use_only_spare_budget(tmp_path):
    r,calls,frozen,cache=harness(tmp_path,[fixture(),fixture(102,"EL"),fixture(103)])
    assert len(calls)==20==r["extra_adapter_calls"]
    assert len(set(calls))==20
    assert set(r["source_groups"])=={code+":2031" for code in DOMESTIC}
    assert all(n==2 for n in r["source_groups"].values())
    assert len(frozen)==3 and all(n==20 for _,n in frozen)
    assert r["outcomes"]=={"frozen":3}

@pytest.mark.parametrize("limit,remaining",[(0,100),(20,0),(20,1)])
def test_no_complete_source_group_fits_budget(tmp_path,limit,remaining):
    r,calls,frozen,_=harness(tmp_path,[fixture()],limit=limit,remaining=remaining)
    assert calls==frozen==[]
    assert r["outcomes"]=={"european_cache_or_budget_unavailable":1}

def test_cache_only_reuses_sources_and_excludes_past_or_duplicate_fixtures(tmp_path):
    _,_,_,cache=harness(tmp_path,[fixture()])
    r,calls,frozen,_=harness(tmp_path,[fixture(),fixture(),fixture(102,minutes=-5),
        fixture(103,minutes=3),fixture(104,minutes=3000)],limit=0,cache=cache)
    assert calls==[] and frozen==[(101,18)]
    assert r["eligible_fixtures"]==1
    assert r["extra_adapter_calls"]==0

def test_time_limit_prevents_research_calls(tmp_path):
    r,calls,frozen,_=harness(tmp_path,[fixture()],available=False)
    assert not calls and not frozen
    assert r["outcomes"]=={"time_or_kickoff_limit":1}

def test_extra_budget_never_exceeds_existing_remaining_calls(tmp_path):
    r,calls,frozen,_=harness(tmp_path,[fixture()],remaining=5)
    assert len(calls)==4 and len(calls)<=5
    assert r["extra_adapter_calls"]==4


def test_domestic_failed_cache_entry_retries_with_spare_budget(tmp_path):
    _,_,_,cache=harness(tmp_path,[fixture()])
    league=DOMESTIC["PL"]
    key=("API-Football",str(league),2031)
    cache[key]=([],{"freshness_known":False},"ProviderAccessError")
    r,calls,frozen,_=harness(tmp_path,[fixture()],limit=1,cache=cache)
    assert calls==[(league,2031)]
    assert r["domestic_retry_entries"]==1
    assert r["extra_adapter_calls"]==1
    assert r["source_group_states"]["PL:2031"]["2031"]=="qualified_empty"
    assert frozen==[(101,18)]


def test_light_run_does_not_retry_and_reports_failure_separately(tmp_path):
    _,_,_,cache=harness(tmp_path,[fixture()])
    league=DOMESTIC["PL"]
    cache[("API-Football",str(league),2031)]=([],{"freshness_known":False},"ProviderAccessError")
    r,calls,_,_=harness(tmp_path,[fixture()],limit=0,cache=cache)
    assert calls==[]
    assert r["domestic_retry_entries"]==0
    assert r["source_group_states"]["PL:2031"]=={"2031":"provider_error","2030":"qualified_empty"}
    assert r["source_groups"]["PL:2031"]==1


def test_stale_domestic_retry_respects_remaining_main_budget(tmp_path):
    _,_,_,cache=harness(tmp_path,[fixture()])
    league=DOMESTIC["PL"]
    for year in (2031,2030):
        cache[("API-Football",str(league),year)]=([],{"freshness_known":True,
            "stale":True,"fetched_at_utc":NOW.isoformat()},None)
    r,calls,_,_=harness(tmp_path,[fixture()],remaining=1,cache=cache)
    assert calls==[]
    assert r["domestic_retry_entries"]==0
    assert set(r["source_group_states"]["PL:2031"].values())=={"unqualified_freshness"}
