"""Cache recovery tests use synthetic history only; no provider calls."""
from datetime import datetime,timedelta,timezone
from pathlib import Path

import v44_cache_only_recovery as c

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
PRIOR=NOW-timedelta(hours=1)

def fixture(i=1):
    return {"provider":"API-Football","id":i,"utcDate":(NOW+timedelta(hours=2)).isoformat(),
            "homeTeam":{"id":10*i,"name":"Home"+str(i)},
            "awayTeam":{"id":10*i+1,"name":"Away"+str(i)},
            "competition":{"id":39,"season":2024,"code":"PL"}}

def cache():
    f=fixture()
    past={**f,"id":99,"utcDate":"2025-01-20T12:00:00Z","status":"FINISHED",
          "score":{"fullTime":{"home":2,"away":1},"duration":"REGULAR"}}
    result={}
    for season in (2024,2023):
        result[("API-Football","39",season)]=([past] if season==2024 else [],{
            "provider":"API-Football","season":season,"freshness_known":True,
            "fetched_at_utc":(NOW-timedelta(minutes=10)).isoformat(),
            "retrieved_at_utc":(NOW-timedelta(minutes=5)).isoformat(),
            "age_seconds":300,"ttl_seconds":3600,"stale":False,
            "source_season_mismatches":0},None)
    return result

def near(f):
    return {"provider":f["provider"],"fixture_id":str(f["id"]),
            "home_team_id":str(f["homeTeam"]["id"]),"away_team_id":str(f["awayTeam"]["id"]),
            "kickoff_utc":f["utcDate"],"captured_at_utc":PRIOR.isoformat(),"model_key":"v8_near"}

def test_complete_cache_builds_exact_evidence_without_mutation():
    history=cache()
    before=repr(history)
    packet,reason=c.packet_from_cache(fixture(),history,NOW,PRIOR)
    assert reason=="ready" and len(packet["history"])==1
    assert packet["as_of_utc"]==NOW.isoformat()
    assert len(packet["provenance"]["sources"])==2
    assert repr(history)==before

def test_missing_failed_stale_or_old_cache_never_qualifies():
    history=cache()
    del history[("API-Football","39",2023)]
    assert c.packet_from_cache(fixture(),history,NOW,PRIOR)==(None,"season_not_in_run_cache")
    history=cache()
    rows,meta,_=history[("API-Football","39",2023)]
    history[("API-Football","39",2023)]=(rows,meta,"SourceError")
    assert c.packet_from_cache(fixture(),history,NOW,PRIOR)==(None,"unusable_cached_season")
    history=cache();history[("API-Football","39",2024)][1]["stale"]=True
    assert c.packet_from_cache(fixture(),history,NOW,PRIOR)==(None,"cache_not_fully_adapter_fresh")
    assert c.packet_from_cache(fixture(),cache(),NOW,NOW-timedelta(minutes=5))==(None,"current_season_fetch_not_newer_than_prior")

def test_one_attempt_only_and_prior_recovered_skipped(tmp_path,monkeypatch):
    fs=[fixture(i) for i in (1,2,3)]
    rows=[(c.utc(f["utcDate"]),f,"PL") for f in fs]
    keys={c.fixture_key(f):c.utc(f["utcDate"]) for f in fs}
    monkeypatch.setattr(c,"targets",lambda **kw:keys)
    monkeypatch.setattr(c,"inspect_leg",lambda *a:"one_shot_near_without_exact_history")
    originals={str(f["id"]):near(f) for f in fs}
    def read(**kwargs):
        if kwargs["root"]==tmp_path/"archive":
            return [{"model_key":c.recovery.MODEL_KEY}] if kwargs["fixture_id"]=="1" else []
        return [originals[kwargs["fixture_id"]]]
    monkeypatch.setattr(c,"read_file_records",read)
    called=[]
    def refresh(f,e,observed):
        called.append(f["id"])
        return {"status":"recovery_written"}
    r=c.run(rows,cache(),clock=lambda:NOW,time_available=lambda:True,
            refresher=refresh,root=tmp_path)
    assert called==[2] and r["attempted"]==1
    assert r["reasons"]["already_recovered"]==1
    assert r["reasons"]["recovery_written"]==1 and r["external_requests"]==0

def test_empty_cache_and_time_budget_do_not_attempt(tmp_path):
    def unexpected(*args,**kwargs):
        raise AssertionError("Unexpected recovery attempt")
    r=c.run([],{},clock=unexpected,time_available=lambda:True,refresher=unexpected,root=tmp_path)
    assert r["attempted"]==0 and r["reasons"]=={"no_in_run_history_cache":1}
    r=c.run([],cache(),clock=unexpected,time_available=lambda:False,refresher=unexpected,root=tmp_path)
    assert r["attempted"]==0 and r["reasons"]=={"time_budget":1}
    assert not list(tmp_path.iterdir())

def test_workflow_uses_separate_default_and_existing_checkpoint():
    source=Path("capture_prematch.py").read_text()
    workflow=Path(".github/workflows/prematch-evidence.yml").read_text()
    assert 'if os.environ.get("V44_CACHE_RECOVERY_ENABLED")=="true":' in source
    assert "fixtures.values(),history_cache,clock=current_time," in source
    assert "refresher=cache_recovery_refresher" in source
    assert "vars.V44_CACHE_RECOVERY_ENABLED != 'false'" in workflow
    assert "steps.cadence.outputs.heavy == 'true' && vars.V44_RECOVERY_CAPTURE_ENABLED != 'false'" in workflow
    assert "V10_MARKET_REFRESH_LIMIT: ${{ steps.cadence.outputs.heavy == 'true' && '8' || '0' }}" in workflow
    assert "--diff-filter=DMRTUXB -- data/v44_recovery_sources/" in workflow
