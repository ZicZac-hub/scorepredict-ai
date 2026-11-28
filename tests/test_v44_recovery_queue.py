from datetime import datetime,timedelta,timezone
from pathlib import Path
import v44_recovery_queue as q

NOW=datetime(2026,1,1,tzinfo=timezone.utc)

def row(i,h):
    kickoff=NOW+timedelta(hours=h)
    f={"provider":"API-Football","id":i,"utcDate":kickoff.isoformat(),
       "homeTeam":{"id":i*2},"awayTeam":{"id":i*2+1}}
    return kickoff,f,"PL"

def test_recovery_reuses_existing_limit_and_preserves_normal_priority(tmp_path,monkeypatch):
    normal=[row(i,i) for i in (1,2,3,4)]
    recovery=row(5,5)
    key=q.fixture_key(recovery[1])
    shadow={(*key[:2],"v8_near"):{"fixture_id":"5"}}
    monkeypatch.setattr(q,"targets",lambda **kw:{key:key[2]})
    monkeypatch.setattr(q,"inspect_leg",lambda *a:"one_shot_near_without_exact_history")
    queue,keys=q.plan(normal+[recovery],normal,shadow,now=NOW,limit=4,root=tmp_path)
    assert keys=={key}
    assert [x[1]["id"] for x in queue[:4]]==[1,2,5,3]
    assert q.plan(normal+[recovery],normal,shadow,now=NOW,limit=2,root=tmp_path)==(normal,set())

def test_existing_recovery_and_usable_history_are_not_repeated(tmp_path,monkeypatch):
    a,b=row(1,1),row(2,2)
    ka,kb=q.fixture_key(a[1]),q.fixture_key(b[1])
    shadow={(*k[:2],"v8_near"):{"fixture_id":k[1]} for k in (ka,kb)}
    monkeypatch.setattr(q,"targets",lambda **kw:{ka:ka[2],kb:kb[2]})
    monkeypatch.setattr(q,"inspect_leg",lambda m,*a:"exact_history_valid_at_window_open" if m["fixture_id"]=="1" else "one_shot_near_expired")
    monkeypatch.setattr(q,"read_file_records",lambda **kw:[{"model_key":q.recovery.MODEL_KEY}] if kw["fixture_id"]=="2" else [])
    assert q.plan([a,b],[a,b],shadow,now=NOW,limit=3,root=tmp_path)==([a,b],set())

def test_recovery_already_in_normal_queue_is_not_duplicated(tmp_path,monkeypatch):
    normal=[row(i,i) for i in (1,2,3,4)]
    key=q.fixture_key(normal[3][1])
    monkeypatch.setattr(q,"targets",lambda **kw:{key:key[2]})
    monkeypatch.setattr(q,"inspect_leg",lambda *a:"one_shot_near_expired")
    queue,keys=q.plan(normal,normal,{(*key[:2],"v8_near"):{}},now=NOW,limit=4,root=tmp_path)
    assert keys=={key}
    assert [r[1]["id"] for r in queue]==[1,2,4,3]

def test_workflow_recovery_defaults_on_only_for_heavy_runs_and_checkpoint_is_append_only():
    source=Path("capture_prematch.py").read_text()
    workflow=Path(".github/workflows/prematch-evidence.yml").read_text()
    assert 'if os.environ.get("V44_RECOVERY_CAPTURE_ENABLED")=="true":' in source
    assert "for kickoff,fixture,code in awaiting[:research_market_refresh_limit]:" in source
    assert "recovery_refresher(fixture,packet,_time(current_time()))" in source
    assert "steps.cadence.outputs.heavy == 'true' && vars.V44_RECOVERY_CAPTURE_ENABLED != 'false'" in workflow
    assert "V10_MARKET_REFRESH_LIMIT: ${{ steps.cadence.outputs.heavy == 'true' && '8' || '0' }}" in workflow
    assert "--diff-filter=DMRTUXB -- data/v44_recovery_sources/" in workflow
    assert "data/v44_recovery_sources/archive/forecasts" in workflow
    assert "data/v44_recovery_sources/history/receipts" in workflow


def test_protected_normal_slots_are_not_converted_to_recovery(tmp_path,monkeypatch):
    normal=[row(i,i) for i in (1,2,3,4)]
    a,b=q.fixture_key(normal[0][1]),q.fixture_key(normal[3][1])
    shadow={(*k[:2],"v8_near"):{"fixture_id":k[1]} for k in (a,b)}
    monkeypatch.setattr(q,"targets",lambda **kw:{a:a[2],b:b[2]})
    monkeypatch.setattr(q,"inspect_leg",lambda *a:"one_shot_near_without_exact_history")
    queue,keys=q.plan(normal,normal,shadow,now=NOW,limit=4,root=tmp_path)
    assert keys=={b}
    assert queue[:2]==normal[:2]
    monkeypatch.setattr(q,"targets",lambda **kw:{a:a[2]})
    assert q.plan(normal,normal,shadow,now=NOW,limit=4,root=tmp_path)==(normal,set())
