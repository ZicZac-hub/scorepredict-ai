"""Batch V37 settlement without new forecasts or extra request allowance."""
from datetime import timedelta
import json

import backfill_verified_results as backfill
from v37_sparse_evaluation import evaluate
from test_v37_independent_settlement import record, verified_final
from test_v37_sparse_history_research import NOW
from test_v31_slate_settlement import fixture_response

def configure(monkeypatch,tmp_path):
    root=tmp_path/"original"
    row=record(root)
    result_path=tmp_path/"prematch"/"verified_results.json"
    monkeypatch.setattr(backfill,"ROOT",root)
    monkeypatch.setattr(backfill,"RESULTS",result_path)
    monkeypatch.setattr(backfill,"STATUS",tmp_path/"status.json")
    monkeypatch.setattr(backfill,"enrolled_due",lambda *a: ({},{}))
    monkeypatch.setenv("API_FOOTBALL_KEY","test-key")
    return root,row,result_path

def test_v37_only_batched_then_graded(monkeypatch,tmp_path):
    root,row,path=configure(monkeypatch,tmp_path)
    originals={p:p.read_bytes() for p in root.rglob("*.json")}
    request,seen=fixture_response([row])
    now=NOW+timedelta(days=3)
    report=backfill.run(request=request,now=now)
    assert report["state"]=="complete"
    assert report["v37_settlement"]["selected"]==1
    assert report["requests"]==1
    assert report["new_verified"]==[str(row["fixture_id"])]
    assert all(p.read_bytes()==b for p,b in originals.items())
    assert evaluate(root=root,prematch=path.parent,now=now)["verified_matched_physical_fixtures"]==1
    assert backfill.run(request=request,now=now)["requests"]==0
    assert seen==[str(row["fixture_id"])]

def test_v37_not_due_or_already_saved_no_requests(monkeypatch,tmp_path):
    root,row,path=configure(monkeypatch,tmp_path)
    def forbidden(*a,**kw):
        raise AssertionError("unnecessary provider request")
    early=backfill.run(request=forbidden,now=NOW+timedelta(days=1))
    assert early["v37_settlement"]["not_yet_due"]==1
    assert early["requests"]==0
    (root/"verified_results.json").write_text(json.dumps({str(row["fixture_id"]):verified_final(row)}))
    late=backfill.run(request=forbidden,now=NOW+timedelta(days=3))
    assert late["v37_settlement"]["already_saved"]==1
    assert late["requests"]==0
    assert not path.exists()
