from datetime import timedelta
import json
import numpy as np
from history_pipeline import build_evidence
import cross_competition_capture as capture
from test_cross_competition_strength import FIXTURE, NOW, P, cache


def source():
    sources = cache()
    meta = {"freshness_known": True, "stale": False, "fetched_at_utc": NOW.isoformat()}
    sources[("API-Football","2",2031)] = ([], dict(meta), None)
    sources[("API-Football","2",2030)] = ([], dict(meta), None)
    return sources


def test_freeze_binds_sources_and_preserves_original(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "predict", lambda *a, **k: {"matrix": P})
    evidence = build_evidence(FIXTURE, [], NOW)
    sources = source()
    assert capture.freeze(FIXTURE,evidence,sources,root=tmp_path,clock=lambda: NOW) == "frozen"
    path = tmp_path/"trials"/"9999.json"
    before = path.read_bytes()
    record = json.loads(before)
    assert record["research_only"] and record["external_requests"] == 0
    assert capture.digest({k:v for k,v in record.items() if k!="record_sha256"}) == record["record_sha256"]
    for sha in record["source_sha256"]:
        assert capture.digest(json.loads((tmp_path/"sources"/(sha+".json")).read_text())) == sha
    assert np.isclose(np.array(record["model"]["challenger"]).sum(),1)
    assert capture.freeze(FIXTURE,evidence,sources,root=tmp_path,clock=lambda: NOW) == "original_preserved"
    assert path.read_bytes() == before


def test_bad_digest_missing_source_and_late_clock_abstain(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "predict", lambda *a, **k: {"matrix": P})
    evidence = build_evidence(FIXTURE, [], NOW)
    assert capture.freeze(FIXTURE,{**evidence,"input_digest":"altered"},source(),root=tmp_path,clock=lambda: NOW) == "european_digest_mismatch"
    sources=source()
    sources.pop(("API-Football","2",2030))
    assert capture.freeze(FIXTURE,evidence,sources,root=tmp_path,clock=lambda: NOW) == "european_source_missing"
    times=iter([NOW, NOW+timedelta(hours=2)])
    assert capture.freeze(FIXTURE,evidence,source(),root=tmp_path,clock=lambda: next(times)) == "calculation_after_kickoff"
    assert not (tmp_path/"trials"/"9999.json").exists()


def test_evaluation_keeps_pending_distinct_and_rejects_modified_receipt(tmp_path, monkeypatch):
    import cross_competition_evaluate as grader
    monkeypatch.setattr(capture,"predict",lambda *a,**k:{"matrix":P})
    monkeypatch.setattr(grader,"predict",lambda *a,**k:{"matrix":P})
    monkeypatch.setattr(grader,"index_verified_results",lambda *a,**k:({},set(),{}))
    assert capture.freeze(FIXTURE,build_evidence(FIXTURE,[],NOW),source(),root=tmp_path,clock=lambda:NOW)=="frozen"
    report=grader.evaluate(root=tmp_path,now=NOW)
    assert report["frozen"]==report["pending"]==1
    assert report["verified"]==report["invalid"]==0
    path=tmp_path/"trials"/"9999.json"
    row=json.loads(path.read_text())
    row["model"]["home_tilt"]=999
    path.write_text(json.dumps(row))
    report=grader.evaluate(root=tmp_path,now=NOW)
    assert report["invalid"]==1 and report["pending"]==0
