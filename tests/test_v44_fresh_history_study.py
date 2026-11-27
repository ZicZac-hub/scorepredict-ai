from datetime import datetime,timedelta,timezone
import json
import numpy as np
import pytest
import v44_fresh_history_study as s
from forecast_ledger import _forecast

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
def model(i="1",captured=NOW,kind="v8_near"):
    return _forecast({"provider":"API-Football","fixture_id":i,"home_team_id":i+"1","away_team_id":i+"2",
        "home_team":"H"+i,"away_team":"A"+i,"kickoff_utc":(NOW+timedelta(hours=2)).isoformat(),
        "captured_at_utc":captured.isoformat(),"evidence_as_of_utc":(captured-timedelta(minutes=1)).isoformat(),
        "input_digest":"digest-"+captured.isoformat(),"model_key":kind,"model_version":"test-"+kind,
        "horizon":"near","score_basis":"regulation","score_matrix":[[.04]*5 for _ in range(5)]})

def save(root,m):
    p=root/"forecasts"/m["provider"]/m["fixture_id"]/(m["forecast_id"]+".json")
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m))

def receipt(root,m):
    key=s.early.receipt_key(list(s.event(m)),m["input_digest"])
    p=root/"receipts"/(key+".json");p.parent.mkdir(parents=True,exist_ok=True)
    row={"event":list(s.event(m)),"history_input_digest":m["input_digest"],
         "as_of_utc":m["evidence_as_of_utc"],"captured_at_utc":m["evidence_as_of_utc"],
         "fixture":{},"history":[],"receipt_sha256":key}
    p.write_text(json.dumps(row));return p

def test_fresh_capture_does_not_repair_original_and_excludes_future(tmp_path,monkeypatch):
    source=tmp_path/"source";history=tmp_path/"history"
    original=model(captured=NOW-timedelta(hours=4),kind="v8")
    fresh=model(captured=NOW-timedelta(minutes=1))
    future=model(captured=NOW+timedelta(seconds=1))
    for m in (original,fresh,future):save(source,m)
    for m in (fresh,future):receipt(history,m)
    monkeypatch.setattr(s.early,"load",lambda path,**kw:json.loads(path.read_text()))
    chosen,r,key=s.source_for(original,NOW,source,history)
    assert chosen["forecast_id"]==fresh["forecast_id"]
    assert r["history_input_digest"]!=original["input_digest"]
    assert not (history/"receipts"/(s.early.receipt_key(list(s.event(original)),original["input_digest"])+".json")).exists()

def test_missing_history_abstains(tmp_path):
    m=model();save(tmp_path,m)
    with pytest.raises(ValueError,match="missing_fresh_exact_history"):
        s.source_for(m,NOW,tmp_path,tmp_path/"history")

def test_unregistered_read_only_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(s,"_result_index",lambda *a:({},set(),{}))
    r=s.evaluate(root=tmp_path/"study",now=NOW)
    assert not r["registered"]
    assert r["comparisons"]["fresh_blend"]["hit_rate"] is None
    assert not (tmp_path/"study").exists()

def test_matched_arms_normalized(monkeypatch):
    models=[model("1"),model("2")]
    monkeypatch.setattr(s,"source_for",lambda m,*a:(m,{"fixture":{},"history":[],"as_of_utc":m["evidence_as_of_utc"],"receipt_sha256":"hash"},"key"))
    monkeypatch.setattr(s.ratings,"predict",lambda *a:{"status":"ready","grid":np.full((5,5),.04)})
    refs,arms=s.calculate(models,NOW,None,None)
    assert len(refs)==2
    for a in arms.values():
        assert len(a["five_doubles"])==5
        assert all(np.isclose(np.sum(g),1.) for g in a["full_score_grids"])


def test_readiness_checks_future_without_issuing_or_registering(tmp_path,monkeypatch):
    pairs=tmp_path/"pairs";(pairs/"sets").mkdir(parents=True)
    (pairs/"sets"/"2025-02-01.json").write_text("{}")
    models=[model("1"),model("2")]
    for m in models:
        m["kickoff_utc"]=(NOW+timedelta(hours=5)).isoformat()
    monkeypatch.setattr(s.sources,"_pair",lambda *a,**k:({"predicted_at_utc":NOW.isoformat()},models))
    def get(m,*a):
        if m["fixture_id"]=="2":
            raise ValueError("missing_fresh_exact_history")
        return m,{"fixture":{},"history":[],"as_of_utc":NOW.isoformat()},"key"
    monkeypatch.setattr(s,"source_for",get)
    monkeypatch.setattr(s.ratings,"predict",lambda *a:{"status":"ready"})
    result=s.readiness(root=tmp_path/"study",pair_root=pairs,now=NOW)
    assert result["counts"]["future_pairs_with_1_ready_legs"]==1
    assert result["leg_statuses"]["future:missing_fresh_exact_history"]==1
    assert not (tmp_path/"study").exists()


def test_scheduler_preserves_isolated_protocol_and_trials():
    from pathlib import Path
    text=Path(".github/workflows/v42-research-watchdog.yml").read_text()
    step=text.split("      - name: Register and verify isolated fresh-history research",1)[1].split("      - name: Run separate late-cutoff trial",1)[0]
    assert "vars.V44_FRESH_HISTORY_ENABLED != 'false'" in step
    assert 'OPENAI_API_KEY: ""' in step
    assert 'API_FOOTBALL_KEY: ""' in step
    assert "python v44_fresh_history_study.py --freeze" in step
    assert 'run["evaluation"]["counts"].get("rejected")' in step
    assert '("readiness",run["readiness"])' in step
    assert "github.ref == 'refs/heads/main'" in text
    assert 'if [ "$V44_FRESH_OUTCOME" = "success" ]; then' in text
    assert "--diff-filter=DMRTUXB -- data/v44_fresh_history_study/protocol.json data/v44_fresh_history_study/trials/" in text


def test_full_archive_to_verified_settlement_without_backfill(tmp_path,monkeypatch):
    from history_pipeline import build_evidence
    from forecast_ledger import append_file_record
    import professional_double_shadow as shadow

    source=tmp_path/"source";history=tmp_path/"history";pairs=tmp_path/"pairs";root=tmp_path/"study"
    original_time=NOW-timedelta(hours=2)
    fixtures=[]
    for i in (1,2):
        fixture={"provider":"API-Football","id":i,"utcDate":(NOW+timedelta(hours=2)).isoformat(),
                 "homeTeam":{"id":i*2,"name":"Home "+str(i)},
                 "awayTeam":{"id":i*2+1,"name":"Away "+str(i)},
                 "competition":{"id":39,"season":2024}}
        fixtures.append(fixture)
        original={"provider":fixture["provider"],"fixture_id":str(i),
                  "home_team_id":str(i*2),"away_team_id":str(i*2+1),
                  "home_team":"Home "+str(i),"away_team":"Away "+str(i),
                  "fixture_name":"Home "+str(i)+" vs Away "+str(i),
                  "competition":"PL","model_key":"v8","model_version":"original-test-v8",
                  "input_digest":"original-"+str(i),"horizon":"early","score_basis":"regulation",
                  "kickoff_utc":fixture["utcDate"],
                  "captured_at_utc":(original_time-timedelta(minutes=1)).isoformat(),
                  "evidence_as_of_utc":(original_time-timedelta(minutes=2)).isoformat(),
                  "score_matrix":np.outer([.4,.3,.2,.075,.025],[.4,.3,.2,.075,.025]).tolist()}
        # The real archive reader requires top10_scores; derive them from the
        # full matrix instead of inventing unrelated score probabilities.
        original["top10_scores"]=sorted(
            [[h,a,float(prob) ] for h,line in enumerate(original["score_matrix"])
             for a,prob in enumerate(line)],key=lambda row:(-row[2],row[0],row[1]))[:10]
        assert append_file_record(original,root=source)
    assert len(shadow.archive_rows(now=original_time,root=source))==2
    status=shadow.freeze(root=pairs,source_root=source,now=original_time,clock=lambda:original_time)
    assert len(status["new_sets"])==1,status
    original_path=next((pairs/"sets").glob("*.json"))
    original_bytes=original_path.read_bytes()
    original_pair,original_models=s.sources._pair(original_path,source_root=source,decision=NOW)
    cutoff=NOW-timedelta(minutes=5)
    for f in fixtures:
        past={**f,"id":100+f["id"],"utcDate":"2025-01-20T12:00:00Z","status":"FINISHED",
              "score":{"fullTime":{"home":2,"away":1},"duration":"REGULAR"}}
        evidence=build_evidence(f,[past],cutoff)
        assert s.early.freeze(f,evidence,root=history,clock=lambda:cutoff+timedelta(seconds=1))=="written"
        m=next(m for m in original_models if str(m["fixture_id"])==str(f["id"]))
        fresh={k:v for k,v in m.items() if k not in ("forecast_id","record_id")}
        fresh.update({"model_key":"v8_near","model_version":"fresh-test-v8","horizon":"near",
                      "input_digest":evidence["input_digest"],"evidence_as_of_utc":evidence["as_of_utc"],
                      "captured_at_utc":(cutoff+timedelta(seconds=2)).isoformat()})
        assert append_file_record(fresh,root=source)
    # Isolate orchestration from the rating estimator's minimum-history threshold.
    monkeypatch.setattr(s.ratings,"predict",lambda *a:{"status":"ready","grid":np.full((5,5),.04)})
    kwargs=dict(root=root,source_root=source,history_root=history,pair_root=pairs)
    assert s.freeze(**kwargs,now=NOW,clock=lambda:NOW)["issued"]==1
    frozen=next((root/"trials").glob("*.json"))
    saved=frozen.read_bytes()
    assert s.freeze(**kwargs,now=NOW,clock=lambda:NOW)["preserved"]==1
    assert frozen.read_bytes()==saved and original_path.read_bytes()==original_bytes
    assert s.evaluate(**kwargs,prematch_root=tmp_path/"none",now=NOW)["counts"]["pending"]==1
    finals={}
    for m in original_models:
        result={k:m[k] for k in ("provider","fixture_id","home_team_id","away_team_id","kickoff_utc")}
        result.update({"status":"FT","evidence_version":2,"score_basis":"regulation",
                       "home_goals":0,"away_goals":0,"verified_at_utc":(NOW+timedelta(hours=5)).isoformat()})
        finals[str(m["fixture_id"])]=result
    (source/"verified_results.json").write_text(json.dumps(finals))
    settled=s.evaluate(**kwargs,prematch_root=tmp_path/"none",now=NOW+timedelta(hours=6))
    assert settled["counts"].get("rejected",0)==0
    assert settled["comparisons"]["fresh_v8"]["pairs"]==1
    assert settled["comparisons"]["fresh_v8"]["hits"]==1
    altered=json.loads(saved)
    altered["references"][0]["receipt_sha256"]="tampered"
    altered["sha256"]=s.early.digest({k:v for k,v in altered.items() if k!="sha256"})
    frozen.write_text(json.dumps(altered))
    rejected=s.evaluate(**kwargs,prematch_root=tmp_path/"none",now=NOW+timedelta(hours=6))
    assert rejected["counts"]["rejected"]==1
    assert rejected["comparisons"]["fresh_v8"]["pairs"]==0
