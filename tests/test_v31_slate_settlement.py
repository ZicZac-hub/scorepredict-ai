"""V31 coverage of original full-slate result settlement, without score edits."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

import backfill_verified_results as backfill
from forecast_ledger import forecast_id
from v28_independent_pair_trial import freeze as freeze_v28
from v31_slate_settlement import enrolled_due

SELECT = datetime(2035, 7, 8, 9, tzinfo=timezone.utc)
MATRIX = np.outer([.20,.32,.25,.15,.08],
                  [.31,.37,.19,.10,.03]).tolist()


def original(fid, *, kickoff=None):
    capture = SELECT-timedelta(hours=2)
    kickoff = kickoff or SELECT+timedelta(hours=6,minutes=fid*8)
    record = {
        "provider":"API-Football",
        "fixture_id":str(fid),
        "model_key":"v8","model_version":"frozen-v8-research",
        "mode":"prospective","score_basis":"regulation",
        "home_team_id":str(fid*2),"away_team_id":str(fid*2+1),
        "home_team":f"Home {fid}","away_team":f"Away {fid}",
        "kickoff_utc":kickoff.isoformat(),
        "captured_at_utc":capture.isoformat(),
        "evidence_as_of_utc":(capture-timedelta(seconds=30)).isoformat(),
        "history_matches":55,"input_digest":f"original-input-{fid}",
        "score_matrix":MATRIX,
    }
    record["forecast_id"]=forecast_id(record)
    return record


def prepare(tmp_path, count=4):
    root = tmp_path/"original_forecasts"
    v28 = tmp_path/"v28"
    results = tmp_path/"results.json"
    diagnostic = tmp_path/"status.json"
    records=[original(i) for i in range(1,count+1)]
    for r in records:
        file=root/"forecasts"/r["provider"]/r["fixture_id"]/(
            r["forecast_id"]+".json")
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(r))
    selected=freeze_v28(root=root,out=v28,now=SELECT)
    assert len(selected["created"])==1
    return root,v28,results,diagnostic,records


def configure(monkeypatch,root,v28,results,diagnostic,*,originals=None):
    monkeypatch.setattr(backfill,"ROOT",root)
    monkeypatch.setattr(backfill,"RESULTS",results)
    monkeypatch.setattr(backfill,"STATUS",diagnostic)
    monkeypatch.setattr(backfill,"read_latest",lambda root: originals or {})
    monkeypatch.setattr(
        backfill,"enrolled_due",
        lambda source_root,existing,clock: enrolled_due(
            source_root,existing,clock,v28_out=v28))
    monkeypatch.setenv("API_FOOTBALL_KEY","test-key")


def fixture_response(records, *, mismatched_ids=()):
    by_id={r["fixture_id"]:r for r in records}
    seen=[]
    def request(key,path,params,with_metadata):
        assert key=="test-key" and path=="/fixtures"
        assert with_metadata is True
        ids=params["ids"].split("-")
        assert len(ids)<=20
        seen.extend(ids)
        matches=[]
        for fid in ids:
            row=by_id[fid]
            bad=fid in mismatched_ids
            matches.append({
                "fixture":{
                    "id":int(fid),"date":row["kickoff_utc"],
                    "status":{"short":"FT"},
                },
                "teams":{
                    "home":{"id":int(row["home_team_id"])+int(bad),
                            "name":row["home_team"]},
                    "away":{"id":int(row["away_team_id"]),
                            "name":row["away_team"]},
                },
                "goals":{"home":1,"away":1},
            })
        return {"metadata":{"stale":False},"data":matches}
    return request,seen


def test_existing_v28_sources_stay_eligible_when_latest_pointer_is_empty(tmp_path,monkeypatch):
    root,v28,results,diagnostic,records=prepare(tmp_path)
    configure(monkeypatch,root,v28,results,diagnostic)
    snapshots=(v28/"decisions").glob("*.json")
    immutable=next(snapshots).read_bytes()
    due=SELECT+timedelta(hours=11)
    request,seen=fixture_response(records)
    status=backfill.run(request=request,now=due)
    assert status["state"]=="complete"
    assert status["eligible_unverified"]==4
    assert status["requests"]==1
    assert status["original_slate_settlement"]["validated_active_slate_days"]==1
    assert status["original_slate_settlement"]["unique_original_fixtures_enrolled"]==4
    assert status["original_slate_settlement"]["due_frozen_not_in_latest_forecast_index"]==4
    assert status["original_slate_settlement"]["due_frozen_ids_prioritized"]==4
    assert status["original_slate_settlement"]["due_frozen_ids_deferred_by_budget"]==0
    assert set(seen)=={"1","2","3","4"}
    assert sorted(status["new_verified"])==["1","2","3","4"]
    received=json.loads(results.read_text())
    assert set(received)=={"1","2","3","4"}
    assert all(record["status"]=="FT" for record in received.values())
    assert all(record["score_basis"]=="regulation" for record in received.values())
    assert next((v28/"decisions").glob("*.json")).read_bytes()==immutable
    assert backfill.run(request=request,now=due+timedelta(minutes=1))[
        "new_verified"]==[]
    assert len(seen)==4


def test_before_kickoff_results_are_not_requested(tmp_path,monkeypatch):
    root,v28,results,diagnostic,records=prepare(tmp_path)
    configure(monkeypatch,root,v28,results,diagnostic)
    def forbidden(*args,**kwargs):
        raise AssertionError("Provider call before independent result due time")
    status=backfill.run(request=forbidden,now=SELECT+timedelta(minutes=1))
    slate=status["original_slate_settlement"]
    assert status["eligible_unverified"]==0
    assert status["requests"]==0
    assert slate["original_fixtures_not_yet_due"]==4
    assert slate["original_fixtures_due_without_saved_result"]==0
    assert slate["next_original_fixture_settlement_due_utc"] is not None
    assert not results.exists()


def test_invalid_provider_identity_does_not_become_verified_result(tmp_path,monkeypatch):
    root,v28,results,diagnostic,records=prepare(tmp_path)
    configure(monkeypatch,root,v28,results,diagnostic)
    request,seen=fixture_response(records,mismatched_ids={"2"})
    status=backfill.run(request=request,now=SELECT+timedelta(hours=11))
    assert "2" in seen
    assert status["skipped_identity"]==1
    assert "2" not in status["new_verified"]
    assert "2" not in json.loads(results.read_text())
    assert status["original_slate_settlement"]["due_frozen_ids_prioritized"]==4


def test_corrupted_original_frozen_slate_is_rejected_before_api_query(tmp_path,monkeypatch):
    root,v28,results,diagnostic,records=prepare(tmp_path)
    file=next((v28/"decisions").glob("*.json"))
    saved=json.loads(file.read_text())
    saved["candidate_catalog"][0]["five_original_v25_robust_scores"][0][
        "score1"]="9-9"
    file.write_text(json.dumps(saved))
    configure(monkeypatch,root,v28,results,diagnostic)
    status=backfill.run(
        request=lambda *args,**kwargs: pytest.fail("tampered V28 source queried"),
        now=SELECT+timedelta(hours=11))
    assert status["requests"]==0
    assert status["original_slate_settlement"]["unique_original_fixtures_enrolled"]==0
    assert status["original_slate_settlement"]["invalid_original_slates_excluded"]


def test_due_v28_fixtures_precede_legacy_fixtures_under_existing_budget(tmp_path,monkeypatch):
    root,v28,results,diagnostic,records=prepare(tmp_path)
    legacy=[original(fid,kickoff=SELECT+timedelta(
        hours=6,minutes=fid%5)) for fid in range(101,126)]
    for r in legacy:
        # The source must exist for the original_model() identity validator.
        file=root/"forecasts"/r["provider"]/r["fixture_id"]/(
            r["forecast_id"]+".json")
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(r))
    configure(monkeypatch,root,v28,results,diagnostic,
              originals={str(r["fixture_id"]):r for r in legacy})
    monkeypatch.setattr(backfill,"MAX_CALLS",1)
    requested=[]
    def request(key,path,params,with_metadata):
        requested.extend(params["ids"].split("-"))
        return {"metadata":{"stale":False},"data":[]}
    status=backfill.run(request=request,now=SELECT+timedelta(hours=11))
    assert status["requests"]==1
    assert len(requested)==20
    assert set(["1","2","3","4"]).issubset(requested)
    assert status["original_slate_settlement"]["due_frozen_ids_prioritized"]==4
    assert status["eligible_unverified"]==29
    assert status["requested_fixture_ids"]==20


def test_enrollment_deduplicates_already_saved_physical_result(tmp_path):
    root,v28,_,_,records=prepare(tmp_path)
    candidate,summary=enrolled_due(
        root,{"1":{"recorded":"elsewhere"}},
        SELECT+timedelta(hours=11),v28_out=v28)
    assert len(candidate)==3
    assert "1" not in candidate
    assert summary["already_in_saved_result_file"]==1
    assert summary["unique_original_fixtures_enrolled"]==4
    assert summary["new_five_score_predictions_issued"] is False
