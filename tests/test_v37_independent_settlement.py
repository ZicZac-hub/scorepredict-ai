"""Original V37 sparse CL/EL tickets must receive independent score settlement.

These are research-only source references; no retrospective score issuance,
no change to V8/V28 source reader, and no increase to the provider-call cap.
"""
from datetime import timedelta
import json

import v10_prospective as v10
import v10_settlement as settlement
import v37_sparse_history_research as sparse
import v37_sparse_evaluation as audit
from forecast_ledger import read_file_records

from test_v37_sparse_history_research import NOW, fixture, sources, evidence


def record(tmp_path):
    result=sparse.freeze(fixture(),evidence(sources()),NOW,root=tmp_path)
    assert result["v37_recorded"] is True
    return read_file_records(tmp_path,kind="forecasts")[0]


def verified_final(row):
    return {
        "fixture_id":int(row["fixture_id"]), "provider":"API-Football",
        "kickoff_utc":row["kickoff_utc"],"home_team_id":row["home_team_id"],
        "away_team_id":row["away_team_id"],
        "home_team":row["home_team"],"away_team":row["away_team"],
        "status":"FT","score_basis":"regulation","evidence_version":2,
        "verified_at_utc":(NOW+timedelta(days=3)).isoformat(),
        "home_goals":3,"away_goals":1,
    }


def test_real_v37_only_forecast_reaches_independent_settlement(tmp_path,monkeypatch):
    original=record(tmp_path)
    assert v10.read_latest(tmp_path)=={}  # original live source filter retained
    due=NOW+timedelta(days=3)
    calls=[]
    def provider(key,identifier,reference,request=None):
        calls.append(str(identifier))
        assert reference["model_key"]=="v37_sparse"
        return verified_final(reference)
    monkeypatch.setattr(settlement,"_provider_match",provider)
    report=settlement.collect("test",root=tmp_path,now=due,
                              limit=2,legacy_results={})
    assert report["v37_only_references_added"]==1
    assert report["v37_invalid_originals_excluded"]==0
    assert report["eligible_frozen_fixtures"]==1
    assert report["checked"]==1
    assert report["newly_verified"]==1
    assert calls==[str(original["fixture_id"])]
    results=json.loads((tmp_path/"verified_results.json").read_text())
    assert str(original["fixture_id"]) in results
    graded=audit.evaluate(root=tmp_path,prematch=tmp_path,now=due)
    assert graded["frozen_unique_physical_fixtures"]==1
    assert graded["verified_matched_physical_fixtures"]==1
    assert graded["pending"]==0
    assert graded["invalid_or_tampered_excluded"]==0
    assert all(graded["methods"][name]["verified_leg_slots"]==1
               for name in audit.METHODS)
    assert graded["not_a_five_double_trial"]
    again=settlement.collect("test",root=tmp_path,now=due+timedelta(minutes=5),
                             limit=2,legacy_results={})
    assert again["newly_verified"]==0
    assert again["already_verified"]==1
    assert again["checked"]==0
    assert len(calls)==1


def test_v37_provider_is_never_called_before_regulation_final_lag(tmp_path,monkeypatch):
    row=record(tmp_path)
    def forbidden(*a,**kw):
        raise AssertionError("provider must not be queried before full-time")
    monkeypatch.setattr(settlement,"_provider_match",forbidden)
    report=settlement.collect(
        "test",root=tmp_path,
        now=v10.utc(row["kickoff_utc"])+timedelta(minutes=90),
        limit=2,legacy_results={})
    assert report["v37_only_references_added"]==1
    assert report["checked"]==0
    assert report["waiting_for_full_time"]==1
    assert report["still_pending"]==1


def test_v37_existing_result_uses_no_duplicate_api_calls(tmp_path,monkeypatch):
    row=record(tmp_path)
    verified={str(row["fixture_id"]):verified_final(row)}
    def forbidden(*a,**kw):
        raise AssertionError("already independently verified score needs no API")
    monkeypatch.setattr(settlement,"_provider_match",forbidden)
    report=settlement.collect(
        "test",root=tmp_path,now=NOW+timedelta(days=3),
        limit=1,legacy_results=verified)
    assert report["already_verified"]==1
    assert report["checked"]==0
    assert report["newly_verified"]==0


def test_v37_conflicting_or_tampered_originals_cannot_request_results(
    tmp_path,monkeypatch):
    original=record(tmp_path)
    mutated={**original,"v37_arm_digest_sha256":"wrong-hash"}
    monkeypatch.setattr(settlement,"read_file_records",
                        lambda *args,**kw:[mutated])
    report=settlement.collect(
        "test",root=tmp_path,now=NOW+timedelta(days=3),
        limit=2,legacy_results={})
    assert report["v37_only_references_added"]==0
    assert report["v37_invalid_originals_excluded"]==1
    assert report["checked"]==0
    monkeypatch.setattr(settlement,"read_file_records",
                        lambda *args,**kw:[original,dict(original)])
    report=settlement.collect(
        "test",root=tmp_path,now=NOW+timedelta(days=3),
        limit=2,legacy_results={})
    assert report["v37_only_references_added"]==0
    assert report["v37_invalid_originals_excluded"]==1
    assert report["checked"]==0


def test_sparse_due_priority_preserves_frozen_doubles_and_request_limit(
    tmp_path):
    original=record(tmp_path)
    # More recent unrelated V8 fixture cannot silently deprioritize a
    # V37 match whose original result is already due.
    refs={str(original["fixture_id"]):original}
    other={**original,"model_key":"v8","fixture_id":"999123",
           "kickoff_utc":(NOW+timedelta(days=2,hours=1)).isoformat()}
    refs["999123"]=other
    ordered=settlement._verification_order(refs,tmp_path,
                                          NOW+timedelta(days=3))
    assert ordered[0][0]==str(original["fixture_id"])
    # The full verification cap remains enforced by collect(), independently
    # of this lookup priority. Existing quota regression tests also apply.
