"""Future-only V37 sparse UEFA research does not relax V8 or add API calls."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from forecast_ledger import read_file_records
import v10_prospective as v10
import v37_sparse_history_research as sparse
import v37_sparse_evaluation as audit


NOW = datetime(2035,1,10,12,tzinfo=timezone.utc)


def fixture():
    return {
        "id":555101, "provider":"API-Football",
        "utcDate":(NOW+timedelta(days=2)).isoformat(),
        "status":"SCHEDULED", "provider_status":"NS",
        "homeTeam":{"id":1,"name":"Home"},
        "awayTeam":{"id":2,"name":"Away"},
        "competition":{"id":2,"code":"CL"},
    }


def sources():
    rows=[]
    def row(i,home,away,when):
        return {
            "id":10000+i, "provider":"API-Football",
            "utcDate":when.isoformat(), "status":"FINISHED",
            "homeTeam":{"id":home,"name":str(home)},
            "awayTeam":{"id":away,"name":str(away)},
            "competition":{"id":2,"code":"CL"},
            "score":{"fullTime":{"home":1+(i%3),"away":i%2},
                     "duration":"REGULAR"},
            "score_basis":"regulation",
        }
    for i in range(40):
        rows.append(row(i,10+(i%12),25+(i%12),NOW-timedelta(days=40-i)))
    for i in range(3):
        rows.append(row(100+i,1,45+i,NOW-timedelta(days=14-i)))
        rows.append(row(200+i,55+i,2,NOW-timedelta(days=10-i)))
    return rows


def evidence(rows):
    return {
        "history":rows,"league_history":rows,
        "input_digest":"signed-authenticated-league-history",
        "as_of_utc":NOW.isoformat(),"policy_version":"two-season-asof-v1",
    }


def test_v37_two_fixed_methods_and_proper_normalization():
    data=sparse.predict(fixture(),sources(),sources(),as_of=NOW)
    assert data is not None
    assert data["v37_research_only"] is True
    assert data["v37_not_a_v8_forecast"] is True
    assert data["model_version"]==sparse.VERSION
    challenger=np.asarray(data["matrix"])
    control=np.asarray(data["v37_control_grid"])
    assert challenger.shape == control.shape
    assert challenger.shape[0]>=11
    assert np.isclose(challenger.sum(),1) and np.isclose(control.sum(),1)
    assert (challenger>=0).all() and (control>=0).all()
    assert not np.allclose(challenger,control)


def test_v37_does_not_use_future_matches_or_unverified_competitions():
    game=fixture()
    rows=sources()
    future=[{**r,"utcDate":(NOW+timedelta(hours=2)).isoformat()}
            for r in rows if 1 in (r["homeTeam"]["id"],r["awayTeam"]["id"]) or
                             2 in (r["homeTeam"]["id"],r["awayTeam"]["id"])]
    kept=[r for r in rows if 1 not in (r["homeTeam"]["id"],r["awayTeam"]["id"])
          and 2 not in (r["homeTeam"]["id"],r["awayTeam"]["id"])]
    assert sparse.predict(game,kept+future,kept+future,as_of=NOW) is None
    no_league={**game,"competition":{"id":39,"code":"PL"}}
    assert sparse.predict(no_league,rows,rows,as_of=NOW) is None
    assert sparse.predict(game,rows,rows,as_of=NOW+timedelta(days=3)) is None
    insufficient=[r for r in rows if r["id"]<10035]
    assert sparse.predict(game,insufficient,insufficient,as_of=NOW) is None


def test_v37_atomic_single_record_and_duplicate_ignored(tmp_path):
    game=fixture()
    records=sources()
    outcome=sparse.freeze(game,evidence(records),NOW,root=tmp_path)
    assert outcome["v37_recorded"] is True
    saved=read_file_records(tmp_path,kind="forecasts",
                            provider="API-Football",fixture_id="555101")
    assert len(saved)==1
    item=saved[0]
    assert item["model_key"]=="v37_sparse"
    assert item["v37_not_eligible_for_v28"] is True
    assert item["v37_arm_digest_sha256"]==sparse.frozen_arm_digest(item)
    assert item["evidence_as_of_utc"]==NOW.isoformat()
    assert np.isclose(np.asarray(item["v37_control_grid"]).sum(),1)
    retry=sparse.freeze(game,evidence(records),NOW,root=tmp_path)
    assert retry=={"v37_recorded":False,
                   "v37_reason":"original_v37_already_issued"}
    assert len(read_file_records(tmp_path,kind="forecasts"))==1


def test_v8_abstention_stays_abstention_with_v37_shadow(monkeypatch,tmp_path):
    monkeypatch.setattr(v10,"research_predict",lambda *a,**kw:None)
    result=v10.capture(fixture(),evidence(sources()),NOW,root=tmp_path,
                       market_calls_remaining=0)
    assert result["v8_recorded"] is False
    assert result["readiness_reason"]=="team_history_below_five"
    assert result["v37_recorded"] is True
    assert result["market_calls"]==0
    assert len(read_file_records(tmp_path,kind="forecasts"))==1
    assert all(r["model_key"]!="v8" for r in
               read_file_records(tmp_path,kind="forecasts"))


def test_v37_graded_on_same_verified_match_or_pending(tmp_path,monkeypatch):
    sparse.freeze(fixture(),evidence(sources()),NOW,root=tmp_path)
    item=read_file_records(tmp_path,kind="forecasts")[0]
    key=(item["provider"],item["fixture_id"],item["model_key"])
    pending=audit.evaluate(rows={key:item},finals={},
                           now=NOW+timedelta(days=4))
    assert pending["pending"]==1
    assert pending["verified_matched_physical_fixtures"]==0
    assert pending["methods"]["league_only"]["verified_leg_slots"]==0
    assert pending["ready_for_model_promotion"] is False
    final={"home_goals":3,"away_goals":2}
    monkeypatch.setattr(audit,"_qualified",lambda *a:True)
    verified=audit.evaluate(rows={key:item},
            finals={(item["provider"],item["fixture_id"]):final},
            now=NOW+timedelta(days=4))
    assert verified["verified_matched_physical_fixtures"]==1
    assert verified["pending"]==0
    assert all(verified["methods"][m]["verified_leg_slots"]==1
               for m in audit.METHODS)
    assert all(verified["methods"][m]["mean_exact_score_log_loss"] is not None
               for m in audit.METHODS)
    assert verified["not_a_five_double_trial"]


def test_v37_tampered_arm_fails_closed(tmp_path):
    sparse.freeze(fixture(),evidence(sources()),NOW,root=tmp_path)
    item=read_file_records(tmp_path,kind="forecasts")[0]
    item["v37_control_grid"][0][0] += .001
    key=(item["provider"],item["fixture_id"],item["model_key"])
    report=audit.evaluate(rows={key:item},finals={},now=NOW+timedelta(days=4))
    assert report["invalid_or_tampered_excluded"]==1
    assert report["verified_matched_physical_fixtures"]==0
    assert report["pending"]==0


def test_v37_production_workflow_only_uses_original_future_source():
    from pathlib import Path
    workflow=Path(".github/workflows/prematch-evidence.yml").read_text()
    assert "run: python capture_prematch.py" in workflow
    assert "run: python v37_sparse_evaluation.py" in workflow
    assert workflow.index("run: python capture_prematch.py") < (
        workflow.index("run: python v37_sparse_evaluation.py"))
    assert "data/v37_sparse_prospective/" in workflow
    assert "run: python v28_independent_pair_trial.py" in workflow
    assert "run: python v36_prospective_double_trials.py" in workflow
    from pathlib import Path
    src=Path("v37_sparse_history_research.py").read_text()
    assert "requests.get" not in src
    assert "requests.post" not in src
    assert "append_file_record(row,root=root,kind=\"forecasts\")" in src
