"""Full-slate V30 audit: all frozen rows, true finals, no selection hindsight."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from forecast_ledger import forecast_id
from v28_independent_pair_trial import freeze as freeze_v28
from v30_full_slate_audit import VERSION, analyze, grade_day, run
from verified_result_index import index_verified_results

NOW = datetime(2035,4,10,9,tzinfo=timezone.utc)
MATRIX = np.outer([.20,.32,.25,.15,.08],
                  [.31,.37,.19,.10,.03]).tolist()


def original(fid, *, hours_before=2):
    captured=NOW-timedelta(hours=hours_before)
    source={
        "provider":"API-Football", "fixture_id":str(fid),
        "model_key":"v8", "model_version":"frozen-test-v8",
        "mode":"prospective", "score_basis":"regulation",
        "home_team_id":str(1000+fid*2), "away_team_id":str(1001+fid*2),
        "home_team":f"Home {fid}", "away_team":f"Away {fid}",
        "kickoff_utc":(NOW+timedelta(hours=5,minutes=fid*6)).isoformat(),
        "captured_at_utc":captured.isoformat(),
        "evidence_as_of_utc":(captured-timedelta(seconds=5)).isoformat(),
        "history_matches":65, "input_digest":f"source-digest-{fid}",
        "score_matrix":MATRIX,
    }
    source["forecast_id"]=forecast_id(source)
    return source


def final(row, goals=(1,1), *, verified_at=None):
    kickoff=datetime.fromisoformat(row["kickoff_utc"])
    return {
        **{k:row[k] for k in (
            "provider","fixture_id","home_team_id","away_team_id","kickoff_utc")},
        "score_basis":"regulation","status":"FT","evidence_version":2,
        "verified_at_utc":(verified_at or
            kickoff+timedelta(hours=3)).isoformat(),
        "home_goals":goals[0],"away_goals":goals[1],
    }


def fixture(tmp_path, count=4):
    root=tmp_path/"original"
    prematch=tmp_path/"prematch"
    out=tmp_path/"v28"
    prematch.mkdir()
    source=[original(i) for i in range(1,count+1)]
    for row in source:
        file=root/"forecasts"/row["provider"]/row["fixture_id"]/(
            row["forecast_id"]+".json")
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(row))
    r=freeze_v28(root=root,out=out,now=NOW)
    assert len(r["created"])==1
    selected=next((out/"decisions").glob("*.json"))
    return root,prematch,out,source,selected


def report(tmp_path,*,root,prematch,out,now):
    return run(root=root,prematch=prematch,v28_out=out,
               out=tmp_path/"v30",now=now)


def test_original_full_candidate_slate_is_prospectively_frozen_and_pending(tmp_path):
    root,prematch,out,source,selected=fixture(tmp_path)
    saved=selected.read_bytes()
    data=report(tmp_path,root=root,prematch=prematch,out=out,now=NOW)
    assert data["version"]==VERSION
    assert data["original_frozen_days"]==1
    assert data["frozen_candidate_pair_rows"]==6
    assert data["candidate_pair_rows_independently_verified"]==0
    assert data["candidate_pair_rows_awaiting_verified_finals"]==6
    assert data["fully_verified_slate_days"]==0
    assert data["days_still_missing_results"]==1
    assert data["complete_day_unweighted_mean_candidate_hit_fraction"] is None
    assert data["days"][0]["status"]=="awaiting_verified_finals"
    assert len(data["days"][0]["missing_fixture_evidence"])==4
    assert all(x["status"]=="before_kickoff"
               for x in data["days"][0]["missing_fixture_evidence"])
    assert "full_slate_diagnostics" not in data["days"][0]
    assert data["model_estimated_mass_is_not_real_win_rate"] is True
    assert data["overlapping_pairs_are_not_independent_samples"] is True
    assert data["seventy_percent_five_double_hit_rate_not_proven"] is True
    assert selected.read_bytes()==saved
    assert not (tmp_path/"v30"/"decisions").exists()


def test_full_verified_slate_measures_all_6_pre_registered_pairs(tmp_path):
    root,prematch,out,source,selected=fixture(tmp_path)
    frozen=selected.read_bytes()
    goals={x["fixture_id"]:final(x) for x in source}
    (prematch/"verified_results.json").write_text(json.dumps(goals))
    clock=NOW+timedelta(hours=14)
    data=report(tmp_path,root=root,prematch=prematch,out=out,now=clock)
    day=data["days"][0]
    assert data["fully_verified_slate_days"]==1
    assert data["frozen_candidate_pair_rows"]==6
    assert data["candidate_pair_rows_independently_verified"]==6
    assert data["candidate_pair_rows_awaiting_verified_finals"]==0
    assert day["complete_slate"] is True
    assert day["status"]=="complete_verified_catalog"
    rows=day["candidate_rows"]
    assert len(rows)==6
    assert sorted(x["quality_rank"] for x in rows)==list(range(1,7))
    assert sorted(x["robustness_rank"] for x in rows)==list(range(1,7))
    assert all(isinstance(x["robust_five_hit"],bool) for x in rows)
    assert all(isinstance(x["v8_baseline_five_hit"],bool) for x in rows)
    assert all(x["actual_scores"]==["1-1","1-1"] for x in rows)
    summary=day["full_slate_diagnostics"]
    assert summary["all_candidate_pairs"]["verified_pair_rows"]==6
    assert sum(x["verified_pair_rows"] for x in
               summary["predefined_quality_quartiles"])==6
    checks=summary["equal_budget_top_k_comparisons"]
    assert list(checks)==["1","5","10"]
    assert checks["1"]["quality_ranked_pairs"]["pair_count"]==1
    assert checks["1"]["quality_ranked_pairs"]["ticket_budget_if_all_pairs_used"]==5
    assert checks["5"]["quality_ranked_pairs"]["ticket_budget_if_all_pairs_used"]==25
    assert checks["10"]["quality_ranked_pairs"]["ticket_budget_if_all_pairs_used"]==30
    assert checks["10"]["top_k"]==6
    assert checks["5"]["not_comparable_to_one_five_double_ticket_set"] is True
    assert checks["5"]["quality_ranked_pairs"]["unique_underlying_fixtures"] <= 4
    assert summary["full_slate_learning_only_not_independent_policy_win_rate"] is True
    assert data["initial_review_ready"] is False
    assert selected.read_bytes()==frozen
    assert report(tmp_path,root=root,prematch=prematch,out=out,now=clock)==data


def test_partial_results_are_diagnostics_not_full_slate_learning(tmp_path):
    root,prematch,out,source,_=fixture(tmp_path)
    # Two independently verified originals give ONE candidate pair, not six.
    goals={x["fixture_id"]:final(x) for x in source[:2]}
    (prematch/"verified_results.json").write_text(json.dumps(goals))
    data=report(tmp_path,root=root,prematch=prematch,out=out,
                now=NOW+timedelta(hours=14))
    day=data["days"][0]
    assert data["partially_verified_slate_days"]==1
    assert data["fully_verified_slate_days"]==0
    assert day["status"]=="partially_verified_catalog"
    assert day["candidate_pairs_verified"]==1
    assert day["candidate_pairs_pending"]==5
    assert day["unique_fixtures_missing_verified_final"]==2
    assert all(x["status"]=="verified_regulation_final_unavailable"
               for x in day["missing_fixture_evidence"])
    assert day["exploratory_verified_candidate_rows"]["verified_pair_rows"]==1
    assert "full_slate_diagnostics" not in day
    assert data["complete_day_unweighted_mean_candidate_hit_fraction"] is None


def test_future_verified_timestamp_stays_pending_not_hit_or_loss(tmp_path):
    root,prematch,out,source,_=fixture(tmp_path)
    future=NOW+timedelta(days=3)
    goals={x["fixture_id"]:final(x,verified_at=future) for x in source}
    (prematch/"verified_results.json").write_text(json.dumps(goals))
    data=report(tmp_path,root=root,prematch=prematch,out=out,
                now=NOW+timedelta(hours=14))
    assert data["candidate_pair_rows_independently_verified"]==0
    assert data["fully_verified_slate_days"]==0
    assert data["days"][0]["candidate_pairs_pending"]==6


def test_original_forecast_tampering_excluded_without_backfill(tmp_path):
    root,prematch,out,source,frozen=fixture(tmp_path)
    before=frozen.read_bytes()
    source_file=root/"forecasts"/source[0]["provider"]/source[0]["fixture_id"]/(
        source[0]["forecast_id"]+".json")
    changed=json.loads(source_file.read_text())
    changed["input_digest"]="tampered-after-freeze"
    source_file.write_text(json.dumps(changed))
    data=report(tmp_path,root=root,prematch=prematch,out=out,
                now=NOW+timedelta(hours=14))
    assert data["original_frozen_days"]==0
    assert data["invalid_or_tampered_frozen_days_excluded"]
    assert data["frozen_candidate_pair_rows"]==0
    assert frozen.read_bytes()==before


def test_edited_saved_rank_or_five_scores_cannot_get_graded(tmp_path):
    root,prematch,out,source,frozen=fixture(tmp_path)
    packet=json.loads(frozen.read_text())
    packet["candidate_catalog"][0]["five_original_v25_robust_scores"][0]["score1"]="4-4"
    frozen.write_text(json.dumps(packet))
    data=report(tmp_path,root=root,prematch=prematch,out=out,
                now=NOW+timedelta(hours=14))
    assert data["original_frozen_days"]==0
    assert data["invalid_or_tampered_frozen_days_excluded"]


def test_conflicting_verified_finals_not_assumed_valid(tmp_path):
    root,prematch,out,source,_=fixture(tmp_path)
    root.mkdir(parents=True,exist_ok=True)
    one={x["fixture_id"]:final(x) for x in source}
    two={x["fixture_id"]:final(x) for x in source}
    two[source[0]["fixture_id"]]=final(source[0],(3,2))
    (root/"verified_results.json").write_text(json.dumps(one))
    (prematch/"verified_results.json").write_text(json.dumps(two))
    data=report(tmp_path,root=root,prematch=prematch,out=out,
                now=NOW+timedelta(hours=14))
    assert data["verified_result_conflicts_excluded"]>=1
    assert data["candidate_pair_rows_independently_verified"]==3
    assert data["fully_verified_slate_days"]==0
    assert data["days"][0]["unique_fixtures_missing_verified_final"]==1


def test_unverified_identity_mismatch_rejected(tmp_path):
    root,prematch,out,source,_=fixture(tmp_path)
    wrong={x["fixture_id"]:final(x) for x in source}
    wrong["1"]["away_team_id"]="9999"
    (prematch/"verified_results.json").write_text(json.dumps(wrong))
    data=report(tmp_path,root=root,prematch=prematch,out=out,
                now=NOW+timedelta(hours=14))
    assert data["fully_verified_slate_days"]==0
    assert data["candidate_pair_rows_independently_verified"]==3


def test_frozen_catalog_not_issued_by_new_readonly_grade(tmp_path):
    root,prematch,out,source,frozen=fixture(tmp_path,count=3)
    frozen_before=frozen.read_bytes()
    assert analyze(root=root,prematch=prematch,v28_out=out,now=NOW)[
        "frozen_candidate_pair_rows"]==3
    assert frozen.read_bytes()==frozen_before
    assert not (out/"decisions"/"2020-01-01.json").exists()
