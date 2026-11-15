"""V18 source and temporal calibration regression cases."""
from datetime import datetime, timedelta, timezone
import json
import math

import pytest

from v18_accuracy_audit import (
    bucket, score_type, source_observations, reliability,
    time_safe_calibration, report, MIN_GLOBAL_TRAIN,
)

T=datetime(2035,1,1,12,tzinfo=timezone.utc)


def forecast(fid=1,days=0,est=None):
    capture=T+timedelta(days=days)
    # 2x2 score grid: 90% draws + 6% home win + 4% away win.
    grid=est or [[0.45,0.04],[0.06,0.45]]
    totals={"home":grid[1][0],"draw":grid[0][0]+grid[1][1],
            "away":grid[0][1]}
    return {"mode":"prospective","model_key":"v8","provider":"API-Football",
            "fixture_id":str(fid),"forecast_id":"id"+str(fid),
            "input_digest":"digest"+str(fid),
            "home_team_id":"101","away_team_id":"102",
            "competition":"TST",
            "captured_at_utc":capture.isoformat(),
            "evidence_as_of_utc":(capture-timedelta(minutes=3)).isoformat(),
            "kickoff_utc":(capture+timedelta(hours=2)).isoformat(),
            "score_matrix":grid,"outcome_probabilities":totals,
            "history_matches":135,"league_history_matches":200,
            "component_weights":{"v6":0.6,"v7":0.4},
            "research_context":{"home":{"last10_observed":10},
                                "away":{"last10_observed":10},
                                "verified_xg_xga":{"available":False},
                                "verified_lineup_adjustments":{"applied":False}},
            "market_used":False}


def save_source(root,row):
    p=root/"forecasts"/row["provider"]/row["fixture_id"]
    p.mkdir(parents=True,exist_ok=True)
    (p/(row["forecast_id"]+".json")).write_text(json.dumps(row),encoding="utf-8")


def finished(row,h=0,a=0,verified_hours=4):
    return {"provider":row["provider"],"fixture_id":int(row["fixture_id"]),
            "home_team_id":row["home_team_id"],
            "away_team_id":row["away_team_id"],
            "kickoff_utc":row["kickoff_utc"],"home_goals":h,"away_goals":a,
            "score_basis":"regulation","status":"FT","evidence_version":2,
            "verified_at_utc":(datetime.fromisoformat(row["captured_at_utc"])+
                                timedelta(hours=verified_hours)).isoformat()}


def test_bucket_edges_and_regulation_outcome():
    assert bucket(.50)==0
    assert bucket(.70)==2
    assert bucket(.99)==4
    assert bucket(1)==4
    with pytest.raises(ValueError):
        bucket(float("nan"))
    assert score_type(2,1)=="1"
    assert score_type(1,1)=="X"
    assert score_type(1,3)=="2"


def test_true_original_source_no_future_rewrites_and_pending(tmp_path):
    row=forecast()
    save_source(tmp_path,row)
    verified={row["fixture_id"]:finished(row)}
    observed,diagnostics=source_observations(tmp_path,verified,[row])
    assert len(observed)==1
    assert observed[0]["hit"]==1
    assert diagnostics["verified_results"]==1
    assert diagnostics["missing_feature_counts"]["shot_based_xg_not_verified"]==1
    assert diagnostics["missing_feature_counts"]["lineup_adjustment_not_validated"]==1
    assert diagnostics["missing_feature_counts"]["independent_market_snapshot_unavailable"]==1
    pending,counts=source_observations(tmp_path,{},[row])
    assert pending==[]
    assert counts["awaiting_or_unverified"]==1
    forged=finished(row)
    forged["verified_at_utc"]=(T-timedelta(days=1)).isoformat()
    future,counts=source_observations(tmp_path,{row["fixture_id"]:forged},[row])
    assert future==[] and counts["awaiting_or_unverified"]==1
    altered=forecast()
    altered["input_digest"]="not-source"
    wrong,counts=source_observations(tmp_path,verified,[altered])
    assert wrong==[]
    assert counts["source_rejections"]["original_forecast_id_or_digest_unverified"]==1


def test_wrong_match_ids_or_team_never_count_as_win(tmp_path):
    row=forecast()
    save_source(tmp_path,row)
    wrong=finished(row)
    wrong["home_team_id"]="not-real"
    observations,diagnostics=source_observations(
        tmp_path,{row["fixture_id"]:wrong},[row])
    assert not observations
    assert diagnostics["awaiting_or_unverified"]==1


def test_no_duplicate_identity_and_invalid_grids_not_included(tmp_path):
    row=forecast()
    save_source(tmp_path,row)
    observations,counts=source_observations(
        tmp_path,{row["fixture_id"]:finished(row)},[row,row])
    assert len(observations)==1
    assert counts["source_rejections"]["duplicate_v8_fixture"]==1
    bad=forecast(22,est=[[float("nan"),0],[0,0]])
    save_source(tmp_path,bad)
    obs,counts=source_observations(tmp_path,{},[bad])
    assert not obs
    assert counts["source_rejections"]


def test_reliability_is_observed_not_model_prob():
    rows=[{"bin":2,"hit":0,"estimated_probability":0.75},
          {"bin":2,"hit":1,"estimated_probability":0.75}]
    x=reliability(rows)
    assert x["verified"]==2 and x["hits"]==1
    assert x["observed_win_rate"]==pytest.approx(.5)
    assert x["brier"]==pytest.approx((.75**2+.25**2)/2)
    assert x["expected_calibration_error"]==pytest.approx(.25)
    assert x["sample_too_small_for_reliable_calibration"] is True
    assert reliability([])["observed_win_rate"] is None


def _row(i,hit,verify_delay=5):
    clock=T+timedelta(days=i)
    return {"provider":"API-Football","fixture_id":str(i),
            "captured_at_utc":clock.isoformat(),
            "kickoff_utc":(clock+timedelta(hours=1)).isoformat(),
            "verified_at_utc":(clock+timedelta(hours=verify_delay)).isoformat(),
            "bin":3,"hit":int(hit),"estimated_probability":.83}


def test_walkforward_requires_earlier_verified_results_and_no_auto_promotion():
    rows=[_row(i,i%2==0) for i in range(155)]
    out=time_safe_calibration(rows)
    assert out["walk_forward_verified_test_matches"]>=50
    assert out["raw_brier"] is not None
    assert out["shrunk_calibrated_brier"] is not None
    assert out["model_promotion"] is False
    # If finals were only verified AFTER the last forecast, no calibration
    # training is allowed even if results are present at report time.
    all_late=[_row(i,i%2==0,verify_delay=24*250) for i in range(155)]
    blocked=time_safe_calibration(all_late)
    assert blocked["walk_forward_verified_test_matches"]==0
    assert blocked["calibration_candidate_eligible_for_review"] is False


def test_no_accuracy_claim_without_real_results(tmp_path):
    doc=report(tmp_path,verified={},forecast_rows=[])
    assert doc["baseline_double_chance"]["verified"]==0
    assert doc["baseline_double_chance"]["observed_win_rate"] is None
    assert doc["target_70_percent_actual_claim"] is False
    assert not doc["time_safe_shrinkage_research"]["deployed"]
