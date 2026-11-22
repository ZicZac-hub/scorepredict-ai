"""V29 tests: future-only distinct pairing, independent result readiness and refusal to backfill."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from forecast_ledger import forecast_id
from v28_independent_pair_trial import freeze as freeze28
from v29_pair_sensitivity import (
    VERSION, evaluate, freeze, run, _distinct_alternative,
)

AT=datetime(2036,2,10,8,tzinfo=timezone.utc)
GOALS=np.outer([.20,.32,.25,.15,.08],[.31,.37,.19,.10,.03]).tolist()


def model(fid, *, age=2):
    captured=AT-timedelta(hours=age)
    row={
        "provider":"API-Football","fixture_id":str(fid),
        "model_key":"v8","model_version":"independent-v8-original",
        "mode":"prospective","score_basis":"regulation",
        "home_team_id":str(fid*2),"away_team_id":str(fid*2+1),
        "home_team":f"Home {fid}","away_team":f"Away {fid}",
        "history_matches":55, "input_digest":f"source-{fid}",
        "kickoff_utc":(AT+timedelta(hours=6,minutes=(fid-1)*7)).isoformat(),
        "captured_at_utc":captured.isoformat(),
        "evidence_as_of_utc":(captured-timedelta(minutes=1)).isoformat(),
        "score_matrix":GOALS,
    }
    row["forecast_id"]=forecast_id(row)
    return row


def final(row,goals,*,verified_at=None):
    return {
        **{k:row[k] for k in (
            "provider","fixture_id","home_team_id","away_team_id","kickoff_utc")},
        "status":"FT","evidence_version":2,"score_basis":"regulation",
        "verified_at_utc":(verified_at or
            datetime.fromisoformat(row["kickoff_utc"])+timedelta(hours=3)).isoformat(),
        "home_goals":goals[0],"away_goals":goals[1],
    }


def make(tmp, ages=(110,100,2,1)):
    root,prematch,out28,out29=[tmp/p for p in ("root","prematch","v28","v29")]
    prematch.mkdir()
    models=[model(i,age=age) for i,age in enumerate(ages,1)]
    for record in models:
        path=root/"forecasts"/record["provider"]/record["fixture_id"]/(
            record["forecast_id"]+".json")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(record))
    source=freeze28(root=root,out=out28,now=AT)
    assert len(source["created"])==1
    path=next((out28/"decisions").glob("*.json"))
    return root,prematch,out28,out29,models,path


def test_v29_freezes_new_distinct_pair_and_grades_without_rewriting_v28(tmp_path):
    root,prematch,out28,out29,models,v28_file=make(tmp_path)
    original=v28_file.read_bytes()
    now=AT+timedelta(minutes=1)
    reply=freeze(root=root,v28_out=out28,out=out29,now=now)
    assert len(reply["created"])==1
    v29_file=next((out29/"decisions").glob("*.json"))
    saved=v29_file.read_bytes()
    packet=json.loads(saved)
    reference=json.loads(original)
    assert packet["version"]==VERSION
    assert packet["decision"]=="DISTINCT_PAIR_FROZEN"
    assert packet["comparison"]["different_pair_from_both_v28_arms"] is True
    both_same = (
        reference["quality_selection"]["fixture_keys"] ==
        reference["robustness_control"]["fixture_keys"]
    )
    unique_reference = {
        tuple(pair) for policy in ("quality_selection","robustness_control")
        for pair in reference[policy]["fixture_keys"]
    }
    assert packet["comparison"]["quality_and_control_same_pair"] is both_same
    assert packet["comparison"]["shared_fixtures_with_v28_quality"] in (0,1)
    assert packet["alternative"]["fixture_keys"]!=reference["quality_selection"]["fixture_keys"]
    assert packet["70_percent_accuracy_not_demonstrated"] is True
    assert freeze(root=root,v28_out=out28,out=out29,now=now)["created"]==[]
    assert v29_file.read_bytes()==saved
    assert v28_file.read_bytes()==original

    pending=evaluate(root=root,prematch=prematch,v28_out=out28,out=out29,now=now)
    readiness=pending["v28_capture_and_result_readiness"]
    assert readiness["v28_frozen_active_days"]==1
    assert readiness["v28_identical_policy_pair_days"]==int(both_same)
    assert readiness["v28_distinct_policy_pair_days"]==int(not both_same)
    assert readiness["v28_fixtures_before_kickoff"]==len(unique_reference)
    assert readiness["v28_verified_distinct_fixture_count"]==0
    assert readiness["v28_fixtures_past_kickoff_awaiting_verified_result"]==0
    assert pending["v29_distinct_pair_experiment"]["pending_alternatives"]==1

    # Compare once all matching physical match results are independently confirmed.
    finals={row["fixture_id"]:final(row,(0,0)) for row in models}
    best=packet["alternative"]["five_original_v25_robust_scores"][0]
    for (_,fid),score in zip(
        packet["alternative"]["fixture_keys"],(best["score1"],best["score2"])):
        target=next(x for x in models if x["fixture_id"]==fid)
        finals[fid]=final(target,tuple(int(i) for i in score.split("-")))
    (prematch/"verified_results.json").write_text(json.dumps(finals))
    done=evaluate(root=root,prematch=prematch,v28_out=out28,out=out29,
                  now=AT+timedelta(hours=13))
    study=done["v29_distinct_pair_experiment"]
    assert study["both_arms_verified_days"]==1
    assert study["alternative_five_wins"]==1
    assert study["initial_review_ready"] is False
    assert done["v28_capture_and_result_readiness"][
        "v28_verified_distinct_fixture_count"]==len(unique_reference)
    assert done["verified_final_evidence_conflicts_excluded"]==0
    assert study["invalid_or_modified_set_exclusions"]=={}
    assert v29_file.read_bytes()==saved and v28_file.read_bytes()==original


def test_no_retrospective_v29_alternatives(tmp_path):
    root,prematch,out28,out29,_,_=make(tmp_path)
    too_late=freeze(root=root,v28_out=out28,out=out29,
                    now=AT+timedelta(minutes=21))
    assert too_late["created"]==[]
    assert too_late["excluded"]["original_v28_older_than_20_minute_window"]==1
    assert not (out29/"decisions").exists()


def test_source_was_written_only_from_v28_verified_slate(tmp_path):
    root,prematch,out28,out29,_,_=make(tmp_path)
    freeze(root=root,v28_out=out28,out=out29,now=AT+timedelta(minutes=1))
    v29_file=next((out29/"decisions").glob("*.json"))
    original=v29_file.read_bytes()
    data=json.loads(original)
    data["alternative"]["quality_rank_score"]=0.99999999
    v29_file.write_text(json.dumps(data))
    invalid=evaluate(root=root,prematch=prematch,v28_out=out28,
                     out=out29,now=AT+timedelta(minutes=1))
    assert invalid["v29_distinct_pair_experiment"]["both_arms_verified_days"]==0
    assert invalid["v29_distinct_pair_experiment"]["invalid_or_modified_set_exclusions"]=={
        "v29_alternative_not_reproducible":1
    }
    v29_file.write_bytes(original)


def test_readiness_transitions_from_future_to_unverified_to_verified(tmp_path):
    root,prematch,out28,out29,models,_=make(tmp_path)
    freeze(root=root,v28_out=out28,out=out29,now=AT+timedelta(minutes=1))
    before=evaluate(root=root,prematch=prematch,v28_out=out28,
                    out=out29,now=AT+timedelta(minutes=1))
    reference=json.loads(next((out28/"decisions").glob("*.json")).read_text())
    unique_fixtures = {
        tuple(key) for arm in ("quality_selection","robustness_control")
        for key in reference[arm]["fixture_keys"]
    }
    assert before["v28_capture_and_result_readiness"]["v28_fixtures_before_kickoff"]==len(unique_fixtures)

    elapsed=AT+timedelta(hours=9)
    not_verified=evaluate(root=root,prematch=prematch,v28_out=out28,
                          out=out29,now=elapsed)
    ready=not_verified["v28_capture_and_result_readiness"]
    assert ready["v28_fixtures_before_kickoff"]==0
    assert ready["v28_fixtures_past_kickoff_awaiting_verified_result"]==len(unique_fixtures)
    assert not_verified["v29_distinct_pair_experiment"]["both_arms_verified_days"]==0

    results={r["fixture_id"]:final(r,(1,1)) for r in models}
    (prematch/"verified_results.json").write_text(json.dumps(results))
    after=evaluate(root=root,prematch=prematch,v28_out=out28,
                   out=out29,now=AT+timedelta(hours=13))
    ready=after["v28_capture_and_result_readiness"]
    assert ready["v28_fixtures_past_kickoff_awaiting_verified_result"]==0
    assert ready["v28_verified_distinct_fixture_count"]==len(unique_fixtures)


def test_invalid_original_v28_eliminates_v29_from_matched_denominator(tmp_path):
    root,prematch,out28,out29,models,v28_file=make(tmp_path)
    freeze(root=root,v28_out=out28,out=out29,now=AT+timedelta(minutes=1))
    record=models[2]
    path=root/"forecasts"/record["provider"]/record["fixture_id"]/(
        record["forecast_id"]+".json")
    changed=json.loads(path.read_text())
    changed["input_digest"]="hindsight-edited"
    path.write_text(json.dumps(changed))
    report=evaluate(root=root,prematch=prematch,v28_out=out28,
                    out=out29,now=AT+timedelta(hours=13))
    assert report["v28_capture_and_result_readiness"][
        "invalid_original_v28_days_excluded"]
    assert report["v29_distinct_pair_experiment"][
        "invalid_or_modified_set_exclusions"]


def test_verify_only_does_not_issue_future_forecasts(tmp_path):
    root,prematch,out28,out29,_,_=make(tmp_path)
    report=run(root=root,prematch=prematch,v28_out=out28,out=out29,
               now=AT+timedelta(minutes=1),verify_only=True)
    assert report["freeze"]["created"]==[]
    assert not (out29/"decisions").exists()
    assert report["research"]["v28_capture_and_result_readiness"][
        "v28_frozen_active_days"]==1


def test_no_spurious_exploration_when_fewer_than_two_distinct_pairs(tmp_path):
    # Synthetic one-candidate slate is ineligible as V28 already refuses to
    # freeze with fewer than 2 eligible four-team pair options.
    root,prematch,out28,out29,_,_=make(tmp_path,ages=(2,1,1))
    v28=json.loads(next((out28/"decisions").glob("*.json")).read_text())
    assert v28["eligible_four_team_pair_count"]==3
    assert v28["policy"]=="FROZEN"
    assert v28["quality_selection"]["fixture_keys"]==v28["robustness_control"]["fixture_keys"]
    later=freeze(root=root,v28_out=out28,out=out29,now=AT+timedelta(minutes=1))
    assert len(later["created"])==1
    artifact=json.loads(next((out29/"decisions").glob("*.json")).read_text())
    assert artifact["alternative"]["fixture_keys"]!=v28["quality_selection"]["fixture_keys"]
