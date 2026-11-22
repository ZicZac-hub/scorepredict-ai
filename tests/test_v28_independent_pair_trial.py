"""V28 genuine independent pair-selection proof-of-freeze and results audit."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from forecast_ledger import forecast_id
from v28_independent_pair_trial import (
    VERSION, _discordance_p, eligible_slate, plan, freeze, evaluate, run,
)

SELECT = datetime(2035, 2, 21, 9, tzinfo=timezone.utc)
GRID = np.outer([.20, .32, .25, .15, .08],
                [.31, .37, .19, .10, .03]).tolist()


def match(fid, *, age_hours=2, capture=None, lead_hours=6):
    captured = (SELECT - timedelta(hours=age_hours)
                if capture is None else capture)
    data = {
        "provider":"API-Football", "fixture_id":str(fid),
        "model_key":"v8", "model_version":"archived-v8",
        "mode":"prospective", "score_basis":"regulation",
        "home_team_id":str(fid*2), "away_team_id":str(fid*2+1),
        "home_team":f"Home {fid}", "away_team":f"Away {fid}",
        "kickoff_utc":(SELECT+timedelta(
            hours=lead_hours, minutes=(fid-1)*8)).isoformat(),
        "captured_at_utc":captured.isoformat(),
        "evidence_as_of_utc":(captured-timedelta(minutes=1)).isoformat(),
        "history_matches":55,
        "input_digest":f"fixed-authentic-evidence-{fid}",
        "score_matrix":GRID,
    }
    data["forecast_id"]=forecast_id(data)
    return data


def result(row, score, *, verified_at=None):
    kickoff=datetime.fromisoformat(row["kickoff_utc"])
    return {
        **{k:row[k] for k in (
            "provider","fixture_id","home_team_id","away_team_id",
            "kickoff_utc")},
        "status":"FT","evidence_version":2,"score_basis":"regulation",
        "verified_at_utc":(verified_at or
                           kickoff+timedelta(hours=3)).isoformat(),
        "home_goals":score[0],"away_goals":score[1],
    }


def sample(tmp, ages=(110,100,2,1), *, lead_hours=6):
    root=tmp/"source"
    prematch=tmp/"prematch"
    prematch.mkdir()
    out=tmp/"v28"
    rows=[match(i, age_hours=age, lead_hours=lead_hours)
          for i,age in enumerate(ages,1)]
    for row in rows:
        file=root/"forecasts"/row["provider"]/row["fixture_id"]/(row["forecast_id"]+".json")
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(row))
    return root,prematch,out,rows


def test_freeze_independently_of_any_v25_archives_and_verify_original_arms(tmp_path):
    root,prematch,out,models=sample(tmp_path)
    originals=[deepcopy(row) for row in models]
    response=freeze(root=root,out=out,now=SELECT)
    assert len(response["created"])==1
    assert response["created"][0]["policy"]=="FROZEN"
    file=next((out/"decisions").glob("*.json"))
    stored=file.read_bytes()
    packet=json.loads(stored)
    assert packet["version"]==VERSION
    assert packet["eligible_fixture_count"]==4
    assert packet["eligible_four_team_pair_count"]==6
    assert packet["computed_pairs"]==6
    assert len(packet["candidate_catalog"])==6
    assert packet["all_eligible_pair_candidates_exhaustively_checked"] is True
    assert packet["quality_selection"]["fixture_keys"]==[
        ["API-Football","3"],["API-Football","4"]]
    assert packet["robustness_control"]["fixture_keys"]==[
        ["API-Football","1"],["API-Football","2"]]
    assert len(packet["quality_selection"]["five_original_v25_robust_scores"])==5
    assert packet["arms_use_identical_v25_robust_score_choice_policy"] is True
    assert packet["at_least_70_percent_accuracy_proven"] is False
    assert models==originals
    assert freeze(root=root,out=out,now=SELECT)["created"]==[]
    assert file.read_bytes()==stored
    pending=evaluate(root=root,prematch=prematch,out=out,now=SELECT)
    assert pending["original_frozen_decisions"]==1
    assert pending["verified_paired_days"]==0
    assert pending["pending_days"]==1

    chosen=packet["quality_selection"]
    option=chosen["five_original_v25_robust_scores"][0]
    finals={r["fixture_id"]:result(r,(0,0)) for r in models}
    for (provider,fid),score in zip(
        chosen["fixture_keys"],(option["score1"],option["score2"])):
        row=next(r for r in models if r["fixture_id"]==fid)
        finals[fid]=result(row,tuple(map(int,score.split("-"))))
    (prematch/"verified_results.json").write_text(json.dumps(finals))
    later=SELECT+timedelta(hours=14)
    report=evaluate(root=root,prematch=prematch,out=out,now=later)
    assert report["verified_paired_days"]==1
    assert report["quality_wins"]==1
    assert report["robustness_control_wins"]==0
    assert report["quality_only_wins"]==1
    assert report["control_only_wins"]==0
    assert report["exact_paired_test_two_sided_p"]==pytest.approx(1)
    assert report["different_pair_days"]==1
    assert report["seventy_percent_accuracy_demonstrated"] is False
    assert report["initial_review_ready"] is False
    assert report["excluded_invalid_or_tampered_files"]=={}
    assert file.read_bytes()==stored


def test_not_enough_fixtures_is_retryable_not_permanent_abstention(tmp_path):
    root,prematch,out,models=sample(tmp_path,ages=(2,1))
    reply=freeze(root=root,out=out,now=SELECT)
    assert reply["created"]==[]
    assert reply["skipped"]["insufficient_eligible_fixtures_retry_later"]==1
    assert not (out/"decisions").exists()
    new=match(3,age_hours=1)
    file=root/"forecasts"/new["provider"]/new["fixture_id"]/(new["forecast_id"]+".json")
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(new))
    next_response=freeze(root=root,out=out,now=SELECT+timedelta(minutes=5))
    assert len(next_response["created"])==1


def test_new_source_captured_after_cutoff_does_not_enter_candidate_slate(tmp_path):
    root,prematch,out,models=sample(tmp_path)
    groups,excluded=eligible_slate(models+[match(99,capture=SELECT+timedelta(
        minutes=10))],SELECT)
    assert sum(map(len,groups.values()))==4
    assert excluded["invalid_source_time"]>=1
    selection=plan(next(iter(groups.values())))
    assert selection["eligible_four_team_pair_count"]==6


def test_very_large_complete_slate_abstains_without_partial_rankings(tmp_path):
    root,prematch,out,models=sample(tmp_path)
    groups,_=eligible_slate(models,SELECT)
    answer=plan(next(iter(groups.values())),max_pairs=2)
    assert answer["policy"]=="ABSTAIN"
    assert answer["reason"]=="full_pair_slate_exceeds_budget"
    assert answer["eligible_four_team_pair_count"]==6
    assert answer["computed_pairs"]==0
    assert answer["candidate_catalog"]==[]


def test_tampered_selection_rejected_and_unverified_stays_pending(tmp_path):
    root,prematch,out,models=sample(tmp_path)
    freeze(root=root,out=out,now=SELECT)
    file=next((out/"decisions").glob("*.json"))
    original=file.read_bytes()
    bad=json.loads(original)
    bad["quality_selection"]["quality_rank_score"]=.999
    file.write_text(json.dumps(bad))
    invalid=evaluate(root=root,prematch=prematch,out=out,now=SELECT)
    assert invalid["original_frozen_decisions"]==0
    assert "frozen_slate_changed_quality_selection" in invalid[
        "excluded_invalid_or_tampered_files"]
    file.write_bytes(original)
    prematch_json={r["fixture_id"]:result(r,(1,1)) for r in models}
    # One source has a result verification timestamp that has not occurred.
    prematch_json["4"]["verified_at_utc"]=(
        SELECT+timedelta(days=2)).isoformat()
    (prematch/"verified_results.json").write_text(json.dumps(prematch_json))
    pending=evaluate(root=root,prematch=prematch,out=out,now=SELECT+timedelta(hours=14))
    assert pending["pending_days"]==1
    assert pending["verified_paired_days"]==0
    assert pending["quality_wins"]==0


def test_original_immutable_forecast_change_invalidates_saved_selection(tmp_path):
    root,prematch,out,models=sample(tmp_path)
    freeze(root=root,out=out,now=SELECT)
    source=models[2]
    file=root/"forecasts"/source["provider"]/source["fixture_id"]/(source["forecast_id"]+".json")
    changed=json.loads(file.read_text())
    changed["input_digest"]="tampered"
    file.write_text(json.dumps(changed))
    invalid=evaluate(root=root,prematch=prematch,out=out,now=SELECT)
    assert invalid["original_frozen_decisions"]==0
    assert invalid["excluded_invalid_or_tampered_files"]


def test_45minute_lead_and_forecast_window_enforced(tmp_path):
    root,prematch,out,models=sample(tmp_path,lead_hours=0.5)
    reply=freeze(root=root,out=out,now=SELECT)
    assert reply["created"]==[]
    assert reply["skipped"]["insufficient_eligible_fixtures_retry_later"]==1
    assert reply["source_ineligibility"].get("outside_research_freeze_window")==2


def test_statistical_significance_is_not_guaranteed_by_one_win():
    assert _discordance_p(0,0) is None
    assert _discordance_p(1,0)==pytest.approx(1)
    assert _discordance_p(2,1)==pytest.approx(1)
    assert _discordance_p(10,0)==pytest.approx(2/1024)
    assert _discordance_p(0,10)==pytest.approx(2/1024)


def test_readonly_run_does_not_issue_new_predictions(tmp_path):
    root,prematch,out,models=sample(tmp_path)
    out_data=run(root=root,prematch=prematch,out=out,now=SELECT,
                 verify_only=True)
    assert out_data["freeze"]["created"]==[]
    assert out_data["verified_comparison"]["original_frozen_decisions"]==0
    assert not (out/"decisions").exists()
    assert out_data["prospective_accuracy_improvement_not_established"] is True
