"""Regression tests for source-verified, research-only professional analyst gate."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from professional_analyst_shadow_v25 import propose, run, _pair, _eligible

NOW = datetime(2035, 2, 10, 10, tzinfo=timezone.utc)


def forecast(fid, lead=3, *, captured=None, home=None, away=None):
    left=np.array([.18,.27,.26,.16,.08,.035,.012,.003])
    right=np.array([.29,.36,.22,.09,.03,.008,.001,.001])
    matrix=np.outer(left,right)
    matrix/=matrix.sum()
    recorded=captured or (NOW-timedelta(minutes=10))
    return {
        "provider":"API-Football", "fixture_id":str(fid),
        "forecast_id":f"original-v8-{fid}",
        "home_team_id":str(home or fid*2),
        "away_team_id":str(away or fid*2+1),
        "home_team":f"Home {fid}", "away_team":f"Away {fid}",
        "model_key":"v8", "model_version":"test-v8",
        "mode":"prospective", "score_basis":"regulation",
        "score_matrix":matrix.tolist(), "input_digest":"frozen-evidence",
        "captured_at_utc":recorded.isoformat(),
        "evidence_as_of_utc":(recorded-timedelta(minutes=2)).isoformat(),
        "kickoff_utc":(NOW+timedelta(hours=lead)).isoformat(),
        "history_matches":150,
    }


def results(row, home=1, away=0):
    kickoff=datetime.fromisoformat(row["kickoff_utc"])
    return {
        "provider":row["provider"],"fixture_id":row["fixture_id"],
        "home_team_id":row["home_team_id"],"away_team_id":row["away_team_id"],
        "kickoff_utc":row["kickoff_utc"],"status":"FT",
        "evidence_version":2,"score_basis":"regulation",
        "verified_at_utc":(kickoff+timedelta(hours=3)).isoformat(),
        "home_goals":home,"away_goals":away,
    }


def test_pro_analyst_selects_originals_and_robust_alternatives_without_mutation():
    first,second=forecast(100),forecast(200,lead=4)
    orig=json.dumps([first,second],sort_keys=True)
    packet=propose([first,second],NOW,min_envelope=0)
    sets=[v for v in packet["days"].values() if v["decision"]=="FROZEN_RESEARCH_ONLY"]
    assert len(sets)==1
    chosen=sets[0]
    assert chosen["model_version"].startswith("pro-analyst-v25")
    assert chosen["first"]["forecast_id"]=="original-v8-100"
    assert len(chosen["five_doubles"]) == len(chosen["baseline_five"]) == 5
    assert len({(p["score1"],p["score2"]) for p in chosen["five_doubles"]})==5
    assert chosen["robust_lower_envelope_model_mass"] <= chosen["v8_optimal_five_model_mass"]
    assert chosen["robust_five_v8_model_mass"] <= chosen["v8_optimal_five_model_mass"]+1e-10
    assert chosen["lower_envelope_not_a_calibrated_probability"] is True
    assert json.dumps([first,second],sort_keys=True)==orig


def test_reject_late_missing_history_and_shared_teams():
    late=forecast(300,captured=NOW+timedelta(hours=5))
    assert _eligible(late,NOW)[1]=="invalid_source_time"
    thin=forecast(301)
    thin["history_matches"]=4
    assert _eligible(thin,NOW)[1]=="insufficient_historical_fixtures"
    a=forecast(100)
    b=forecast(200,lead=4,home=int(a["away_team_id"]))
    output=propose([a,b],NOW,min_envelope=0)
    assert not any(p["decision"]=="FROZEN_RESEARCH_ONLY" for p in output["days"].values())


def test_reject_non_normalized_matrix_and_wrong_score_basis():
    bad=forecast(100)
    bad["score_matrix"][0][0]=-0.1
    assert _eligible(bad,NOW)[1]=="invalid_probability_grid"
    bad=forecast(100)
    bad["score_basis"]="extra_time"
    assert _eligible(bad,NOW)[1]=="missing_verified_pre_match_origin"


def test_write_once_and_settle_two_legs_using_identity_matched_result_files(tmp_path):
    first,second=forecast(100),forecast(200,lead=4)
    original=tmp_path/"original"
    prematch=tmp_path/"prematch"
    prematch.mkdir()
    out=tmp_path/"research"
    initial=run(root=original, prematch_root=prematch,out=out,now=NOW,
                forecasts=[first,second],min_envelope=0)
    assert initial["total_frozen_sets"] == 1
    assert initial["verified_sets"] == 0
    assert initial["pending_sets"] == 1
    path=next((out/"sets").glob("*.json"))
    untouched=path.read_bytes()
    packet=json.loads(untouched)
    one,two=packet["five_doubles"][0].values()
    # Selection contains two string scores, not a result or post-kickoff feature.
    goals1=tuple(map(int,packet["five_doubles"][0]["score1"].split("-")))
    goals2=tuple(map(int,packet["five_doubles"][0]["score2"].split("-")))
    actuals={first["fixture_id"]:results(first,*goals1),
             second["fixture_id"]:results(second,*goals2)}
    (prematch/"verified_results.json").write_text(json.dumps(actuals))
    later=NOW+timedelta(hours=9)
    after=run(root=original,prematch_root=prematch,out=out,now=later,
              forecasts=[first,second],min_envelope=0)
    assert after["total_frozen_sets"]==1
    assert after["verified_sets"]==1
    assert after["robust_wins"]==1
    assert after["outcomes"][0]["actual_scores"]==[
        packet["five_doubles"][0]["score1"],packet["five_doubles"][0]["score2"]]
    assert path.read_bytes()==untouched


def test_identity_mismatch_and_conflicting_scores_remain_pending(tmp_path):
    first,second=forecast(100),forecast(200,lead=4)
    original=tmp_path/"original"
    prematch=tmp_path/"prematch"
    original.mkdir()
    prematch.mkdir()
    out=tmp_path/"shadow"
    run(root=original,prematch_root=prematch,out=out,now=NOW,
        forecasts=[first,second],min_envelope=0)
    a,b=results(first),results(second)
    b["away_team_id"]="999"
    (original/"verified_results.json").write_text(json.dumps({"100":a,"200":b}))
    out1=run(root=original,prematch_root=prematch,out=out,
             now=NOW+timedelta(hours=9),forecasts=[first,second],min_envelope=0)
    assert out1["verified_sets"]==0
    b["away_team_id"]=second["away_team_id"]
    (original/"verified_results.json").write_text(json.dumps({"100":a,"200":b}))
    alternative=dict(b,home_goals=b["home_goals"]+1)
    (prematch/"verified_results.json").write_text(json.dumps({"200":alternative}))
    out2=run(root=original,prematch_root=prematch,out=out,
             now=NOW+timedelta(hours=9),forecasts=[first,second],min_envelope=0)
    assert out2["conflicting_verified_results_excluded"]==1
    assert out2["verified_sets"]==0

def test_future_verified_result_is_not_visible_before_its_timestamp(tmp_path):
    first,second=forecast(100),forecast(200,lead=4)
    root=tmp_path/"original"
    root.mkdir()
    out=tmp_path/"shadow"
    prematch=tmp_path/"prematch"
    prematch.mkdir()
    (root/"verified_results.json").write_text(
        json.dumps({"100":results(first),"200":results(second)}))
    # Source file already contains finals but audit time precedes verification:
    # never introduce look-ahead just because a future test row exists.
    early=run(root=root, prematch_root=prematch,out=out,now=NOW,
              forecasts=[first,second],min_envelope=0)
    assert early["verified_sets"]==0
    assert early["pending_sets"]==1
    settled=run(root=root,prematch_root=prematch,out=out,
                now=NOW+timedelta(hours=9),forecasts=[first,second],
                min_envelope=0)
    assert settled["verified_sets"]==1

def test_human_style_notes_only_restate_available_evidence_and_stale_abstains():
    recent=forecast(100)
    recent["research_context"]={
        "home":{"last10_observed":10,"rest_days_since_last_kickoff":6.5},
        "away":{"last10_observed":8,"rest_days_since_last_kickoff":2},
        "verified_xg_xga":{"available":False},
        "verified_lineup_adjustments":{"applied":False}}
    second=forecast(200,lead=4)
    selected=next(iter(propose([recent,second],NOW,min_envelope=0)["days"].values()))
    memo=selected["first"]["human_research_note"]
    assert memo["home_recent_matches_observed"]==10
    assert memo["away_recent_matches_observed"]==8
    assert memo["home_rest_days"]==6.5
    assert memo["away_rest_days"]==2
    assert memo["genuine_shot_based_xg_available"] is False
    assert memo["validated_lineup_adjustment_applied"] is False
    assert memo["numerical_context_weight"]==0
    old=forecast(300,captured=NOW-timedelta(days=5))
    assert _eligible(old,NOW)[1]=="stale_pre_match_source_evidence"
