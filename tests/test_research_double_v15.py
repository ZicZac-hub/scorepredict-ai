"""V15 deep multi-AI pair acceptance, no-future-leakage, and result grading."""
import json
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from pathlib import Path
from research_double_v15 import assemble_pair,freeze,evaluate,VERSION

NOW=datetime(2031,11,20,0,tzinfo=timezone.utc)
DAY="2031-11-20"
SCORES=[[1,1,.13],[1,0,.115],[2,1,.104],[0,0,.092],
        [2,0,.089],[0,1,.073],[1,2,.062],[2,2,.051],
        [3,1,.042],[0,2,.025]]


def leg(fid,home,away,hid,aid,hour):
    return {"provider":"API-Football","fixture_id":str(fid),
        "home_team":home,"away_team":away,
        "home_team_id":str(hid),"away_team_id":str(aid),
        "kickoff_utc":f"2031-11-20T0{hour}:00:00Z",
        "forecast_id":"forecast-"+str(fid),
        "model_key":"v8","model_version":"ziczac-v8"}


def setup(tmp_path):
    a=leg(101,"Team A","Team B",11,12,3)
    b=leg(102,"Team C","Team D",13,14,4)
    first=[{"score1":"1-1","score2":"1-1","joint_probability":.016},
           {"score1":"1-0","score2":"1-1","joint_probability":.014},
           {"score1":"1-1","score2":"1-0","joint_probability":.013},
           {"score1":"1-0","score2":"1-0","joint_probability":.012},
           {"score1":"2-1","score2":"1-1","joint_probability":.011}]
    frozen={"mode":"prospective","status":"frozen_pre_kickoff",
            "captured_at_utc":"2031-11-19T12:00:00Z",
            "sydney_date":DAY,"first":a,"second":b,
            "five_doubles":first}
    base=tmp_path/"base"
    d=base/"doubles";d.mkdir(parents=True)
    (d/(DAY+".json")).write_text(json.dumps(frozen))
    models={}
    for entry in (a,b):
        models[("API-Football",entry["fixture_id"],"v8")]={
            **entry,"top10_scores":deepcopy(SCORES),"model_version":"ziczac-v8",
            "input_digest":"frozen-digest-"+entry["fixture_id"],
            "captured_at_utc":"2031-11-19T11:00:00Z"}
    return frozen,models,base


def saved_vote(folder,entry,*,providers=("gemini","openai"),after=False):
    d=folder/"forecasts";d.mkdir(parents=True,exist_ok=True)
    stamp="2031-11-20T05:00:00Z" if after else "2031-11-19T23:00:00Z"
    row={**entry,"research_only":True,"captured_at_utc":stamp,
         "source_forecast_id":entry["forecast_id"],
         "input_digest":"frozen-digest-"+entry["fixture_id"],
         "votes":{name:{"score":[1,1],"home":.4,"draw":.3,"away":.3}
                  for name in providers}}
    (d/(entry["fixture_id"]+".json")).write_text(json.dumps(row))


def final_record(entry,home,away):
    return {**entry,"home_goals":home,"away_goals":away,
            "score_basis":"regulation","status":"FT",
            "evidence_version":2,
            "verified_at_utc":"2031-11-20T08:00:00Z"}


def test_v15_refuses_one_provider_or_missing_partner(tmp_path):
    original,models,base=setup(tmp_path)
    ar=tmp_path/"llm"
    saved_vote(ar,original["first"],providers=("gemini",))
    packet,reason=assemble_pair(original,models,ai_root=ar,now=NOW)
    assert packet is None and reason=="missing_original_earlier_ai_vote_for_both_matches"
    saved_vote(ar,original["second"])
    packet,reason=assemble_pair(original,models,ai_root=ar,now=NOW)
    assert packet is None and reason=="too_few_successful_ai_providers_per_match"


def test_v15_real_two_leg_votes_select_five_without_rewriting_baseline(tmp_path):
    original,models,base=setup(tmp_path)
    unchanged=deepcopy(original)
    ar=tmp_path/"llm"
    for entry in (original["first"],original["second"]):
        saved_vote(ar,entry)
    result,reason=assemble_pair(original,models,ai_root=ar,now=NOW)
    assert reason=="ready"
    assert result["model_version"]==VERSION
    assert result["original_predictions_modified"] is False
    assert result["ai_provider_names"]["first"]==["gemini","openai"]
    assert result["ai_provider_names"]["second"]==["gemini","openai"]
    assert result["proven_accuracy_improvement"] is False
    assert len(result["five_doubles"])==5
    assert len({(x["score1"],x["score2"]) for x in result["five_doubles"]})==5
    assert result["selected_statistical_coverage"]<=result["baseline_coverage"]+1e-12
    assert unchanged==original
    root=tmp_path/"v15"
    status=freeze(now=NOW,root=root,source_root=base,ai_root=ar,model_rows=models)
    assert status["frozen_new"]==1 and status["real_ai_requests_performed"]==0
    path=root/"forecasts"/(DAY+".json")
    archived=path.read_bytes()
    repeat=freeze(now=NOW,root=root,source_root=base,ai_root=ar,model_rows=models)
    assert repeat["frozen_new"]==0 and repeat["previously_frozen"]==1
    assert path.read_bytes()==archived
    finals={str(entry["fixture_id"]):final_record(entry,1,1)
            for entry in (original["first"],original["second"])}
    performance=evaluate(root=root,base_root=base,results=finals)
    assert performance["verified_pairs"]==1
    assert performance["rows"][0]["actual_scores"]==["1-1","1-1"]
    assert performance["ai_assisted_hits"] in (0,1)
    assert performance["verified_hit_rate"] in (0.,1.)
    assert len(performance["v15_win_rate_wilson_95"])==2
    assert performance["automatic_production_promotion"] is False
    assert performance["v15_only_wins"]+performance["v8_only_wins"]<=1


def test_v15_future_or_mismatched_ids_never_count_as_ai_votes(tmp_path):
    original,models,base=setup(tmp_path)
    ar=tmp_path/"llm"
    for entry in (original["first"],original["second"]):
        saved_vote(ar,entry)
    path=ar/"forecasts"/"102.json"
    changed=json.loads(path.read_text())
    changed["home_team_id"]="wrong"
    path.write_text(json.dumps(changed))
    result,reason=assemble_pair(original,models,ai_root=ar,now=NOW)
    assert result is None and reason=="missing_original_earlier_ai_vote_for_both_matches"
    saved_vote(ar,original["second"],after=True)
    result,_=assemble_pair(original,models,ai_root=ar,now=NOW)
    assert result is None
    assert assemble_pair(original,models,ai_root=ar,
                         now=NOW+timedelta(hours=4))[0] is None


def test_v15_does_not_mark_unverified_pairs_as_losses(tmp_path):
    original,models,base=setup(tmp_path)
    ar=tmp_path/"llm"
    for entry in (original["first"],original["second"]):
        saved_vote(ar,entry)
    root=tmp_path/"v15"
    freeze(now=NOW,root=root,source_root=base,ai_root=ar,model_rows=models)
    report=evaluate(root=root,base_root=base,results={})
    assert report["frozen_pairs"]==1
    assert report["pending_pairs"]==1
    assert report["verified_pairs"]==0
    assert report["verified_hit_rate"] is None


def test_v16_rejects_another_horizon_or_input_digest(tmp_path):
    original,models,_=setup(tmp_path)
    folder=tmp_path/"llm"
    for entry in (original["first"],original["second"]):
        saved_vote(folder,entry)
    changed=folder/"forecasts"/"101.json"
    row=json.loads(changed.read_text())
    row["source_forecast_id"]="other-forecast-for-same-fixture"
    changed.write_text(json.dumps(row))
    answer,reason=assemble_pair(original,models,ai_root=folder,now=NOW)
    assert answer is None
    assert reason=="missing_original_earlier_ai_vote_for_both_matches"
    row["source_forecast_id"]=original["first"]["forecast_id"]
    row["input_digest"]="digest-from-a-later-refreshed-model"
    changed.write_text(json.dumps(row))
    answer,reason=assemble_pair(original,models,ai_root=folder,now=NOW)
    assert answer is None
    assert reason=="missing_original_earlier_ai_vote_for_both_matches"


def test_v16_selects_earliest_complete_pre_match_panel(tmp_path):
    original,models,_=setup(tmp_path)
    folder=tmp_path/"llm"
    for entry in (original["first"],original["second"]):
        saved_vote(folder,entry)
    first=folder/"forecasts"/"101.json"
    complete=json.loads(first.read_text())
    early=dict(complete)
    early["captured_at_utc"]="2031-11-19T20:00:00Z"
    early["votes"]={"gemini":complete["votes"]["gemini"]}
    (first.parent/"101-partial-earlier.json").write_text(json.dumps(early))
    answer,reason=assemble_pair(original,models,ai_root=folder,now=NOW)
    assert reason=="ready" and answer is not None
    assert answer["ai_source_capture_utc"]["first"]=="2031-11-19T23:00:00Z"
    assert answer["source_forecast_ids"]["first"]==original["first"]["forecast_id"]
    assert answer["shared_ai_providers_across_both_legs"]==["gemini","openai"]


def test_v16_rejects_impossible_or_non_normalized_votes(tmp_path):
    original,models,_=setup(tmp_path)
    folder=tmp_path/"llm"
    for entry in (original["first"],original["second"]):
        saved_vote(folder,entry)
    path=folder/"forecasts"/"101.json"
    row=json.loads(path.read_text())
    row["votes"]["openai"]["home"]=.99
    row["votes"]["openai"]["draw"]=.99
    path.write_text(json.dumps(row))
    answer,reason=assemble_pair(original,models,ai_root=folder,now=NOW)
    assert answer is None and reason=="too_few_successful_ai_providers_per_match"
