"""Real saved-source audit: no fake provider votes or wrong source panels."""
import json

from v15_real_source_audit import audit


def _write(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def _leg(fid,home,away):
    return {"provider":"API-Football","fixture_id":str(fid),
            "home_team_id":str(home),"away_team_id":str(away),
            "kickoff_utc":"2031-11-20T03:00:00Z",
            "forecast_id":"forecast-"+str(fid)}


def _setup(tmp_path):
    base=tmp_path/"base"; ai=tmp_path/"llm"
    _write(ai/"status.json",{
        "configured_providers":["openai","gemini","claude"],"attempted":0,"captured":0})
    _write(ai/"evaluation.json",{"complete_panels_with_verified_results":0,
                                  "ten_provider_complete_verified":0})
    _write(base/"performance.json",{"five_doubles":{
        "verified_pairs":1,"wins":0,"awaiting_both_results":1}})
    first=_leg(10,11,12);second=_leg(20,13,14)
    _write(base/"doubles"/"2031-11-20.json",{
        "mode":"prospective","status":"frozen_pre_kickoff",
        "captured_at_utc":"2031-11-19T12:00:00Z",
        "sydney_date":"2031-11-20","first":first,"second":second})
    for leg in (first,second):
        _write(base/"forecasts"/"API-Football"/leg["fixture_id"]/
               (leg["forecast_id"]+".json"),{
                   **leg,"captured_at_utc":"2031-11-19T11:00:00Z",
                   "input_digest":"digest-"+leg["fixture_id"]})
    return base,ai,first,second


def _vote(leg):
    return {**leg,"research_only":True,
            "source_forecast_id":leg["forecast_id"],
            "input_digest":"digest-"+leg["fixture_id"],
            "captured_at_utc":"2031-11-19T22:00:00Z",
            "votes":{
                "gemini":{"score":[1,1],"home":.4,"draw":.3,"away":.3},
                "openai":{"score":[2,1],"home":.5,"draw":.2,"away":.3}}}


def test_only_genuine_double_leg_responses_qualify(tmp_path):
    base,ai,first,second=_setup(tmp_path)
    for leg in (first,second):
        _write(ai/"forecasts"/(leg["fixture_id"]+".json"),_vote(leg))
    report=audit(base,ai)
    assert report["configured_provider_count"]==3
    assert report["saved_valid_individual_ai_fixture_records"]==2
    assert report["actual_successful_provider_counts_by_name"]=={
        "gemini":2,"openai":2}
    assert report["original_sets_with_at_least_two_ai_votes_on_both_legs"]==1
    assert report["source_rows"][0]["actual_provider_votes_per_leg"]==[2,2]
    assert report["source_rows"][0]["shared_provider_names"]==["gemini","openai"]
    assert report["v15_real_ai_double_accuracy"] is None
    assert report["v15_approval_for_production"] is False
    # Replayed same-provider packet isn't an additional independent fixture.
    _write(ai/"forecasts"/"replay.json",_vote(first))
    assert audit(base,ai)["actual_successful_provider_counts_by_name"]=={
        "gemini":2,"openai":2}


def test_wrong_frozen_forecast_or_input_digest_is_not_ai_consensus(tmp_path):
    base,ai,first,second=_setup(tmp_path)
    row=_vote(first)
    row["source_forecast_id"]="a-different-v8-horizon"
    _write(ai/"forecasts"/"wrong-source.json",row)
    row=_vote(first)
    row["input_digest"]="refreshed-after-original-frozen"
    _write(ai/"forecasts"/"wrong-digest.json",row)
    _write(ai/"forecasts"/"second.json",_vote(second))
    report=audit(base,ai)
    assert report["original_sets_with_at_least_two_ai_votes_on_both_legs"]==0
    assert report["source_rows"][0]["actual_provider_votes_per_leg"]==[0,2]
    assert report["source_rows"][0]["leg_reasons"][0]=="no_source_matched_pre_kickoff_ai_panel"


def test_malformed_votes_and_after_kickoff_are_not_counted(tmp_path):
    base,ai,first,second=_setup(tmp_path)
    row=_vote(first)
    row["votes"]["openai"]["home"]=1.1
    _write(ai/"forecasts"/"invalid.json",row)
    row=_vote(second)
    row["captured_at_utc"]="2031-11-20T04:00:00Z"
    _write(ai/"forecasts"/"late.json",row)
    report=audit(base,ai)
    assert report["saved_valid_individual_ai_fixture_records"]==1
    assert report["actual_successful_provider_counts_by_name"]=={"gemini":1}
    assert report["original_sets_with_at_least_two_ai_votes_on_both_legs"]==0
    assert report["source_rows"][0]["actual_provider_votes_per_leg"]==[1,0]
