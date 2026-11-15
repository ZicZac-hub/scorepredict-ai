"""Real V15 AI source audit counts actual valid provider votes, not keys."""
import json
from v15_real_source_audit import audit


def test_only_real_two_leg_answers_qualify_and_provider_counts_are_integers(tmp_path):
    base=tmp_path/"base"; ai=tmp_path/"llm"
    (base/"doubles").mkdir(parents=True)
    (ai/"forecasts").mkdir(parents=True)
    (ai/"status.json").write_text(json.dumps({
        "configured_providers":["openai","gemini","claude"],
        "attempted":0,"captured":0}))
    (ai/"evaluation.json").write_text(json.dumps({
        "complete_panels_with_verified_results":0,
        "ten_provider_complete_verified":0}))
    (base/"performance.json").write_text(json.dumps({
        "five_doubles":{"verified_pairs":1,"wins":0,
                        "awaiting_both_results":1}}))
    pair={"mode":"prospective","status":"frozen_pre_kickoff",
          "first":{"provider":"API-Football","fixture_id":"10",
                   "kickoff_utc":"2031-11-20T03:00:00Z"},
          "second":{"provider":"API-Football","fixture_id":"20",
                    "kickoff_utc":"2031-11-20T04:00:00Z"},
          "sydney_date":"2031-11-20"}
    (base/"doubles"/"2031-11-20.json").write_text(json.dumps(pair))
    for fid,kick in (("10","2031-11-20T03:00:00Z"),
                     ("20","2031-11-20T04:00:00Z")):
        row={"research_only":True,"provider":"API-Football",
             "fixture_id":fid,"kickoff_utc":kick,
             "captured_at_utc":"2031-11-19T22:00:00Z",
             "votes":{"gemini":{"score":[1,1]},
                      "openai":{"score":[2,1]}}}
        (ai/"forecasts"/(fid+".json")).write_text(json.dumps(row))
    result=audit(base,ai)
    assert result["configured_provider_count"]==3
    assert result["saved_valid_individual_ai_fixture_records"]==2
    assert result["actual_successful_provider_counts_by_name"]=={
        "gemini":2,"openai":2}
    assert result["original_sets_with_at_least_two_ai_votes_on_both_legs"]==1
    assert result["v15_real_ai_double_accuracy"] is None
    assert result["v15_approval_for_production"] is False
