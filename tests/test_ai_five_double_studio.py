"""Regression tests for the separate experimental AI Five-Double Studio."""
from datetime import datetime,timezone,timedelta
import copy
import json

import pytest

from ai_five_double_engine import (eligible,identity,merged_five,pair_check,
                                   endorsements)
from ai_five_double_ui import (options,analyze_pair,archive_rows,_csv_bytes)


NOW=datetime(2031,11,20,0,0,tzinfo=timezone.utc)
# 11 AM/2 PM Australia/Sydney on 20 Nov 2031, safely pre-match.
def fixture(fid,home,away,kick="2031-11-20T03:00:00Z",provider="API-Football",
            scores=None):
    scores=scores if scores is not None else [
        [1,1,.13],[1,0,.115],[2,1,.104],[0,0,.092],[2,0,.089],
        [0,1,.073],[1,2,.062],[2,2,.051],[3,1,.042],[0,2,.025]
    ]
    return {
        "fixture":{"id":fid,"provider":provider,"utcDate":kick,
                   "homeTeam":{"id":fid*2,"name":home},
                   "awayTeam":{"id":fid*2+1,"name":away},
                   "competition":{"name":"Test league"}},
        "kickoff":kick,"result":{
            "model_version":"existing-statistical-v8",
            "scores":scores,
            "home_xg":1.3,"away_xg":1.0,
            "outcome_probabilities":{"home":.44,"draw":.28,"away":.28}
        },
        "past":[],"league":"Test league"
    }


def pair():
    return fixture(101,"Alpha","Beta"),fixture(102,"Gamma","Delta")


def test_two_match_model_baseline_survives_no_external_ai():
    a,b=pair()
    report=merged_five(a,b,now=NOW)
    assert report["status"]=="ready"
    assert len(report["doubles"])==5
    assert [(p["Score 1"],p["Score 2"]) for p in report["doubles"]]==[
        (p["Score 1"],p["Score 2"]) for p in report["baseline"]]
    assert report["selected_model_coverage"]==pytest.approx(
        report["baseline_coverage"])
    assert report["ai_votes"]=={"1":0,"2":0}
    assert report["original_immutable_predictions_changed"] is False


def test_ai_votes_adjust_order_without_inventing_probabilities():
    a,b=pair()
    a["result"]["scores"][1][2]=.119
    base=merged_five(a,b,now=NOW)
    ai_a={"status":"completed","votes":{
        "openai":{"score":[1,0],"home":.49,"draw":.25,"away":.26},
        "gemini":{"score":[1,0],"home":.48,"draw":.25,"away":.27}}}
    ai_b={"status":"completed","votes":{
        "gemini":{"score":[1,1],"home":.4,"draw":.4,"away":.2}}}
    merged=merged_five(a,b,ai_a,ai_b,now=NOW)
    assert merged["ai_votes"]=={"1":2,"2":1}
    assert merged["status"]=="ready"
    assert merged["doubles"][0]["Score 1"]=="1-0"
    assert merged["doubles"][0]["Score 2"]=="1-1"
    assert merged["doubles"][0]["AI endorsements 1"]==2
    assert merged["doubles"][0]["Model joint"]==pytest.approx(.119*.13)
    assert merged["doubles"][0]["Model joint"]<=1
    assert merged["selected_model_coverage"]<=1
    assert merged["baseline_coverage"]>=merged["selected_model_coverage"]
    assert "observed" in str(merged["estimated_coverage_is_not_observed_win_rate"]) or (
        merged["estimated_coverage_is_not_observed_win_rate"] is True)


def test_unknown_ai_score_never_becomes_fake_statistical_probability():
    a,b=pair()
    opinion={"status":"completed","votes":{
        "grok":{"score":[9,9],"home":.4,"draw":.3,"away":.3}}}
    merged=merged_five(a,b,opinion,now=NOW)
    assert all(p["Score 1"]!="9-9" for p in merged["doubles"])
    assert merged["ai_votes"]["1"]==1
    assert all(p["AI endorsements 1"]==0 for p in merged["doubles"])


def test_same_sydney_day_required_including_utc_midnight():
    a,b=pair()
    a["kickoff"]="2031-11-19T23:30:00Z"
    a["fixture"]["utcDate"]=a["kickoff"]
    b["kickoff"]="2031-11-20T03:00:00Z"
    b["fixture"]["utcDate"]=b["kickoff"]
    earlier=datetime(2031,11,19,20,tzinfo=timezone.utc)
    assert pair_check(a,b,now=earlier) is None  # Sydney 20 Nov
    b["kickoff"]="2031-11-20T13:30:00Z"
    b["fixture"]["utcDate"]=b["kickoff"]
    assert "same Sydney" in pair_check(a,b,now=earlier)


def test_no_duplicate_fixture_or_shared_team():
    a,b=pair()
    assert "same fixture" in pair_check(a,a,now=NOW) or "different matches" in pair_check(a,a,now=NOW)
    b["fixture"]["homeTeam"]["name"]="Alpha"
    assert "four different teams" in pair_check(a,b,now=NOW)


def test_match_must_be_pre_kickoff_with_real_scores():
    a,b=pair()
    assert eligible(a,now=NOW)
    assert not eligible(a,now=datetime(2031,11,20,2,57,tzinfo=timezone.utc))
    b["result"]["scores"]=[[0,0,.4]]
    assert "at least five" in pair_check(a,b,now=NOW)


def test_saved_archive_fallback_returns_immutable_original_forecast(monkeypatch):
    from v10_prospective import read_latest
    import v10_prospective
    saved={
        "provider":"API-Football", "fixture_id":"501", "model_key":"v8",
        "model_version":"research-v8-frozen",
        "home_team":"Lions", "away_team":"Tigers",
        "home_team_id":"10", "away_team_id":"11",
        "kickoff_utc":"2031-11-20T03:00:00Z",
        "captured_at_utc":"2031-11-19T00:00:00Z",
        "top10_scores":fixture(501,"Lions","Tigers")["result"]["scores"],
        "competition":"PL","history_matches":35,
        "expected_goals":{"home":1.1,"away":1.2}
    }
    monkeypatch.setattr(v10_prospective,"read_latest",lambda root:{
        ("API-Football","501","v8"):saved
    })
    out=archive_rows(now=NOW)
    assert len(out)==1
    assert out[0]["archive_frozen"]
    assert out[0]["frozen_at_utc"]=="2031-11-19T00:00:00+00:00"
    current=fixture(501,"Lions","Tigers")
    assert len(options([current],now=NOW,archived=out))==1


def test_analyze_pair_uses_only_matching_saved_provider_votes_without_live_calls(monkeypatch):
    import ai_five_double_ui as ui
    def no_paid(*args,**kwargs):
        raise AssertionError("Never call paid APIs without explicit consent")
    monkeypatch.setattr(ui,"run_selected",no_paid)
    first,second=pair()
    def archive(f):
        if f["id"]==101:
            return {"research_only":True,
                    "votes":{"gemini":{"score":[1,1],"home":.4,"draw":.3,"away":.3}}}
        return None
    report=analyze_pair(first,second,{},now=NOW,archiver=archive,
                        configs={},use_live=False)
    assert report["status"]=="ready"
    assert report["ai_research"][0]["votes"]["gemini"]["score"]==[1,1]
    assert report["model_and_ai_result"]["ai_votes"]=={"1":1,"2":0}
    assert report["session_only"] is True
    assert report["independent_prospective_accuracy_record"] is False
    assert b"Match 1 exact score" in _csv_bytes(report["model_and_ai_result"])


def test_invalid_after_kickoff_pair_does_not_read_ai_sources():
    first,second=pair()
    def prohibit(_):
        raise AssertionError("Cannot retrieve AI for an invalid pair")
    report=analyze_pair(first,second,{},now=NOW+timedelta(days=5),archiver=prohibit)
    assert report["status"]=="invalid_pair"


def test_balanced_challenger_does_not_rewrite_or_invent_scores():
    first,second=pair()
    unchanged=(copy.deepcopy(first),copy.deepcopy(second))
    original=merged_five(first,second,now=NOW)
    balanced=merged_five(first,second,now=NOW,
                         selection_mode="balanced_challenger")
    assert original["status"]==balanced["status"]=="ready"
    assert balanced["selection_mode"]=="balanced_challenger"
    assert balanced["diversity_is_experimental"] is True
    assert balanced["no_promotion_or_archive_rewrite"] is True
    assert len(balanced["doubles"])==5
    options_1={f"{h}-{a}" for h,a,_ in first["result"]["scores"]}
    options_2={f"{h}-{a}" for h,a,_ in second["result"]["scores"]}
    assert all(p["Score 1"] in options_1 and p["Score 2"] in options_2
               for p in balanced["doubles"])
    assert len({(p["Score 1"],p["Score 2"]) for p in balanced["doubles"]})==5
    assert balanced["selected_diversity"]["match_1_scores"] >= original["baseline_diversity"]["match_1_scores"]
    assert balanced["selected_diversity"]["match_2_scores"] >= original["baseline_diversity"]["match_2_scores"]
    assert balanced["selected_diversity"] != original["baseline_diversity"]
    assert balanced["selected_model_coverage"] <= original["baseline_coverage"]+1e-12
    assert first==unchanged[0] and second==unchanged[1]


def test_challenger_stays_session_only_and_baseline_is_default():
    first,second=pair()
    baseline=analyze_pair(first,second,{},now=NOW,archiver=lambda fixture:None)
    challenger=analyze_pair(first,second,{},now=NOW,archiver=lambda fixture:None,
                            selection_mode="balanced_challenger")
    assert baseline["model_and_ai_result"]["selection_mode"]=="probability"
    assert challenger["model_and_ai_result"]["selection_mode"]=="balanced_challenger"
    assert challenger["session_only"] is True
    assert challenger["original_forecasts_overwritten"] is False
    assert challenger["independent_prospective_accuracy_record"] is False
    assert b"Match 1 exact score" in _csv_bytes(challenger["model_and_ai_result"])


def test_invalid_selector_rejected_without_creating_guesses():
    first,second=pair()
    bad=merged_five(first,second,now=NOW,selection_mode="unvalidated_magic")
    assert bad["status"]=="invalid_selection_mode"
    assert bad["doubles"]==[]
