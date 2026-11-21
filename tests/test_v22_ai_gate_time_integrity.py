"""AI credibility counts only available, non-repeated real-event trials."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json

import pytest

import v22_ai_gate as gate

NOW=datetime(2034,6,1,12,tzinfo=timezone.utc)


def trial(number=1,*,provider="API-Football",kickoff=None):
    at=kickoff or NOW-timedelta(days=1)
    source={"provider":provider,"fixture_id":str(number),
            "forecast_id":"original-"+str(number),"input_digest":"source-"+str(number),
            "home_team_id":"11","away_team_id":"22",
            "home_team":"Home Club","away_team":"Away Club",
            "kickoff_utc":at.isoformat(),"captured_at_utc":(at-timedelta(hours=5)).isoformat(),
            "outcome_probabilities":{"home":.46,"draw":.24,"away":.30}}
    panel={key:source[key] for key in ("provider","fixture_id","home_team_id",
                                     "away_team_id","kickoff_utc")}
    panel.update(source_forecast_id=source["forecast_id"],input_digest=source["input_digest"],
                 captured_at_utc=(at-timedelta(hours=2)).isoformat(),
                 votes={"gemini":{"home":.8,"draw":.1,"away":.1}})
    final={key:source[key] for key in ("provider","fixture_id","home_team_id",
                                     "away_team_id","home_team","away_team","kickoff_utc")}
    final.update(evidence_version=2,score_basis="regulation",status="FT",home_goals=2,away_goals=1,
                 verified_at_utc=(at+timedelta(hours=4)).isoformat())
    return panel,source,final


def evaluate_one(panel,source,final,**kwargs):
    return gate.evaluate([panel],[source],{source["fixture_id"]:final},as_of=NOW,**kwargs)


def test_future_final_is_not_available_even_when_label_and_identity_match():
    panel,source,final=trial()
    final["verified_at_utc"]=(NOW+timedelta(seconds=1)).isoformat()
    before=evaluate_one(panel,source,final)
    assert before["provider_metrics"]=={}
    assert before["excluded"]["unmatched_identity_result_or_time"]==1
    after=gate.evaluate([panel],[source],{source["fixture_id"]:final},
                        as_of=NOW+timedelta(minutes=1))
    assert after["provider_metrics"]["gemini"]["source_matched_verified_fixtures"]==1
    assert after["provider_metrics"]["gemini"]["production_weight"]==0


def test_future_original_and_panel_cannot_be_counted_by_backfilled_final():
    panel,source,final=trial(kickoff=NOW+timedelta(days=1))
    assert evaluate_one(panel,source,final)["provider_metrics"]=={}
    assert gate.evaluate([panel],[source],{source["fixture_id"]:final},
                         as_of=NOW+timedelta(days=2))["provider_metrics"]


@pytest.mark.parametrize("status",["LIVE","HT","CANCELLED",None])
def test_nonterminal_or_unavailable_result_never_becomes_verified_ai_trial(status):
    panel,source,final=trial()
    final["status"]=status
    assert evaluate_one(panel,source,final)["provider_metrics"]=={}


def test_fixture_id_aliases_cannot_turn_one_match_into_one_hundred_trials():
    panels,sources,finals=[],[],{}
    for number in range(100):
        panel,source,final=trial(number)
        # Different source IDs do not make identical normalized teams/kickoff new events.
        source["home_team"]="  HOME   club "
        source["away_team"]="Away CLUB"
        panels.append(panel);sources.append(source);finals[str(number)]=final
    report=gate.evaluate(panels,sources,finals,as_of=NOW)
    measured=report["provider_metrics"]["gemini"]
    assert measured["source_matched_verified_fixtures"]==1
    assert measured["passes_statistical_shadow_gate"] is False
    assert report["excluded"]["repeated_provider_fixture_or_event"]==99
    assert report["fixture_reuse_in_provider_denominator_allowed"] is False
    assert report["independence_of_distinct_events_not_established"] is True


def test_each_ai_provider_keeps_its_own_non_repeated_event_cohort():
    panel,source,final=trial()
    repeated=deepcopy(panel)
    repeated["votes"]["claude"]={"home":.7,"draw":.2,"away":.1}
    report=gate.evaluate([panel,repeated],[source],{source["fixture_id"]:final},as_of=NOW)
    assert {key:row["source_matched_verified_fixtures"]
            for key,row in report["provider_metrics"].items()}=={"claude":1,"gemini":1}


def test_same_teams_on_distinct_days_remain_distinct_future_trials():
    rows=[trial(n,kickoff=NOW-timedelta(days=n+1)) for n in range(2)]
    report=gate.evaluate([r[0] for r in rows],[r[1] for r in rows],
                         {r[1]["fixture_id"]:r[2] for r in rows},as_of=NOW)
    assert report["provider_metrics"]["gemini"]["source_matched_verified_fixtures"]==2


def test_first_vote_uses_actual_utc_order_across_offsets_and_invalid_times():
    early,source,final=trial()
    kickoff=gate.utc(source["kickoff_utc"])
    early["captured_at_utc"]=(kickoff-timedelta(hours=2)).isoformat()
    early["votes"]["gemini"]={"home":.2,"draw":.4,"away":.4}
    later=deepcopy(early)
    # This later UTC observation sorts before the earlier one as raw text.
    later["captured_at_utc"]=(kickoff-timedelta(hours=1)).astimezone(
        timezone(timedelta(hours=-12))).isoformat()
    later["votes"]["gemini"]={"home":.8,"draw":.1,"away":.1}
    invalid=deepcopy(later)
    invalid["captured_at_utc"]="invalid timestamp"
    expected=evaluate_one(early,source,final)
    report=gate.evaluate([later,invalid,early],[source],
                         {source["fixture_id"]:final},as_of=NOW)
    assert report["provider_metrics"]==expected["provider_metrics"]
    assert report["excluded"]=={"repeated_provider_fixture_or_event":1,
                                 "unmatched_identity_result_or_time":1}


def test_missing_event_names_and_boolean_probabilities_are_not_valid_evidence():
    panel,source,final=trial()
    source.pop("home_team")
    assert evaluate_one(panel,source,final)["excluded"]=={"event_identity_missing":1}
    panel,source,final=trial()
    panel["votes"]["gemini"]={"home":True,"draw":False,"away":False}
    result=evaluate_one(panel,source,final)
    assert result["provider_metrics"]=={}
    assert result["excluded"]=={"invalid_provider_vote":1}


def test_malformed_panel_or_vote_cannot_crash_report():
    panel,source,final=trial()
    panel["votes"]=["not a provider mapping"]
    report=gate.evaluate([None,[],panel],[source,None],{source["fixture_id"]:final},as_of=NOW)
    assert report["provider_metrics"]=={}
    assert report["excluded"]=={"invalid_provider_votes":1}


def test_main_uses_requested_evaluation_cutoff_without_relabelling_future_results(monkeypatch,tmp_path):
    panel,source,final=trial()
    final["verified_at_utc"]=(NOW+timedelta(seconds=1)).isoformat()
    panels=tmp_path/"panels";originals=tmp_path/"originals"
    (panels/"forecasts").mkdir(parents=True);originals.mkdir()
    (panels/"forecasts"/"panel.json").write_text(json.dumps(panel))
    (originals/"verified_results.json").write_text(json.dumps({source["fixture_id"]:final}))
    monkeypatch.setattr(gate,"read_latest",lambda root:{"source":source})
    out=tmp_path/"ai_status.json"
    assert gate.main(panel_root=panels,original_root=originals,out=out,as_of=NOW)=={}
    report=json.loads(out.read_text())
    assert report["evaluated_at_utc"]==NOW.isoformat()
    assert report["saved_provider_panel_count"]==1
