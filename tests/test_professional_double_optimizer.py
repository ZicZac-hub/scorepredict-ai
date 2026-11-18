from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from professional_double_optimizer import (
    POLICY, AUTO_SELECTION, optimal_five, manual, premium, calibrate, auto_shadow,
    calibration_trials, trials_from_verified_ledger, compare_score_prices,
)
from ziczac_interactive_capture import pair
from ziczac_unified_audit import private_records

NOW = datetime(2035, 11, 20, tzinfo=timezone.utc)


def row(number):
    p = np.array([.20, .30, .25, .15, .06, .025, .01, .005])
    grid = np.outer(p, p)
    scores = sorted([(h, a, float(v)) for h, line in enumerate(grid)
                     for a, v in enumerate(line)], key=lambda x: -x[2])[:10]
    kickoff = (NOW+timedelta(hours=5+number)).isoformat()
    return {"fixture": {"provider": "API-Football", "id": number, "utcDate": kickoff,
                        "homeTeam": {"id": number*2, "name": f"Home {number}"},
                        "awayTeam": {"id": number*2+1, "name": f"Away {number}"}},
            "kickoff": kickoff, "archive_frozen": True,
            "frozen_at_utc": (NOW-timedelta(hours=1)).isoformat(),
            "result": {"scores": scores, "matrix": grid.tolist(), "model_version": "v1"}}


def test_full_grid_probability_optimum_matches_exhaustive_cartesian_search():
    first = np.array([[.3, .1], [.4, .2]])
    second = np.array([[.1, .2, .15], [.3, .15, .1]])
    expected = sorted((float(p*q) for p in first.ravel() for q in second.ravel()), reverse=True)[:5]
    chosen = optimal_five(first, second)
    assert [r["joint_probability"] for r in chosen] == pytest.approx(expected)
    assert len({(r["score1"], r["score2"]) for r in chosen}) == 5


@pytest.mark.parametrize("bad", [[[float("nan")]], [[-1, 2]], [[.4]], [[1], [0, 0]]])
def test_invalid_grids_do_not_create_five_doubles(bad):
    with pytest.raises((ValueError, TypeError)):
        optimal_five(bad, [[.5, .5]])


def test_manual_is_source_preserving_future_same_day_and_probability_only():
    a, b = row(1), row(2)
    original = deepcopy((a, b))
    report = manual(a, b, now=NOW)
    assert report["status"] == "ready"
    assert len(report["choices"]) == 5
    assert report["estimated_coverage"] == pytest.approx(sum(p["joint_probability"] for p in report["choices"]))
    assert report["coverage_is_calibrated"] is False
    assert report["target_70_percent_proven"] is False
    assert report["realized_roi"] is None
    assert (a, b) == original
    assert manual(a, a, now=NOW)["status"] == "abstained"
    assert manual(a, b, now=NOW+timedelta(hours=7))["status"] == "abstained"
    a["dependence_warning"] = True
    assert manual(a, b, now=NOW)["reason"] == "validated_joint_dependence_model_required"


def test_premium_cannot_substitute_high_model_coverage_for_calibration_evidence():
    result = premium([row(1), row(2)], now=NOW)
    assert result["status"] == "abstained"
    assert result["selection"] is None
    assert result["eligible_pairs_compared"] == 1
    assert result["target_70_percent_proven"] is False
    old = row(3)
    old["frozen_at_utc"] = (NOW-timedelta(hours=25)).isoformat()
    assert premium([row(1), old], now=NOW)["qualified_fixtures"] == 1


def test_slow_model_computation_cannot_be_backdated_before_kickoff(monkeypatch):
    import professional_double_optimizer as optimizer
    at=[NOW]
    original=optimizer.optimal_five
    def delayed(*args):
        result=original(*args)
        at[0]=NOW+timedelta(hours=8)
        return result
    monkeypatch.setattr(optimizer,"optimal_five",delayed)
    report=manual(row(1),row(2),now=NOW,clock=lambda:at[0])
    assert report["status"]=="abstained"
    assert report["reason"]=="analysis_completed_outside_pre_match_window"


def test_two_digit_optimal_scores_are_saved_without_clipping():
    a,b=row(1),row(2)
    choices=[{"score1":f"{n}-0","score2":"1-0"} for n in range(10,15)]
    captured=pair(a["fixture"],b["fixture"],choices,now=NOW,method=POLICY)
    assert captured["choices"][0]["score1"]=="10-0"


def trials():
    rows = []
    start = NOW-timedelta(days=150)
    for n in range(110):
        predicted = start+timedelta(days=n)
        rows.append({"policy_id": POLICY, "prospective": True,
                     "selection_scope":AUTO_SELECTION,
                     "verification_status": "verified", "model_versions": ["v1", "v1"],
                     "fixture_identities": [["API-Football", str(2*n)], ["API-Football", str(2*n+1)]],
                     "fixture_team_names": [[f"Home{2*n}",f"Away{2*n}"],
                                            [f"Home{2*n+1}",f"Away{2*n+1}"]],
                     "predicted_at_utc": predicted.isoformat(),
                     "kickoffs_utc": [(predicted+timedelta(hours=h)).isoformat() for h in (1, 2)],
                     "verified_at_utc": (predicted+timedelta(hours=5)).isoformat(),
                     "estimated_coverage": .8, "hit": False})
    return rows


def test_calibration_is_prequential_and_does_not_prove_a_seventy_percent_hit_rate():
    result = calibrate(.8, trials(), now=NOW, model_versions=["v1", "v1"])
    assert result["ready"]
    assert result["prequential_test_pairs"] == 60
    assert result["calibrated_coverage_estimate"] < .3
    assert result["target_70_percent_proven"] is False
    # A sparsely observed probability range cannot borrow an unrelated bucket.
    assert not calibrate(.2, trials(), now=NOW, model_versions=["v1", "v1"])["ready"]
    late = trials()
    for trial in late:
        trial["verified_at_utc"] = (NOW+timedelta(days=1)).isoformat()
    assert not calibrate(.8, late, now=NOW, model_versions=["v1", "v1"])["ready"]


def test_repeated_wrong_policy_or_wrong_model_trials_cannot_inflate_sample():
    t = trials()[0]
    accepted, _ = calibration_trials([deepcopy(t)]*100, now=NOW, model_versions=["v1", "v1"])
    assert len(accepted) == 1
    assert not calibrate(.8, [deepcopy(t)]*100, now=NOW, model_versions=["v1", "v1"])["ready"]
    assert calibration_trials(trials(), now=NOW, model_versions=["different", "v1"])[0] == []
    duplicate=deepcopy(t)
    duplicate["fixture_identities"]=[["football-data.org","200"],["football-data.org","201"]]
    assert len(calibration_trials([t,duplicate],now=NOW,model_versions=["v1","v1"])[0])==1


def test_new_policy_and_existing_v25_are_included_in_verified_private_audit():
    a, b = row(1), row(2)
    report = manual(a, b, now=NOW)
    metadata = {k: report[k] for k in ("estimated_coverage", "model_versions", "source_times_utc")}
    metadata["selection_scope"]=AUTO_SELECTION
    saved = pair(a["fixture"], b["fixture"], report["choices"], method=POLICY,
                 now=NOW, policy_metadata=metadata)
    legacy = pair(a["fixture"], b["fixture"], report["choices"], method="v25_professional_analyst", now=NOW)
    assert len(private_records([saved, legacy])) == 2
    assert saved["record_id"] != legacy["record_id"]
    finals = {}
    for i, leg in enumerate(saved["legs"]):
        goals = list(map(int, report["choices"][0][f"score{i+1}"].split("-")))
        finals[(leg["provider"], leg["fixture_id"])] = {
            **leg, "evidence_version": 2, "score_basis": "regulation", "status": "FT",
            "home_goals": goals[0], "away_goals": goals[1],
            "verified_at_utc": (NOW+timedelta(days=1)).isoformat()}
    observed = trials_from_verified_ledger([saved], finals)
    # Metadata without immutable source-grid references cannot qualify calibration.
    assert observed==[]
    finals[(saved["legs"][1]["provider"], saved["legs"][1]["fixture_id"])]["away_team_id"] = "wrong"
    assert trials_from_verified_ledger([saved], finals) == []


def test_real_score_quotes_keep_value_distinct_from_hit_probability_and_roi():
    report = manual(row(1), row(2), now=NOW)
    quotes = []
    for i, fixture in enumerate((report["first_fixture"], report["second_fixture"])):
        score = list(map(int, report["choices"][0][f"score{i+1}"].split("-")))
        quotes.append({"provider": fixture["provider"], "fixture_id": fixture["id"],
                       "kickoff_utc": fixture["utcDate"], "bookmaker": "Actual source book",
                       "score": score, "decimal_odds": 8, "market": "correct_score_90min",
                       "updated_at_utc": NOW.isoformat(), "observed_at_utc": NOW.isoformat()})
    result = compare_score_prices(report, quotes, now=NOW)
    assert result["available"]
    assert result["comparisons"][0]["leg_price_product"] == 64
    assert result["comparisons"][0]["executable_double_quote_verified"] is False
    assert result["realized_roi"] is None
    quotes[1]["kickoff_utc"] = (NOW+timedelta(days=1)).isoformat()
    assert not compare_score_prices(report, quotes, now=NOW)["available"]


def authenticated_private_trial(tmp_path, *, source_age=timedelta(hours=1)):
    from ai_five_double_ui import archive_rows
    from forecast_ledger import append_file_record
    source=tmp_path/"original"
    for number in (1,2):
        template=row(number)
        fixture=template["fixture"]
        model={"provider":fixture["provider"],"fixture_id":str(fixture["id"]),
               "model_key":"v8","model_version":"v1","input_digest":"original-input-"+str(number),
               "home_team_id":str(fixture["homeTeam"]["id"]),
               "away_team_id":str(fixture["awayTeam"]["id"]),
               "home_team":fixture["homeTeam"]["name"],"away_team":fixture["awayTeam"]["name"],
               "score_matrix":template["result"]["matrix"],
               "top10_scores":[list(p) for p in template["result"]["scores"]],
               "captured_at_utc":(NOW-source_age).isoformat(),
               "evidence_as_of_utc":(NOW-source_age-timedelta(minutes=1)).isoformat(),
               "kickoff_utc":fixture["utcDate"]}
        append_file_record(model,root=source)
    candidates=archive_rows(now=NOW,root=source)
    report=manual(candidates[0],candidates[1],now=NOW,clock=lambda:NOW)
    metadata={k:report[k] for k in
              ("estimated_coverage","model_versions","source_times_utc","source_forecast_refs")}
    metadata["selection_scope"]=AUTO_SELECTION
    saved=pair(report["first_fixture"],report["second_fixture"],report["choices"],
               method=POLICY,now=NOW,policy_metadata=metadata)
    finals={}
    for i,leg in enumerate(saved["legs"]):
        goals=list(map(int,report["choices"][0][f"score{i+1}"].split("-")))
        finals[(leg["provider"],leg["fixture_id"])]=dict(
            leg,evidence_version=2,score_basis="regulation",status="FT",
            home_goals=goals[0],away_goals=goals[1],
            verified_at_utc=(NOW+timedelta(days=1)).isoformat())
    return source,saved,finals


def test_private_calibration_requires_authenticated_source_times_and_whole_row_result(tmp_path):
    source,saved,finals=authenticated_private_trial(tmp_path)
    observed=trials_from_verified_ledger([saved],finals,source_root=source)
    assert len(observed)==1 and observed[0]["hit"] is True
    assert len(calibration_trials(observed,now=NOW+timedelta(days=2),
                                  model_versions=["v1","v1"])[0])==1
    saved["policy_metadata"]["source_times_utc"][0]=NOW.isoformat()
    assert trials_from_verified_ledger([saved],finals,source_root=source)==[]


def test_private_calibration_rejects_stale_original_even_when_owner_metadata_claims_fresh(tmp_path):
    source,saved,finals=authenticated_private_trial(tmp_path,source_age=timedelta(hours=25))
    assert trials_from_verified_ledger([saved],finals,source_root=source)==[]
    saved["policy_metadata"]["source_times_utc"]=[NOW.isoformat(),NOW.isoformat()]
    assert trials_from_verified_ledger([saved],finals,source_root=source)==[]


@pytest.mark.parametrize("field",["kickoff_utc","home_team_id","home_team"])
def test_private_trial_fixture_and_result_cannot_collude_against_original_source(tmp_path,field):
    source,saved,finals=authenticated_private_trial(tmp_path)
    replacement=(NOW+timedelta(hours=8)).isoformat() if field=="kickoff_utc" else "wrong"
    saved["legs"][0][field]=replacement
    first=saved["legs"][0]
    finals[(first["provider"],first["fixture_id"])][field]=replacement
    assert trials_from_verified_ledger([saved],finals,source_root=source)==[]


def test_private_trial_source_input_digest_and_selection_scope_are_verified(tmp_path):
    source,saved,finals=authenticated_private_trial(tmp_path)
    saved["policy_metadata"]["source_forecast_refs"]["first"]["input_digest"]="tampered"
    assert trials_from_verified_ledger([saved],finals,source_root=source)==[]
    source2,saved2,finals2=authenticated_private_trial(tmp_path/"second")
    saved2["policy_metadata"]["selection_scope"]="manual_same_day_v1"
    assert trials_from_verified_ledger([saved2],finals2,source_root=source2)==[]
    saved2["policy_metadata"]="corrupt metadata"
    assert trials_from_verified_ledger([None,"corrupt record",saved2],finals2,source_root=source2)==[]
