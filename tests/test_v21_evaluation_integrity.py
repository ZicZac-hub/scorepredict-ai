"""A saved source, not a result-friendly label, authenticates V21 accuracy."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json

import numpy as np
import pytest

import v21_prospective as research
from v21_goal_model_comparison import _poisson


NOW=datetime(2033,1,20,3,tzinfo=timezone.utc)
KICK=NOW+timedelta(hours=6)
END=KICK+timedelta(hours=3)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def archived_pair(tmp_path):
    original=tmp_path/"original"
    tournament=tmp_path/"tournament"
    grid=np.outer(_poisson(1.6,11),_poisson(1.3,11)).tolist()
    sources=[]
    for fid,home,away in ((100,10,11),(200,20,21)):
        row={"model_key":"v8","model_version":"fixed-v8-test-version",
             "provider":"API-Football","fixture_id":str(fid),
             "forecast_id":f"original{fid}","input_digest":f"digest{fid}",
             "home_team_id":str(home),"away_team_id":str(away),
             "home_team":f"Home {home}","away_team":f"Away {away}",
             "captured_at_utc":(NOW-timedelta(hours=2)).isoformat(),
             "evidence_as_of_utc":(NOW-timedelta(hours=3)).isoformat(),
             "kickoff_utc":KICK.isoformat(),"competition":"BSA",
             "league_history_matches":130,"score_matrix":grid}
        write(original/"forecasts"/row["provider"]/row["fixture_id"]/
              (row["forecast_id"]+".json"),row)
        sources.append(row)
    research.freeze_models(root=tournament,source_root=original,now=NOW,originals=sources)
    research.freeze_pairs(root=tournament,now=NOW,min_coverage=.001)
    pair_path=next((tournament/"pairs").glob("*.json"))
    pair=json.loads(pair_path.read_text())
    selected=pair["methods"]["baseline_v8"]["doubles"][0]
    finals={}
    for index,source in enumerate(sources,1):
        h,a=map(int,selected[f"score{index}"].split("-"))
        finals[source["fixture_id"]]={**source,"evidence_version":2,
            "score_basis":"regulation","status":"FT","home_goals":h,
            "away_goals":a,"verified_at_utc":(KICK+timedelta(hours=2)).isoformat()}
    return tournament,original,pair_path,sources,finals


def grade(tournament,original,finals,*,now=END):
    return research.evaluate(root=tournament,source_root=original,verified=finals,now=now)


def test_real_archived_grids_settle_without_changing_any_evidence(tmp_path):
    root,original,pair_path,sources,finals=archived_pair(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob("*.json")}
    report=grade(root,original,finals)
    assert report["methods"]["baseline_v8"]=={
        "wins":1,"verified":1,"pending":0,"observed_hit_rate":1.0}
    assert report["valid_unique_pairs"]==1
    assert report["evidence_rejections"]=={}
    assert report["pairs"][0]["source_model_versions"]==["fixed-v8-test-version"]*2
    assert all(p.read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize("mutation,reason",[
    ("after_kickoff","pair_outside_predeclared_pre_match_window"),
    ("too_close_to_kickoff","pair_outside_predeclared_pre_match_window"),
    ("future_capture","pair_not_available_at_evaluation"),
    ("result_friendly_choices","five_doubles_or_coverage_do_not_reproduce_from_original"),
    ("inflated_coverage","five_doubles_or_coverage_do_not_reproduce_from_original"),
    ("wrong_digest","original_source_identity_or_digest_mismatch"),
    ("same_fixture_twice","two_events_and_four_distinct_known_teams_required"),
])
def test_tampered_pair_is_not_a_verified_trial(tmp_path,mutation,reason):
    root,original,path,sources,finals=archived_pair(tmp_path)
    pair=json.loads(path.read_text())
    if mutation=="after_kickoff":
        pair["captured_at_utc"]=(KICK+timedelta(hours=1)).isoformat()
    elif mutation=="too_close_to_kickoff":
        pair["captured_at_utc"]=(KICK-timedelta(minutes=44)).isoformat()
    elif mutation=="future_capture":
        pair["captured_at_utc"]=(END+timedelta(seconds=1)).isoformat()
    elif mutation=="result_friendly_choices":
        pair["methods"]["baseline_v8"]["doubles"][0]["score1"]="9-9"
    elif mutation=="inflated_coverage":
        pair["methods"]["baseline_v8"]["coverage"]+=.1
    elif mutation=="wrong_digest":
        pair["first"]["input_digest"]="wrong-source"
    else:
        pair["second"]=deepcopy(pair["first"])
    write(path,pair)
    report=grade(root,original,finals)
    assert report["methods"]["baseline_v8"]["verified"]==0
    assert report["methods"]["baseline_v8"]["pending"]==0
    assert report["evidence_rejections"]=={reason:1}


@pytest.mark.parametrize("mutation,reason",[
    ("original_later","original_evidence_observed_after_selection"),
    ("variant_later","frozen_variant_source_or_time_mismatch"),
    ("variant_grid","frozen_variants_do_not_reproduce_from_original"),
])
def test_later_or_modified_source_never_authenticates_earlier_selection(tmp_path,mutation,reason):
    root,original,path,sources,finals=archived_pair(tmp_path)
    source=sources[0]
    original_path=original/"forecasts"/source["provider"]/source["fixture_id"]/(source["forecast_id"]+".json")
    variant_path=root/"forecasts"/(research._row_key(source)+".json")
    if mutation=="original_later":
        source["captured_at_utc"]=(NOW+timedelta(minutes=1)).isoformat()
        write(original_path,source)
    else:
        variant=json.loads(variant_path.read_text())
        if mutation=="variant_later":
            variant["captured_at_utc"]=(NOW+timedelta(minutes=1)).isoformat()
        else:
            variant["model_results"]["poisson_marginal"]["score_matrix"][0][0]+=.01
        write(variant_path,variant)
    assert grade(root,original,finals)["evidence_rejections"]=={reason:1}


def test_copies_and_relabelled_dates_cannot_manufacture_a_large_hit_rate(tmp_path):
    root,original,path,sources,finals=archived_pair(tmp_path)
    pair=json.loads(path.read_text())
    for number in range(1,100):
        (root/"pairs"/f"copy{number:03}.json").write_text(json.dumps(pair))
    report=grade(root,original,finals)
    assert report["methods"]["baseline_v8"]["verified"]==1
    assert report["evidence_rejections"]=={"filename_sydney_date_mismatch":99}
    assert report["fixture_reuse_in_denominator_allowed"] is False
    pair["sydney_date"]="2033-01-21"
    write(root/"pairs"/"2033-01-21.json",pair)
    report=grade(root,original,finals)
    assert report["methods"]["baseline_v8"]["verified"]==1
    assert report["evidence_rejections"]["fixture_outside_declared_sydney_date"]==1


def test_future_verification_and_partial_results_remain_pending(tmp_path):
    root,original,path,sources,finals=archived_pair(tmp_path)
    future=deepcopy(finals)
    future["200"]["verified_at_utc"]=(END+timedelta(minutes=1)).isoformat()
    report=grade(root,original,future)
    assert report["methods"]["baseline_v8"]["verified"]==0
    assert report["methods"]["baseline_v8"]["pending"]==1
    assert report["result_rejections"]=={"result_observed_after_evaluation":1}
    missing=grade(root,original,{"100":finals["100"]})
    assert missing["methods"]["baseline_v8"]["pending"]==1
    assert missing["methods"]["baseline_v8"]["wins"]==0
    assert missing["methods"]["baseline_v8"]["observed_hit_rate"] is None


def test_replay_accepts_machine_epsilon_without_accepting_material_probability_changes(tmp_path):
    root,original,path,sources,finals=archived_pair(tmp_path)
    pair=json.loads(path.read_text())
    pair["methods"]["baseline_v8"]["coverage"]+=1e-14
    write(path,pair)
    assert grade(root,original,finals)["valid_unique_pairs"]==1
    pair["methods"]["baseline_v8"]["coverage"]+=1e-6
    write(path,pair)
    assert grade(root,original,finals)["valid_unique_pairs"]==0
