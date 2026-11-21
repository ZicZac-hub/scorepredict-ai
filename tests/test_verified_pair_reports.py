"""Outcome counts require original grids, exact packets and pre-match choices."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json

import numpy as np
import pytest

from forecast_ledger import append_file_record,read_file_records
from v21_capture_sources import source_packet_path
from v21_goal_model_comparison import top_scores
import v22_prospective
import v23_prospective
import verified_pair_reports as audit

NOW=datetime(2035,1,1,0,tzinfo=timezone.utc)


def setup(tmp_path,method="v23_market_form"):
    source=tmp_path/"original";packets=tmp_path/"packets";out=tmp_path/"pairs"
    grid=np.outer([.4,.3,.2,.075,.025],[.4,.3,.2,.075,.025])
    for fid in (1,2):
        model={"mode":"prospective","provider":"API-Football","fixture_id":str(fid),
               "model_key":"v8","model_version":"original-test-v8","input_digest":"digest"+str(fid),
               "home_team_id":str(fid*2),"away_team_id":str(fid*2+1),
               "home_team":f"Home {fid}","away_team":f"Away {fid}",
               "fixture_name":f"Home {fid} vs Away {fid}","competition":"PL",
               "score_matrix":grid.tolist(),"top10_scores":top_scores(grid,10),
               "league_history_matches":100,"captured_at_utc":(NOW-timedelta(hours=1)).isoformat(),
               "evidence_as_of_utc":(NOW-timedelta(hours=1,minutes=5)).isoformat(),
               "kickoff_utc":(NOW+timedelta(hours=3+fid)).isoformat()}
        assert append_file_record(model,root=source)
    models=list(read_file_records(root=source))
    for model in models:
        xg=[]
        for team in (model["home_team_id"],model["away_team_id"]):
            for n in range(8):
                home,away=(team,str(1000+n)) if n%2==0 else (str(1000+n),team)
                xg.append({"fixture_id":f"{team}-{n}","kickoff_utc":(NOW-timedelta(days=n+2)).isoformat(),
                           "home_team_id":home,"away_team_id":away,"home_xg":1.3,"away_xg":.9})
        packet={"status":"captured_pre_kickoff","provider":"API-Football",
                **{key:model[key] for key in ("fixture_id","home_team_id","away_team_id","kickoff_utc")},
                "captured_at_utc":NOW.isoformat(),"verified_historical_fixture_xg":xg,
                "confirmed_starting_xi":[],"multi_market":{
                    "1x2":{"probabilities":{"home":.5,"draw":.25,"away":.25},
                           "last_source_update_utc":(NOW-timedelta(minutes=1)).isoformat()},
                    "over_2_5":{"probabilities":{"yes":.6,"no":.4},
                                 "last_source_update_utc":(NOW-timedelta(minutes=1)).isoformat()}}}
        seal(packet)
        record={"source_forecast":{key:model[key] for key in (
            "provider","fixture_id","forecast_id","input_digest","home_team_id","away_team_id","kickoff_utc")},
                "provider_snapshot":packet}
        path=source_packet_path(packets,model);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(record))
    freezer=v22_prospective if method=="v22_research" else v23_prospective
    status=freezer.freeze(now=NOW+timedelta(minutes=1),root=out,source_root=source,provider_root=packets)
    assert status["newly_frozen_days"]==["2035-01-01"]
    path=next((out/"pairs").glob("*.json"));pair=json.loads(path.read_text())
    return source,packets,out,path,pair


def seal(packet):
    packet.pop("snapshot_sha256",None)
    packet["snapshot_sha256"]=hashlib.sha256(json.dumps(
        packet,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def results(pair):
    output={}
    for number,side in enumerate(("first","second"),1):
        leg=pair[side];goals=list(map(int,pair["doubles"]["baseline_v8"][0][f"score{number}"].split("-")))
        output[leg["fixture_id"]]={**leg,"status":"FT","evidence_version":2,"score_basis":"regulation",
            "home_goals":goals[0],"away_goals":goals[1],"verified_at_utc":(NOW+timedelta(hours=8)).isoformat()}
    return output


def evaluate(state,method="v23_market_form",verified=None,now=None):
    source,packets,out,_,_=state
    return audit.evaluate(root=out,source_root=source,provider_root=packets,
        prematch_root=out/"no-results",research_method=method,verified=verified,
        now=now or NOW+timedelta(days=1))


@pytest.mark.parametrize("method",["v22_research","v23_market_form"])
def test_genuine_frozen_choices_grade_same_row_without_mutating_originals(tmp_path,method):
    state=setup(tmp_path,method);source,packets,out,path,pair=state
    before={p:p.read_bytes() for folder in (source,packets,out) for p in folder.rglob("*.json")}
    report=evaluate(state,method,results(pair))
    assert report["methods"]["baseline_v8"]["verified"]==1
    assert report["methods"]["baseline_v8"]["hits"]==1
    assert report["evidence_rejections"]=={}
    assert all(path.read_bytes()==data for path,data in before.items())
    assert report["model_promoted"] is False


@pytest.mark.parametrize("change",["choices","coverage","input_digest","packet_hash","late_capture","duplicate_team"])
def test_altered_saved_record_never_establishes_a_win(tmp_path,change):
    state=setup(tmp_path);source,packets,out,path,pair=state
    actual=results(pair)
    if change=="choices":pair["doubles"]["v23_market_form"][0]["score1"]="9-9"
    elif change=="coverage":pair["model_estimated_coverage"]["baseline_v8"]+=.01
    elif change=="input_digest":pair["first"]["input_digest"]="wrong-source"
    elif change=="packet_hash":pair["source_packet_hashes"][0]="a"*64
    elif change=="late_capture":pair["captured_at_utc"]=(NOW+timedelta(hours=6)).isoformat()
    else:pair["second"]["home_team_id"]=pair["first"]["home_team_id"]
    path.write_text(json.dumps(pair))
    report=evaluate(state,verified=actual)
    assert report["methods"]["baseline_v8"]["verified"]==0
    assert sum(report["evidence_rejections"].values())==1


def test_copied_file_is_not_an_independent_trial(tmp_path):
    state=setup(tmp_path);source,packets,out,path,pair=state
    (path.parent/"2035-01-02.json").write_text(path.read_text())
    report=evaluate(state,verified=results(pair))
    assert report["methods"]["baseline_v8"]["verified"]==1
    assert report["evidence_rejections"]=={"pair_filename_sydney_date_mismatch":1}


@pytest.mark.parametrize("change",["future_result","one_result","wrong_provider","nonterminal"])
def test_incomplete_or_unavailable_results_stay_pending(tmp_path,change):
    state=setup(tmp_path);actual=results(state[-1])
    if change=="future_result":actual["1"]["verified_at_utc"]=(NOW+timedelta(days=2)).isoformat()
    elif change=="one_result":actual.pop("2")
    elif change=="wrong_provider":actual["1"]["provider"]="different-provider"
    else:actual["1"]["status"]="NS"
    report=evaluate(state,verified=actual)
    assert report["methods"]["baseline_v8"]["verified"]==0
    assert report["methods"]["baseline_v8"]["pending"]==1
    assert report["methods"]["baseline_v8"]["hit_rate"] is None


def test_exact_packet_version_is_replayed_and_later_packet_cannot_backfill(tmp_path):
    state=setup(tmp_path);source,packets,out,path,pair=state
    model=read_file_records(root=source)[0];base=source_packet_path(packets,model)
    version=packets/"versions"/(base.stem+"_version.json")
    version.parent.mkdir();base.rename(version)
    assert evaluate(state,verified=results(pair))["methods"]["baseline_v8"]["verified"]==1
    record=json.loads(version.read_text());record["provider_snapshot"]["captured_at_utc"]=(NOW+timedelta(minutes=2)).isoformat()
    seal(record["provider_snapshot"])
    pair["source_packet_hashes"][0]=record["provider_snapshot"]["snapshot_sha256"]
    path.write_text(json.dumps(pair));version.write_text(json.dumps(record))
    assert evaluate(state,verified=results(pair))["methods"]["baseline_v8"]["verified"]==0


def test_v22_recorded_calibration_must_reproduce_pre_decision_fit(tmp_path):
    state=setup(tmp_path,"v22_research");pair=state[-1]
    pair["calibration"][0]={"status":"past_only_shadow_not_promoted","temperature":1.3,
        "training_samples":100,"training_ended_at_utc":NOW.isoformat()}
    state[-2].write_text(json.dumps(pair))
    report=evaluate(state,"v22_research",results(pair))
    assert report["evidence_rejections"]=={"calibration_not_reproduced_from_pre_decision_results":1}
