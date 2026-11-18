"""Fresh matched-pair evidence keeps original packets and bounded quotas."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np

from forecast_ledger import append_file_record
import professional_double_shadow as shadow
import v21_capture_sources as collector

NOW=datetime(2031,6,1,10,tzinfo=timezone.utc)


def model(number=1, minutes=60):
    values=np.asarray([.4,.3,.2,.075,.025])
    grid=np.outer(values,values)
    scores=sorted([[h,a,float(value)] for h,row in enumerate(grid)
                   for a,value in enumerate(row)],key=lambda row:(-row[2],row[0],row[1]))[:10]
    return {"mode":"prospective","provider":"API-Football","fixture_id":str(number),
            "model_key":"v8","model_version":"test-v8","input_digest":"source-"+str(number),
            "home_team_id":str(number*2),"away_team_id":str(number*2+1),
            "home_team":f"Home {number}","away_team":f"Away {number}",
            "fixture_name":f"Home {number} vs Away {number}","competition":"PL",
            "score_matrix":grid.tolist(),"top10_scores":scores,
            "captured_at_utc":(NOW-timedelta(hours=2)).isoformat(),
            "evidence_as_of_utc":(NOW-timedelta(hours=2,minutes=5)).isoformat(),
            "kickoff_utc":(NOW+timedelta(minutes=minutes)).isoformat()}


def seal(packet):
    packet=deepcopy(packet)
    packet.pop("snapshot_sha256",None)
    packet["snapshot_sha256"]=hashlib.sha256(json.dumps(packet,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return packet


def packet(leg, *, captured=None, complete=False):
    stamp=captured or NOW-timedelta(hours=4)
    row={"status":"captured_pre_kickoff","provider":leg["provider"],
         "fixture_id":leg["fixture_id"],"home_team_id":leg["home_team_id"],
         "away_team_id":leg["away_team_id"],"kickoff_utc":leg["kickoff_utc"],
         "captured_at_utc":stamp.isoformat(),"verified_historical_fixture_xg":[],
         "confirmed_starting_xi":[],"multi_market":{},
         "available_sources":{"historical_xg":True},"diagnostics":{"queries":3}}
    if complete:
        for team in (leg["home_team_id"],leg["away_team_id"]):
            for number in range(6):
                row["verified_historical_fixture_xg"].append({
                    "fixture_id":team+"-"+str(number),
                    "kickoff_utc":(stamp-timedelta(days=number+2)).isoformat(),
                    "home_team_id":team,"away_team_id":"opponent-"+str(number),
                    "home_xg":1.2,"away_xg":.8})
            row["confirmed_starting_xi"].append({"team_id":team,
                "starting_xi":[f"Player {team}-{number}" for number in range(11)]})
        row["multi_market"]={
            "1x2":{"last_source_update_utc":stamp.isoformat(),
                   "probabilities":{"home":.5,"draw":.25,"away":.25}},
            "over_2_5":{"last_source_update_utc":stamp.isoformat(),
                        "probabilities":{"yes":.55,"no":.45}}}
    return seal(row)


def save_packet(root, leg, value):
    path=collector.source_packet_path(root,leg)
    path.parent.mkdir(parents=True,exist_ok=True)
    record={"source_forecast":{key:leg[key] for key in (
        "provider","fixture_id","forecast_id","input_digest","home_team_id",
        "away_team_id","kickoff_utc")},"provider_snapshot":value}
    path.write_text(json.dumps(record))
    return path


def fixture_sources(tmp_path):
    source=tmp_path/"original"
    for row in (model(1,60),model(2,150)):
        assert append_file_record(row,root=source)
    pair_root=tmp_path/"fixed-pair"
    saved=shadow.freeze(root=pair_root,source_root=source,
        now=NOW-timedelta(hours=1),clock=lambda:NOW-timedelta(hours=1))
    assert saved["new_sets"]==["2031-06-01"]
    legs=collector.targeted_pair_models(source_root=source,pair_root=pair_root,now=NOW)
    assert len(legs)==2
    return source,pair_root,legs


def arrange_capture(monkeypatch, legs, *, at=None):
    current=[NOW] if at is None else at
    monkeypatch.setattr(collector,"current_pair",lambda root,now:list(legs))
    monkeypatch.setattr(collector,"load_evidence",lambda *a,**kw:{"history":[],"input_digest":"new-history"})
    seen=[]
    def capture(fixture, history, key, **kwargs):
        seen.append(str(fixture["id"]))
        leg=next(row for row in legs if row["fixture_id"]==str(fixture["id"]))
        return packet(leg,captured=current[0])
    monkeypatch.setattr(collector,"capture",capture)
    monkeypatch.setenv("V21_DAILY_CAPTURE_ATTEMPTS","6")
    return current,seen


def test_sixty_minute_refresh_preserves_early_packet_and_only_attempts_once(monkeypatch,tmp_path):
    source,pairs,legs=fixture_sources(tmp_path)
    root=tmp_path/"packets"
    leg=legs[0]
    path=save_packet(root,leg,packet(leg))
    original=path.read_bytes()
    current,seen=arrange_capture(monkeypatch,[leg])
    result=collector.capture_pair(root=root,source_root=source,pair_root=tmp_path/"none",
        now=NOW,clock=lambda:current[0],enabled=True,api_key="test")
    assert result["attempted_legs"]==1 and seen==[leg["fixture_id"]]
    assert result["near_refresh_window_minutes"]==[8,65]
    assert path.read_bytes()==original
    assert len(list((root/"versions").glob("*.json")))==1
    current[0]=NOW+timedelta(minutes=15)
    repeated=collector.capture_pair(root=root,source_root=source,pair_root=tmp_path/"none",
        now=current[0],clock=lambda:current[0],enabled=True,api_key="test")
    assert repeated["attempted_legs"]==0
    assert repeated["skipped"]==[leg["fixture_id"]+":refresh_already_attempted"]
    assert len(seen)==1


def test_fresh_complete_packet_avoids_unnecessary_near_kickoff_request(monkeypatch,tmp_path):
    source,pairs,legs=fixture_sources(tmp_path)
    root=tmp_path/"packets"
    leg=legs[0]
    save_packet(root,leg,packet(leg,captured=NOW-timedelta(minutes=15),complete=True))
    current,seen=arrange_capture(monkeypatch,[leg])
    result=collector.capture_pair(root=root,source_root=source,pair_root=tmp_path/"none",
        now=NOW,clock=lambda:current[0],enabled=True,api_key="test")
    assert result["attempted_legs"]==0 and seen==[]
    assert result["source_availability"][leg["fixture_id"]]["refresh_reasons"]==[]


def test_one_team_xg_or_duplicated_XI_cannot_look_like_both_teams():
    leg=model()
    raw=packet(leg,captured=NOW,complete=True)
    first_team=leg["home_team_id"]
    raw["verified_historical_fixture_xg"]=[r for r in raw["verified_historical_fixture_xg"]
        if r["home_team_id"]==first_team]*2
    raw["confirmed_starting_xi"]=[raw["confirmed_starting_xi"][0]]*2
    status=collector.source_availability(leg,raw,now=NOW)
    assert status["xg_fixtures_by_team"]=={first_team:6,leg["away_team_id"]:0}
    assert status["confirmed_xi_team_ids"]==[first_team]
    assert len(status["refresh_reasons"])==2


def test_stale_source_or_market_remains_explicitly_refreshable():
    leg=model()
    stale=packet(leg,captured=NOW-timedelta(hours=4),complete=True)
    status=collector.source_availability(leg,stale,now=NOW)
    assert status["refresh_reasons"]==["source_not_observed_within_3_hours"]
    stale["multi_market"]["1x2"]["last_source_update_utc"]=(NOW-timedelta(hours=13)).isoformat()
    status=collector.source_availability(leg,stale,now=NOW)
    assert "fresh_1x2_and_goal_total_markets_unavailable" in status["refresh_reasons"]


def test_conflicting_XIs_and_unfinished_historical_xg_are_not_complete_evidence():
    leg=model()
    raw=packet(leg,captured=NOW,complete=True)
    changed=deepcopy(raw["confirmed_starting_xi"][0])
    changed["starting_xi"][0]="Different player"
    raw["confirmed_starting_xi"].append(changed)
    raw["verified_historical_fixture_xg"][0]["kickoff_utc"]=(NOW-timedelta(hours=1)).isoformat()
    status=collector.source_availability(leg,raw,now=NOW)
    assert status["conflicting_xi_team_ids"]==[leg["home_team_id"]]
    assert status["confirmed_xi_team_ids"]==[leg["away_team_id"]]
    assert status["xg_fixtures_by_team"][leg["home_team_id"]]==5
    assert len(status["refresh_reasons"])==2


def test_budget_reserved_pair_refreshes_both_originals_before_common_cutoff(monkeypatch,tmp_path):
    source,pairs,legs=fixture_sources(tmp_path)
    root=tmp_path/"packets"
    originals={save_packet(root,leg,packet(leg)):None for leg in legs}
    originals={path:path.read_bytes() for path in originals}
    usage=root/"usage"/"2031-06-01.json"
    usage.parent.mkdir()
    usage.write_text(json.dumps({"attempts":4}))
    current,seen=arrange_capture(monkeypatch,list(reversed(legs)))
    result=collector.capture_pair(root=root,source_root=source,pair_root=pairs,
        now=NOW,clock=lambda:current[0],enabled=True,api_key="test")
    assert result["attempted_legs"]==2 and result["daily_attempts_used"]==6
    assert result["targeted_pair_fixture_ids"]==[leg["fixture_id"] for leg in legs]
    assert seen==result["targeted_pair_fixture_ids"]
    assert len(list((root/"versions").glob("*.json")))==2
    assert all(path.read_bytes()==value for path,value in originals.items())
    assert json.loads(usage.read_text())["attempts"]==6
    for path in (root/"refresh_attempts").glob("*.json"):
        assert json.loads(path.read_text())["fixed_pair_common_cutoff_priority"]


def test_corrupt_fixed_pair_cannot_promote_later_match_into_reserved_quota(monkeypatch,tmp_path):
    source,pairs,legs=fixture_sources(tmp_path)
    path=pairs/"sets"/"2031-06-01.json"
    record=json.loads(path.read_text())
    record["selection"]["source_forecast_refs"]["first"]["input_digest"]="wrong"
    record["record_sha256"]=shadow._digest(record)
    path.write_text(json.dumps(record))
    assert collector.targeted_pair_models(source_root=source,pair_root=pairs,now=NOW)==[]
    root=tmp_path/"packets"
    later=legs[1]
    current,seen=arrange_capture(monkeypatch,[later])
    usage=root/"usage"/"2031-06-01.json"
    usage.parent.mkdir(parents=True)
    usage.write_text(json.dumps({"attempts":4}))
    status=collector.capture_pair(root=root,source_root=source,pair_root=pairs,
        now=NOW,clock=lambda:current[0],enabled=True,api_key="test")
    assert status["targeted_pair_fixture_ids"]==[]
    assert status["attempted_legs"]==0 and seen==[]
    assert status["skipped"]==[later["fixture_id"]+":near_kickoff_budget_reserved"]


def test_history_finishing_after_window_does_not_start_further_provider_requests(monkeypatch,tmp_path):
    source,pairs,legs=fixture_sources(tmp_path)
    current,seen=arrange_capture(monkeypatch,[legs[0]])
    def history(*args,**kwargs):
        current[0]=NOW+timedelta(hours=2)
        return {"history":[],"input_digest":"late-history"}
    monkeypatch.setattr(collector,"load_evidence",history)
    result=collector.capture_pair(root=tmp_path/"packets",source_root=source,pair_root=tmp_path/"none",
        now=NOW,clock=lambda:current[0],enabled=True,api_key="test")
    assert result["attempted_legs"]==1 and seen==[]
    assert "history_completed_outside_capture_window" in result["skipped"][0]
    assert result["completed_at_utc"]==current[0].isoformat()


def test_exhausted_daily_budget_blocks_target_pair_refresh(monkeypatch,tmp_path):
    source,pairs,legs=fixture_sources(tmp_path)
    root=tmp_path/"packets"
    for leg in legs:
        save_packet(root,leg,packet(leg))
    usage=root/"usage"/"2031-06-01.json"
    usage.parent.mkdir()
    usage.write_text(json.dumps({"attempts":6}))
    current,seen=arrange_capture(monkeypatch,legs)
    result=collector.capture_pair(root=root,source_root=source,pair_root=pairs,
        now=NOW,clock=lambda:current[0],enabled=True,api_key="test")
    assert result["attempted_legs"]==0 and seen==[]
    assert result["daily_attempts_used"]==6
    assert len(result["skipped"])==2
    assert all(reason.endswith(":daily_attempt_budget_exhausted") for reason in result["skipped"])


def test_pair_priority_does_not_expand_the_two_attempt_run_budget(monkeypatch,tmp_path):
    source,pairs,legs=fixture_sources(tmp_path)
    other={**model(3,70),"forecast_id":"third-original"}
    root=tmp_path/"packets"
    for leg in legs:
        save_packet(root,leg,packet(leg))
    current,seen=arrange_capture(monkeypatch,[other,*legs])
    result=collector.capture_pair(root=root,source_root=source,pair_root=pairs,
        now=NOW,clock=lambda:current[0],enabled=True,api_key="test")
    assert result["attempted_legs"]==2 and len(seen)==2
    assert result["daily_attempts_used"]==2
    assert result["skipped"]==["3:per_run_attempt_budget_exhausted"]


def test_provider_response_after_kickoff_cannot_replace_or_add_packet(monkeypatch,tmp_path):
    source,pairs,legs=fixture_sources(tmp_path)
    root=tmp_path/"packets"
    leg=legs[0]
    base=save_packet(root,leg,packet(leg))
    original=base.read_bytes()
    current,seen=arrange_capture(monkeypatch,[leg])
    def late_response(*args,**kwargs):
        current[0]=NOW+timedelta(hours=2)
        return packet(leg,captured=current[0])
    monkeypatch.setattr(collector,"capture",late_response)
    status=collector.capture_pair(root=root,source_root=source,pair_root=tmp_path/"none",
        now=NOW,clock=lambda:current[0],enabled=True,api_key="test")
    assert status["attempted_legs"]==1 and status["new_packets"]==[]
    assert status["skipped"]==[leg["fixture_id"]+":no_qualified_packet"]
    assert base.read_bytes()==original
    assert list((root/"versions").glob("*.json"))==[]
    assert json.loads((root/"usage"/"2031-06-01.json").read_text())["attempts"]==1
