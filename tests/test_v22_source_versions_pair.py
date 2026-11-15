"""V22 append-only source version selection and API-free auto-pair ranking."""
import hashlib
import json
from datetime import datetime,timedelta,timezone

import numpy as np
import pytest

from v21_capture_sources import best_packet,source_packet_path
from v21_goal_model_comparison import _poisson,top_scores
import v22_pair_selector as pair_selector

NOW=datetime(2032,10,3,3,tzinfo=timezone.utc)
KICK=(NOW+timedelta(hours=4)).isoformat()
GRID=np.outer(_poisson(1.6,11),_poisson(1.2,11))


def original_leg():
    return {"provider":"API-Football","fixture_id":"900",
            "forecast_id":"ziczacv8original",
            "input_digest":"frozen-original-sha",
            "home_team_id":"11","away_team_id":"22",
            "kickoff_utc":KICK}


def packet(leg,*,at,quality):
    xg=[]
    for team in ("11","22"):
        for i in range(quality):
            xg.append({"fixture_id":team+"-"+str(i),
                       "home_team_id":team,"away_team_id":"900",
                       "home_xg":1.9,"away_xg":1.0,
                       "kickoff_utc":(NOW-timedelta(days=i+2)).isoformat()})
    data={"status":"captured_pre_kickoff","schema":1,
          "provider":leg["provider"],"fixture_id":leg["fixture_id"],
          "home_team_id":leg["home_team_id"],"away_team_id":leg["away_team_id"],
          "kickoff_utc":leg["kickoff_utc"],"captured_at_utc":at.isoformat(),
          "verified_historical_fixture_xg":xg,
          "multi_market":({"1x2":{},"over_2_5":{}} if quality>=6 else {})}
    data["snapshot_sha256"]=hashlib.sha256(json.dumps(
        data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return {"source_forecast":leg,"provider_snapshot":data}


def test_best_source_version_is_source_matched_and_as_of_only(tmp_path):
    leg=original_leg()
    old=packet(leg,at=NOW-timedelta(minutes=20),quality=2)
    new=packet(leg,at=NOW-timedelta(minutes=1),quality=8)
    future=packet(leg,at=NOW+timedelta(minutes=10),quality=12)
    base=source_packet_path(tmp_path,leg)
    base.parent.mkdir(parents=True,exist_ok=True)
    base.write_text(json.dumps(old))
    versions=tmp_path/"versions"
    versions.mkdir()
    (versions/(base.stem+"_new.json")).write_text(json.dumps(new))
    (versions/(base.stem+"_future.json")).write_text(json.dumps(future))
    picked=best_packet(tmp_path,leg,now=NOW)
    assert picked["provider_snapshot"]["snapshot_sha256"]==new["provider_snapshot"]["snapshot_sha256"]
    assert json.loads(base.read_text())==old
    assert best_packet(tmp_path,leg,now=NOW-timedelta(hours=1)) is None


def fixture(fid,home,away):
    return {"fixture":{"id":fid,"provider":"API-Football","utcDate":KICK,
                       "homeTeam":{"id":home,"name":str(home)},
                       "awayTeam":{"id":away,"name":str(away)}},
            "kickoff":KICK,
            "result":{"matrix":GRID,"scores":top_scores(GRID),
                      "model_version":"research-v8"},
            "past":[]}


def test_v22_auto_pair_rejects_missing_data_and_no_api_calls(monkeypatch):
    rows=[fixture(11,101,102),fixture(12,201,202),fixture(13,301,302)]
    assert pair_selector.candidates(rows,lambda f:None,now=NOW)["abstained"]
    queried=[]
    def found(f):
        queried.append(f["id"])
        return {"snapshot_sha256":"a"*64}
    def shadow_only(row,packet,as_of):
        assert packet["snapshot_sha256"]=="a"*64
        return {"ready":True,"matrix":GRID,
                "scores":top_scores(GRID),
                "source_snapshot_digest":packet["snapshot_sha256"],
                "prediction_method":"v22_test_shadow"}
    monkeypatch.setattr(pair_selector,"shadow",shadow_only)
    ranked=pair_selector.candidates(rows,found,now=NOW,min_coverage=.001)
    assert ranked["total_qualifying_pairs"]==3
    assert ranked["source_qualified_fixtures"]==3
    assert sorted(queried)==[11,12,13]
    assert ranked["no_external_calls"]
    assert ranked["pairs"][0]["not_observed_accuracy"]
