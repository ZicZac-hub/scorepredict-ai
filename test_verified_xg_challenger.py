"""Synthetic source-integrity tests; no claim of historical skill."""
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np

from verified_xg_challenger import predict_shadow
from verified_xg_calibration import evaluate,fit_bands


START=datetime(2025,1,1,tzinfo=timezone.utc)


def match(i):
    h=1+i%6
    a=1+(i+1)%6
    return {"id":1000+i,"provider":"API-Football",
            "utcDate":(START+timedelta(days=i)).isoformat(),
            "status":"FINISHED","competition":{"id":1,"code":"PL"},
            "homeTeam":{"id":h},"awayTeam":{"id":a},
            "score":{"fullTime":{"home":i%4,"away":(i//2)%3}}}


def evidence():
    history=[match(i) for i in range(1,161)]
    fixture={"id":9001,"provider":"API-Football",
             "utcDate":(START+timedelta(days=180)).isoformat(),
             "competition":{"id":1,"code":"PL"},
             "homeTeam":{"id":1},"awayTeam":{"id":2},"status":"SCHEDULED"}
    cutoff=(START+timedelta(days=179,hours=12)).isoformat()
    selected=[m for m in history if 1 in (m["homeTeam"]["id"],m["awayTeam"]["id"])
              or 2 in (m["homeTeam"]["id"],m["awayTeam"]["id"])][-12:]
    xg=[{"fixture_id":str(m["id"]),"kickoff_utc":m["utcDate"],
         "home_team_id":str(m["homeTeam"]["id"]),
         "away_team_id":str(m["awayTeam"]["id"]),
         "home_xg":1.3+(m["id"]%3)*.2,
         "away_xg":.7+(m["id"]%2)*.15,
         "type":"api_football_reported_expected_goals"}
        for m in selected]
    snap={"schema":1,"status":"captured_pre_kickoff","provider":"API-Football",
          "fixture_id":str(fixture["id"]),
          "home_team_id":"1","away_team_id":"2",
          "kickoff_utc":fixture["utcDate"],"captured_at_utc":cutoff,
          "verified_historical_fixture_xg":xg,
          "confirmed_starting_xi":[]}
    return fixture,history,snap


def seal(snapshot):
    x={k:v for k,v in snapshot.items() if k!="snapshot_sha256"}
    snapshot["snapshot_sha256"]=hashlib.sha256(
        json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()
    return snapshot


def test_verified_xg_and_normalization():
    fixture,history,snapshot=evidence()
    result=predict_shadow(fixture,history,seal(snapshot))
    assert result is not None
    assert result["verified_xg_matches"]["home"]>=3
    assert result["verified_xg_matches"]["away"]>=3
    assert result["lineup_status"]=="unavailable"
    assert result["lineup_numeric_weight"]==0.
    assert np.isclose(result["matrix"].sum(),1.)
    assert np.isfinite(result["matrix"]).all()
    assert np.min(result["matrix"])>=0
    assert np.isclose(result["home_win"]+result["draw"]+result["away_win"],1.)


def test_unverified_and_late_data_rejected():
    fixture,history,snapshot=evidence()
    authenticated=seal(snapshot)
    authenticated["verified_historical_fixture_xg"][0]["home_xg"]=14.0
    assert predict_shadow(fixture,history,authenticated) is None
    fixture,history,snapshot=evidence()
    assert predict_shadow(fixture,history,seal(snapshot),as_of=fixture["utcDate"]) is None


def test_future_results_and_unrelated_leagues_do_not_change_output():
    fixture,history,snapshot=evidence()
    packet=seal(snapshot)
    base=predict_shadow(fixture,history,packet)
    future=match(179)
    future["id"]=50000
    outside=match(75)
    outside["id"]=50001
    outside["competition"]={"id":99,"code":"OUT"}
    candidate=predict_shadow(fixture,history+[future,outside],packet)
    np.testing.assert_array_equal(base["matrix"],candidate["matrix"])


def test_confirmed_lineups_only_change_evidence_status():
    fixture,history,snapshot=evidence()
    baseline=predict_shadow(fixture,history,seal(snapshot))
    fixture,history,snapshot=evidence()
    snapshot["confirmed_starting_xi"]=[
        {"team_id":str(t),"provider_reported":True,
         "starting_xi":[f"player-{t}-{i}" for i in range(11)]}
        for t in (1,2)]
    both=predict_shadow(fixture,history,seal(snapshot))
    assert both["lineup_status"]=="both_confirmed"
    assert both["lineup_numeric_weight"]==0.
    np.testing.assert_array_equal(baseline["matrix"],both["matrix"])


def test_misidentified_xg_row_does_not_count():
    fixture,history,snapshot=evidence()
    for row in snapshot["verified_historical_fixture_xg"]:
        row["home_team_id"]="9999"
    assert predict_shadow(fixture,history,seal(snapshot)) is None


def test_independent_calibration_reports_no_accuracy_without_real_cases():
    report=evaluate([],cutoff="2026-01-01")
    assert report["holdout_cases"]==0
    assert report["factors_from_training_only"]==[1.,1.,1.]
    assert report["five_double"]["verified_xg"]["hit_rate"] is None
    assert report["model_metrics"]["verified_xg"]["score_log_loss"] is None


def test_calibration_training_only():
    from engine import _grid
    model=_grid(.9,.6)
    train=[{"xg":model,"actual":(4,2)} for _ in range(30)]
    factors=fit_bands(train)
    assert factors[2]>1.
    assert fit_bands(train[:29])==[1.,1.,1.]
