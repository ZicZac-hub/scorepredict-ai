"""V22 no-hindsight calibration, AI credibility and immutable pair settlement."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json

import numpy as np
import pytest

import v22_calibration as calibration
import v22_ai_gate as gate
import v22_prospective as tournament
from v21_goal_model_comparison import _poisson,top_scores


NOW=datetime(2033,4,1,12,tzinfo=timezone.utc)
GRID=np.outer(_poisson(1.6,11),_poisson(1.1,11))


def historical(fid,when,*,confirmed=True):
    kickoff=when
    original={
        "provider":"API-Football","fixture_id":str(fid),
        "home_team_id":"11","away_team_id":"22",
        "home_team":"Home","away_team":"Away",
        "kickoff_utc":kickoff.isoformat(),"competition":"PL",
        "score_basis":"regulation","model_key":"v8",
        "forecast_id":f"forecast_{fid}","input_digest":f"sha_{fid}",
        "captured_at_utc":(when-timedelta(hours=4)).isoformat(),
        "score_matrix":GRID.tolist(),
        "outcome_probabilities":{"home":.46,"draw":.24,"away":.30}
    }
    verified={
        "provider":"API-Football","fixture_id":str(fid),
        "home_team_id":"11","away_team_id":"22",
        "home_team":"Home","away_team":"Away",
        "kickoff_utc":kickoff.isoformat(),"score_basis":"regulation",
        "evidence_version":2,"status":"FT",
        "verified_at_utc":(when+timedelta(hours=6)).isoformat(),
        "home_goals":2,"away_goals":1
    }
    return original,verified


def test_calibration_uses_strictly_verified_past_not_future():
    rows=[]; finals={}
    for i in range(103):
        row,final=historical(i,NOW-timedelta(days=150-i))
        rows.append(row);finals[str(i)]=final
    report=calibration.fitted_temperature(rows,finals,
        as_of=NOW-timedelta(days=120),league="PL",min_samples=100)
    assert report["ready"] is False
    later=calibration.fitted_temperature(rows,finals,
        as_of=NOW,league="PL",min_samples=100)
    assert later["ready"] and later["sample_size"]==103
    assert .8<=later["temperature"]<=1.3
    simulated=calibration.chronological_evaluation(rows,finals,min_samples=100,as_of=NOW)
    assert simulated["leagues"]["PL"]["verified_evaluated"]==3
    assert not simulated["automatic_promotion"]


def test_ai_provider_vote_must_match_frozen_original_forecast():
    original,result=historical(999,NOW-timedelta(days=6))
    base_fields={k:original[k] for k in
        ("provider","fixture_id","home_team_id","away_team_id","kickoff_utc")}
    panel={**base_fields,"source_forecast_id":original["forecast_id"],
           "input_digest":original["input_digest"],
           "captured_at_utc":(NOW-timedelta(days=6,hours=-1)).isoformat(),
           "votes":{"gemini":{"home":.8,"draw":.1,"away":.1}}}
    # Prediction after original but before kickoff: actual kickoff is NOW-6d.
    panel["captured_at_utc"]=(NOW-timedelta(days=6,hours=2)).isoformat()
    report=gate.evaluate([panel],[original],{"999":result},min_paired=1)
    assert report["provider_metrics"]["gemini"]["source_matched_verified_fixtures"]==1
    assert report["provider_metrics"]["gemini"]["production_weight"]==0
    panel["input_digest"]="future_or_other_source"
    assert not gate.evaluate([panel],[original],{"999":result},
                             min_paired=1)["provider_metrics"]


def candidate(fid,home,away):
    kickoff=(NOW+timedelta(hours=3)).isoformat()
    return {"source":{"provider":"API-Football","fixture_id":str(fid),
            "forecast_id":f"forecast_{fid}","input_digest":f"sha_{fid}",
            "home_team_id":str(home),"away_team_id":str(away),
            "model_key":"v8","kickoff_utc":kickoff},
            "snapshot_hash":"a"*64,
            "v8":{"scores":top_scores(GRID)},
            "v22":{"scores":top_scores(GRID),
                   "historical_xg_counts":[8,8]}}


def test_v22_pair_selection_excludes_shared_teams_and_weak_coverage():
    a=candidate(1,11,22)
    b=candidate(2,33,44)
    c=candidate(3,11,45)
    pairs=tournament.rank_pairs([a,b,c],min_coverage=.001)
    assert len(pairs)==2
    assert all(not ({p["a"]["source"]["fixture_id"],p["b"]["source"]["fixture_id"]}==
                     {"1","3"}) for p in pairs)
    assert tournament.rank_pairs([a,b],min_coverage=1.0)==[]


def test_v22_frozen_pair_write_once_and_pending_not_a_loss(monkeypatch,tmp_path):
    a,b=candidate(1,11,22),candidate(2,33,44)
    monkeypatch.setattr(tournament,"frozen_candidates",
                        lambda **kwargs:[a,b])
    frozen=tournament.freeze(now=NOW,root=tmp_path)
    assert len(frozen["newly_frozen_days"])==1
    assert tournament.freeze(now=NOW,root=tmp_path)["newly_frozen_days"]==[]
    monkeypatch.setattr(tournament,"original_model",
                        lambda root,leg:leg if "input_digest" in leg else None)
    report=tournament.evaluate(root=tmp_path,source_root=tmp_path,verified={})
    assert report["methods"]["v22_research"]["pending"]==1
    assert report["methods"]["v22_research"]["hits"]==0
    assert report["methods"]["v22_research"]["hit_rate"] is None
    assert not report["model_promoted"]
