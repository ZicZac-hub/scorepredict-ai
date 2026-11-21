"""V21 prospective freeze: distinct source forecast IDs, abstention and verified outcomes."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import numpy as np
import pytest

import v21_prospective as research
from v21_goal_model_comparison import _poisson


NOW=datetime(2032,11,15,2,tzinfo=timezone.utc)
KICK=(NOW+timedelta(hours=6)).isoformat()
GRID=np.outer(_poisson(1.6,11),_poisson(1.3,11)).tolist()


def source(fid,home,away):
    return {"model_key":"v8","provider":"API-Football","fixture_id":str(fid),
            "forecast_id":f"forecast{fid}","input_digest":f"digest{fid}",
            "home_team_id":str(home),"away_team_id":str(away),
            "home_team":str(home),"away_team":str(away),
            "kickoff_utc":KICK,"captured_at_utc":(NOW-timedelta(hours=2)).isoformat(),
            "evidence_as_of_utc":(NOW-timedelta(hours=3)).isoformat(),
            "competition":"BSA","league_history_matches":130,
            "score_matrix":GRID}


def test_all_same_day_pairs_ranked_and_source_frozen_once(monkeypatch,tmp_path):
    rows=[source(100,10,11),source(200,20,21),source(300,30,31)]
    byid={x["fixture_id"]:x for x in rows}
    monkeypatch.setattr(research,"original_model",
                        lambda root,leg:byid.get(str(leg["fixture_id"])))
    new=research.freeze_models(root=tmp_path,source_root=tmp_path,now=NOW,originals=rows)
    assert len(new["new_frozen_fixtures"])==3
    again=research.freeze_models(root=tmp_path,source_root=tmp_path,now=NOW,originals=rows)
    assert again["new_frozen_fixtures"]==[]
    frozen=[json.loads(x.read_text()) for x in (tmp_path/"forecasts").glob("*.json")]
    options=research.rank_pairs(frozen,min_coverage=0.001)
    assert len(options)==3
    run=research.freeze_pairs(root=tmp_path,now=NOW,min_coverage=.001)
    assert run["newly_frozen_days"]
    assert len(list((tmp_path/"pairs").glob("*.json")))==1
    assert research.freeze_pairs(root=tmp_path,now=NOW)["newly_frozen_days"]==[]
    result=research.evaluate(root=tmp_path,source_root=tmp_path,verified={},now=NOW)
    assert result["methods"]["baseline_v8"]["verified"]==0
    assert result["methods"]["baseline_v8"]["pending"]==1
    assert not result["models_promoted"]


def test_no_pair_when_three_teams_duplicate_or_coverage_low():
    a=source(100,10,11)
    b=source(200,10,22)
    models=research._models(a)
    for row in (a,b):
        row.update({"status":"frozen_pre_kickoff","model_results":models})
    assert research.rank_pairs([a,b],min_coverage=0)==[]
    b["home_team_id"]="30"
    assert research.rank_pairs([a,b],min_coverage=1)==[]


def test_no_postkickoff_model_freeze(monkeypatch,tmp_path):
    old=source(100,10,11)
    monkeypatch.setattr(research,"original_model",lambda root,leg:old)
    result=research.freeze_models(root=tmp_path,source_root=tmp_path,
                                   now=NOW+timedelta(hours=7),originals=[old])
    assert not result["new_frozen_fixtures"]
