from datetime import datetime, timedelta, timezone

import numpy as np
from v41_weight_audit import audit, distributions, METHODS, MIN_PRIOR

NOW=datetime(2035,6,1,tzinfo=timezone.utc)
GRID=np.outer([.30,.35,.20,.10,.04,.01],[.30,.35,.20,.10,.04,.01])


def row(i, captured, *, delay=timedelta(hours=4), provider="API-Football"):
    return {"source":{"provider":provider,"fixture_id":str(i),
                      "forecast_id":"original-"+str(i),"home_team":"Home "+str(i),
                      "away_team":"Away "+str(i)},
            "grid":GRID.copy(),"score":(1,0),"captured":captured,
            "kickoff":captured+timedelta(hours=1),"verified_at":captured+delay}


def test_fixed_sensitivities_preserve_normalization_and_original():
    original=GRID.copy()
    result=distributions(GRID)
    assert set(result)==set(METHODS)
    assert np.array_equal(original,GRID)
    assert all(np.isclose(p.sum(),1) and np.all(p>=0) for p in result.values())


def test_future_finals_cannot_train_past_capture():
    rows=[row(i,NOW-timedelta(days=2),delay=timedelta(hours=4))
          for i in range(MIN_PRIOR+10)]
    result=audit(rows,now=NOW)
    assert result["verified_unique_fixtures"]==MIN_PRIOR+10
    assert result["maximum_prior_before_historical_capture"]==0
    assert result["chronological_holdout_count"]==0
    assert result["chronological_selected"]["mean_exact_score_log_loss"] is None
    assert result["no_live_weight_change"] is True


def test_chronological_training_and_cross_provider_duplicates():
    rows=[row(i,NOW-timedelta(days=3)) for i in range(MIN_PRIOR)]
    duplicate={**rows[0],"source":{**rows[0]["source"],"provider":"other"}}
    future=row(900,NOW-timedelta(days=1))
    result=audit(rows+[duplicate,future],now=NOW)
    assert result["verified_unique_fixtures"]==MIN_PRIOR+1
    assert result["excluded"]==1
    assert result["chronological_holdout_count"]==1
    assert result["holdout_receipts"][0]["prior_verified_count"]==MIN_PRIOR
    assert result["holdout_receipts"][0]["fixture_id"]=="900"
    assert result["holdout_minimum_met"] is False
    assert result["target_70_percent_proven"] is False
