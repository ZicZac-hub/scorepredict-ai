from datetime import datetime,timedelta,timezone
import numpy as np
from competition_adaptive_audit import audit,mixture_weights,MODEL_NAMES

NOW=datetime(2032,1,1,tzinfo=timezone.utc)

def row(i,league="PL"):
    stamp=NOW+timedelta(days=i)
    p=np.array([.15,.35,.25,.15,.07,.03])
    return {"source":{"provider":"API-Football","fixture_id":str(i),"forecast_id":str(i),
                      "competition":league,"home_team":"H"+str(i),"away_team":"A"+str(i)},
            "captured":stamp,"kickoff":stamp+timedelta(hours=1),
            "verified_at":stamp+timedelta(hours=4),"grid":np.outer(p,p),"score":(1,1)}

def test_adaptation_uses_prior_verified_results_and_equal_denominators():
    rows=[row(i) for i in range(32)]
    result=audit(rows,now=NOW+timedelta(days=40))
    assert result["chronological_holdout_count"]==2
    assert result["competition_specific_holdout_count"]==2
    assert result["receipts"][0]["prior_global"]==30
    assert result["receipts"][0]["competition_weight"]==30/80
    assert all(m["verified_leg_slots"]==2 for m in result["same_fixture_metrics"].values())
    assert result["receipts"][0]["competition_weights"]["baseline_v8"]>=.5
    assert not result["automatic_promotion"]

def test_future_result_cannot_enter_training_and_unknown_competition_falls_back():
    rows=[row(i) for i in range(31)]
    rows[0]["verified_at"]=NOW+timedelta(days=35)
    assert audit(rows,now=NOW+timedelta(days=40))["chronological_holdout_count"]==0
    rows=[row(i) for i in range(30)]+[row(30,None)]
    r=audit(rows,now=NOW+timedelta(days=40))
    assert r["chronological_holdout_count"]==1
    assert r["competition_specific_holdout_count"]==0
    assert r["receipts"][0]["global_weights"]==r["receipts"][0]["competition_weights"]

def test_empty_prior_is_not_an_accuracy_result():
    assert mixture_weights([]) is None
    r=audit([],now=NOW)
    assert r["same_fixture_metrics"]["competition"]["mean_exact_score_log_loss"] is None
