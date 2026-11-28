from datetime import timedelta
import numpy as np
import pytest
import v44_joint_pair_transfer as m
from test_v44_v9_pair_replay import NOW,model

def fixture(i,kind):
    return {**model(i,kind),"competition":"PL","score_matrix":(np.ones((5,5))/25).tolist()}

def test_both_targets_aliases_and_late_finals_excluded():
    models=[fixture(i,"v8_control") for i in ("1","2")]
    legs=[(x,fixture(i,"v9")) for x,i in zip(models,("1","2"))]
    old=NOW-timedelta(days=1)
    prior=[{"identity":("provider",str(i)),"event":("prior",str(i)),
            "known":old,"league":"PL","actual":(1,0),"means":(2.,2.)}
           for i in range(30)]
    half,full,sizes=m.corrections(prior,models,legs,NOW)
    assert all(s["prior_fixtures"]==30 for s in sizes)
    contaminants=[{**prior[0],"identity":m.inventory._identity(models[1])},
                  {**prior[0],"event":m.inventory._event(models[0])},
                  {**prior[0],"known":NOW}]
    h,f,s=m.corrections(prior+contaminants,models,legs,NOW)
    assert s==sizes
    assert all(np.allclose(x,y) for x,y in zip(half,h))
    assert all(np.allclose(x,y) for x,y in zip(full,f))
    with pytest.raises(ValueError,match="insufficient_prior_fixtures"):
        m.corrections(prior[:29],models,legs,NOW)
    with pytest.raises(ValueError,match="missing_predecision_market_match"):
        m.corrections(prior,models,legs,NOW-timedelta(seconds=1))

def test_empty_archive_read_only(tmp_path):
    r=m.evaluate(source_root=tmp_path/"a",prematch_root=tmp_path/"b",pair_root=tmp_path/"c",now=NOW)
    assert r["verified_training_fixtures"]==0
    assert all(v["pairs"]==0 and v["hit_rate"] is None for v in r["matched_pair_comparisons"].values())
    assert not r["automatic_promotion"] and r["external_requests"]==0
    assert not list(tmp_path.iterdir())
