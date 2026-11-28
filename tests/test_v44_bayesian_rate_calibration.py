from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
import v44_bayesian_rate_calibration as m

NOW=datetime(2035,1,1,tzinfo=timezone.utc)

def rec(i,league="PL",known=None):
    d=NOW+timedelta(days=i)
    p=np.zeros((5,5));p[1,1]=1.
    return {"identity":("provider",str(i)),"event":(str(i),),"league":league,
            "decision":d,"kickoff":d+timedelta(hours=1),
            "known":known or d+timedelta(hours=3),
            "actual":(0,1),"means":(1.,1.),"a":p,"b":p}

def test_predictive_normalizes_and_uses_uncertainty():
    p=m.predictive(2.,20.,20.)
    assert np.isclose(p.sum(),1)
    assert np.isclose(np.dot(np.arange(len(p)),p),2.)
    assert np.dot((np.arange(len(p))-2)**2,p)>2.
    with pytest.raises(ValueError):
        m.predictive(0.,20.,20.)

def test_strictly_prior_results_and_league_shrinkage():
    records=[rec(i) for i in range(30)]
    current=rec(31)
    fit,n,local=m.fit(records,current)
    assert n==30 and local==30 and fit is not None
    assert fit[2][0]<1 and np.isclose(fit[2][1],1)
    future=rec(30,known=current["decision"])
    changed,n2,_=m.fit(records+[future],current)
    assert n2==n and np.allclose(fit[0],changed[0])
    assert m.fit(records[:29],current)[0] is None

def test_walk_compares_same_rows_and_does_not_mutate():
    records=[rec(i) for i in range(34)]
    report=m.walk(records)
    assert report["matched_comparison_fixtures"]==4
    assert all(a["fixtures"]==4 for a in report["comparisons"].values())
    assert records[0]["b"][1,1]==1
    with pytest.raises(ValueError):
        m.walk(records+[records[0]])

def test_empty_archive(tmp_path):
    report=m.evaluate(source_root=tmp_path/"a",prematch_root=tmp_path/"b",now=NOW)
    assert report["verified_inputs"]==0
    assert report["external_requests"]==0 and not report["automatic_promotion"]
    assert not list(tmp_path.iterdir())
