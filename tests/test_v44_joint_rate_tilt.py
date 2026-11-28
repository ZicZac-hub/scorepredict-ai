import numpy as np
import pytest
import v44_joint_rate_tilt as m
from test_v44_bayesian_rate_calibration import rec,NOW

def test_joint_tilt_preserves_odds_ratio_and_zeros():
    p=np.array([[.1,.2,0.],[.3,.4,0.]])
    q=m.tilt(p,(.7,1.2),1.)
    assert np.isclose(q.sum(),1)
    assert np.array_equal(p==0,q==0)
    assert np.isclose(q[0,0]*q[1,1]/(q[0,1]*q[1,0]),
                      p[0,0]*p[1,1]/(p[0,1]*p[1,0]))
    assert np.dot(q.sum(axis=1),[0,1])<np.dot(p.sum(axis=1),[0,1])
    assert np.allclose(m.tilt(p,(1.,1.),.5),p)
    assert np.allclose(m.tilt(p,(.7,1.2),0.),p)
    assert np.isclose(p[1,1],.4)

def test_invalid_parameters():
    p=np.ones((2,2))/4
    for scale,strength in [((0.,1.),1.),((1.,float("inf")),.5),((1.,1.),.25)]:
        with pytest.raises(ValueError):
            m.tilt(p,scale,strength)

def test_same_fixture_comparison_and_warmup():
    rows=[rec(i) for i in range(34)]
    report=m.walk(rows)
    assert report["matched_comparison_fixtures"]==4
    assert all(v["fixtures"]==4 for v in report["comparisons"].values())
    assert report["exclusions"]["insufficient_prior_fixtures"]==30
    with pytest.raises(ValueError):
        m.walk(rows+[rows[0]])

def test_empty_archive(tmp_path):
    report=m.evaluate(source_root=tmp_path/"a",prematch_root=tmp_path/"b",now=NOW)
    assert report["verified_inputs"]==0
    assert not report["automatic_promotion"]
    assert report["external_requests"]==0
    assert not list(tmp_path.iterdir())
