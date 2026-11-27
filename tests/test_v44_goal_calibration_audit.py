from datetime import datetime,timezone
import numpy as np
import pytest
import v44_goal_calibration_audit as g

def test_marginals_preserve_joint_total_distribution():
    p=np.zeros((5,5));p[0,0]=.25;p[2,3]=.75
    m=g.marginals(p)
    assert np.allclose(m["home_goals"],[.25,0,.75,0,0])
    assert np.allclose(m["away_goals"],[.25,0,0,.75,0])
    assert m["total_goals"][5]==.75
    assert all(np.isclose(x.sum(),1) for x in m.values())

def test_nonrandomized_pit_is_uniform_in_exact_weighted_population():
    p=[.2,.3,.5]
    h=np.zeros(g.BINS)
    for y,w in enumerate(p):
        h+=w*np.array(g.diagnose(p,y)["pit_bin_mass"])
    assert np.allclose(h,np.full(g.BINS,1/g.BINS))

def test_ranked_probability_score_including_tail():
    assert g.diagnose([1.,0],0)["ranked_probability_score"]==0
    r=g.diagnose([1.,0],4)
    assert r["ranked_probability_score"]==4
    assert r["outside_support"] and r["zero_probability_observation"]
    assert r["pit_bin_mass"][-1]==1
    assert g.diagnose([0.,1],0)["pit_bin_mass"][0]==1
    assert g.diagnose([.5,.5],0)["ranked_probability_score"]==.25

def test_bad_input_and_empty_archive(tmp_path):
    for p,y in [([.3,.3],0),([1,0],-1),([1,0],True),([float("nan"),1],0)]:
        with pytest.raises(ValueError):
            g.diagnose(p,y)
    r=g.evaluate(source_root=tmp_path/"a",prematch_root=tmp_path/"b",now=datetime(2025,1,1,tzinfo=timezone.utc))
    assert r["verified_matched_fixtures"]==0
    assert r["comparisons"]["v9"]["total_goals"]["nonrandomized_pit_bin_fractions"] is None
    assert r["external_requests"]==0 and r["automatic_promotion"] is False
    assert not list(tmp_path.iterdir())
