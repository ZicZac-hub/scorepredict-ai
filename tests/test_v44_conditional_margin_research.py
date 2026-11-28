from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
import v44_conditional_margin_research as m

NOW=datetime(2035,1,1,tzinfo=timezone.utc)

def rec(i):
    d=NOW+timedelta(days=i)
    p=np.ones((5,5))/25
    return {"identity":("provider",str(i)),"event":(str(i),),"decision":d,
            "kickoff":d+timedelta(hours=1),"known":d+timedelta(hours=3),
            "actual":(3,0),"a":p,"b":p.copy()}

def test_adjust_preserves_every_total_and_zeros():
    p=np.ones((5,5));p[0,0]=0;p/=p.sum()
    q=m.adjust(p,.3)
    h,a=np.indices(p.shape)
    for t in range(9):
        assert np.isclose(p[h+a==t].sum(),q[h+a==t].sum())
    assert np.array_equal(p==0,q==0)
    assert q[abs(h-a)>=3].sum()>p[abs(h-a)>=3].sum()
    assert np.allclose(m.adjust(p,0),p)
    with pytest.raises(ValueError):
        m.adjust(p,100)

def test_fit_only_prior_final_excludes_target_and_alias():
    prior=[rec(i) for i in range(20)]
    target=rec(22)
    offset,n=m.fit(prior,target,"a")
    assert n==20 and offset>0
    late={**rec(21),"known":target["decision"]}
    alias={**rec(0),"event":target["event"]}
    assert m.fit(prior+[late,alias,target],target,"a")== (offset,n)
    assert m.fit(prior[:19],target,"a")== (None,19)

def test_matched_comparisons_and_empty_archive(tmp_path):
    report=m.walk([rec(i) for i in range(23)])
    assert report["matched_comparison_fixtures"]==3
    assert all(v["fixtures"]==3 for v in report["comparisons"].values())
    with pytest.raises(ValueError):
        m.walk([rec(0),rec(0)])
    report=m.evaluate(source_root=tmp_path/"a",prematch_root=tmp_path/"b",now=NOW)
    assert report["verified_inputs"]==0 and not report["automatic_promotion"]
    assert not list(tmp_path.iterdir())
