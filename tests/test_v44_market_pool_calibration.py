from datetime import datetime,timedelta,timezone
import numpy as np
import v44_market_pool_calibration as m

NOW=datetime(2025,1,1,tzinfo=timezone.utc)

def record(i,known,p0=.05,p1=.2):
    return {"identity":("p",str(i)),"event":("event",str(i)),"known":known,
            "p0":p0,"p1":p1}

def test_training_excludes_future_equal_cutoff_and_target():
    rows=[record(i,NOW-timedelta(days=1)) for i in range(30)]
    rows += [record(40,NOW),record(41,NOW+timedelta(seconds=1))]
    assert m.fit(rows,NOW,("p","0"),("event","0"))==(None,29)
    w,n=m.fit(rows,NOW,("p","99"),("event","99"))
    assert n==30 and w==1.

def test_regularization_prefers_equal_when_models_agree():
    rows=[record(i,NOW-timedelta(days=1),.1,.1) for i in range(30)]
    assert m.fit(rows,NOW,("p","99"),("event","99"))==(.5,30)

def test_pool_preserves_mass_and_differing_score_support():
    a=np.full((5,6),1/30)
    b=np.full((7,5),1/35)
    p=m.pool(a,b,.25)
    assert p.shape==(7,6)
    assert np.isclose(p.sum(),1)
    assert np.allclose(p[:5,:5],.75*a[:,:5]+.25*b[:5,:])
