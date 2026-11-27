import math
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
import v44_shared_tempo_research as tempo


def test_shared_gamma_theoretical_moments():
    h,a=np.indices((64,64))
    mu,nu,r=1.8,1.1,6.
    p=tempo.shared_gamma(mu,nu,(64,64),r)
    assert np.isclose(p.sum(),1)
    assert np.isclose((p*h).sum(),mu,atol=1e-8)
    assert np.isclose((p*a).sum(),nu,atol=1e-8)
    assert np.isclose((p*(h-mu)*(a-nu)).sum(),mu*nu/r,atol=1e-8)
    assert np.isclose((p*(h-mu)**2).sum(),mu+mu**2/r,atol=1e-8)
    assert np.isclose(p[0,0],(r/(r+mu+nu))**r)


def test_independent_poisson_limit():
    mu,nu=1.7,.9
    p=tempo.shared_gamma(mu,nu,(32,32),1e6)
    expected=math.exp(-mu-nu)*mu**2/2*nu
    assert np.isclose(p[2,1],expected,rtol=1e-5)


def test_preserve_original_grid_and_reject_large_tail():
    g=np.full((5,6),1/30)
    before=g.copy()
    f=tempo.families(g)
    assert np.array_equal(g,before)
    assert np.allclose(f["anchor"][:5,:6],g)
    assert f["anchor"][5:,:].sum()==0
    assert f["anchor"][:,6:].sum()==0
    assert np.isclose(f["shared_tempo"].sum(),1)
    with pytest.raises(ValueError,match="tail"):
        tempo.shared_gamma(5,5,(5,5),6)


def test_single_evaluation_no_future_training(monkeypatch):
    monkeypatch.setattr(tempo,"MIN_TRAIN_FIXTURES",1)
    t=datetime(2025,1,1,tzinfo=timezone.utc)
    g=np.full((5,5),1/25)
    f=tempo.families(g)
    def row(i,decision,known):
        return {"provider_id":("p",str(i)),"event":(str(i),),
                "decision":decision,"known":known,"roster":tuple(f),
                "base_grid":g,"groups":f,"actual":[1,1],
                "likelihood":np.array([[.04,.07]])}
    records=[row(1,t,t+timedelta(hours=3)),row(2,t+timedelta(hours=2),t+timedelta(hours=5)),
             row(3,t+timedelta(hours=6),t+timedelta(hours=9))]
    result=tempo.single_evaluation(records)
    assert all(r["fixtures"]==1 for r in result.values())
