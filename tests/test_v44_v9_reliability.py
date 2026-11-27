import numpy as np
import pytest
from v44_v9_reliability import checked_projection,partitions,TOL

BASE=np.full((5,5),.04)
MARKET={"winner":[.55,.25,.2],"totals":[.45,.55]}

def test_checked_projection_hits_both_soft_targets():
    result,diag=checked_projection(BASE,MARKET)
    for masks,values in zip(partitions(BASE.shape),(MARKET["winner"],MARKET["totals"])):
        expected=.55*np.array([BASE[m].sum() for m in masks])+.45*np.array(values)
        assert np.allclose([result[m].sum() for m in masks],expected,atol=TOL,rtol=0)
    assert np.isclose(result.sum(),1)
    assert diag["max_marginal_error"]<=TOL
    assert np.all(BASE==.04)

def test_zero_weight_preserves_input_and_missing_totals_allowed():
    result,_=checked_projection(BASE,{"winner":MARKET["winner"]},weight=0.)
    assert np.allclose(result,BASE)
    result,_=checked_projection(BASE,{"winner":MARKET["winner"]})
    assert np.isclose(result.sum(),1)

def test_fails_closed_on_invalid_or_unsupported_constraints():
    with pytest.raises(ValueError,match="invalid_winner"):
        checked_projection(BASE,{"winner":[.6,.5,.2]})
    with pytest.raises(ValueError,match="unsupported"):
        checked_projection(np.eye(5)/5,MARKET)
    with pytest.raises(ValueError,match="not_converged"):
        checked_projection(BASE,MARKET,max_iter=1)
