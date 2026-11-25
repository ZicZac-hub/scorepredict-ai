import numpy as np
import pytest
from v42_research_engine import hypotheses, ensemble
from v42_research_trial import PROTOCOL, _arms

def grid():
    return np.outer([.30,.35,.20,.10,.04,.01],[.40,.30,.17,.08,.04,.01])

def test_normalized_deterministic_and_immutable():
    p=grid()
    before=p.copy()
    for q in [*hypotheses(p).values(),ensemble(p)]:
        assert np.isclose(q.sum(),1) and np.all(q>=0) and q.shape==p.shape
    assert np.array_equal(p,before)
    assert np.array_equal(ensemble(p),ensemble(p))

def test_directional_hypotheses():
    p=grid()
    h,a=np.indices(p.shape)
    arms=hypotheses(p)
    assert arms["high_goal"][h+a>=4].sum()>p[h+a>=4].sum()
    assert arms["low_goal"][h+a<=1].sum()>p[h+a<=1].sum()
    assert arms["wide_margin"][abs(h-a)>=3].sum()>p[abs(h-a)>=3].sum()

def test_invalid_and_missing_evidence():
    p=grid()
    with pytest.raises(ValueError):
        ensemble(p*2)
    arms=_arms([{"score_matrix":p.tolist()}]*2,[{},{}])
    final=arms["unified_final"]
    assert final["status"]=="issued"
    assert final["paired_xg_used"] is False and final["paired_market_used"] is False
    assert len({(x["score1"],x["score2"]) for x in final["five_doubles"]})==5
    assert PROTOCOL["automatic_model_promotion"] is False
