from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
import v44_rest_context_research as rest
from pathlib import Path

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
KICK=NOW+timedelta(hours=2)

def model():
    def side(days):
        return {"last_match_utc":(KICK-timedelta(days=days)).isoformat(),
                "rest_days_since_last_kickoff":days,"finished_matches":10}
    return {"evidence_as_of_utc":NOW.isoformat(),
            "captured_at_utc":NOW.isoformat(),"kickoff_utc":KICK.isoformat(),
            "research_context":{"as_of_utc":NOW.isoformat(),"home":side(6),"away":side(2)}}

def test_qualified_context_and_missing_or_future_rejected():
    m=model()
    assert rest.feature(m)==.5
    m["research_context"]["away"]["rest_days_since_last_kickoff"]=3
    with pytest.raises(ValueError,match="mismatch"):
        rest.feature(m)
    m=model()
    m["research_context"]["away"]["last_match_utc"]=(NOW+timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError,match="insufficient"):
        rest.feature(m)

def test_zero_adjustment_and_normalization():
    p=np.full((5,5),.04)
    assert np.allclose(rest.adjust(p,.5,0.),p)
    q=rest.adjust(p,.5,.12)
    assert np.isclose(q.sum(),1)
    assert q[2,0]>q[0,2]
    assert np.allclose(rest.adjust(p,0.,.12),p)

def test_shrinkage_and_minimum_informative_samples():
    r={"feature":.5,"loss":{b:2. for b in rest.COEFFICIENTS}}
    assert rest.fit([r]*59) is None
    assert rest.fit([r]*60)==0.
    assert rest.fit([{**r,"feature":0.}]*60) is None

def test_near_history_hook_precedes_calculation():
    source=Path("capture_prematch.py").read_text()
    block=source.split("    def _refresh_market(fixture,evidence,observed,code,request,remaining):",1)[1].split("    def _record_daily",1)[0]
    assert block.index('_record_history(fixture,evidence,"near")')<block.index('updated=refresh_v8_near(')
    assert 'load_evidence(' not in block
