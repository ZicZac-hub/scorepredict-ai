"""Guard against accidentally replacing the production forecast with calibration candidate."""
import numpy as np
from engine import _grid

def test_temperature_candidate_math():
    base=_grid(1.4,1.1)
    logits=np.log(np.maximum(base,1e-12))/1.35
    logits-=logits.max()
    smooth=np.exp(logits)
    smooth/=smooth.sum()
    assert np.isclose(smooth.sum(),1.0)
    assert smooth.max()<base.max()
    assert np.unravel_index(smooth.argmax(),smooth.shape)==np.unravel_index(base.argmax(),base.shape)

def test_baseline_default_remains_unchanged():
    import inspect
    from engine import predict
    assert inspect.signature(predict).parameters["variant"].default=="baseline"
