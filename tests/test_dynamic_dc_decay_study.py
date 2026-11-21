"""Research-only decay selection and score comparison mechanics."""
import numpy as np
from run_dynamic_dc_decay_study import score_pair, select_half_life

def test_decay_choice_uses_log_loss_not_hits():
    validation={"matched_pairs":40,"models":{
        "90":{"score_log_loss":3.0,"hits":1},
        "180":{"score_log_loss":3.2,"hits":10},
        "365":{"score_log_loss":3.1,"hits":2}}}
    assert select_half_life(validation)==90
    validation["matched_pairs"]=29
    assert select_half_life(validation) is None

def test_score_pair_uniform_grid():
    a=np.full((3,3),1/9)
    hit,loss,coverage=score_pair([a,a],((0,0),(0,0)))
    assert hit==1
    assert np.isclose(loss,np.log(9))
    assert np.isclose(coverage,5/81)
