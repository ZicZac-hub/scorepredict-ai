"""Fixed V41 upset-aware sensitivity, never a guaranteed result.

Favorite means the side with higher modeled outright win probability, not a
subjective label or a prediction of certain victory. Opposite outcomes are
already possible in the original grid. This bounded challenger tests whether
their probabilities are underestimated.
"""
import numpy as np
from v41_unified_ensemble import normalized

POLICY = {"favorite_minimum_win_probability": .50,
          "minimum_win_probability_gap": .15,
          "underdog_win_multiplier": 1.25, "draw_multiplier": 1.10,
          "maximum_total_variation": .05}


def adjust(grid):
    base=normalized(grid)
    h,a=np.indices(base.shape)
    home,away=h>a,a>h
    draw=h==a
    ph,pa=float(base[home].sum()),float(base[away].sum())
    side="home" if ph>pa else "away"
    if max(ph,pa)<POLICY["favorite_minimum_win_probability"] or abs(ph-pa)<POLICY["minimum_win_probability_gap"]:
        return base.copy(),{"applied":False,"reason":"no_clear_model_favorite","favorite":None}
    favorite,underdog=(home,away) if side=="home" else (away,home)
    changed=base.copy()
    changed[underdog]*=POLICY["underdog_win_multiplier"]
    changed[draw]*=POLICY["draw_multiplier"]
    changed/=changed.sum()
    tv=float(np.abs(changed-base).sum()/2)
    if tv>POLICY["maximum_total_variation"]:
        changed=base+(changed-base)*(POLICY["maximum_total_variation"]/tv)
    changed=normalized(changed,base.shape)
    return changed,{"applied":True,"favorite":side,
        "favorite_before":float(base[favorite].sum()),"favorite_after":float(changed[favorite].sum()),
        "underdog_before":float(base[underdog].sum()),"underdog_after":float(changed[underdog].sum()),
        "draw_before":float(base[draw].sum()),"draw_after":float(changed[draw].sum()),
        "total_variation":float(np.abs(changed-base).sum()/2)}
