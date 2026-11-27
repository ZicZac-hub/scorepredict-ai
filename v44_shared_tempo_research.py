"""Shared match-tempo Gamma-Poisson challenger (research only).

Conditional on a common mean-one Gamma variable, the two goal counts are
independent Poisson; marginally their counts are overdispersed and positively
dependent. Inspired by JRSS-C 74(3), 717 (2025), shared Gamma random effect:
https://academic.oup.com/jrsssc/article/74/3/717/7929974
Unlike V21's independent NB or additive bivariate Poisson, this is a common
multiplicative tempo. Rates still derive from V8, not independent evidence.
Fixed r=6,20 form ONE challenger family. No retrospective model selection.
"""
import json
import math
import numpy as np

from v44_family_pool_research import _grid
import v44_pair_transfer_research as transfer

SHAPES=(6.,20.)
MIN_SUPPORT=32
MAX_TAIL=1e-4


def shared_gamma(mu_h,mu_a,shape,dispersion):
    """Exact finite-support cells of the negative multinomial distribution.

    No overflow folded into an exact score; discarded tail must be tiny.
    Normalization is explicitly conditional on the represented score grid.
    """
    if (not all(math.isfinite(v) and v>0 for v in (mu_h,mu_a,dispersion))
            or len(shape)!=2 or any(type(v) is not int or not 5<=v<=256 for v in shape)):
        raise ValueError("invalid shared-tempo parameters")
    r=dispersion
    denominator=r+mu_h+mu_a
    p0,ph,pa=r/denominator,mu_h/denominator,mu_a/denominator
    out=np.zeros(shape)
    out[0,0]=math.exp(r*math.log(p0))
    for h in range(1,shape[0]):
        out[h,0]=out[h-1,0]*(r+h-1)/h*ph
    for h in range(shape[0]):
        for a in range(1,shape[1]):
            out[h,a]=out[h,a-1]*(r+h+a-1)/a*pa
    mass=float(out.sum())
    if not math.isfinite(mass) or not 1-MAX_TAIL<=mass<=1+1e-10:
        raise ValueError("shared_tempo_tail_exceeds_limit")
    return out/mass


def families(raw):
    base=_grid(raw)
    h,a=np.indices(base.shape)
    mu_h=float((h*base).sum())
    mu_a=float((a*base).sum())
    if not (.15<=mu_h<=5 and .15<=mu_a<=5):
        raise ValueError("shared_tempo_means_outside_scope")
    shape=tuple(max(MIN_SUPPORT,n) for n in base.shape)
    anchor=np.pad(base,[(0,shape[d]-base.shape[d]) for d in (0,1)])
    tempo=np.mean([shared_gamma(mu_h,mu_a,shape,r) for r in SHAPES],axis=0)
    # Dictionary order is stable and agrees between training and pair replay.
    return {"anchor":anchor,"shared_tempo":tempo}


def evaluate(**kwargs):
    report=transfer.evaluate(family_fn=families,**kwargs)
    report["study_type"]="shared_gamma_tempo_original_pair_replay"
    report["comparisons"]={
        {"v44_equal":"tempo_equal","v44_transfer":"tempo_adaptive"}.get(name,name):value
        for name,value in report["comparisons"].items()}
    for row in report["rows"]:
        row["metrics"]={
            {"v44_equal":"tempo_equal","v44_transfer":"tempo_adaptive"}.get(name,name):value
            for name,value in row["metrics"].items()}
    report["method"]={"gamma_shapes":list(SHAPES),"minimum_support":MIN_SUPPORT,
                      "max_discarded_tail":MAX_TAIL,
                      "training":"same chronological regularized mixture likelihood as V44 transfer",
                      "family_roster":["anchor","shared_tempo"]}
    report["limitations"].append("Tempo introduces within-match goal dependence, not dependence between distinct fixtures.")
    report["limitations"].append("The tempo family is derived from baseline expected goals, not an independently fitted team-strength model.")
    return report


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
