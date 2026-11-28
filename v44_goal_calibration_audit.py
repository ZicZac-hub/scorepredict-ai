"""Read-only calibration diagnostics for discrete goal distributions.

Use the nonrandomized PIT construction for count forecasts (Czado,
Gneiting and Held, 2009): distribute each observation uniformly over
[F(y-1), F(y)]. This avoids adding random noise to a small sample.
Marginal home/away/total checks cannot establish joint calibration.
No new forecasts, provider requests, parameter fitting or promotion.
"""
from datetime import datetime,timezone
from collections import Counter
import json
import numpy as np

from v10_prospective import utc
from five_double_accuracy import _result_index,_final
from v44_family_pool_research import _grid
from v44_fixture_evidence import _identity
import v44_market_ablation as archive
import v42_research_trial as sources

BINS=10
MODELS=("v8_control","v9")
MARGINALS=("home_goals","away_goals","total_goals")

def marginals(raw):
    p=_grid(raw)
    h,a=np.indices(p.shape)
    total=np.bincount((h+a).ravel(),weights=p.ravel(),minlength=sum(p.shape)-1)
    return dict(zip(MARGINALS,(p.sum(axis=1),p.sum(axis=0),total)))

def diagnose(raw,actual):
    p=np.asarray(raw,dtype=float)
    if (p.ndim!=1 or not len(p) or not np.isfinite(p).all()
            or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-6,rtol=0)):
        raise ValueError("invalid_marginal")
    if type(actual) is not int or actual<0:
        raise ValueError("invalid_actual_count")
    p=p/p.sum()
    cdf=np.cumsum(p)
    cdf[-1]=1.
    lower=float(cdf[actual-1]) if 0<actual<=len(p) else float(actual>len(p))
    upper=float(cdf[actual]) if actual<len(p) else 1.
    lower=max(0.,min(1.,lower));upper=max(lower,min(1.,upper))
    mass=upper-lower
    edges=np.linspace(0,1,BINS+1)
    hist=np.zeros(BINS)
    if mass>0:
        hist=np.maximum(0.,np.minimum(upper,edges[1:])-np.maximum(lower,edges[:-1]))/mass
        hist/=hist.sum()
    else:
        hist[min(BINS-1,int(lower*BINS))]=1.
    support=np.arange(len(p))
    mean=float(np.dot(p,support))
    # Ranked probability score on ordered counts; include the entire
    # out-of-support error without allocating an unbounded array.
    rps=float(np.square(cdf-(support>=actual)).sum()+max(0,actual-len(p)))
    return {"expected":mean,"actual":actual,"error_observed_minus_expected":actual-mean,
            "variance":float(np.dot(p,np.square(support-mean))),
            "ranked_probability_score":rps,"pit_bin_mass":hist.tolist(),
            "outside_support":bool(actual>=len(p)),
            "zero_probability_observation":bool(actual>=len(p) or p[actual]==0)}

def summarize(rows):
    out={}
    for model in MODELS:
        out[model]={}
        for name in MARGINALS:
            items=[r["diagnostics"][model][name] for r in rows]
            n=len(items)
            hist=np.mean([x["pit_bin_mass"] for x in items],axis=0).tolist() if n else None
            out[model][name]={"fixtures":n,
                **{"mean_"+key:float(np.mean([x[key] for x in items])) if n else None
                   for key in ("expected","actual","error_observed_minus_expected","variance","ranked_probability_score")},
                "nonrandomized_pit_bin_fractions":hist,
                "outside_support":int(sum(x["outside_support"] for x in items)),
                "zero_probability_observations":int(sum(x["zero_probability_observation"] for x in items))}
    return out

def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matches,counts,rejects=archive.match_archive(source_root,clock)
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    rows=[]
    for control,market in matches:
        actual=_final(control,finals.get(_identity(control)),clock)
        if actual is None or _final(market,finals.get(_identity(market)),clock)!=actual:
            counts["without_verified_final"]+=1
            continue
        observed=dict(zip(MARGINALS,(actual[0],actual[1],sum(actual))))
        diagnostics={}
        for m in (control,market):
            diagnostics[m["model_key"]]={name:diagnose(p,observed[name])
                                         for name,p in marginals(m["score_matrix"]).items()}
        rows.append({"identity":list(_identity(control)),
                     "kickoff_utc":control["kickoff_utc"],"diagnostics":diagnostics})
    return {"study_type":"archived_goal_count_calibration_v1",
            "verified_matched_fixtures":len(rows),"counts":dict(counts),
            "source_rejections":dict(rejects),"result_conflicts":len(conflicts),
            "result_rejections":dict(result_rejects),"comparisons":summarize(rows),
            "external_requests":0,"automatic_promotion":False,"rows":rows,
            "limitations":["Descriptive marginal calibration does not imply correct joint score probabilities.",
                           "PIT histograms from small or dependent samples are not formal uniformity tests.",
                           "Historical individual-fixture metrics are not five-double hit rates.",
                           "No forecasts, fitted parameters, or release decisions are produced."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
