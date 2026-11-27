"""V9 numerical reliability and reproducibility research; no live changes.

Use the original archived V8 control, not a recomputed later history.
Check all marginal residuals after projection. Never silently accept
nonconvergence or unsupported cells. Compare on the same verified fixtures.
References: Egidi et al. (2018), Statistical Modelling 18, 436–459;
Clarke et al. (2017), American Journal of Sports Science 5, 45–49.
The latter motivates future raw-price calibration; no raw prices are
invented from rounded archived probabilities.
"""
from collections import Counter
from datetime import datetime,timezone
import json,math
import numpy as np

from five_double_accuracy import _final,_result_index,wilson
from v10_prospective import utc
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics
import v44_market_ablation as audit
import v44_fixture_evidence as inventory
import v42_research_trial as sources

WEIGHT=.45
TOL=1e-9
MAX_ITER=1000


def partitions(shape):
    h,a=np.indices(shape)
    return ([h>a,h==a,h<a],[(h+a)<=2,(h+a)>2])


def checked_projection(raw,market,*,weight=WEIGHT,max_iter=MAX_ITER):
    base=_grid(raw)
    if not math.isfinite(weight) or not 0<=weight<=1:
        raise ValueError("invalid_projection_weight")
    if type(max_iter) is not int or not 1<=max_iter<=MAX_ITER:
        raise ValueError("invalid_iteration_limit")
    if not audit._distribution(market.get("winner"),3):
        raise ValueError("invalid_winner_distribution")
    winner,totals=partitions(base.shape)
    groups=[(winner,np.array(market["winner"],dtype=float))]
    if market.get("totals") is not None:
        if not audit._distribution(market["totals"],2):
            raise ValueError("invalid_total_distribution")
        groups.append((totals,np.array(market["totals"],dtype=float)))
    # Target the intended final blended marginals, rather than projecting to
    # potentially inconsistent hard targets and then blending the grid.
    targets=[]
    for masks,p in groups:
        p=p/p.sum()
        observed=np.array([base[m].sum() for m in masks])
        target=(1-weight)*observed+weight*p
        if any(t>TOL and base[m].sum()==0 for m,t in zip(masks,target)):
            raise ValueError("unsupported_market_partition")
        targets.append((masks,target))
    q=base.copy()
    for iteration in range(1,max_iter+1):
        for masks,target in targets:
            for m,t in zip(masks,target):
                mass=float(q[m].sum())
                if t==0:
                    q[m]=0
                elif mass<=0:
                    raise ValueError("unsupported_projection_cell")
                else:
                    q[m]*=t/mass
        error=max(abs(float(q[m].sum())-t) for masks,target in targets for m,t in zip(masks,target))
        if error<=TOL:
            if not np.isfinite(q).all() or np.any(q<0) or abs(q.sum()-1)>TOL:
                raise ValueError("invalid_projected_grid")
            return q,{"iterations":iteration,"max_marginal_error":float(error),
                      "total_variation":float(np.abs(q-base).sum()/2)}
    raise ValueError("projection_not_converged")


def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matches,counts,rejects=audit.match_archive(source_root,clock)
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    rows=[];failures=Counter();residuals=[];reproduction=[]
    from market_score_v9 import _project,MARKET_WEIGHT
    for control,stored in matches:
        try:
            base=_grid(control["score_matrix"])
            market=audit.qualify_market(stored)
            original=_grid(stored["score_matrix"])
            legacy=(1-MARKET_WEIGHT)*base+MARKET_WEIGHT*_project(base,market)
            legacy/=legacy.sum()
            if legacy.shape!=original.shape:
                raise ValueError("legacy_support_mismatch")
            reproduction.append(float(np.abs(legacy-original).sum()/2))
            candidate,diagnostics=checked_projection(base,market)
            residuals.append(diagnostics["max_marginal_error"])
            result=finals.get(inventory._identity(control))
            actual=_final(control,result,clock)
            if actual is None or _final(stored,result,clock)!=actual:
                counts["checked_without_verified_final"]+=1
                continue
            arms={"v8_control":base,"stored_v9":original,"same_base_v9":legacy,
                  "checked_projection":candidate}
            rows.append({"fixture_id":control["fixture_id"],"diagnostics":diagnostics,
                         "metrics":{k:_metrics(g,actual) for k,g in arms.items()}})
        except (ValueError,KeyError,TypeError,ArithmeticError) as exc:
            failures[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    summary={}
    for name in ("v8_control","stored_v9","same_base_v9","checked_projection"):
        m=[r["metrics"][name] for r in rows]
        n=len(m);hits=sum(v["top_score_hit"] for v in m)
        summary[name]={"fixtures":n,"hits":hits,"hit_rate":hits/n if n else None,
            "wilson_95":wilson(hits,n),
            **{k:math.fsum(v[k] for v in m)/n if n else None
               for k in ("exact_log_loss","score_brier","outcome_brier")}}
    return {"study_type":"v9_same_base_checked_projection_replay","counts":dict(counts),
        "verified_matched_fixtures":len(rows),"comparisons":summary,
        "legacy_reproduction":{"checked":len(reproduction),
          "within_1e_8":sum(x<=1e-8 for x in reproduction),
          "maximum_total_variation":max(reproduction,default=None)},
        "checked_projection_max_residual":max(residuals,default=None),
        "projection_rejections":dict(failures),"source_rejections":dict(rejects),
        "conflicting_results":len(conflicts),"result_rejections":dict(result_rejects),
        "external_requests":0,"automatic_promotion":False,"rows":rows,
        "limitations":["Fixed 45 percent target blend is an experimental design choice.",
                       "This is an archived replay, not a prospective accuracy result.",
                       "Stored normalized probabilities do not independently prove the original raw market prices.",
                       "No new match selections or production predictions are generated."]}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
