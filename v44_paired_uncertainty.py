"""Read-only paired forecast uncertainty diagnostic on authenticated archives.

Compare identical fixtures and retain all observations within each calendar
block. Percentile resampling intervals are descriptive sensitivity estimates,
not calibrated prospective guarantees or a release gate. No forecasts,
pair selection, provider requests, or archive mutations.
"""
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,math
import numpy as np

from v10_prospective import utc
from five_double_accuracy import _result_index,_final
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics
from v44_fixture_evidence import _identity
import v44_market_ablation as archive
import v42_research_trial as sources

SEED=440151
REPLICATES=4000
MIN_BLOCKS=10
MIN_FIXTURES=30
METRICS=("exact_log_loss","score_brier","outcome_brier","top_score_hit")


def summarize(rows,*,period="day"):
    if period not in ("day","week"):
        raise ValueError("invalid_block_period")
    groups=defaultdict(list);seen=set()
    for row in rows:
        identity=tuple(row["identity"])
        if identity in seen:
            raise ValueError("duplicate_fixture")
        seen.add(identity)
        at=utc(row["kickoff_utc"])
        key=at.date().isoformat() if period=="day" else tuple(at.isocalendar()[:2])
        diff=[]
        for name in METRICS:
            a,b=(row["metrics"][m][name] for m in ("v8_control","v9"))
            if name=="top_score_hit":
                if type(a) is not bool or type(b) is not bool:
                    raise ValueError("invalid_hit")
            elif any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (a,b)):
                raise ValueError("invalid_loss")
            diff.append(float(b)-float(a))
        groups[key].append(diff)
    n=len(seen);g=len(groups)
    arrays=[np.asarray(groups[k],dtype=float) for k in sorted(groups)]
    total=np.sum([a.sum(axis=0) for a in arrays],axis=0) if n else np.zeros(len(METRICS))
    averages=total/n if n else None
    status="descriptive_resampling_available" if n>=MIN_FIXTURES and g>=MIN_BLOCKS else "insufficient_distinct_blocks_or_fixtures"
    out={"status":status,"period":period,"fixtures":n,"blocks":g,
         "minimum_blocks":MIN_BLOCKS,"minimum_fixtures":MIN_FIXTURES,
         "metrics":{name:{"mean_v9_minus_v8":float(averages[i]) if n else None,
                          "descriptive_95_percentile_interval":None}
                    for i,name in enumerate(METRICS)}}
    if status!="descriptive_resampling_available":
        return out
    sums=np.stack([a.sum(axis=0) for a in arrays])
    sizes=np.array([len(a) for a in arrays],dtype=float)
    rng=np.random.default_rng(SEED)
    picks=rng.integers(0,g,size=(REPLICATES,g))
    draws=sums[picks].sum(axis=1)/sizes[picks].sum(axis=1)[:,None]
    bounds=np.quantile(draws,[.025,.975],axis=0)
    for i,name in enumerate(METRICS):
        out["metrics"][name]["descriptive_95_percentile_interval"]=[float(v) for v in bounds[:,i]]
    out["replicates"]=REPLICATES
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
        rows.append({"identity":list(_identity(control)),
                     "kickoff_utc":control["kickoff_utc"],
                     "metrics":{m["model_key"]:_metrics(_grid(m["score_matrix"]),actual)
                                for m in (control,market)}})
    return {"study_type":"paired_archived_forecast_uncertainty_v1",
            "counts":dict(counts),"source_rejections":dict(rejects),
            "result_conflicts":len(conflicts),"result_rejections":dict(result_rejects),
            "daily_blocks":summarize(rows,period="day"),
            "weekly_blocks":summarize(rows,period="week"),
            "external_requests":0,"automatic_promotion":False,
            "limitations":["Historical matched fixture results are not prospective five-double results.",
                           "Calendar-block resampling preserves within-block dependence but cannot establish between-block independence.",
                           "Intervals are exploratory, not formal significance tests or repeated-look-adjusted inference.",
                           "The fixed minimum sample thresholds are conservative reporting choices, not statistical guarantees.",
                           "Missing finals are excluded and reported; they are not losses."],"rows":rows}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
