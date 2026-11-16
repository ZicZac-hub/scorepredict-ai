"""Comparable, pre-match-only benchmark for football prediction strategies.

Input rows contain a fixture identity, original forecast timestamp, kickoff,
verified regulation-time score and per-model frozen score distributions.
No provider or AI is credited without a stored, timestamped prediction.
"""
from datetime import datetime, timezone
import math

def _time(value):
    try:
        t=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except (TypeError,ValueError,OverflowError):
        return None

def _actual(value):
    if not isinstance(value,(tuple,list)) or len(value)!=2:
        return None
    if any(type(x) is not int or x<0 for x in value):
        return None
    return tuple(value)

def benchmark(rows, min_sample=100):
    """Return exact-hit, top-five coverage and multiclass Brier for each model.

    Each row: fixture_id, kickoff, actual=[home,away],
    forecasts={model: {created_at, scores:[[home,away,prob],...]}}.
    This is a per-match benchmark; five-double success is a separate metric.
    """
    by_model={}
    seen=set()
    rejected={"unsettled":0,"duplicate_fixture":0,"late_or_untimed":0,
              "invalid_distribution":0}
    for row in rows:
        fid=str(row.get("fixture_id") or "")
        actual=_actual(row.get("actual"))
        kickoff=_time(row.get("kickoff"))
        if not fid or actual is None or kickoff is None:
            rejected["unsettled"]+=1
            continue
        if fid in seen:
            rejected["duplicate_fixture"]+=1
            continue
        seen.add(fid)
        for name,forecast in (row.get("forecasts") or {}).items():
            if not isinstance(forecast,dict) or (stamp:=_time(forecast.get("created_at"))) is None or stamp>=kickoff:
                rejected["late_or_untimed"]+=1
                continue
            probs={}
            valid=True
            for item in forecast.get("scores") or []:
                try:
                    h,a,p=item
                    if type(h) is not int or type(a) is not int or min(h,a)<0:
                        valid=False;break
                    p=float(p)
                    if not math.isfinite(p) or p<0 or p>1 or (h,a) in probs:
                        valid=False;break
                    probs[h,a]=p
                except (ValueError,TypeError):
                    valid=False;break
            mass=sum(probs.values())
            # Accept a complete distribution or a declared truncated top-k,
            # but never normalize a truncated list into a fake full distribution.
            if not valid or not probs or mass>1.000001:
                rejected["invalid_distribution"]+=1
                continue
            top=sorted(probs,key=lambda score:(-probs[score],score))
            bucket=by_model.setdefault(str(name),{"n":0,"hits":0,"top5_hits":0,
                                                 "top5_probability_sum":0.0,
                                                 "brier_sum":0.0,"brier_n":0})
            bucket["n"]+=1
            bucket["hits"]+=int(top[0]==actual)
            bucket["top5_hits"]+=int(actual in top[:5])
            bucket["top5_probability_sum"]+=sum(probs[k] for k in top[:5])
            if mass>=.999999:
                bucket["brier_sum"]+=sum(p*p for p in probs.values())+1-2*probs.get(actual,0)
                bucket["brier_n"]+=1
    models={}
    for name,b in sorted(by_model.items()):
        n=b["n"]
        models[name]={"fixtures":n,"exact_hit_rate":b["hits"]/n,
                      "top5_hit_rate":b["top5_hits"]/n,
                      "mean_top5_coverage":b["top5_probability_sum"]/n,
                      "multiclass_brier":b["brier_sum"]/b["brier_n"] if b["brier_n"] else None,
                      "full_distribution_count":b["brier_n"],
                      "sample_status":"sufficient_for_initial_review" if n>=min_sample else "insufficient",
                      "eligible_for_promotion":False}
    return {"models":models,"rejected":rejected,
            "note":"Never compare models on different fixture cohorts; intersect fixture IDs first. "
                   "A holdout, calibration, uncertainty and five-double audit are required before promotion."}


def paired_benchmark(rows):
    """Compare only forecasts stored together on unique settled fixtures.

    Duplicate fixture IDs are excluded altogether, including conflicting
    scores/kickoffs. This prevents an appended or altered record from
    silently replacing a genuinely frozen forecast in the leaderboard.
    """
    from collections import Counter
    ids=[str(row.get("fixture_id") or "") for row in rows if isinstance(row,dict)]
    counts=Counter(ids)
    unique=[row for row in rows if isinstance(row,dict)
            and row.get("fixture_id") and counts[str(row["fixture_id"])]==1]
    eligible={}
    for row in unique:
        valid=set()
        for name,forecast in (row.get("forecasts") or {}).items():
            candidate=dict(row,forecasts={name:forecast})
            if name in benchmark([candidate])["models"]:
                valid.add(name)
        eligible[str(row["fixture_id"])]=(row,valid)
    model_names=set().union(*(valid for _,valid in eligible.values())) if eligible else set()
    if len(model_names)<2:
        return {"shared_fixtures":0,"models":{},"duplicate_fixtures_excluded":sum(v>1 for v in counts.values())}
    shared=[row for row,valid in eligible.values() if model_names.issubset(valid)]
    if not shared:
        return {"shared_fixtures":0,"models":{},"duplicate_fixtures_excluded":sum(v>1 for v in counts.values())}
    models={}
    for name in sorted(model_names):
        sample=[dict(row,forecasts={name:row["forecasts"][name]}) for row in shared]
        result=benchmark(sample)
        if name in result["models"]:
            models[name]=result["models"][name]
    return {"shared_fixtures":len(shared),"models":models,
            "duplicate_fixtures_excluded":sum(v>1 for v in counts.values())}
