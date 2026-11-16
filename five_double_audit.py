"""Leakage-safe audit metrics for frozen five-double predictions.

This module NEVER edits a saved forecast or promotes a challenger model.
It evaluates only settled, uniquely identified fixture pairs with predictions
created before BOTH kickoffs. A reported hit rate is observed, not modelled.
"""
import math
from datetime import datetime, timezone

def _utc(value):
    try:
        result=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if result.tzinfo is None:
            return None
        return result.astimezone(timezone.utc)
    except (ValueError,TypeError):
        return None

def _score(value):
    if isinstance(value,str):
        parts=value.split("-")
    elif isinstance(value,(tuple,list)):
        parts=value
    else:
        return None
    if len(parts)!=2:
        return None
    try:
        result=tuple(int(x) for x in parts)
    except (TypeError,ValueError):
        return None
    return result if min(result)>=0 else None

def evaluate(records):
    """records: id, predicted_at, kickoff_1/2, actual_1/2, doubles.

    Each double has Score 1 and Score 2, optionally Joint. Missing or
    post-kickoff predictions are excluded, never counted as losses.
    """
    seen=set()
    settled=[]
    exclusions={}
    for record in records:
        identity=str(record.get("id") or "")
        if not identity or identity in seen:
            exclusions["missing_or_duplicate_id"]=exclusions.get("missing_or_duplicate_id",0)+1
            continue
        seen.add(identity)
        created=_utc(record.get("predicted_at"))
        kick1=_utc(record.get("kickoff_1"))
        kick2=_utc(record.get("kickoff_2"))
        if not created or not kick1 or not kick2 or created>=min(kick1,kick2):
            exclusions["not_provably_pre_match"]=exclusions.get("not_provably_pre_match",0)+1
            continue
        actual1=_score(record.get("actual_1"))
        actual2=_score(record.get("actual_2"))
        if actual1 is None or actual2 is None:
            exclusions["unsettled"]=exclusions.get("unsettled",0)+1
            continue
        options=record.get("doubles") or []
        if len(options)!=5:
            exclusions["not_five_doubles"]=exclusions.get("not_five_doubles",0)+1
            continue
        unique=set()
        mass=0.0
        valid=True
        for option in options:
            pair=(_score(option.get("Score 1")),_score(option.get("Score 2")))
            if None in pair or pair in unique:
                valid=False
                break
            unique.add(pair)
            if "Joint" in option:
                try:
                    probability=float(option["Joint"])
                    if not math.isfinite(probability) or probability<0 or probability>1:
                        valid=False
                        break
                    mass+=probability
                except (TypeError,ValueError):
                    valid=False
                    break
        if not valid or mass>1+1e-8:
            exclusions["invalid_options"]=exclusions.get("invalid_options",0)+1
            continue
        settled.append({"id":identity,"hit":(actual1,actual2) in unique,
                        "predicted_coverage":mass if all("Joint" in o for o in options) else None})
    n=len(settled)
    hits=sum(item["hit"] for item in settled)
    coverages=[x["predicted_coverage"] for x in settled if x["predicted_coverage"] is not None]
    return {"settled":n,"hits":hits,"observed_hit_rate":hits/n if n else None,
            "mean_model_coverage":sum(coverages)/len(coverages) if coverages else None,
            "coverage_sample":len(coverages),"excluded":exclusions,"records":settled}
