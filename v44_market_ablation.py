"""Audit archived V9 market forecasts against their same-time V8 controls.

An exact fixture, evidence digest, evidence cutoff and capture timestamp match
is required. This is a research comparison of already archived forecasts,
not a new forecast generator. No external calls or model promotion.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from forecast_ledger import _forecast, LedgerValidationError
from five_double_accuracy import _final, _result_index, wilson
from v10_prospective import utc
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics
import v44_fixture_evidence as inventory
import v42_research_trial as sources

MODELS=("v8_control","v9")
MAX_LEAD=timedelta(hours=48)
MIN_LEAD=timedelta(minutes=5)
MARKET_MAX_AGE=timedelta(hours=12)


def match_key(model):
    return (*inventory._identity(model),str(model["home_team_id"]),
            str(model["away_team_id"]),utc(model["kickoff_utc"]),
            utc(model["evidence_as_of_utc"]),utc(model["captured_at_utc"]),
            model["input_digest"],inventory._event(model))


def _distribution(value,size):
    return (isinstance(value,(list,tuple)) and len(value)==size
            and all(type(p) in (int,float) and math.isfinite(p) and 0<p<1 for p in value)
            and abs(math.fsum(value)-1)<=1e-6)


def qualify_market(model):
    if model.get("model_key")!="v9" or model.get("market_used") is not True:
        raise ValueError("not_market_forecast")
    market=model["market_snapshot"]
    stamp=utc(market["snapshot_utc"])
    captured=utc(model["captured_at_utc"])
    if not timedelta(0)<=captured-stamp<=MARKET_MAX_AGE:
        raise ValueError("market_outside_capture_window")
    if market.get("settlement")!="regulation_90_minutes":
        raise ValueError("non_regulation_market")
    if not _distribution(market.get("winner"),3):
        raise ValueError("invalid_winner_distribution")
    if market.get("totals") is not None and not _distribution(market["totals"],2):
        raise ValueError("invalid_total_distribution")
    return market


def match_archive(source_root,clock):
    groups=defaultdict(lambda:defaultdict(list))
    counts,rejects=Counter(),Counter()
    for path in sorted((Path(source_root)/"forecasts").glob("*/*/*.json")):
        try:
            raw=json.loads(path.read_text(encoding="utf-8"))
            if raw.get("model_key") not in MODELS:
                continue
            counts["candidate_records"]+=1
            if not raw.get("forecast_id") or not raw.get("input_digest"):
                raise ValueError("missing_original_identity")
            model=_forecast(raw)
            if (path.parent.parent.name,path.parent.name,path.stem)!=(*inventory._identity(model),model["forecast_id"]):
                raise ValueError("archive_path_mismatch")
            captured,kickoff=utc(model["captured_at_utc"]),utc(model["kickoff_utc"])
            if captured>clock or not MIN_LEAD<kickoff-captured<=MAX_LEAD:
                raise ValueError("outside_fixed_capture_window")
            if model.get("score_basis")!="regulation":
                raise ValueError("non_regulation_forecast")
            if model["model_key"]=="v9":
                qualify_market(model)
            elif model.get("market_used") is not False:
                raise ValueError("market_in_control")
            groups[match_key(model)][model["model_key"]].append(model)
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError,LedgerValidationError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    candidates=[]
    for key,group in groups.items():
        if any(len(group.get(name,[]))!=1 for name in MODELS):
            counts["unmatched_or_ambiguous_groups"]+=1
            continue
        control,market=(group[name][0] for name in MODELS)
        candidates.append((utc(control["captured_at_utc"]),control["forecast_id"],control,market))
    # Select earliest matched capture, without consulting finals.
    rows=[];seen_ids=set();seen_events=set()
    for _,_,control,market in sorted(candidates,key=lambda r:r[:2]):
        identity=inventory._identity(control)
        event=inventory._event(control)
        if identity in seen_ids or event in seen_events:
            counts["duplicate_matched_capture"]+=1
            continue
        seen_ids.add(identity);seen_events.add(event)
        rows.append((control,market))
    counts["matched_fixtures"]=len(rows)
    return rows,counts,rejects


def _summary(rows):
    output={}
    for name in MODELS:
        metrics=[r["metrics"][name] for r in rows]
        n=len(metrics);hits=sum(m["top_score_hit"] for m in metrics)
        output[name]={"fixtures":n,"hits":hits,"hit_rate":hits/n if n else None,
                      "wilson_95":wilson(hits,n),
                      **{k:math.fsum(m[k] for m in metrics)/n if n else None
                         for k in ("exact_log_loss","score_brier","outcome_brier")}}
    differences=[r["metrics"]["v9"]["exact_log_loss"]-r["metrics"]["v8_control"]["exact_log_loss"] for r in rows]
    return output,{"mean_market_minus_control_log_loss":math.fsum(differences)/len(differences) if differences else None,
                   "market_lower_loss":sum(d<0 for d in differences),"control_lower_loss":sum(d>0 for d in differences),
                   "ties":sum(d==0 for d in differences)}


def evaluate(*,source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    matches,counts,rejects=match_archive(source_root,clock)
    finals,conflicts,result_rejections=_result_index(source_root,prematch_root)
    rows=[]
    for control,market in matches:
        actual=_final(control,finals.get(inventory._identity(control)),clock)
        if actual is None or _final(market,finals.get(inventory._identity(market)),clock)!=actual:
            counts["without_verified_final"]+=1
            continue
        rows.append({"fixture_id":control["fixture_id"],"capture_utc":control["captured_at_utc"],
                     "metrics":{m["model_key"]:_metrics(_grid(m["score_matrix"]),actual)
                                for m in (control,market)}})
    comparisons,paired=_summary(rows)
    return {"study_type":"archived_same_time_market_ablation",
            "counts":dict(counts),"verified_matched_fixtures":len(rows),
            "comparisons":comparisons,"paired_log_loss":paired,"source_rejections":dict(rejects),
            "conflicting_results":len(conflicts),"result_rejections":dict(result_rejections),
            "external_requests":0,"automatic_promotion":False,"rows":rows,
            "limitations":["Uses stored original market forecasts and their matched controls, not an independent re-fetch of market data.",
                           "Market-informed forecasts can differ for reasons other than the market if the original computation used a different internal as-of time.",
                           "Individual exact-score results are not five-double hit rates."]}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
