"""Independent, read-only evaluation of isolated recovery archives.

No new forecasts or provider calls. Reproduce the saved statistical control
from its exact receipt, authenticate its original pre-match identity, and
grade only source-matched verified regulation-time results. Earlier versus
later controls are a timing comparison, not evidence of model superiority.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json,math,re

import numpy as np

from forecast_ledger import _forecast,LedgerValidationError
from five_double_accuracy import _final,_result_index,wilson
from history_pipeline import build_evidence
from v10_prospective import utc,research_predict
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics
from v44_early_history_audit import qualifies
from v44_refresh_priority import model_key
import v44_early_history as early
import v44_recovery_capture as recovery
import v42_research_trial as sources

HEX=re.compile(r"^[a-f0-9]{64}$")

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _original_pairs(pair_root,source_root,at):
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            _,models=sources._pair(path,source_root=source_root,decision=at)
            first=min(utc(m["kickoff_utc"]) for m in models)
            if timedelta(minutes=5)<first-at<=timedelta(hours=24):
                yield models
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            continue

def authenticate(path,*,root,source_root,pair_root,now):
    m=_forecast(_read(path))
    if (m["model_key"]!=recovery.MODEL_KEY or m["model_version"]!=recovery.VERSION
            or m.get("score_basis")!="regulation" or m.get("horizon")!="isolated_recovery"
            or m["forecast_id"]!=path.stem or not HEX.fullmatch(path.stem)):
        raise ValueError("invalid_recovery_policy_or_identity")
    captured=utc(m["captured_at_utc"])
    cutoff=utc(m["evidence_as_of_utc"])
    key=model_key(m)
    if not cutoff<=captured<=now or not timedelta(minutes=5)<key[2]-captured<=timedelta(hours=48):
        raise ValueError("invalid_recovery_timestamp")
    if path.parent.name!=key[1] or path.parent.parent.name!=key[0]:
        raise ValueError("recovery_path_mismatch")
    prior_id=m["previous_near_forecast_id"]
    if not isinstance(prior_id,str) or not HEX.fullmatch(prior_id):
        raise ValueError("invalid_prior_id")
    original_path=Path(source_root)/"forecasts"/key[0]/key[1]/(prior_id+".json")
    original=_forecast(_read(original_path))
    if (original["forecast_id"]!=prior_id or original["model_key"]!="v8_near"
            or original.get("score_basis")!="regulation" or model_key(original)!=key
            or not utc(original["captured_at_utc"])<cutoff):
        raise ValueError("prior_forecast_mismatch")
    if not any(key in [model_key(x) for x in models]
               for models in _original_pairs(pair_root,source_root,captured)):
        raise ValueError("no_original_pair_at_capture")
    e=[key[0],key[1],key[2].isoformat(),key[3],key[4]]
    rid=early.receipt_key(e,m["input_digest"])
    history_root=Path(root)/"history"
    receipt=early.load(history_root/"receipts"/(rid+".json"),root=history_root,now=captured)
    if (not qualifies(receipt,m,captured)
            or m["recovery_receipt_sha256"]!=receipt["receipt_sha256"]):
        raise ValueError("recovery_receipt_mismatch")
    evidence=build_evidence(receipt["fixture"],receipt["history"],cutoff)
    forecast=research_predict(receipt["fixture"],evidence["history"],
                              league_history=evidence["league_history"],as_of=cutoff)
    if (forecast is None or len(forecast.get("component_weights") or {})<2
            or forecast["model_version"]!=m["numerical_model_version"]):
        raise ValueError("numerical_model_not_reproducible")
    grid=_grid(m["score_matrix"])
    reproduced=_grid(forecast["matrix"])
    if grid.shape!=reproduced.shape or not np.allclose(grid,reproduced,rtol=0,atol=1e-12):
        raise ValueError("recovery_grid_not_reproduced")
    return m,original,grid

def evaluate(*,root=recovery.ROOT,source_root=sources.SOURCE_ROOT,
             pair_root=sources.PAIR_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc));root=Path(root)
    paths=sorted((root/"archive"/"forecasts").glob("*/*/*.json"))
    counts=Counter();rejects=Counter();rows=[];seen=set();physical=set()
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    candidates=[]
    for path in paths:
        try:
            m,original,grid=authenticate(path,root=root,source_root=source_root,pair_root=pair_root,now=clock)
            candidates.append((m,original,grid))
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError,LedgerValidationError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    for m,original,grid in sorted(candidates,key=lambda r:(utc(r[0]["captured_at_utc"]),r[0]["forecast_id"])):
        key=model_key(m)[:2]
        event=(utc(m["kickoff_utc"])," ".join(m["home_team"].casefold().split()),
               " ".join(m["away_team"].casefold().split()))
        if key in seen or event in physical:
            rejects["duplicate_recovery_event"]+=1;continue
        seen.add(key);physical.add(event)
        counts["authenticated"]+=1
        result=finals.get(key)
        actual=_final(m,result,clock)
        if actual is None or _final(original,result,clock)!=actual:
            counts["pending_or_unverified"]+=1;continue
        counts["verified"]+=1
        rows.append({"fixture_id":m["fixture_id"],"original_forecast_id":original["forecast_id"],
                     "recovery_forecast_id":m["forecast_id"],
                     "metrics":{"earlier_near":_metrics(_grid(original["score_matrix"]),actual),
                                "later_recovery":_metrics(grid,actual)}})
    summary={}
    for name in ("earlier_near","later_recovery"):
        metrics=[r["metrics"][name] for r in rows]
        n=len(metrics);hits=sum(r["top_score_hit"] for r in metrics)
        summary[name]={"fixtures":n,"hits":hits,"hit_rate":hits/n if n else None,
                       "wilson_95":wilson(hits,n),
                       **{k:math.fsum(r[k] for r in metrics)/n if n else None
                          for k in ("exact_log_loss","score_brier","outcome_brier")}}
    return {"study_type":"recovery_source_reproduction_audit","archive_files":len(paths),
            "counts":dict(counts),"rejections":dict(rejects),"comparisons":summary,
            "result_conflicts":len(conflicts),"result_rejections":dict(result_rejects),
            "external_requests":0,"automatic_promotion":False,
            "limitations":["Different source times confound earlier-versus-later comparison.",
                           "Receipt hashes do not independently establish provider fetch freshness.",
                           "Single-fixture metrics do not demonstrate five-double accuracy.",
                           "No new forecasts are issued."],"rows":rows}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
