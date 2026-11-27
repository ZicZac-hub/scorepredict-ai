"""Separate fresh-history prospective study. Never repair original receipts.

Same original fixed fixtures, independently registered protocol, newer genuine
V8-family capture allowed only before the common decision. Both arms use the
same fresh history. No new pair selection, API requests or production changes.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
import argparse,hashlib,json,math
from forecast_ledger import _forecast,LedgerValidationError
from five_double_accuracy import _final,_result_index,wilson
from source_capture_clock import capture_clock
from v10_prospective import utc
import v42_research_trial as sources
import v44_early_history as early
from v44_early_history_audit import qualifies
import v44_score_driven_research as ratings
from v44_family_pool_research import _grid
from v44_pair_transfer_research import _arm

ROOT=Path("data/v44_fresh_history_study")
VERSION="v44-fresh-history-v1"
METHODS=("fresh_v8","fresh_ratings","fresh_blend")
PROTOCOL={"version":VERSION,"methods":list(METHODS),"blend_weight":ratings.BLEND,
          "selection":"unchanged_original_pair","source":"latest_qualified_v8_or_v8_near_with_exact_history",
          "source_max_age_hours":24,"window":"5_to_180_minutes_before_first_kickoff",
          "no_retroactive_issuance":True,"automatic_promotion":False}

def protocol(root,now,register=False):
    names=("v44_fresh_history_study.py","v44_early_history.py","v44_early_history_audit.py",
           "v44_score_driven_research.py","history_pipeline.py","v44_family_pool_research.py",
           "v44_pair_transfer_research.py","professional_double_optimizer.py","professional_double_shadow.py",
           "v42_research_trial.py","forecast_ledger.py","five_double_accuracy.py","source_capture_clock.py")
    here=Path(__file__).resolve().parent
    body={"protocol":PROTOCOL,"implementation_sha256":{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names}}
    path=Path(root)/"protocol.json"
    if register and not path.exists():
        record={**body,"registered_at_utc":now.isoformat()}
        record["sha256"]=early.digest(record)
        early._once(path,record)
    record=early._read(path)
    if (any(record.get(k)!=v for k,v in body.items())
            or record["sha256"]!=early.digest({k:v for k,v in record.items() if k!="sha256"})
            or utc(record["registered_at_utc"])>now):
        raise ValueError("protocol_integrity_failure")
    return record

def event(model):
    return (str(model["provider"]),str(model["fixture_id"]),utc(model["kickoff_utc"]).isoformat(),
            str(model["home_team_id"]),str(model["away_team_id"]))

def source_for(original,cutoff,source_root,history_root):
    cutoff=utc(cutoff)
    e=event(original)
    candidates=[]
    folder=Path(source_root)/"forecasts"/e[0]/e[1]
    for path in sorted(folder.glob("*.json")):
        try:
            raw=early._read(path)
            if raw.get("model_key") not in ("v8","v8_near") or not raw.get("forecast_id"):
                continue
            m=_forecast(raw)
            captured=utc(m["captured_at_utc"])
            if (m["forecast_id"]!=path.stem or event(m)!=e
                    or not timedelta(0)<=cutoff-captured<=timedelta(hours=24)):
                continue
            if m.get("score_basis")!="regulation":
                continue
            keys=(early.receipt_key(list(e),m["input_digest"]),early.digest(list(e)))
            for key in keys:
                receipt_path=Path(history_root)/"receipts"/(key+".json")
                if not receipt_path.is_file():
                    continue
                try:
                    receipt=early.load(receipt_path,root=history_root,now=cutoff)
                    if qualifies(receipt,m,cutoff):
                        candidates.append((captured,m["forecast_id"],key,m,receipt))
                except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
                    continue
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError,LedgerValidationError):
            continue
    if not candidates:
        raise ValueError("missing_fresh_exact_history")
    # Select before inspecting model readiness or finals; no best-result selection.
    _,_,key,m,receipt=max(candidates,key=lambda r:r[:3])
    return m,receipt,key

def calculate(models,cutoff,source_root,history_root):
    refs=[];base=[];other=[]
    for original in models:
        m,receipt,key=source_for(original,cutoff,source_root,history_root)
        r=ratings.predict(receipt["fixture"],receipt["history"],receipt["as_of_utc"])
        if r["status"]!="ready":
            raise ValueError("insufficient_fresh_team_history")
        base.append(_grid(m["score_matrix"]));other.append(_grid(r["grid"]))
        refs.append({"forecast_id":m["forecast_id"],"forecast_sha256":early.digest(m),
                     "receipt_key":key,"receipt_sha256":receipt["receipt_sha256"]})
    arms={name:_arm(grids) for name,grids in zip(METHODS,(base,other,[ratings._blend(a,b) for a,b in zip(base,other)]))}
    for arm in arms.values():
        arm["full_score_grids"]=[g.tolist() for g in arm["full_score_grids"]]
    return refs,arms

def freeze(*,root=ROOT,source_root=sources.SOURCE_ROOT,history_root=early.ROOT,pair_root=sources.PAIR_ROOT,now=None,clock=None):
    started,observed=capture_clock(now,clock)
    root=Path(root); p=protocol(root,started,register=True); counts=Counter()
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        dst=root/"trials"/path.name
        if dst.exists():
            counts["preserved"]+=1;continue
        try:
            pair,models=sources._pair(path,source_root=source_root,decision=started)
            if not sources._window(models,started):
                counts["outside_window"]+=1;continue
            refs,arms=calculate(models,started,source_root,history_root)
            ended=observed()
            if not sources._window(models,ended):
                raise ValueError("calculation_finished_outside_window")
            row={"version":VERSION,"day":path.stem,"protocol_sha256":p["sha256"],
                 "pair_sha256":pair["record_sha256"],"decision_utc":started.isoformat(),
                 "predicted_at_utc":ended.isoformat(),"references":refs,"arms":arms}
            row["sha256"]=early.digest(row)
            counts["issued"]+=int(early._once(dst,row))
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            counts[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    return dict(counts)

def evaluate(*,root=ROOT,source_root=sources.SOURCE_ROOT,history_root=early.ROOT,pair_root=sources.PAIR_ROOT,
             prematch_root=sources.PREMATCH_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc));root=Path(root)
    files=sorted((root/"trials").glob("*.json"))
    p=protocol(root,clock) if files or (root/"protocol.json").exists() else None
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    counts=Counter(); rows=[];seen=set();physical=set()
    for path in files:
        try:
            rec=early._read(path)
            start,end=map(utc,(rec["decision_utc"],rec["predicted_at_utc"]))
            if (rec["version"]!=VERSION or rec["day"]!=path.stem or rec["protocol_sha256"]!=p["sha256"]
                    or rec["sha256"]!=early.digest({k:v for k,v in rec.items() if k!="sha256"})
                    or not utc(p["registered_at_utc"])<=start<=end<=clock):
                raise ValueError("invalid_trial")
            pair,models=sources._pair(Path(pair_root)/"sets"/path.name,source_root=source_root,decision=start)
            if (rec["pair_sha256"]!=pair["record_sha256"] or not sources._window(models,start)
                    or not sources._window(models,end)):
                raise ValueError("invalid_original_pair")
            refs,arms=calculate(models,start,source_root,history_root)
            if refs!=rec["references"] or not sources._same_calculation(arms,rec["arms"]):
                raise ValueError("calculation_mismatch")
            ids={event(m)[:2] for m in models}
            events={(utc(m["kickoff_utc"])," ".join(m["home_team"].casefold().split())," ".join(m["away_team"].casefold().split())) for m in models}
            if len(ids)!=2 or len(events)!=2 or seen&ids or physical&events:
                raise ValueError("repeat_fixture")
            seen.update(ids);physical.update(events);counts["authenticated"]+=1
            actual=[_final(m,finals.get(event(m)[:2]),clock) for m in models]
            if any(s is None for s in actual):
                counts["pending"]+=1;continue
            rows.append({"day":path.stem,"metrics":{n:sources._metric(a,actual) for n,a in arms.items()}})
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            counts["rejected"]+=1
    out={}
    for n in METHODS:
        metrics=[r["metrics"][n] for r in rows]; count=len(metrics);hits=sum(m["hit"] for m in metrics)
        out[n]={"pairs":count,"hits":hits,"hit_rate":hits/count if count else None,"wilson_95":wilson(hits,count),
                "mean_log_loss":math.fsum(m["score_log_loss"] for m in metrics)/count if count else None}
    return {"study_type":VERSION,"registered":p is not None,"counts":dict(counts),"comparisons":out,
            "result_conflicts":len(conflicts),"result_rejections":dict(result_rejects),"external_requests":0,
            "automatic_promotion":False,"rows":rows}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--freeze",action="store_true");args=parser.parse_args()
    print(json.dumps({"capture":freeze(),"evaluation":evaluate()} if args.freeze else evaluate(),indent=2,allow_nan=False))
