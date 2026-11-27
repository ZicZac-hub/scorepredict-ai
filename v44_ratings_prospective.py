"""Fixed-policy prospective V44 opponent-rating experiment.
No new fixture selection, paid calls, retro-issued predictions or production
promotion. Both exact original history receipts must qualify before kickoff.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import numpy as np

from v10_prospective import utc
from source_capture_clock import capture_clock
from five_double_accuracy import _final,_result_index,wilson
from v44_early_history_audit import qualifies
import v44_early_history as early
import v44_score_driven_research as ratings
import v42_research_trial as sources
from v44_family_pool_research import _grid
from v44_pair_transfer_research import _arm

ROOT=Path("data/v44_ratings_prospective")
VERSION="v44-exact-history-ratings-v1"
METHODS=("original_v8","ratings_only","ratings_blend")
PROTOCOL={"version":VERSION,"methods":list(METHODS),
          "blend_weight":ratings.BLEND,"decision_window":sources.PROTOCOL["decision_rule"],
          "missing_history":"abstain_both_legs_required","automatic_promotion":False}


def _protocol(root,now,register=False):
    path=root/"protocol.json"
    names=("v44_ratings_prospective.py","v44_early_history.py","v44_early_history_audit.py",
           "v44_score_driven_research.py","history_pipeline.py","v44_pair_transfer_research.py",
           "v44_family_pool_research.py","professional_double_shadow.py",
           "professional_double_optimizer.py","source_capture_clock.py")
    here=Path(__file__).resolve().parent
    body={"protocol":PROTOCOL,"implementation_sha256":{
          n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names}}
    if register and not path.exists():
        record={**body,"registered_at_utc":now.isoformat()}
        record["sha256"]=early.digest(record)
        early._once(path,record)
    record=early._read(path)
    if (any(record.get(k)!=v for k,v in body.items())
            or record["sha256"]!=early.digest({k:v for k,v in record.items() if k!="sha256"})
            or utc(record["registered_at_utc"])>now):
        raise ValueError("protocol_or_implementation_mismatch")
    return record


def receipt_for(model,cutoff,history_root):
    event=[str(model["provider"]),str(model["fixture_id"]),utc(model["kickoff_utc"]).isoformat(),
           str(model["home_team_id"]),str(model["away_team_id"])]
    keys=(early.receipt_key(event,model["input_digest"]),early.digest(event))
    options=[]
    for key in keys:
        path=Path(history_root)/"receipts"/(key+".json")
        if path.is_file():
            record=early.load(path,root=history_root,now=cutoff)
            if qualifies(record,model,cutoff):
                options.append((utc(record["captured_at_utc"]),key,record))
    return min(options,key=lambda r:r[:2]) if options else None


def _arms(pair,models,cutoff,history_root):
    references=[]
    challengers=[]
    for model in models:
        match=receipt_for(model,cutoff,history_root)
        if match is None:
            raise ValueError("missing_exact_early_history")
        _,key,receipt=match
        pred=ratings.predict(receipt["fixture"],receipt["history"],receipt["as_of_utc"])
        if pred["status"]!="ready":
            raise ValueError("insufficient_qualified_team_history")
        challengers.append(pred["grid"])
        references.append({"receipt_key":key,"receipt_sha256":receipt["receipt_sha256"]})
    base=[_grid(m["score_matrix"]) for m in models]
    arms={"original_v8":{"full_score_grids":base,
                         "five_doubles":pair["selection"]["choices"],
                         "estimated_coverage":pair["selection"]["estimated_coverage"]},
          "ratings_only":_arm(challengers),
          "ratings_blend":_arm([ratings._blend(b,c) for b,c in zip(base,challengers)])}
    for arm in arms.values():
        arm["full_score_grids"]=[np.asarray(g).tolist() for g in arm["full_score_grids"]]
    return arms,references


def freeze(*,root=ROOT,history_root=early.ROOT,pair_root=sources.PAIR_ROOT,
           source_root=sources.SOURCE_ROOT,now=None,clock=None):
    root=Path(root)
    started,observed=capture_clock(now,clock)
    protocol=_protocol(root,started,register=True)
    counts=Counter()
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        counts["original_pair_files"]+=1
        destination=root/"trials"/path.name
        if destination.exists():
            counts["preserved"]+=1
            continue
        try:
            pair,models=sources._pair(path,source_root=source_root,decision=started)
            if not sources._window(models,started):
                counts["outside_decision_window"]+=1
                continue
            counts["eligible_window"]+=1
            arms,references=_arms(pair,models,started,history_root)
            predicted=observed()
            if not sources._window(models,predicted):
                raise ValueError("calculation_finished_outside_window")
            record={"version":VERSION,"sydney_date":path.stem,
                    "decision_utc":started.isoformat(),"predicted_at_utc":predicted.isoformat(),
                    "protocol_sha256":protocol["sha256"],"pair_sha256":pair["record_sha256"],
                    "history_refs":references,"arms":arms}
            record["sha256"]=early.digest(record)
            if early._once(destination,record):
                counts["issued"]+=1
            else:
                counts["preserved"]+=1
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            counts[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    for key in ("original_pair_files","preserved","outside_decision_window","eligible_window","issued"):
        counts.setdefault(key,0)
    return dict(counts)


def evaluate(*,root=ROOT,history_root=early.ROOT,pair_root=sources.PAIR_ROOT,
             source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,now=None):
    root=Path(root)
    now=utc(now or datetime.now(timezone.utc))
    paths=sorted((root/"trials").glob("*.json"))
    protocol=_protocol(root,now) if (root/"protocol.json").exists() or paths else None
    finals,conflicts,result_rejections=_result_index(source_root,prematch_root)
    rows=[]
    counts=Counter()
    seen=set()
    events=set()
    for path in paths:
        try:
            rec=early._read(path)
            cutoff,predicted=map(utc,(rec["decision_utc"],rec["predicted_at_utc"]))
            if (rec["version"]!=VERSION or rec["sydney_date"]!=path.stem
                    or rec["protocol_sha256"]!=protocol["sha256"]
                    or rec["sha256"]!=early.digest({k:v for k,v in rec.items() if k!="sha256"})
                    or not utc(protocol["registered_at_utc"])<=cutoff<=predicted<=now):
                raise ValueError("invalid_trial")
            pair,models=sources._pair(Path(pair_root)/"sets"/path.name,source_root=source_root,decision=cutoff)
            if (pair["record_sha256"]!=rec["pair_sha256"]
                    or not sources._window(models,cutoff) or not sources._window(models,predicted)):
                raise ValueError("invalid_original_pair_or_window")
            arms,refs=_arms(pair,models,cutoff,history_root)
            if refs!=rec["history_refs"] or not sources._same_calculation(arms,rec["arms"]):
                raise ValueError("trial_does_not_reproduce")
            ids={(str(m["provider"]),str(m["fixture_id"])) for m in models}
            physical={(utc(m["kickoff_utc"]),m["home_team"].casefold().strip(),m["away_team"].casefold().strip()) for m in models}
            if len(ids)!=2 or len(physical)!=2 or seen&ids or events&physical:
                raise ValueError("repeat_fixture")
            seen.update(ids);events.update(physical)
            counts["authenticated_issued"]+=1
            actual=[_final(m,finals.get((str(m["provider"]),str(m["fixture_id"]))),now) for m in models]
            if any(s is None for s in actual):
                counts["pending"]+=1
                continue
            rows.append({"day":path.stem,"metrics":{n:sources._metric(a,actual) for n,a in arms.items()}})
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError) as exc:
            counts["rejected"]+=1
    comparisons={}
    for name in METHODS:
        m=[r["metrics"][name] for r in rows]
        n=len(m);hits=sum(v["hit"] for v in m)
        comparisons[name]={"pairs":n,"hits":hits,"hit_rate":hits/n if n else None,
            "wilson_95":wilson(hits,n),"mean_log_loss":math.fsum(v["score_log_loss"] for v in m)/n if n else None}
    return {"study_type":VERSION,"counts":dict(counts),"comparisons":comparisons,
            "result_conflicts":len(conflicts),"result_rejections":dict(result_rejections),
            "external_requests":0,"automatic_promotion":False,"rows":rows}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    args=parser.parse_args()
    if args.freeze:
        print(json.dumps({"capture":freeze(),"evaluation":evaluate()},indent=2,allow_nan=False))
    else:
        print(json.dumps(evaluate(),indent=2,allow_nan=False))
