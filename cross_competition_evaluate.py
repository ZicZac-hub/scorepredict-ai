"""Read-only matched single-fixture evaluation of original frozen research receipts."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from cross_competition_capture import ROOT, digest
from cross_competition_strength import research, VERSION
from history_pipeline import build_evidence, _date, season_year
from v37_sparse_history_research import predict
from verified_result_index import index_verified_results
from five_double_accuracy import _final
from persist_prematch import write_json

IMPLEMENTATION = ("cross_competition_capture.py","cross_competition_strength.py","history_pipeline.py",
                  "v37_sparse_history_research.py","research_ensemble_v8.py","engine.py")


def validated(path, root, clock):
    row = json.loads(path.read_text())
    if row["record_sha256"] != digest({k:v for k,v in row.items() if k!="record_sha256"}):
        raise ValueError("receipt_hash")
    if row["version"] != VERSION or row["research_only"] is not True or row["external_requests"] != 0:
        raise ValueError("wrong_protocol")
    here = Path(__file__).resolve().parent
    if row["implementation_sha256"] != {n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in IMPLEMENTATION}:
        raise ValueError("implementation_changed")
    fixture = row["fixture"]
    kickoff, decision, predicted, early = map(_date, (fixture["utcDate"],row["decision_as_of_utc"],
                    row["predicted_at_utc"],row["european_as_of_utc"]))
    if not early <= decision <= predicted < kickoff or predicted > clock:
        raise ValueError("invalid_timing")
    if path.stem != str(fixture["id"]) or not path.stem.isdecimal():
        raise ValueError("fixture_identity")
    cache = {}
    for sha in row["source_sha256"]:
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid_source_ref")
        packet = json.loads((Path(root)/"sources"/(sha+".json")).read_text())
        if digest(packet)!=sha:
            raise ValueError("source_hash")
        key = tuple(packet["key"])
        meta = packet["metadata"]
        if key in cache or not meta["freshness_known"] or meta["stale"] is not False or _date(meta["fetched_at_utc"]) > decision:
            raise ValueError("source_metadata")
        cache[key] = (packet["rows"],meta,None)
    league = str(fixture["competition"]["id"])
    season = season_year(fixture)
    european = [r for year in (season,season-1) for r in cache[("API-Football",league,year)][0]]
    evidence = build_evidence(fixture,european,early)
    if evidence["input_digest"] != row["european_input_digest"]:
        raise ValueError("european_digest")
    control = predict(fixture,evidence["history"],as_of=early)
    if control is None:
        raise ValueError("control_missing")
    candidate = research(fixture,control["matrix"],cache,as_of=decision)
    if candidate["status"]!="research_candidate":
        raise ValueError("challenger_missing")
    for name in ("control","challenger"):
        a,b=np.asarray(candidate[name]),np.asarray(row["model"][name])
        if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=1e-12):
            raise ValueError("grid_reproduction")
    return fixture,candidate


def evaluate(*, root=ROOT, source_root=Path("data/v10_prospective"),
             prematch_root=Path("data/prematch"), now=None):
    clock=_date(now or datetime.now(timezone.utc))
    finals, conflicts, rejects=index_verified_results(source_root,prematch_root,clock)
    summary={"frozen":0,"verified":0,"pending":0,"invalid":0,"control_exact_hits":0,
             "challenger_exact_hits":0,"control_mean_log_loss":None,"challenger_mean_log_loss":None,
             "research_only":True,"not_a_five_double_trial":True,"production_promoted":False}
    losses={"control":[],"challenger":[]}
    for path in sorted((Path(root)/"trials").glob("*.json")):
        summary["frozen"]+=1
        try:
            fixture, candidate=validated(path,root,clock)
            model={"provider":fixture["provider"],"fixture_id":fixture["id"],
                   "home_team_id":fixture["homeTeam"]["id"],"away_team_id":fixture["awayTeam"]["id"],
                   "kickoff_utc":fixture["utcDate"]}
            actual=_final(model,finals.get((model["provider"],str(model["fixture_id"]))),clock)
            if actual is None:
                summary["pending"]+=1
                continue
            h,a=actual
            summary["verified"]+=1
            for name in losses:
                p=np.asarray(candidate[name])
                summary[name+"_exact_hits"]+=int(np.unravel_index(np.argmax(p),p.shape)==actual)
                probability=p[h,a] if h<p.shape[0] and a<p.shape[1] else 0.
                losses[name].append(-math.log(max(1e-15,float(probability))))
        except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError):
            summary["invalid"]+=1
    if summary["frozen"]!=summary["verified"]+summary["pending"]+summary["invalid"]:
        raise ValueError("denominator_mismatch")
    for name in losses:
        if losses[name]:
            summary[name+"_mean_log_loss"]=math.fsum(losses[name])/len(losses[name])
    return summary


if __name__=="__main__":
    report=evaluate()
    write_json(ROOT/"status.json",report)
    print(json.dumps(report,indent=2))
