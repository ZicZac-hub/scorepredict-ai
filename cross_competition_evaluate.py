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


def score_metrics(grid, actual):
    """Same observed event, proper probability scores, never a betting return."""
    p = np.asarray(grid, dtype=float)
    if (p.ndim != 2 or not p.size or not np.isfinite(p).all() or
            (p < 0).any() or not math.isclose(float(p.sum()), 1., abs_tol=1e-8)):
        raise ValueError("invalid_normalized_grid")
    if len(actual) != 2 or any(type(n) is not int or n < 0 for n in actual):
        raise ValueError("invalid_regulation_score")
    h, a = actual
    x, y = np.indices(p.shape)
    cell = h * p.shape[1] + a if h < p.shape[0] and a < p.shape[1] else -1
    prob = float(p.ravel()[cell]) if cell >= 0 else 0.
    rank = np.argsort(-p.ravel(), kind="stable")
    masks = (x > y, x == y, x < y)
    actual_outcome = (h > a, h == a, h < a)
    brier = math.fsum((float(p[mask].sum())-observed)**2
                     for mask, observed in zip(masks, actual_outcome))
    high = float(p[x+y>=4].sum())
    low = float(p[x+y<=1].sum())
    return {"exact_log_loss":-math.log(max(1e-15,prob)),
            "exact_multiclass_brier":1+float(np.square(p).sum())-2*prob,
            "outcome_brier":brier,
            "top_one_hit":int(cell==int(rank[0])),
            "top_five_hit":int(cell in rank[:5]),
            "high_goal_brier":(high-int(h+a>=4))**2,
            "low_goal_brier":(low-int(h+a<=1))**2}


def paired_metrics(rows):
    n = len(rows)
    keys = ("exact_log_loss","exact_multiclass_brier","outcome_brier",
            "high_goal_brier","low_goal_brier")
    methods = {}
    for method in ("control","challenger"):
        methods[method] = {"verified_fixtures":n,
            "top_one_hits":sum(r[method]["top_one_hit"] for r in rows),
            "top_five_hits":sum(r[method]["top_five_hit"] for r in rows),
            **{"mean_"+key:math.fsum(r[method][key] for r in rows)/n if n else None
               for key in keys}}
    differences = {}
    for key in keys:
        vals = [r["control"][key]-r["challenger"][key] for r in rows]
        mean = math.fsum(vals)/n if n else None
        # Descriptive interval only; serial clubs can violate independence.
        interval = None
        if n >= 30:
            se = math.sqrt(math.fsum((v-mean)**2 for v in vals)/(n-1)/n)
            interval = [mean-1.96*se,mean+1.96*se]
        differences[key] = {"mean_control_minus_challenger":mean,
                            "descriptive_normal_95_interval":interval}
    return {"same_verified_fixtures":n,"methods":methods,"paired_differences":differences,
            "positive_difference_means_lower_challenger_loss":True,
            "minimum_review_fixtures":100,"minimum_review_count_reached":n>=100,
            "independence_assumption_not_established":True,"automatic_promotion":False}


def evaluate(*, root=ROOT, source_root=Path("data/v10_prospective"),
             prematch_root=Path("data/prematch"), now=None):
    clock=_date(now or datetime.now(timezone.utc))
    finals, conflicts, rejects=index_verified_results(source_root,prematch_root,clock)
    summary={"frozen":0,"verified":0,"pending":0,"invalid":0,"control_exact_hits":0,
             "challenger_exact_hits":0,"control_mean_log_loss":None,"challenger_mean_log_loss":None,
             "research_only":True,"not_a_five_double_trial":True,"production_promoted":False}
    matched=[]
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
            # Validate both before incrementing a shared denominator.
            scored={name:score_metrics(candidate[name],actual)
                    for name in ("control","challenger")}
            matched.append(scored)
            summary["verified"]+=1
        except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError):
            summary["invalid"]+=1
    if summary["frozen"]!=summary["verified"]+summary["pending"]+summary["invalid"]:
        raise ValueError("denominator_mismatch")
    summary["calibration"]=paired_metrics(matched)
    for name, metrics in summary["calibration"]["methods"].items():
        summary[name+"_exact_hits"]=metrics["top_one_hits"]
        summary[name+"_mean_log_loss"]=metrics["mean_exact_log_loss"]
    return summary


if __name__=="__main__":
    report=evaluate()
    write_json(ROOT/"status.json",report)
    print(json.dumps(report,indent=2))
