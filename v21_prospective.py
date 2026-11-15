"""V21 genuinely frozen three-model five-double tournament.

Separate write-once shadow; no original score is overwritten. Each future
candidate is derived solely from a validated V8 full matrix captured BEFORE
kickoff. Rank ALL eligible same-Sydney-day pairs and abstain when original
league evidence or estimated five-pick coverage is insufficient.
"""
import json
import math
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

from persist_prematch import load_json, write_json, verified_result
from v10_prospective import ROOT as ORIGINAL_ROOT, read_latest, utc
from v16_pair_research import original_model
from v21_goal_model_comparison import MODEL_NAMES,variants,top_scores

ROOT=Path("data/v21_prospective")
SYDNEY=ZoneInfo("Australia/Sydney")
MIN_LEAD_MINUTES=45
MAX_LEAD_HOURS=36
MIN_LEAGUE_HISTORY=50
MIN_FIVE_COVERAGE=.055  # model-estimated uncalibrated guardrail, NOT win accuracy


def _valid_source(row,root,now):
    try:
        if row.get("model_key") not in ("v8","v8_near"):
            return False
        true=original_model(root,row)
        if true is None or true.get("input_digest") != row.get("input_digest"):
            return False
        if utc(row["captured_at_utc"]) > now or utc(row["kickoff_utc"]) <= now+timedelta(minutes=MIN_LEAD_MINUTES):
            return False
        if utc(row["kickoff_utc"]) > now+timedelta(hours=MAX_LEAD_HOURS):
            return False
        if int(row.get("league_history_matches") or 0)<MIN_LEAGUE_HISTORY:
            return False
        return True
    except (TypeError,ValueError,KeyError):
        return False


def _first_source_for_fixture(originals,root,now):
    """Never choose a model based on the future final score."""
    best={}
    for row in originals.values() if isinstance(originals,dict) else originals:
        if not _valid_source(row,root,now):
            continue
        key=(str(row["provider"]),str(row["fixture_id"]))
        previous=best.get(key)
        if previous is None or (row["model_key"]=="v8_near" and previous["model_key"]!="v8_near") or (
           row["model_key"]==previous["model_key"] and
           utc(row["captured_at_utc"])>utc(previous["captured_at_utc"])):
            best[key]=row
    return list(best.values())


def _row_key(row):
    return str(row["provider"])+"_"+str(row["fixture_id"])+"_"+str(row["forecast_id"])


def _models(row):
    candidate=variants(row["score_matrix"])
    return {name:{"top10_scores":top_scores(grid,10),
                  "score_matrix":grid.tolist()}
            for name,grid in candidate.items()}


def _write_once(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open("x",encoding="utf-8") as fh:
            fh.write(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n")
        return True
    except FileExistsError:
        return False


def freeze_models(*,root=ROOT,source_root=ORIGINAL_ROOT,now=None,originals=None):
    clock=utc(now or datetime.now(timezone.utc))
    candidates=_first_source_for_fixture(
        originals if originals is not None else read_latest(source_root),source_root,clock)
    saved=[]
    for source in candidates:
        try:
            models=_models(source)
            row={key:source[key] for key in (
                "provider","fixture_id","forecast_id","input_digest","home_team_id",
                "away_team_id","home_team","away_team","kickoff_utc",
                "evidence_as_of_utc","competition")}
            row.update({"schema":1,"status":"frozen_pre_kickoff",
                        "captured_at_utc":clock.isoformat(),
                        "baseline_captured_at_utc":source["captured_at_utc"],
                        "model_results":models,
                        "league_history_matches":source["league_history_matches"],
                        "not_promoted":True,"source":"existing_v8_frozen_matrix_no_new_api_calls"})
            filename=Path(root)/"forecasts"/(_row_key(row)+".json")
            if _write_once(filename,row):
                saved.append(str(row["fixture_id"]))
        except (ValueError,KeyError,TypeError):
            continue
    return {"candidates":len(candidates),"new_frozen_fixtures":saved}


def _join(first,second,model):
    left=first["model_results"][model]["top10_scores"][:5]
    right=second["model_results"][model]["top10_scores"][:5]
    options=sorted(({"score1":f"{a}-{b}","score2":f"{c}-{d}",
                     "joint_probability":float(p*q)}
                    for a,b,p in left for c,d,q in right),
                   key=lambda r:(-r["joint_probability"],r["score1"],r["score2"]))[:5]
    return options,sum(r["joint_probability"] for r in options)


def rank_pairs(rows,min_coverage=MIN_FIVE_COVERAGE):
    """All pairs, four different teams, same Sydney date, no forced picks."""
    eligible=defaultdict(list)
    for r in rows:
        try:
            if r["status"]!="frozen_pre_kickoff":continue
            day=utc(r["kickoff_utc"]).astimezone(SYDNEY).date().isoformat()
            eligible[day].append(r)
        except (KeyError,TypeError,ValueError):
            continue
    results=[]
    for day,matches in eligible.items():
        for left,right in combinations(matches,2):
            teams=[(str(r["provider"]),str(r[key])) for r in (left,right)
                   for key in ("home_team_id","away_team_id")]
            if len(set(teams))!=4 or left["forecast_id"]==right["forecast_id"]:
                continue
            grids={}
            try:
                for method in MODEL_NAMES:
                    picks,coverage=_join(left,right,method)
                    grids[method]={"doubles":picks,"coverage":coverage}
            except (TypeError,ValueError,KeyError):
                continue
            # rank only by pre-declared statistical estimates, never by
            # actual results, scoreline preference, or unfrozen AI votes.
            robust=min(grids["baseline_v8"]["coverage"],
                       grids["negative_binomial_r3"]["coverage"],
                       grids["bivariate_poisson_c012"]["coverage"])
            if robust<min_coverage:
                continue
            results.append({"day":day,"first":left,"second":right,
                            "methods":grids,"robust_coverage":robust})
    return sorted(results,key=lambda r:(-r["robust_coverage"],r["day"],
                   str(r["first"]["fixture_id"]),str(r["second"]["fixture_id"])))


def freeze_pairs(*,root=ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    rows=[load_json(x,{}) for x in sorted((Path(root)/"forecasts").glob("*.json"))]
    eligible=[]
    for row in rows:
        try:
            if row.get("status")=="frozen_pre_kickoff" and (
                utc(row["captured_at_utc"])<=clock and
                clock+timedelta(minutes=MIN_LEAD_MINUTES)<utc(row["kickoff_utc"])<=clock+timedelta(hours=MAX_LEAD_HOURS)):
                eligible.append(row)
        except (KeyError,ValueError,TypeError):
            continue
    proposals=rank_pairs(eligible)
    chosen={}
    for result in proposals:
        if result["day"] in chosen:
            continue
        chosen[result["day"]]=result
    saved=[]
    for day,pick in chosen.items():
        record={"schema":1,"status":"frozen_pre_kickoff",
                "sydney_date":day,"captured_at_utc":clock.isoformat(),
                "first":{k:pick["first"][k] for k in (
                    "provider","fixture_id","forecast_id","input_digest",
                    "home_team_id","away_team_id","kickoff_utc")},
                "second":{k:pick["second"][k] for k in (
                    "provider","fixture_id","forecast_id","input_digest",
                    "home_team_id","away_team_id","kickoff_utc")},
                "methods":pick["methods"],
                "robust_model_coverage":pick["robust_coverage"],
                "policy":"rank_all_eligible_pairs_or_abstain; no_future_results",
                "not_promoted":True}
        if _write_once(Path(root)/"pairs"/(day+".json"),record):
            saved.append(day)
    return {"eligible_forecasts":len(eligible),
            "qualifying_pairs":len(proposals),"newly_frozen_days":saved,
            "abstained":not bool(proposals)}


def _verified(leg,actual):
    if not actual:
        return False
    proxy={"provider":leg["provider"],"fixture_id":str(leg["fixture_id"]),
           "home_team_id":str(leg["home_team_id"]),
           "away_team_id":str(leg["away_team_id"]),
           "kickoff_utc":leg["kickoff_utc"]}
    try:
        return (verified_result(proxy,actual,str(leg["fixture_id"])) and
                actual.get("status") in ("FT","FINISHED","AET","PEN") and
                utc(actual["verified_at_utc"])>utc(leg["kickoff_utc"]))
    except (ValueError,TypeError,KeyError):
        return False


def evaluate(*,root=ROOT,source_root=ORIGINAL_ROOT,verified=None):
    results=(verified if verified is not None else
             load_json(Path(source_root)/"verified_results.json",{}))
    rows=[]
    counts={m:{"wins":0,"verified":0,"pending":0} for m in MODEL_NAMES}
    for file in sorted((Path(root)/"pairs").glob("*.json")):
        pair=load_json(file,{})
        if pair.get("status")!="frozen_pre_kickoff":continue
        first,second=pair.get("first") or {},pair.get("second") or {}
        source1=original_model(source_root,first)
        source2=original_model(source_root,second)
        if not source1 or not source2 or any(
           actual.get("input_digest") != leg.get("input_digest")
           for actual,leg in ((source1,first),(source2,second))):
            continue
        a=results.get(str(first["fixture_id"]))
        b=results.get(str(second["fixture_id"]))
        finished=_verified(first,a) and _verified(second,b)
        score1=f"{a['home_goals']}-{a['away_goals']}" if finished else None
        score2=f"{b['home_goals']}-{b['away_goals']}" if finished else None
        row={"sydney_date":pair["sydney_date"],"verified":bool(finished),
             "actual_scores":[score1,score2] if finished else None,"model_hits":{}}
        for name in MODEL_NAMES:
            choices=(pair.get("methods") or {}).get(name,{}).get("doubles") or []
            if len(choices)!=5:
                continue
            if not finished:
                counts[name]["pending"]+=1
                row["model_hits"][name]=None
                continue
            hit=int(any(choice["score1"]==score1 and choice["score2"]==score2
                        for choice in choices))
            counts[name]["verified"]+=1
            counts[name]["wins"]+=hit
            row["model_hits"][name]=bool(hit)
        rows.append(row)
    for name,stats in counts.items():
        n=stats["verified"]
        stats["observed_hit_rate"]=stats["wins"]/n if n else None
    return {"schema":1,"kind":"strictly_frozen_before_both_kickoffs",
            "methods":counts,"pairs":rows,"models_promoted":False,
            "measured_improvement_proven":False,
            "note":"Pending results not losses; future model promotion requires independent sample, calibration and uncertainty."}


def main(root=ROOT,source_root=ORIGINAL_ROOT):
    current=datetime.now(timezone.utc)
    root=Path(root)
    snapshots=freeze_models(root=root,source_root=source_root,now=current)
    ranked=freeze_pairs(root=root,now=current)
    performance=evaluate(root=root,source_root=source_root)
    write_json(root/"capture_status.json",{"at_utc":current.isoformat(),**snapshots,**ranked})
    write_json(root/"performance.json",performance)
    return {"capture":snapshots,"selection":ranked,
            "verified_pairs":performance["methods"]["baseline_v8"]["verified"]}


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
