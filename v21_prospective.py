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


def freeze_pairs(*,root=ROOT,now=None,min_coverage=MIN_FIVE_COVERAGE):
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
    proposals=rank_pairs(eligible,min_coverage=min_coverage)
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


def _same_frozen_calculation(saved,expected):
    """Exact keys/choices, with only machine-scale numeric tolerance."""
    if type(saved) in (int,float) and type(expected) in (int,float):
        return (math.isfinite(saved) and math.isfinite(expected) and
                math.isclose(saved,expected,rel_tol=0,abs_tol=1e-12))
    if type(saved) is not type(expected):
        return False
    if isinstance(saved,dict):
        return saved.keys()==expected.keys() and all(
            _same_frozen_calculation(saved[key],expected[key]) for key in saved)
    if isinstance(saved,list):
        return len(saved)==len(expected) and all(
            _same_frozen_calculation(a,b) for a,b in zip(saved,expected))
    return saved==expected


def _evaluation_sources(pair,file,*,root,source_root,clock):
    """Reconstruct every fixed variant from its real pre-selection source.

    A label, filename or saved winning choice cannot authenticate a trial.
    Earlier archived variants and original forecasts are read, never repaired.
    """
    if (not isinstance(pair,dict) or pair.get("schema")!=1 or
            pair.get("status")!="frozen_pre_kickoff"):
        raise ValueError("not_a_frozen_v21_pair")
    captured=utc(pair["captured_at_utc"])
    if captured>clock:
        raise ValueError("pair_not_available_at_evaluation")
    day=pair["sydney_date"]
    if file.stem!=day:
        raise ValueError("filename_sydney_date_mismatch")
    models=[]
    reconstructed=[]
    for leg in (pair["first"],pair["second"]):
        source=original_model(source_root,leg)
        if source is None or source.get("input_digest")!=leg.get("input_digest"):
            raise ValueError("original_source_identity_or_digest_mismatch")
        kickoff=utc(source["kickoff_utc"])
        if not captured+timedelta(minutes=MIN_LEAD_MINUTES)<kickoff<=captured+timedelta(hours=MAX_LEAD_HOURS):
            raise ValueError("pair_outside_predeclared_pre_match_window")
        if kickoff.astimezone(SYDNEY).date().isoformat()!=day:
            raise ValueError("fixture_outside_declared_sydney_date")
        original_capture=utc(source["captured_at_utc"])
        if not utc(source["evidence_as_of_utc"])<=original_capture<=captured:
            raise ValueError("original_evidence_observed_after_selection")
        frozen=load_json(Path(root)/"forecasts"/(_row_key(leg)+".json"),{})
        keys=("provider","fixture_id","forecast_id","input_digest","home_team_id","away_team_id")
        if (not isinstance(frozen,dict) or frozen.get("status")!="frozen_pre_kickoff" or
                any(str(frozen.get(key))!=str(source[key]) for key in keys) or
                utc(frozen["kickoff_utc"])!=kickoff or
                utc(frozen["baseline_captured_at_utc"])!=original_capture or
                not original_capture<=utc(frozen["captured_at_utc"])<=captured or
                frozen.get("league_history_matches")!=source.get("league_history_matches") or
                int(source.get("league_history_matches") or 0)<MIN_LEAGUE_HISTORY):
            raise ValueError("frozen_variant_source_or_time_mismatch")
        for key in ("home_team","away_team","evidence_as_of_utc","competition"):
            if frozen.get(key)!=source.get(key):
                raise ValueError("frozen_variant_context_source_mismatch")
        expected=_models(source)
        if not _same_frozen_calculation(frozen.get("model_results"),expected):
            raise ValueError("frozen_variants_do_not_reproduce_from_original")
        models.append(source)
        reconstructed.append({"model_results":expected})
    identities={(str(m["provider"]),str(m["fixture_id"])) for m in models}
    teams={(str(m["provider"]),str(m[key])) for m in models
           for key in ("home_team_id","away_team_id")}
    names=[" ".join(str(m.get(key) or "").casefold().split()) for m in models
           for key in ("home_team","away_team")]
    if len(identities)!=2 or len(teams)!=4 or not all(names) or len(set(names))!=4:
        raise ValueError("two_events_and_four_distinct_known_teams_required")
    expected_methods={}
    for name in MODEL_NAMES:
        choices,coverage=_join(*reconstructed,name)
        expected_methods[name]={"doubles":choices,"coverage":coverage}
    robust=min(expected_methods[name]["coverage"] for name in
               ("baseline_v8","negative_binomial_r3","bivariate_poisson_c012"))
    if (not _same_frozen_calculation(pair.get("methods"),expected_methods) or
            not _same_frozen_calculation(pair.get("robust_model_coverage"),robust)):
        raise ValueError("five_doubles_or_coverage_do_not_reproduce_from_original")
    events={(utc(m["kickoff_utc"]).isoformat(),
             " ".join(str(m["home_team"]).casefold().split()),
             " ".join(str(m["away_team"]).casefold().split())) for m in models}
    return models,identities,events


def evaluate(*,root=ROOT,source_root=ORIGINAL_ROOT,verified=None,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    results=(verified if verified is not None else
             load_json(Path(source_root)/"verified_results.json",{}))
    rows=[]
    counts={m:{"wins":0,"verified":0,"pending":0} for m in MODEL_NAMES}
    calibration={m:[] for m in MODEL_NAMES}
    rejected=Counter()
    result_rejections=Counter()
    seen_ids=set()
    seen_events=set()
    for file in sorted((Path(root)/"pairs").glob("*.json")):
        try:
            pair=load_json(file,{})
            sources,identities,events=_evaluation_sources(pair,file,root=root,
                source_root=source_root,clock=clock)
            if seen_ids.intersection(identities) or seen_events.intersection(events):
                raise ValueError("repeated_fixture_or_cross_provider_event")
            seen_ids.update(identities)
            seen_events.update(events)
        except (ValueError,TypeError,KeyError,OSError,AttributeError,OverflowError) as exc:
            rejected[str(exc) if isinstance(exc,ValueError) else "malformed_frozen_evidence"]+=1
            continue
        first,second=pair["first"],pair["second"]
        a=results.get(str(first["fixture_id"]))
        b=results.get(str(second["fixture_id"]))
        available=[]
        for leg,result in ((first,a),(second,b)):
            valid=_verified(leg,result)
            if valid and utc(result["verified_at_utc"])>clock:
                valid=False
                result_rejections["result_observed_after_evaluation"]+=1
            available.append(valid)
        finished=all(available)
        score1=f"{a['home_goals']}-{a['away_goals']}" if finished else None
        score2=f"{b['home_goals']}-{b['away_goals']}" if finished else None
        row={"sydney_date":pair["sydney_date"],"verified":bool(finished),
             "actual_scores":[score1,score2] if finished else None,"model_hits":{},
             "source_model_versions":[m.get("model_version") or "unknown" for m in sources]}
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
            # Coverage is the forecast probability of at least one of five
            # mutually exclusive outcomes, not an observed success rate.
            probability=float(pair["methods"][name]["coverage"])
            calibration[name].append((probability,hit))
        rows.append(row)
    calibration_report={}
    for name,stats in counts.items():
        n=stats["verified"]
        stats["observed_hit_rate"]=stats["wins"]/n if n else None
        samples=calibration[name]
        mean=sum(p for p,y in samples)/n if n else None
        calibration_report[name]={
            "calibration_sample_size":len(samples),
            "mean_predicted_five_coverage":mean,
            "five_event_brier_score":sum((p-y)**2 for p,y in samples)/n if n else None,
            "observed_minus_expected_rate":stats["observed_hit_rate"]-mean if n else None,
        }
    return {"schema":1,"kind":"strictly_frozen_before_both_kickoffs",
            "evaluated_at_utc":clock.isoformat(),"valid_unique_pairs":len(rows),
            "evidence_rejections":dict(rejected),"result_rejections":dict(result_rejections),
            "full_variant_and_choice_reconstruction":True,"fixture_reuse_in_denominator_allowed":False,
            "methods":counts,"calibration":calibration_report,"pairs":rows,"models_promoted":False,
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
