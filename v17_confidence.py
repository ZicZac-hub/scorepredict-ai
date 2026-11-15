"""V17 evidence-gated outcome research; separate from five exact-score doubles.

No network requests, paid AI calls, invented xG, or retroactive score changes.
70% is a threshold on *uncalibrated model estimates*, never measured accuracy.
"""
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from persist_prematch import load_json, verified_result, write_json
from v10_prospective import ROOT as SOURCE_ROOT, read_latest, utc
from v16_pair_research import original_model

ROOT = Path("data/v17_confidence")
SYDNEY = ZoneInfo("Australia/Sydney")
VERSION = "v17-unvalidated-outcome-confidence-v1"
ESTIMATED_THRESHOLD = 0.70
MIN_HISTORY = 30
MIN_LEAD_MINUTES = 45
MAX_LEAD_HOURS = 24


def wilson(hits, total):
    if not total:
        return None
    z = 1.959963984540054
    phat = hits / total
    denom = 1 + z*z/total
    half = z * math.sqrt((phat*(1-phat) + z*z/(4*total))/total)
    middle = phat + z*z/(2*total)
    return [max(0., (middle-half)/denom),
            min(1., (middle+half)/denom)]


def probabilities(row):
    """Validate original full score matrix, metadata and chronology."""
    if not isinstance(row, dict) or row.get("model_key") != "v8":
        raise ValueError("original_v8_required")
    if not row.get("forecast_id") or not row.get("input_digest"):
        raise ValueError("original_source_identity_missing")
    if not utc(row["evidence_as_of_utc"]) <= utc(row["captured_at_utc"]) < utc(row["kickoff_utc"]):
        raise ValueError("not_pre_match")
    if str(row["home_team_id"]) == str(row["away_team_id"]):
        raise ValueError("same_team")
    matrix = row["score_matrix"]
    if (not isinstance(matrix, list) or not matrix or
            not isinstance(matrix[0], list) or not matrix[0]):
        raise ValueError("missing_full_score_grid")
    width = len(matrix[0])
    if max(len(matrix), width) > 256:
        raise ValueError("oversized_grid")
    total = 0.
    home = draw = away = 0.
    for i, line in enumerate(matrix):
        if not isinstance(line, list) or len(line) != width:
            raise ValueError("invalid_grid_shape")
        for j, value in enumerate(line):
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid_grid_probability")
            total += value
            if i > j:
                home += value
            elif i == j:
                draw += value
            else:
                away += value
    if abs(total-1.0) > 1e-6:
        raise ValueError("grid_not_normalised")
    output = {"1":home,"X":draw,"2":away,
              "1X":home+draw,"X2":draw+away,"12":home+away}
    original = row.get("outcome_probabilities") or {}
    if any(abs(output[k]-float(original[k]))>1e-6
           for k,key in (("1","home"),("X","draw"),("2","away"))
           for _ in [None] if k in output and key in original):
        # Historical source matrices must agree with recorded outcome summary.
        raise ValueError("model_outcome_mismatch")
    if original and not all(k in original for k in ("home","draw","away")):
        raise ValueError("incomplete_outcome_summary")
    return output


def candidates(forecasts, now):
    """Read genuine V8 full grids only; no live results are consulted."""
    clock = utc(now)
    eligible = defaultdict(list)
    rejected = defaultdict(int)
    for row in forecasts:
        try:
            probabilities(row)
            if (int(row.get("history_matches") or 0)<MIN_HISTORY or
                int(row.get("league_history_matches") or 0)<MIN_HISTORY or
                len(row.get("component_weights") or {})<2):
                raise ValueError("not_enough_verified_history")
            kick = utc(row["kickoff_utc"])
            if not timedelta(minutes=MIN_LEAD_MINUTES)<=kick-clock<=timedelta(hours=MAX_LEAD_HOURS):
                raise ValueError("outside_45m_to_24h")
            if utc(row["captured_at_utc"])>clock:
                raise ValueError("record_not_yet_captured")
            outcomes = probabilities(row)
            kind = max(("1X","X2","12"),key=lambda key:(outcomes[key],key))
            day = kick.astimezone(SYDNEY).date().isoformat()
            eligible[day].append((row,kind,outcomes[kind]))
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            rejected[str(exc)[:100]]+=1
    return eligible,dict(rejected)


def _pick(row,kind,estimated):
    return {
        "provider":row["provider"],"fixture_id":str(row["fixture_id"]),
        "forecast_id":row["forecast_id"],"input_digest":row["input_digest"],
        "home_team_id":str(row["home_team_id"]),
        "away_team_id":str(row["away_team_id"]),
        "home_team":row["home_team"],"away_team":row["away_team"],
        "kickoff_utc":row["kickoff_utc"],"selection":kind,
        "model_estimated_probability":round(float(estimated),10)
    }


def select(forecasts, now, threshold=ESTIMATED_THRESHOLD):
    """Select the strongest predeclared 70%-estimate singles and two-leg pair.

    Two legs are kept on the same Sydney date, use four distinct teams and
    must jointly reach the threshold. Unqualified days abstain.
    """
    if not 0.5<=threshold<1:
        raise ValueError("threshold_invalid")
    days,rejected=candidates(forecasts,now)
    output={}
    for day,rows in sorted(days.items()):
        ranked=sorted(rows,key=lambda e:(-e[2],str(e[0]["provider"]),
                                         str(e[0]["fixture_id"])))
        result={"version":VERSION,"mode":"shadow_pre_match",
                "sydney_date":day,"captured_at_utc":utc(now).isoformat(),
                "estimated_probability_not_calibrated":True,
                "target_measured_accuracy":0.70,
                "eligible_fixtures":len(rows),"single":None,"pair":None,
                "single_abstention":"estimated_below_70_percent",
                "pair_abstention":"no_two_distinct_matches_reach_70_percent_model_estimate"}
        if ranked[0][2]>=threshold:
            row,choice,prob=ranked[0]
            result["single"]=_pick(row,choice,prob)
            result["single_abstention"]=None
        best=None
        for i,(first,choice1,p1) in enumerate(ranked):
            for second,choice2,p2 in ranked[i+1:]:
                if ((first["provider"],str(first["fixture_id"]))==
                        (second["provider"],str(second["fixture_id"]))):
                    continue
                if len({str(first["home_team_id"]),str(first["away_team_id"]),
                        str(second["home_team_id"]),str(second["away_team_id"])})!=4:
                    continue
                mass=p1*p2
                if best is None or mass>best[0]:
                    best=(mass,first,choice1,p1,second,choice2,p2)
        if best and best[0]>=threshold:
            mass,a,choice1,p1,b,choice2,p2=best
            result["pair"]={"first":_pick(a,choice1,p1),
                            "second":_pick(b,choice2,p2),
                            "joint_model_estimate":round(mass,10),
                            "independence_assumption":True}
            result["pair_abstention"]=None
        output[day]=result
    return output,rejected


def freeze(root=ROOT,source=SOURCE_ROOT,now=None,forecasts=None):
    """Write ONCE, only before kickoff, from already frozen source forecasts."""
    clock=utc(now or datetime.now(timezone.utc))
    if forecasts is None:
        forecasts=[r for r in read_latest(source).values() if r.get("model_key")=="v8"]
    selected,rejected=select(forecasts,clock)
    saved=[]
    for day,record in selected.items():
        if record["single"] is None and record["pair"] is None:
            continue
        target=Path(root)/"forecasts"/(day+".json")
        if target.exists():
            continue
        # Original pre-match forecast must exist and match exact frozen
        # identity+digest, rather than a refreshed model or arbitrary payload.
        source_ok=True
        for leg in ([record["single"]] if record["single"] else [])+(
                [record["pair"]["first"],record["pair"]["second"]]
                if record["pair"] else []):
            orig=original_model(source,leg)
            if (orig is None or orig.get("input_digest")!=leg["input_digest"]
                    or utc(leg["kickoff_utc"])<=clock):
                source_ok=False
                break
        if not source_ok:
            rejected["original_source_not_verified"]=rejected.get("original_source_not_verified",0)+1
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            with target.open("x",encoding="utf-8") as f:
                f.write(json.dumps(record,sort_keys=True,indent=2)+"\n")
            saved.append(day)
        except FileExistsError:
            pass
    return {"frozen_dates":saved,"eligible_dates":len(selected),
            "abstention_reasons":rejected,
            "model_threshold_is_not_calibrated_accuracy":True}


def _result(leg,source,verified):
    if not leg:
        return None
    archived=original_model(source,leg)
    if archived is None or archived.get("input_digest")!=leg.get("input_digest"):
        return None
    result=verified.get(str(leg["fixture_id"]))
    if not verified_result(archived,result,str(leg["fixture_id"])):
        return None
    if (utc(result["verified_at_utc"])<=utc(leg["kickoff_utc"]) or
        result.get("status") not in ("FT","FINISHED","AET","PEN")):
        return None
    return result


def _wins(choice,result):
    h,a=result["home_goals"],result["away_goals"]
    outcome="1" if h>a else "2" if h<a else "X"
    return outcome in choice["selection"]


def evaluate(root=ROOT,source=SOURCE_ROOT,verified=None):
    """Use ONLY saved before-kickoff selections and verified regulation scores."""
    if verified is None:
        verified=load_json(Path(source)/"verified_results.json",{})
    items=[];single_n=single_wins=pair_n=pair_wins=0
    pending_single=pending_pair=0
    for path in sorted((Path(root)/"forecasts").glob("*.json")):
        row=load_json(path,{})
        if (row.get("version")!=VERSION or row.get("mode")!="shadow_pre_match"
                or row.get("sydney_date")!=path.stem):
            continue
        single=row.get("single")
        pair=row.get("pair")
        info={"sydney_date":row["sydney_date"],"single_status":"abstained",
              "pair_status":"abstained"}
        if single:
            if utc(row["captured_at_utc"])>=utc(single["kickoff_utc"]):
                continue
            actual=_result(single,source,verified)
            if actual is not None:
                hit=_wins(single,actual);single_n+=1;single_wins+=int(hit)
                info["single_status"]="win" if hit else "loss"
            else:
                pending_single+=1;info["single_status"]="pending"
        if pair:
            legs=(pair["first"],pair["second"])
            if utc(row["captured_at_utc"])>=min(utc(x["kickoff_utc"]) for x in legs):
                continue
            actuals=[_result(x,source,verified) for x in legs]
            if all(x is not None for x in actuals):
                hit=all(_wins(x,y) for x,y in zip(legs,actuals))
                pair_n+=1;pair_wins+=int(hit)
                info["pair_status"]="win" if hit else "loss"
            else:
                pending_pair+=1;info["pair_status"]="pending"
        items.append(info)
    rate=pair_wins/pair_n if pair_n else None
    bound=wilson(pair_wins,pair_n)
    return {"version":VERSION,"research_only":True,
            "threshold_basis":"uncalibrated_pre_match_model_estimate_not_measured_accuracy",
            "single":{"verified":single_n,"wins":single_wins,
                      "rate":single_wins/single_n if single_n else None,
                      "wilson_95":wilson(single_wins,single_n),"pending":pending_single},
            "two_match_double_chance":{"verified":pair_n,"wins":pair_wins,
                      "rate":rate,"wilson_95":bound,"pending":pending_pair},
            "meets_70_percent_claim_gate":bool(pair_n>=100 and bound and bound[0]>=0.70),
            "claim_gate":"At least 100 independent future pairs AND Wilson 95% lower bound at least 70%; no selection means no measured accuracy",
            "rows":items}


def evidence_dashboard(source=SOURCE_ROOT):
    """Report verified data availability; never convert missing inputs to votes."""
    report=load_json(Path(source)/"performance.json",{})
    ai=load_json(Path("data/v15_ai_double/source_audit.json"),{})
    models=report.get("models") or {}
    return {"frozen_v8":(models.get("v8") or {}).get("frozen_fixtures",0),
            "verified_v8":(models.get("v8") or {}).get("verified_fixtures",0),
            "verified_five_double_sets":(report.get("five_doubles") or {}).get("verified_pairs",0),
            "real_xg_feed":False,"verified_lineup_numeric_adjustment":False,
            "timestamped_market_benchmark_validated":False,
            "configured_ai_providers":ai.get("configured_provider_count",0),
            "source_matched_two_match_ai_sets":ai.get("original_sets_with_at_least_two_ai_votes_on_both_legs",0),
            "notes":["xG, injury-adjustment and multi-book odds remain missing or unvalidated",
                     "A model estimate above 70% is NOT 70% observed accuracy",
                     "V17 makes zero additional network or paid provider calls"]}


def main():
    capture=freeze()
    performance=evaluate()
    write_json(ROOT/"performance.json",performance)
    write_json(ROOT/"evidence_status.json",evidence_dashboard())
    print(json.dumps({"frozen":capture,"single":performance["single"],
                      "paired":performance["two_match_double_chance"],
                      "claim_gate":performance["meets_70_percent_claim_gate"]},
                     sort_keys=True))


if __name__=="__main__":
    main()
