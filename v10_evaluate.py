"""Auditable v10 v8/v9 prospective evaluation and five-double selector.

Only original write-once pre-kickoff forecasts and independently verified
regulation-time results enter observed accuracy. Historical replays and
retrospective revisions NEVER count as live wins.
"""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import math
import numpy as np

from daily_pick_ui import SYDNEY
from persist_prematch import RESULTS, load_json, verified_result, write_json
from smart_double_v8 import rank_research_pairs
from smart_double_ai import five_doubles
from v10_prospective import ROOT, read_latest, utc

REPORT = ROOT / "performance.json"
PAIRS = ROOT / "doubles"
MIN_COVERAGE = 0.04  # conservative untuned guardrail, NOT a calibrated hit rate
FREEZE_MAX_LEAD_HOURS = 24  # do not freeze a week early before full fixture discovery
FREEZE_MIN_LEAD_MINUTES = 45  # avoid selecting a pair as kickoff starts


def _qualified(row, result):
    """Validate actual provider, identity, date and regulation score."""
    if not result or not verified_result(row,result,str(row["fixture_id"])):
        return False
    if result.get("status") not in ("FT","AET","PEN","FINISHED"):
        return False
    try:
        return utc(result["verified_at_utc"])>utc(result["kickoff_utc"])
    except (ValueError,TypeError,KeyError):
        return False


def _metric(row, result):
    matrix=np.asarray(row["score_matrix"], dtype=float)
    home,away=result["home_goals"],result["away_goals"]
    predicted=[int(x) for x in row["predicted_score"]]
    prob=float(matrix[home,away]) if home<matrix.shape[0] and away<matrix.shape[1] else 0.0
    p=row["outcome_probabilities"]
    observed={"home":int(home>away),"draw":int(home==away),"away":int(home<away)}
    return {
        "exact":int(predicted==[home,away]),
        "top3":int(any(list(s[:2])==[home,away] for s in row["top10_scores"][:3])),
        "log_loss":-math.log(max(prob,1e-12)),
        "outcome_brier":sum((p[key]-observed[key])**2 for key in observed),
        "predicted_exact_probability":float(row["predicted_score_probability"]),
        "fixture_id":str(row["fixture_id"]),
        "league":row.get("competition"),
        "kickoff_utc":row["kickoff_utc"],
    }


def _summarize(items, *, frozen):
    n=len(items)
    return {
        "frozen_fixtures":frozen,"verified_fixtures":n,
        "pending_fixtures":max(0,frozen-n),
        "exact_score_hits":sum(r["exact"] for r in items),
        "exact_score_rate":sum(r["exact"] for r in items)/n if n else None,
        "top3_hit_rate":sum(r["top3"] for r in items)/n if n else None,
        "mean_score_log_loss":sum(r["log_loss"] for r in items)/n if n else None,
        "mean_outcome_brier":sum(r["outcome_brier"] for r in items)/n if n else None,
        "mean_expected_exact_hits":sum(r["predicted_exact_probability"] for r in items) if n else None,
    }


def _match(row):
    return {"fixture":{"id":row["fixture_id"],"provider":row["provider"],
                       "homeTeam":{"name":row.get("home_team"),"id":row.get("home_team_id")},
                       "awayTeam":{"name":row.get("away_team"),"id":row.get("away_team_id")}},
            "kickoff":row["kickoff_utc"],
            "result":{"scores":[(int(h),int(a),float(p)) for h,a,p in row["top10_scores"]]}}


def _write_once(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open("x",encoding="utf-8") as output:
            output.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
        return True
    except FileExistsError:
        return False


def freeze_doubles(forecasts=None, root=ROOT, now=None):
    """Consider ALL researched future fixtures, not just first 18.

    Freezes maximum five-joint-probability coverage among pairs meeting the
    same-Sydney-day, four-team, independently researched data policy.
    Abstains if coverage is below the explicitly uncalibrated guardrail.
    Never rewrites an original selection.
    """
    instant=utc(now or datetime.now(timezone.utc))
    options=forecasts if forecasts is not None else read_latest(root)
    eligible=defaultdict(list)
    for key,row in options.items():
        if row.get("model_key")!="v8":
            continue
        if (len(row.get("component_weights") or {})<2
                or int(row.get("history_matches") or 0)<10
                or len(row.get("top10_scores") or [])<5):
            continue
        kickoff=utc(row["kickoff_utc"])
        if kickoff<=instant:
            continue
        day=kickoff.astimezone(SYDNEY).date().isoformat()
        eligible[day].append(row)
    stats={"days_evaluated":0,"days_saved":[],"abstained":{}}
    for day,rows in sorted(eligible.items()):
        target=Path(root)/"doubles"/(day+".json")
        if target.exists():
            continue
        stats["days_evaluated"]+=1
        # A discovered pair several days early is NOT necessarily the best
        # pair: the complete day's slate may not be available or researched.
        # Delay first frozen publication until the first eligible kickoff is
        # at most 24 hours away, while preserving a 45-minute safety lead.
        first_kickoff=min(utc(row["kickoff_utc"]) for row in rows)
        if first_kickoff-instant>timedelta(hours=FREEZE_MAX_LEAD_HOURS):
            stats["abstained"][day]="waiting_for_near_matchday_fixture_research"
            continue
        recent=[row for row in rows if
                utc(row["kickoff_utc"])-instant>=timedelta(minutes=FREEZE_MIN_LEAD_MINUTES)]
        if len(recent)<2:
            stats["abstained"][day]="insufficient_two_pre_kickoff_legs"
            continue
        rows=recent
        if len(rows)<2:
            stats["abstained"][day]="fewer_than_two_qualified_matches"
            continue
        ranked=rank_research_pairs([_match(r) for r in rows],limit=1)
        if not ranked:
            stats["abstained"][day]="no_same_day_four_team_pair"
            continue
        best=ranked[0]
        if best["coverage"]<MIN_COVERAGE:
            stats["abstained"][day]="estimated_coverage_below_operating_guardrail"
            continue
        ids=[(m["fixture"]["provider"],str(m["fixture"]["id"]))
             for m in (best["first"],best["second"])]
        by_id={(r["provider"],str(r["fixture_id"])):r for r in rows}
        first,second=(by_id[identifier] for identifier in ids)
        first_kick,second_kick=utc(first["kickoff_utc"]),utc(second["kickoff_utc"])
        if instant>=min(first_kick,second_kick):
            continue
        picks=[{"score1":p["Score 1"],"score2":p["Score 2"],
                "joint_probability":float(p["Joint"])} for p in best["doubles"]]
        # Freeze v9/v10 challengers for the IDENTICAL v8-selected pair
        # only when both alternate model snapshots already exist now.
        challenger_picks={}
        for challenger in ("v9","v10"):
            candidates=[options.get((provider,fixture_id,challenger))
                        for provider,fixture_id in ids]
            if all(candidates) and all(utc(c["captured_at_utc"])<=instant
                                       for c in candidates):
                alternatives=five_doubles(_match(candidates[0]),_match(candidates[1]))
                if len(alternatives)==5:
                    challenger_picks[challenger]=[{
                        "score1":p["Score 1"],"score2":p["Score 2"],
                        "joint_probability":float(p["Joint"])} for p in alternatives]
        result={"policy":"v10-evidence-ranked-shadow-v1",
                "mode":"prospective", "status":"frozen_pre_kickoff",
                "sydney_date":day,"captured_at_utc":instant.isoformat(),
                "first":{field:first[field] for field in
                         ("provider","fixture_id","kickoff_utc","home_team_id","away_team_id",
                          "home_team","away_team","forecast_id","model_version")},
                "second":{field:second[field] for field in
                          ("provider","fixture_id","kickoff_utc","home_team_id","away_team_id",
                           "home_team","away_team","forecast_id","model_version")},
                "five_doubles":picks,"estimated_coverage":float(best["coverage"]),
                "same_pair_challengers":challenger_picks,
                "independence_assumption":True,
                "accuracy_status":"pending_verified_final_scores",
                "selection_note":"Maximize sum of five mutually exclusive joint score probabilities across eligible four-team Sydney-day pairs; no arbitrary diversification or model promotion.",
                "minimum_coverage_policy":MIN_COVERAGE,
                "selection_window":{
                    "max_hours_before_first_eligible_kickoff":FREEZE_MAX_LEAD_HOURS,
                    "minimum_minutes_before_each_selected_match":FREEZE_MIN_LEAD_MINUTES}}
        if _write_once(target,result):
            stats["days_saved"].append(day)
    return stats


def _pair_verified(leg,verified):
    result=verified.get(str(leg["fixture_id"]))
    return result if _qualified(leg,result) else None


def evaluate(root=ROOT,results=None):
    if results is not None:
        verified=results
    else:
        from reconcile_capture_conflicts import merge_verified
        # Legacy and research-only providers share the same verifiable
        # identity standard. Never silently favour conflicting final scores.
        old=load_json(RESULTS,{})
        independent=load_json(Path(root)/"verified_results.json",{})
        verified=merge_verified(old,independent)
    latest=read_latest(root)
    model_rows=defaultdict(list)
    frozen=defaultdict(int)
    hits={}
    pending=defaultdict(int)
    for (provider,fixture_id,model),row in latest.items():
        frozen[model]+=1
        result=verified.get(fixture_id)
        if _qualified(row,result):
            model_rows[model].append(_metric(row,result))
            hits[(provider,fixture_id,model)]=int(row["predicted_score"]==
                                                  [result["home_goals"],result["away_goals"]])
        else:
            pending[model]+=1
    # Pair each market score ONLY against a historical v8 forecast built
    # from the identical evidence packet. If the original v8 was frozen days
    # earlier, the optional v8_control captures the SAME new history and
    # time as the market challenger; no unfair old-vs-new comparison.
    common=[]
    common_v10=[]
    control_count=0
    comparison_excluded=0
    for (provider,fixture_id,model),row in latest.items():
        if model!="v8":
            continue
        for challenger, cohort in (("v9",common),("v10",common_v10)):
            alt=latest.get((provider,fixture_id,challenger))
            if alt is None or hits.get((provider,fixture_id,challenger)) is None:
                continue
            matched=next((
                source for source in (
                    latest.get((provider,fixture_id,"v8_control")),
                    row)
                if source is not None
                and source.get("input_digest")==alt.get("input_digest")
                and utc(source["captured_at_utc"])<=utc(alt["captured_at_utc"])
            ),None)
            if matched is None:
                comparison_excluded+=1
                continue
            baseline_hit=hits.get((provider,fixture_id,matched["model_key"]))
            challenger_hit=hits.get((provider,fixture_id,challenger))
            if baseline_hit is None:
                continue
            cohort.append((baseline_hit,challenger_hit))
            control_count+=int(matched["model_key"]=="v8_control")
    pairs=[]
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        packet=load_json(path,{})
        if packet.get("mode")!="prospective" or packet.get("status")!="frozen_pre_kickoff":
            continue
        try:
            captured=utc(packet["captured_at_utc"])
            if captured>=min(utc(packet["first"]["kickoff_utc"]),utc(packet["second"]["kickoff_utc"])):
                continue
            a,b=(_pair_verified(packet[side],verified) for side in ("first","second"))
            if a is None or b is None:
                pairs.append({"sydney_date":packet["sydney_date"],
                              "status":"awaiting_verified_results"})
                continue
            score1=f'{a["home_goals"]}-{a["away_goals"]}'
            score2=f'{b["home_goals"]}-{b["away_goals"]}'
            hit=any(p["score1"]==score1 and p["score2"]==score2
                    for p in packet["five_doubles"])
            additional={}
            for challenger,choices in (packet.get("same_pair_challengers") or {}).items():
                if challenger not in ("v9","v10") or len(choices)!=5:
                    continue
                additional[challenger]={
                    "hit":any(p["score1"]==score1 and p["score2"]==score2 for p in choices),
                    "coverage":sum(float(p["joint_probability"]) for p in choices)}
            pairs.append({"sydney_date":packet["sydney_date"],
                          "status":"verified","hit":hit,
                          "same_pair_challengers":additional,
                          "actual_scores":[score1,score2],
                          "estimated_coverage":packet["estimated_coverage"]})
        except (KeyError,TypeError,ValueError):
            continue
    completed=[p for p in pairs if p["status"]=="verified"]
    comparison={}
    for challenger in ("v9","v10"):
        cohort=[p for p in completed if challenger in p.get("same_pair_challengers",{})]
        comparison[challenger]={
            "same_pair_verified_days":len(cohort),
            "challenger_wins":sum(p["same_pair_challengers"][challenger]["hit"] for p in cohort),
            "v8_wins_on_those_days":sum(p["hit"] for p in cohort),
            "challenger_hit_rate":sum(p["same_pair_challengers"][challenger]["hit"] for p in cohort)/len(cohort) if cohort else None,
            "note":"Frozen before kickoff on the identical v8-selected fixtures; no post-hoc alternate pairs."}
    report={
        "version":"v10-shadow-performance-v1",
        "evidence":"genuinely_frozen_pre_kickoff_only",
        "verified_result_source":"verified provider-matched union: data/prematch/verified_results.json and data/v10_prospective/verified_results.json",
        "models":{m:_summarize(model_rows[m],frozen=frozen[m]) for m in ("v8","v9","v10")},
        "comparison_diagnostics":{
            "matched_market_time_controls":control_count,
            "excluded_mismatched_evidence_pairs":comparison_excluded,
            "policy":"Only compare v8 versus v9/v10 when input_digest and original evidence match; use preserved v8_control for a later market snapshot."},
        "paired_v8_v9":{"verified_same_fixture_count":len(common),
                        "v8_only_hits":sum(a and not b for a,b in common),
                        "v9_only_hits":sum(b and not a for a,b in common),
                        "insufficient_to_claim_improvement":len(common)<200},
        "paired_v8_v10":{"verified_same_fixture_count":len(common_v10),
                         "v8_only_hits":sum(a and not b for a,b in common_v10),
                         "v10_only_hits":sum(b and not a for a,b in common_v10),
                         "insufficient_to_claim_improvement":len(common_v10)<200},
        "five_doubles":{"frozen_pairs":len(pairs),"verified_pairs":len(completed),
                        "same_pair_challengers":comparison,
                        "awaiting_both_results":len(pairs)-len(completed),
                        "wins":sum(p["hit"] for p in completed),
                        "hit_rate":sum(p["hit"] for p in completed)/len(completed) if completed else None,
                        "average_estimated_coverage":
                            sum(p["estimated_coverage"] for p in completed)/len(completed)
                            if completed else None,"rows":pairs},
        "limitations":"Exact scores remain low probability. Retrospective replay and a market-disabled v8 fallback are not counted as verified v9/v10 results. Pending matches are not losses.",
        "promotion_gate":"Never promote v9/v10 as more accurate without enough paired prospective results, calibration and uncertainty checks."}
    return report


def main():
    stats=freeze_doubles()
    report=evaluate()
    report["freeze_diagnostics"]=stats
    write_json(REPORT,report)
    print(json.dumps({key:val if key!="five_doubles" else {
        k:v for k,v in val.items() if k!="rows"} for key,val in report.items()},indent=2))


if __name__=="__main__":
    main()
