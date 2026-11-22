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
from double_score_selection import select_balanced,select_diversified
from robust_double_v14 import robust_pair, rank_robust_pairs, VERSION as ROBUST_V14
from v10_prospective import ROOT, read_latest, utc

REPORT = ROOT / "performance.json"
PAIRS = ROOT / "doubles"
MIN_COVERAGE = 0.04  # conservative untuned guardrail, NOT a calibrated hit rate
FREEZE_MAX_LEAD_HOURS = 24  # do not freeze a week early before full fixture discovery
FREEZE_MIN_LEAD_MINUTES = 45  # avoid selecting a pair as kickoff starts
SAME_PAIR_CHALLENGERS = ("v9", "v10", "v12", "v13", "v13_balanced", "v13_diversified",
                         "v7_diversified_same_pair", "v14_robust_same_pair")


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


def _same_pair_source(candidate, original, cutoff):
    return (
        utc(candidate["captured_at_utc"]) <= cutoff
        and bool(original.get("input_digest"))
        and candidate.get("input_digest") == original["input_digest"]
        and utc(candidate["kickoff_utc"]) == utc(original["kickoff_utc"])
        and all(str(candidate[key]) == str(original[key]) for key in
                ("provider", "fixture_id", "home_team_id", "away_team_id"))
    )


def _v14_match(row):
    match=_match(row)
    match["result"].update({
        "component_score_matrices":row.get("v14_component_grids") or {},
        "component_versions":row.get("v14_component_versions") or {},
    })
    return match


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
    # The original frozen v8 score remains the reference forecast.
    # For FUTURE pair selection, use a separately frozen v8_near model
    # where that verified before-kickoff research is genuinely newer.
    # An early snapshot must never be silently rewritten as "fresh".
    per_fixture={}
    for key,row in options.items():
        model=row.get("model_key")
        if model not in ("v8","v8_near"):
            continue
        if (len(row.get("component_weights") or {})<2
                or int(row.get("history_matches") or 0)<10
                or len(row.get("top10_scores") or [])<5):
            continue
        kickoff=utc(row["kickoff_utc"])
        recorded=utc(row["captured_at_utc"])
        if kickoff<=instant or recorded>instant or recorded>=kickoff:
            continue
        identity=(row["provider"],str(row["fixture_id"]))
        current=per_fixture.get(identity)
        # Prefer genuinely fresher evidence when it exists. A near-kickoff
        # model is NOT automatically assumed to outperform the old model.
        if (current is None or
            (model=="v8_near" and current["model_key"]!="v8_near") or
            (model==current["model_key"] and recorded>utc(current["captured_at_utc"]))):
            per_fixture[identity]=row
    for row in per_fixture.values():
        kickoff=utc(row["kickoff_utc"])
        day=kickoff.astimezone(SYDNEY).date().isoformat()
        eligible[day].append(row)
    stats={"days_evaluated":0,"days_saved":[],"v14_robust_days_saved":[],
           "abstained":{},
           "near_research_available":sum(r["model_key"]=="v8_near" for r in per_fixture.values())}
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
        # The challenger must match the exact source used for each saved V8
        # leg. An unrelated V8 control cannot qualify the selected baseline.
        challenger_picks={}
        ignored_challengers={}
        for challenger in ("v9", "v10", "v12"):
            candidates=[options.get((provider,fixture_id,challenger))
                        for provider,fixture_id in ids]
            if not all(candidates):
                ignored_challengers[challenger]="missing_one_or_both_challenger_forecasts"
                continue
            if not all(_same_pair_source(candidate, original, instant)
                       for candidate, original in zip(candidates, (first, second))):
                ignored_challengers[challenger]="ineligible_or_different_selected_v8_evidence"
                continue
            alternatives=five_doubles(_match(candidates[0]),_match(candidates[1]))
            if len(alternatives)==5:
                challenger_picks[challenger]=[{
                    "score1":p["Score 1"],"score2":p["Score 2"],
                    "joint_probability":float(p["Joint"])} for p in alternatives]
        # Same-PAIR V13 comparison, with each V13 forecast already frozen
        # before this selection and matched to the selected V8 evidence digest.
        # Never backfill V13 picks after either fixture starts or if a newer
        # v8_near snapshot used different history.
        v13_rows=[options.get((provider,fixture_id,"v13"))
                  for provider,fixture_id in ids]
        if all(v13_rows) and all(
            _same_pair_source(candidate, original, instant)
            for candidate,original in zip(v13_rows,(first,second))):
            left,right=(_match(candidate) for candidate in v13_rows)
            primary=five_doubles(left,right)
            if len(primary)==5:
                sources=[{"Score 1":f"{h}-{a}","Score 2":f"{j}-{k}",
                          "Joint":float(p)*float(q)}
                         for h,a,p in v13_rows[0]["top10_scores"][:10]
                         for j,k,q in v13_rows[1]["top10_scores"][:10]]
                alternates={
                    "v13":primary,
                    "v13_balanced":select_balanced(sources,5,diversity_weight=.18),
                    "v13_diversified":select_diversified(sources,5)}
                for name,options_five in alternates.items():
                    if len(options_five)==5:
                        challenger_picks[name]=[
                            {"score1":p["Score 1"],"score2":p["Score 2"],
                             "joint_probability":float(p["Joint"])}
                            for p in options_five]
            # The historical 65-day research found V7 fully diversified
            # worth prospectively observing; these are REAL fixed V7
            # component scores, not a post-match label on V13 probabilities.
            seven_first=v13_rows[0].get("v7_fitted_top10_scores") or []
            seven_second=v13_rows[1].get("v7_fitted_top10_scores") or []
            if len(seven_first)>=5 and len(seven_second)>=5:
                seven_options=[
                    {"Score 1":f"{h}-{a}","Score 2":f"{j}-{k}",
                     "Joint":float(p)*float(q)}
                    for h,a,p in seven_first[:10]
                    for j,k,q in seven_second[:10]]
                seven=select_diversified(seven_options,5)
                if len(seven)==5:
                    challenger_picks["v7_diversified_same_pair"]=[
                        {"score1":p["Score 1"],"score2":p["Score 2"],
                         "joint_probability":float(p["Joint"])}
                        for p in seven]
            # V14 uses complete V7 and V8 score grids; it must abstain
            # when the genuine frozen model data are unavailable.
            try:
                conservative=robust_pair(
                    _v14_match(v13_rows[0]),_v14_match(v13_rows[1]),
                    now=instant)
                if conservative is not None:
                    challenger_picks["v14_robust_same_pair"]=[
                        {"score1":p["Score 1"],"score2":p["Score 2"],
                         "joint_probability":float(p["Joint"]),
                         "conservative_not_calibrated":True}
                        for p in conservative["doubles"]]
            except (ValueError,TypeError,KeyError,ArithmeticError):
                ignored_challengers["v14_robust_same_pair"]="invalid_score_grid"
        else:
            ignored_challengers["v13"]="missing_same_evidence_pre_match_v13_pair"
        result={"policy":"v10-evidence-ranked-shadow-v2-fresh-research",
                "mode":"prospective", "status":"frozen_pre_kickoff",
                "sydney_date":day,"captured_at_utc":instant.isoformat(),
                "first":{field:first[field] for field in
                         ("provider","fixture_id","kickoff_utc","home_team_id","away_team_id",
                          "home_team","away_team","forecast_id","model_key","model_version")},
                "second":{field:second[field] for field in
                          ("provider","fixture_id","kickoff_utc","home_team_id","away_team_id",
                           "home_team","away_team","forecast_id","model_key","model_version")},
                "five_doubles":picks,"estimated_coverage":float(best["coverage"]),
                "same_pair_challengers":challenger_picks,
                "challenger_exclusions":ignored_challengers,
                "selected_forecasts":{str(key):{
                    "source":r["model_key"],"input_digest":r.get("input_digest"),
                    "captured_at_utc":r["captured_at_utc"]}
                    for key,r in zip((1,2),(first,second))},
                "independence_assumption":True,
                "accuracy_status":"pending_verified_final_scores",
                "selection_note":"Maximize sum of five mutually exclusive joint score probabilities across eligible four-team Sydney-day pairs; no arbitrary diversification or model promotion.",
                "minimum_coverage_policy":MIN_COVERAGE,
                "selection_window":{
                    "max_hours_before_first_eligible_kickoff":FREEZE_MAX_LEAD_HOURS,
                    "minimum_minutes_before_each_selected_match":FREEZE_MIN_LEAD_MINUTES}}
        if _write_once(target,result):
            stats["days_saved"].append(day)
            # Freeze an independently selected V14 two-match pair for this
            # same newly discovered slate/date. Never backfill on older
            # days or invent earlier score grids from newer research.
            robust_pool=[]
            by_v13={}
            for original in rows:
                key=(original["provider"],str(original["fixture_id"]))
                archived=options.get((*key,"v13"))
                if (archived is None or
                    not _same_pair_source(archived, original, instant) or
                    not archived.get("v14_component_grids")):
                    continue
                robust_pool.append(_v14_match(archived))
                by_v13[key]=archived
            shortlisted=rank_robust_pairs(
                robust_pool,now=instant,limit=1,min_coverage=MIN_COVERAGE)
            if shortlisted:
                candidate=shortlisted[0]
                selected_ids=[
                    (part["fixture"]["provider"],str(part["fixture"]["id"]))
                    for part in (candidate["first"],candidate["second"])]
                v14first,v14second=(by_v13[key] for key in selected_ids)
                record={
                    "policy":"v14-robust-two-model-prospective-shadow-v1",
                    "mode":"prospective_robust_v14",
                    "status":"frozen_pre_kickoff",
                    "model_version":ROBUST_V14,
                    "sydney_date":day,
                    "captured_at_utc":instant.isoformat(),
                    "first":{field:v14first[field] for field in
                             ("provider","fixture_id","kickoff_utc",
                              "home_team_id","away_team_id",
                              "home_team","away_team","forecast_id",
                              "model_key","model_version")},
                    "second":{field:v14second[field] for field in
                              ("provider","fixture_id","kickoff_utc",
                               "home_team_id","away_team_id",
                               "home_team","away_team","forecast_id",
                               "model_key","model_version")},
                    "selected_forecast_input_digests":[
                        v14first["input_digest"],v14second["input_digest"]],
                    "five_doubles":[{
                        "score1":opt["Score 1"],"score2":opt["Score 2"],
                        "v7_joint":float(opt["V7 joint"]),
                        "v8_joint":float(opt["V8 joint"]),
                        "conservative_envelope":float(opt["Joint"])}
                        for opt in candidate["doubles"]],
                    "lower_envelope_score":float(
                        candidate["estimated_joint_coverage"]),
                    "coverage_v7":float(candidate["model_coverage_v7"]),
                    "coverage_v8":float(candidate["model_coverage_v8"]),
                    "covered_probability_is_not_verified_win_rate":True,
                    "lower_envelope_is_not_calibrated_probability":True,
                    "model_sources_are_correlated":True,
                    "accuracy_status":"pending_verified_final_scores",
                    "original_five_doubles_modified":False}
                robust_path=Path(root)/"robust_doubles"/(day+".json")
                if _write_once(robust_path,record):
                    stats["v14_robust_days_saved"].append(day)
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
    common_v12=[]
    common_v13=[]
    original_vs_near=[]
    control_count=0
    comparison_excluded=0
    for (provider,fixture_id,model),row in latest.items():
        if model!="v8":
            continue
        early=hits.get((provider,fixture_id,"v8"))
        near=hits.get((provider,fixture_id,"v8_near"))
        if early is not None and near is not None:
            # This tests whether genuinely newer available information
            # helped the SAME real fixture; not a same-as-of model contest.
            original_vs_near.append((early,near))
        # V12 and V8 use exactly the SAME historical input digest and are
        # frozen together, so they can be compared without market-as-of
        # controls or post-hoc fixture selection.
        alt12=latest.get((provider,fixture_id,"v12"))
        if (alt12 is not None and
            alt12.get("input_digest")==row.get("input_digest") and
            utc(row["captured_at_utc"])<=utc(alt12["captured_at_utc"]) and
            hits.get((provider,fixture_id,"v8")) is not None and
            hits.get((provider,fixture_id,"v12")) is not None):
            common_v12.append((
                hits[(provider,fixture_id,"v8")],
                hits[(provider,fixture_id,"v12")]))
        alt13=latest.get((provider,fixture_id,"v13"))
        if (alt13 is not None and
            alt13.get("input_digest")==row.get("input_digest") and
            utc(row["captured_at_utc"])<=utc(alt13["captured_at_utc"]) and
            hits.get((provider,fixture_id,"v8")) is not None and
            hits.get((provider,fixture_id,"v13")) is not None):
            common_v13.append((
                hits[(provider,fixture_id,"v8")],
                hits[(provider,fixture_id,"v13")]))
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
                if challenger not in SAME_PAIR_CHALLENGERS or len(choices)!=5:
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
    for challenger in SAME_PAIR_CHALLENGERS:
        cohort=[p for p in completed if challenger in p.get("same_pair_challengers",{})]
        comparison[challenger]={
            "same_pair_verified_days":len(cohort),
            "challenger_wins":sum(p["same_pair_challengers"][challenger]["hit"] for p in cohort),
            "v8_wins_on_those_days":sum(p["hit"] for p in cohort),
            "challenger_hit_rate":sum(p["same_pair_challengers"][challenger]["hit"] for p in cohort)/len(cohort) if cohort else None,
            "note":("Frozen before kickoff on the same v8-selected matches. "
                    "For V14 the coverage figure is a lower-envelope model "
                    "criterion, NOT a calibrated probability or win rate.")}
    # V14 selects an entirely different pair on a day when its conservative
    # V7/V8 envelope supports it. Never award a hit using the original
    # pair's results; independently verify both V14 fixtures.
    robust_days=[]
    for path in sorted((Path(root)/"robust_doubles").glob("*.json")):
        packet=load_json(path,{})
        if (packet.get("mode")!="prospective_robust_v14" or
            packet.get("status")!="frozen_pre_kickoff"):
            continue
        try:
            recorded_at=utc(packet["captured_at_utc"])
            if recorded_at>=min(utc(packet["first"]["kickoff_utc"]),
                                utc(packet["second"]["kickoff_utc"])):
                continue
            actual_first,actual_second=(
                _pair_verified(packet[side],verified)
                for side in ("first","second"))
            if actual_first is None or actual_second is None:
                robust_days.append({
                    "sydney_date":packet["sydney_date"],
                    "status":"awaiting_verified_results"})
                continue
            home=f'{actual_first["home_goals"]}-{actual_first["away_goals"]}'
            away=f'{actual_second["home_goals"]}-{actual_second["away_goals"]}'
            hit=any(row["score1"]==home and row["score2"]==away
                    for row in packet["five_doubles"])
            robust_days.append({
                "sydney_date":packet["sydney_date"],
                "status":"verified","hit":bool(hit),
                "actual_scores":[home,away],
                "selected_fixture_ids":[str(packet["first"]["fixture_id"]),
                                        str(packet["second"]["fixture_id"])],
                "lower_envelope_score":float(packet["lower_envelope_score"]),
            })
        except (KeyError,TypeError,ValueError):
            continue
    robust_verified=[r for r in robust_days if r["status"]=="verified"]
    original_by_date={r["sydney_date"]:r for r in completed}
    both_on_day=[(original_by_date[r["sydney_date"]],r)
                 for r in robust_verified if r["sydney_date"] in original_by_date]
    robust_report={
        "frozen_pairs":len(robust_days),
        "verified_pairs":len(robust_verified),
        "awaiting_both_results":len(robust_days)-len(robust_verified),
        "wins":sum(int(r["hit"]) for r in robust_verified),
        "hit_rate":(sum(int(r["hit"]) for r in robust_verified)/
                    len(robust_verified) if robust_verified else None),
        "same_sydney_day_v8_comparison_count":len(both_on_day),
        "v14_only_winning_days":sum(
            bool(v14["hit"]) and not bool(original["hit"])
            for original,v14 in both_on_day),
        "v8_only_winning_days":sum(
            bool(original["hit"]) and not bool(v14["hit"])
            for original,v14 in both_on_day),
        "distinct_pair_selection":"V14 may choose different two fixtures on same day",
        "lower_envelope_is_not_a_calibrated_probability":True,
        "experiments_not_promoted":True,
        "rows":robust_days,
    }
    report={
        "version":"v10-shadow-performance-v1",
        "evidence":"genuinely_frozen_pre_kickoff_only",
        "verified_result_source":"verified provider-matched union: data/prematch/verified_results.json and data/v10_prospective/verified_results.json",
        "models":{m:_summarize(model_rows[m],frozen=frozen[m]) for m in ("v8","v8_near","v9","v10","v12","v13")},
        "paired_original_v8_vs_near_v8":{
            "same_fixture_verified":len(original_vs_near),
            "original_only_exact_hits":sum(a and not b for a,b in original_vs_near),
            "fresh_research_only_exact_hits":sum(b and not a for a,b in original_vs_near),
            "purpose":"Observe the effect of newer verified pre-match information; NOT an equal-information model comparison.",
            "insufficient_to_claim_improvement":len(original_vs_near)<200},
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
        "paired_v8_v12":{"verified_same_fixture_count":len(common_v12),
                         "v8_only_hits":sum(a and not b for a,b in common_v12),
                         "v12_only_hits":sum(b and not a for a,b in common_v12),
                         "insufficient_to_claim_improvement":len(common_v12)<200,
                         "same_evidence_required":True,
                         "v12_is_shadow_only":True},
        "paired_v8_v13":{"verified_same_fixture_count":len(common_v13),
                         "v8_only_hits":sum(a and not b for a,b in common_v13),
                         "v13_only_hits":sum(b and not a for a,b in common_v13),
                         "insufficient_to_claim_improvement":len(common_v13)<200,
                         "same_evidence_required":True,
                         "v13_is_shadow_only":True},
        "v14_independently_selected_five_doubles":robust_report,
        "five_doubles":{"frozen_pairs":len(pairs),"verified_pairs":len(completed),
                        "same_pair_challengers":comparison,
                        "awaiting_both_results":len(pairs)-len(completed),
                        "wins":sum(p["hit"] for p in completed),
                        "hit_rate":sum(p["hit"] for p in completed)/len(completed) if completed else None,
                        "average_estimated_coverage":
                            sum(p["estimated_coverage"] for p in completed)/len(completed)
                            if completed else None,"rows":pairs},
        "limitations":"Exact scores remain low probability. Retrospective replay and a market-disabled v8 fallback are not counted as verified v9/v10 results. Pending matches are not losses.",
        "promotion_gate":"Never promote v9/v10/v12/v13/v14 as more accurate without sufficient same-evidence prospective results, calibration and uncertainty checks."}
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
