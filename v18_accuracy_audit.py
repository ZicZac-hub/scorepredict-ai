"""V18: prospective, source-matched accuracy and calibration diagnostics.

No paid API calls, training on future information, rewriting immutable
forecasts, retrospective predictions, or automatic model promotion.

The observational sample is *original saved V8 forecasts* with separately
verified final results. Rolling recalibration, if enough examples ever exist,
uses only results verified BEFORE the evaluated forecast was captured.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from persist_prematch import load_json, verified_result, write_json
from v10_prospective import ROOT as SOURCE_ROOT, read_latest, utc
from v16_pair_research import original_model
from v17_confidence import ROOT as V17_ROOT, probabilities, wilson

ROOT=Path("data/v18_accuracy")
VERSION="v18-immutable-observed-calibration-audit-v1"
MIN_GLOBAL_TRAIN=100
MIN_BUCKET_TRAIN=20
MIN_CALIBRATION_TEST=50
PRIOR_STRENGTH=30
BINS=((0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.001))


def score_type(home,away):
    if type(home) is not int or type(away) is not int or home<0 or away<0:
        raise ValueError("invalid_regulation_goals")
    return "1" if home>away else ("X" if home==away else "2")


def bucket(prob):
    if not math.isfinite(prob) or not 0<=prob<=1:
        raise ValueError("invalid_probability")
    for i,(lo,hi) in enumerate(BINS):
        if lo<=prob<hi:
            return i
    raise ValueError("confidence_outside_double_chance_range")


def _qualified(row,result):
    if not isinstance(result,dict):
        return False
    try:
        if not verified_result(row,result,str(row["fixture_id"])):
            return False
        if result.get("status") not in ("FT","FINISHED","AET","PEN"):
            return False
        if utc(result["verified_at_utc"])<=utc(row["kickoff_utc"]):
            return False
        score_type(result["home_goals"],result["away_goals"])
        return True
    except (ValueError,TypeError,KeyError,OverflowError):
        return False


def source_observations(root=SOURCE_ROOT,verified=None,forecast_rows=None):
    """Build one actual observation per provider-scoped original V8 fixture.

    Missing/unverifiable results are PENDING, not losses. Other model
    snapshots, unrelated forecast source IDs and late records never count.
    """
    if verified is None:
        verified=load_json(Path(root)/"verified_results.json",{})
    if forecast_rows is None:
        forecast_rows=[row for row in read_latest(root).values()
                       if row.get("model_key")=="v8"]
    rows=[];missing=Counter();pending=0
    seen=set()
    quality=Counter()
    for row in forecast_rows:
        try:
            if not isinstance(row,dict) or row.get("model_key")!="v8":
                missing["non_original_v8"]+=1
                continue
            identity=(str(row["provider"]),str(row["fixture_id"]))
            if identity in seen:
                missing["duplicate_v8_fixture"]+=1
                continue
            seen.add(identity)
            p=probabilities(row)
            orig=original_model(root,row)
            if (orig is None or orig.get("input_digest")!=row["input_digest"]
                    or orig.get("model_key")!="v8"):
                missing["original_forecast_id_or_digest_unverified"]+=1
                continue
            if (int(row.get("history_matches") or 0)<30 or
                    int(row.get("league_history_matches") or 0)<30):
                quality["insufficient_history_under_30"]+=1
            context=row.get("research_context") or {}
            hx=(context.get("home") or {}) if isinstance(context,dict) else {}
            ax=(context.get("away") or {}) if isinstance(context,dict) else {}
            if ((hx.get("last10_observed") or 0)<8 or
                (ax.get("last10_observed") or 0)<8):
                quality["short_last_ten_sample"]+=1
            # These must never silently transform model-estimated goals into
            # licensed shot-level xG or display-only team news into coefficients.
            if not ((context.get("verified_xg_xga") or {}).get("available")):
                quality["shot_based_xg_not_verified"]+=1
            if not ((context.get("verified_lineup_adjustments") or {}).get("applied")):
                quality["lineup_adjustment_not_validated"]+=1
            odds = row.get("market_snapshot") or {}
            if not (row.get("market_used") and
                    odds.get("snapshot_utc") and odds.get("bookmaker")):
                quality["independent_market_snapshot_unavailable"]+=1

            kind=max(("1X","X2","12"),key=lambda x:(p[x],x))
            estimate=float(p[kind])
            item={"provider":str(row["provider"]),
                  "fixture_id":str(row["fixture_id"]),
                  "forecast_id":str(row["forecast_id"]),
                  "source_digest":str(row["input_digest"]),
                  "competition":str(row.get("competition") or "UNKNOWN"),
                  "captured_at_utc":utc(row["captured_at_utc"]).isoformat(),
                  "kickoff_utc":utc(row["kickoff_utc"]).isoformat(),
                  "selection":kind,
                  "estimated_probability":estimate,
                  "bin":bucket(estimate)}
            result=verified.get(str(row["fixture_id"]))
            if not _qualified(row,result):
                pending+=1
                continue
            observed=score_type(result["home_goals"],result["away_goals"])
            item["hit"]=int(observed in kind)
            item["actual_outcome"]=observed
            item["verified_at_utc"]=utc(result["verified_at_utc"]).isoformat()
            rows.append(item)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            missing[str(exc)[:70]]+=1
    rows.sort(key=lambda x:(x["captured_at_utc"],x["provider"],x["fixture_id"]))
    return rows,{"frozen_source_matches_valid":len(rows)+pending,
                 "verified_results":len(rows),
                 "awaiting_or_unverified":pending,
                 "source_rejections":dict(missing),
                 "missing_feature_counts":dict(quality),
                 "no_extra_paid_calls":True}


def reliability(rows):
    """Brier/reliability from truly verified original predictions only."""
    distribution=[]
    for i,(lo,hi) in enumerate(BINS):
        matched=[x for x in rows if x["bin"]==i]
        n=len(matched);wins=sum(int(x["hit"]) for x in matched)
        pred=sum(float(x["estimated_probability"]) for x in matched)/n if n else None
        actual=wins/n if n else None
        distribution.append({
            "estimated_band":f"{lo:.0%}–{min(hi,1):.0%}",
            "verified":n,"hits":wins,"mean_estimated":pred,
            "observed_rate":actual,"wilson_95":wilson(wins,n),
            "absolute_gap":abs(actual-pred) if n else None})
    n=len(rows)
    return {"verified":n,
            "hits":sum(int(x["hit"]) for x in rows),
            "observed_win_rate":sum(int(x["hit"]) for x in rows)/n if n else None,
            "mean_model_estimate":sum(float(x["estimated_probability"]) for x in rows)/n if n else None,
            "brier":sum((float(x["estimated_probability"])-int(x["hit"]))**2
                        for x in rows)/n if n else None,
            "expected_calibration_error":sum(
                x["verified"]*x["absolute_gap"] for x in distribution
                if x["absolute_gap"] is not None)/n if n else None,
            "wilson_95":wilson(sum(int(x["hit"]) for x in rows),n),
            "bins":distribution,
            "sample_too_small_for_reliable_calibration":n<200}


def time_safe_calibration(rows):
    """Test only chronological, pre-verification-available bucket correction.

    Training uses results whose independently VERIFIED timestamp is strictly
    earlier than the candidate's forecast CAPTURE timestamp, not kickoff.
    A shrinkage toward the candidate's V8 probability avoids overreacting.
    No parameters are tuned against the evaluation cohort.
    """
    ordered=sorted(rows,key=lambda x:(x["captured_at_utc"],x["provider"],x["fixture_id"]))
    attempts=0
    results=[]
    for row in ordered:
        past=[p for p in ordered
              if p["verified_at_utc"]<row["captured_at_utc"]
              and p["kickoff_utc"]<row["captured_at_utc"]
              and not (p["provider"]==row["provider"] and
                       p["fixture_id"]==row["fixture_id"])]
        if len(past)<MIN_GLOBAL_TRAIN:
            continue
        matched=[p for p in past if p["bin"]==row["bin"]]
        if len(matched)<MIN_BUCKET_TRAIN:
            continue
        attempts+=1
        baseline=float(row["estimated_probability"])
        calibrated=(sum(int(x["hit"]) for x in matched)+PRIOR_STRENGTH*baseline)/(
            len(matched)+PRIOR_STRENGTH)
        results.append({"fixture_id":row["fixture_id"],
                        "raw_brier":(baseline-int(row["hit"]))**2,
                        "calibrated_brier":(calibrated-int(row["hit"]))**2})
    count=len(results)
    raw=sum(x["raw_brier"] for x in results)/count if count else None
    adjusted=sum(x["calibrated_brier"] for x in results)/count if count else None
    return {"walk_forward_verified_test_matches":count,
            "min_global_previous_verified":MIN_GLOBAL_TRAIN,
            "min_prior_verified_in_same_band":MIN_BUCKET_TRAIN,
            "min_eligible_test_matches_to_review":MIN_CALIBRATION_TEST,
            "raw_brier":raw,"shrunk_calibrated_brier":adjusted,
            "calibration_candidate_eligible_for_review":bool(
                count>=MIN_CALIBRATION_TEST and adjusted is not None
                and raw is not None and adjusted<raw),
            "deployed":False,"model_promotion":False,
            "limitations":"Original V8 outputs remain unchanged. The diagnostic is only "
                          "an as-of-safe candidate; passing a retrospective screen does "
                          "not establish a prospective model advantage."}


def report(root=SOURCE_ROOT,verified=None,forecast_rows=None):
    observations,diagnostics=source_observations(root,verified,forecast_rows)
    summaries={}
    for league in sorted({r["competition"] for r in observations}):
        subset=[r for r in observations if r["competition"]==league]
        item=reliability(subset)
        summaries[league]={"verified":item["verified"],
                           "observed_win_rate":item["observed_win_rate"],
                           "brier":item["brier"],
                           "sample_too_small":len(subset)<50}
    base=reliability(observations)
    cal=time_safe_calibration(observations)
    return {"version":VERSION,"evaluation_type":"original_prospective_v8_only",
            "evaluated_forecasts_captured_pre_kickoff":True,
            "verified_only_no_pending_losses":True,
            "target_70_percent_actual_claim":False,
            "baseline_double_chance":base,
            "time_safe_shrinkage_research":cal,
            "per_league":summaries,
            "evidence":diagnostics,
            "unavailable_external_sources":{
                "genuine_shot_quality_xg":"no_audited_feed_integrated",
                "validated_player_lineup_coefficients":"not_trained_or_validated",
                "multi_book_consensus":"not_prospectively_calibrated",
                "two_leg_multi_llm_consensus":"not_source_qualified_yet"},
            "note":"No new football observations are fabricated. Reliable calibration "
                   "needs hundreds of verified pre-match predictions. Original "
                   "five exact-score double results are tracked separately."}


def main():
    result=report()
    write_json(ROOT/"accuracy_report.json",result)
    summary=result["baseline_double_chance"]
    print(json.dumps({
        "verified":summary["verified"],
        "observed_win_rate":summary["observed_win_rate"],
        "brier":summary["brier"],
        "calibration_tested":result["time_safe_shrinkage_research"]["walk_forward_verified_test_matches"],
        "calibration_deployed":False,"70_percent_claim_supported":False},
        sort_keys=True))


if __name__=="__main__":
    main()
