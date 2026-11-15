"""V15 research-only, genuinely paired external AI five-double ledger.

Frozen research must include two independent fixture identities, an actual
saved statistical model for both legs and at least two genuine pre-kickoff
AI provider votes PER leg. AI votes are qualitative suggestions used only
to break nearly tied statistical-score rankings (<=12% per leg);
they are NOT calibrated exact-score likelihoods or independent evidence
because all LLMs read the same original statistics.

Never issue paid API calls, fabricate missing providers, revise the original
five doubles or backfill past kickoffs. Unpaired votes = abstain.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ai_five_double_engine import merged_five
from v16_pair_research import original_model
from llm_research_panel import PROVIDERS
from v10_prospective import ROOT as BASE_ROOT, read_latest, utc
from persist_prematch import load_json, RESULTS
from reconcile_capture_conflicts import merge_verified
from v10_evaluate import _pair_verified

VERSION="ziczac-v15-ai-assisted-pair-shadow"
ROOT=Path("data/v15_ai_double")
AI_ROOT=Path("data/llm_research")
MIN_PROVIDERS_PER_LEG=2


def _fixture(leg):
    return {
        "fixture":{
            "id":leg["fixture_id"],
            "provider":leg["provider"],
            "utcDate":leg["kickoff_utc"],
            "homeTeam":{"id":leg["home_team_id"],"name":leg["home_team"]},
            "awayTeam":{"id":leg["away_team_id"],"name":leg["away_team"]},
        },
        "kickoff":leg["kickoff_utc"],
    }


def _source(leg,models,root=None):
    # In the live ledger, the original archived file (not today's latest
    # per-fixture V8 refresh) is authoritative. Never swap source digests.
    if root is not None:
        return original_model(root,leg)
    key=(str(leg.get("provider")),str(leg.get("fixture_id")))
    for version in ("v8","v8_near"):
        original=models.get((*key,version))
        if original and original.get("forecast_id")==leg.get("forecast_id"):
            return original
    return None


def valid_saved_votes(raw_votes):
    """Count real, well-formed provider responses, not configured API keys.

    Share this exact validation with the source audit so the panel and
    audit cannot disagree about malformed / synthetic provider votes.
    """
    votes={}
    if not isinstance(raw_votes,dict):
        return votes
    for name,v in raw_votes.items():
        if name not in PROVIDERS or not isinstance(v,dict):
            continue
        score=v.get("score")
        if (not isinstance(score,(tuple,list)) or len(score)!=2 or
            any(type(x) is not int or x<0 or x>9 for x in score)):
            continue
        probs=[v.get("home"),v.get("draw"),v.get("away")]
        if (any(type(p) not in (int,float) or
                not math.isfinite(float(p)) or not 0<=p<=1 for p in probs) or
            abs(sum(probs)-1)>0.02):
            continue
        votes[name]=v
    return votes


def _vote_records(leg,root,now,allowed_sources=None,min_votes=1):
    """Only FIRST real pre-kickoff AI record matching verified fixture IDs.

    Do not use later overwritten versions; return validated, named votes only.
    """
    found=[]
    folder=Path(root)/"forecasts"
    for path in folder.glob("*.json"):
        try:
            row=load_json(path,{})
            if not isinstance(row,dict) or not row.get("research_only"):
                continue
            if (str(row.get("provider"))!=str(leg.get("provider")) or
                str(row.get("fixture_id"))!=str(leg.get("fixture_id")) or
                str(row.get("home_team_id"))!=str(leg.get("home_team_id")) or
                str(row.get("away_team_id"))!=str(leg.get("away_team_id"))):
                continue
            start=utc(row["captured_at_utc"])
            kickoff=utc(row["kickoff_utc"])
            if (kickoff!=utc(leg["kickoff_utc"]) or
                start>=kickoff or start>now):
                continue
            # Exactly the original pair's statistical forecast and digest.
            # Another V8 horizon or later refresh for the SAME fixture must
            # not be silently relabelled as the frozen pair's AI evidence.
            if allowed_sources is not None:
                if not any(
                    source.get("forecast_id")==leg.get("forecast_id") and
                    row.get("source_forecast_id")==source.get("forecast_id") and
                    bool(source.get("input_digest")) and
                    row.get("input_digest")==source.get("input_digest") and
                    bool(source.get("captured_at_utc")) and
                    utc(source["captured_at_utc"])<=start
                    for source in allowed_sources):
                    continue
            votes=valid_saved_votes(row.get("votes"))
            if votes:
                found.append((start,path.name,row,votes))
        except (OSError,ValueError,TypeError,KeyError):
            continue
    if not found:
        return None
    # Chronological rule fixed before results: use the FIRST eligible
    # complete panel, not the most flattering after-the-fact answer.
    eligible=[item for item in found if len(item[3])>=min_votes]
    _,_,selected,votes=min(eligible or found,key=lambda v:(v[0],v[1]))
    return {**selected,"votes":votes}


def assemble_pair(packet,models,*,ai_root=AI_ROOT,now=None,
                  min_votes=MIN_PROVIDERS_PER_LEG,source_root=None):
    """Return (record, abstention diagnostic) from archived historical facts."""
    when=utc(now or datetime.now(timezone.utc))
    if packet.get("mode")!="prospective" or packet.get("status")!="frozen_pre_kickoff":
        return None,"not_an_original_frozen_pair"
    try:
        first,second=packet["first"],packet["second"]
        first_kick,second_kick=utc(first["kickoff_utc"]),utc(second["kickoff_utc"])
        captured=utc(packet["captured_at_utc"])
        if not captured<when<min(first_kick,second_kick):
            return None,"past_kickoff_or_not_yet_frozen"
        sources=[_source(leg,models,root=source_root) for leg in (first,second)]
        if any(not source for source in sources):
            return None,"original_statistical_forecast_missing"
        votes=[]
        for leg,source in zip((first,second),sources):
            votes.append(_vote_records(
                leg,ai_root,when,allowed_sources=[source],
                min_votes=min_votes))
        if not all(votes):
            return None,"missing_original_earlier_ai_vote_for_both_matches"
        counts=[len(v.get("votes") or {}) for v in votes]
        if min(counts)<min_votes:
            return None,"too_few_successful_ai_providers_per_match"
        rows=[]
        for leg,model in zip((first,second),sources):
            row=_fixture(leg)
            row["result"]={"scores":model["top10_scores"],
                           "model_version":model["model_version"]}
            rows.append(row)
        result=merged_five(rows[0],rows[1],votes[0],votes[1],now=when)
        if result["status"]!="ready" or len(result["doubles"])!=5:
            return None,"incomplete_statistical_joint_scores"
        rendered=[]
        for outcome in result["doubles"]:
            rendered.append({
                "score1":outcome["Score 1"],"score2":outcome["Score 2"],
                "statistical_joint_estimate":float(outcome["Model joint"]),
                "supporting_ai_votes_match1":outcome["AI endorsements 1"],
                "supporting_ai_votes_match2":outcome["AI endorsements 2"],
            })
        record={
            "mode":"v15_ai_supported_double_shadow",
            "model_version":VERSION,
            "status":"frozen_pre_kickoff",
            "sydney_date":packet["sydney_date"],
            "captured_at_utc":when.isoformat(),
            "source_original_pair_frozen_at_utc":packet["captured_at_utc"],
            "first":first,"second":second,
            "five_doubles":rendered,
            "baseline_five_doubles":packet["five_doubles"],
            "baseline_coverage":result["baseline_coverage"],
            "selected_statistical_coverage":result["selected_model_coverage"],
            "ai_provider_names":{
                "first":sorted(votes[0]["votes"]),
                "second":sorted(votes[1]["votes"])},
            "ai_source_capture_utc":{
                "first":votes[0]["captured_at_utc"],
                "second":votes[1]["captured_at_utc"]},
            "source_forecast_ids":{
                "first":sources[0]["forecast_id"],
                "second":sources[1]["forecast_id"]},
            "source_input_digests":{
                "first":sources[0]["input_digest"],
                "second":sources[1]["input_digest"]},
            "shared_ai_providers_across_both_legs":sorted(
                set(votes[0]["votes"]) & set(votes[1]["votes"])),
            "minimum_verified_providers_per_leg":min_votes,
            "ai_reweight_max_per_leg":0.12,
            "ai_votes_are_not_calibrated_probabilities":True,
            "same_input_briefing_makes_votes_correlated":True,
            "original_predictions_modified":False,
            "proven_accuracy_improvement":False,
            "independent_verification_status":"pending",
        }
        return record,"ready"
    except (KeyError,TypeError,ValueError,IndexError):
        return None,"invalid_source_records"


def _write_once(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open("x",encoding="utf-8") as dst:
            dst.write(json.dumps(record,indent=2,sort_keys=True)+"\n")
        return True
    except FileExistsError:
        return False


def freeze(*,now=None,root=ROOT,source_root=BASE_ROOT,ai_root=AI_ROOT,
           model_rows=None,min_votes=MIN_PROVIDERS_PER_LEG):
    when=utc(now or datetime.now(timezone.utc))
    source=read_latest(source_root) if model_rows is None else model_rows
    status={"policy":VERSION,"checked_at_utc":when.isoformat(),
            "existing_original_days":0,"frozen_new":0,
            "previously_frozen":0,"abstentions":{},
            "saved_days":[],"real_ai_requests_performed":0}
    for path in sorted((Path(source_root)/"doubles").glob("*.json")):
        original=load_json(path,{})
        if original.get("mode")!="prospective":
            continue
        status["existing_original_days"]+=1
        dest=Path(root)/"forecasts"/path.name
        if dest.exists():
            status["previously_frozen"]+=1
            continue
        record,reason=assemble_pair(original,source,ai_root=ai_root,
                                    now=when,min_votes=min_votes,
                                    source_root=source_root if model_rows is None else None)
        if record is None:
            status["abstentions"][path.stem]=reason
        elif _write_once(dest,record):
            status["frozen_new"]+=1
            status["saved_days"].append(path.stem)
    (Path(root)).mkdir(parents=True,exist_ok=True)
    (Path(root)/"status.json").write_text(
        json.dumps(status,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return status


def wilson_95(hits,n):
    """95% Wilson score interval; does not assume a claimed win rate."""
    if not n:
        return None
    z=1.959963984540054
    p=hits/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.0,center-width),min(1.0,center+width)]


def evaluate(*,root=ROOT,base_root=BASE_ROOT,results=None):
    if results is None:
        verified=merge_verified(load_json(RESULTS,{}),
                    load_json(Path(base_root)/"verified_results.json",{}))
    else:
        verified=results
    rows=[]
    for path in sorted((Path(root)/"forecasts").glob("*.json")):
        packet=load_json(path,{})
        try:
            start=utc(packet["captured_at_utc"])
            legs=[packet["first"],packet["second"]]
            if packet.get("mode")!="v15_ai_supported_double_shadow" or (
                start>=min(utc(leg["kickoff_utc"]) for leg in legs)):
                continue
            result=[_pair_verified(leg,verified) for leg in legs]
            if any(x is None for x in result):
                rows.append({"sydney_date":packet["sydney_date"],
                             "status":"awaiting_verified_results"})
                continue
            scores=[f'{x["home_goals"]}-{x["away_goals"]}' for x in result]
            hit=any(row["score1"]==scores[0] and row["score2"]==scores[1]
                    for row in packet["five_doubles"])
            original=any(row["score1"]==scores[0] and row["score2"]==scores[1]
                         for row in packet["baseline_five_doubles"])
            rows.append({"sydney_date":packet["sydney_date"],"status":"verified",
                         "hit":hit,"original_v8_hit_same_pair":original,
                         "actual_scores":scores})
        except (ValueError,KeyError,TypeError):
            continue
    completed=[x for x in rows if x["status"]=="verified"]
    n=len(completed)
    result={"version":VERSION,"frozen_pairs":len(rows),"verified_pairs":n,
            "pending_pairs":len(rows)-n,
            "ai_assisted_hits":sum(x["hit"] for x in completed),
            "original_v8_hits_same_pairs":sum(x["original_v8_hit_same_pair"]
                                              for x in completed),
            "verified_hit_rate":sum(x["hit"] for x in completed)/n if n else None,
            "v15_win_rate_wilson_95":wilson_95(sum(x["hit"] for x in completed),n),
            "v8_same_pair_win_rate_wilson_95":wilson_95(
                sum(x["original_v8_hit_same_pair"] for x in completed),n),
            "v15_only_wins":sum(x["hit"] and not x["original_v8_hit_same_pair"]
                                 for x in completed),
            "v8_only_wins":sum(x["original_v8_hit_same_pair"] and not x["hit"]
                                 for x in completed),
            "paired_comparison_is_pre_match_and_same_fixtures":True,
            "automatic_production_promotion":False,
            "data_limitation":"No performance claim before genuine two-match multi-AI forecasts are frozen and independently verified.",
            "rows":rows}
    Path(root).mkdir(parents=True,exist_ok=True)
    (Path(root)/"evaluation.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",default=str(BASE_ROOT))
    parser.add_argument("--root",default=str(ROOT))
    parser.add_argument("--ai-root",default=str(AI_ROOT))
    args=parser.parse_args()
    output=freeze(source_root=Path(args.source_root),root=Path(args.root),
                  ai_root=Path(args.ai_root))
    comparison=evaluate(root=Path(args.root),base_root=Path(args.source_root))
    print(json.dumps({"capture":output,"verification":{
        k:v for k,v in comparison.items() if k!="rows"}},indent=2))


if __name__=="__main__":
    main()
