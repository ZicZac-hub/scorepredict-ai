"""Strict read-only audit of genuine V15 AI evidence for two-match doubles.

Never calls paid APIs, backfills forecasts, or changes a frozen original.
A vote counts for a frozen pair only if it matches the *exact* source
forecast ID, input digest, teams, provider, kickoff and capture ordering.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from persist_prematch import load_json
from v10_prospective import ROOT as MATCH_ROOT, utc
from llm_research_panel import ROOT as AI_ROOT
from research_double_v15 import valid_saved_votes

SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


def _safe_path(root, leg):
    parts=[str(leg.get(key) or "") for key in
           ("provider", "fixture_id", "forecast_id")]
    if any(not SAFE_PART.fullmatch(value) or value in (".", "..")
           for value in parts):
        return None
    return Path(root)/"forecasts"/parts[0]/parts[1]/(parts[2]+".json")


def _source(root, leg):
    path=_safe_path(root,leg)
    if path is None:
        return None
    source=load_json(path,{})
    try:
        if (not isinstance(source,dict)
            or str(source["provider"])!=str(leg["provider"])
            or str(source["fixture_id"])!=str(leg["fixture_id"])
            or str(source["forecast_id"])!=str(leg["forecast_id"])
            or str(source["home_team_id"])!=str(leg["home_team_id"])
            or str(source["away_team_id"])!=str(leg["away_team_id"])
            or utc(source["kickoff_utc"])!=utc(leg["kickoff_utc"])
            or not source.get("input_digest")
            or utc(source["captured_at_utc"])>=utc(leg["kickoff_utc"])):
            return None
        return source
    except (KeyError,TypeError,ValueError):
        return None


def _leg_audit(leg, records, root):
    """The first time-eligible complete panel wins, never cherry-picked."""
    source=_source(root,leg)
    if source is None:
        return 0, [], "original_source_forecast_or_digest_unavailable"
    eligible=[]
    for row in records:
        try:
            if (str(row["provider"])!=str(leg["provider"])
                or str(row["fixture_id"])!=str(leg["fixture_id"])
                or str(row["home_team_id"])!=str(leg["home_team_id"])
                or str(row["away_team_id"])!=str(leg["away_team_id"])
                or row.get("source_forecast_id")!=source["forecast_id"]
                or row.get("input_digest")!=source["input_digest"]
                or utc(row["kickoff_utc"])!=utc(leg["kickoff_utc"])
                or not utc(source["captured_at_utc"])<=utc(row["captured_at_utc"])<utc(leg["kickoff_utc"])):
                continue
            eligible.append(row)
        except (KeyError,TypeError,ValueError):
            continue
    if not eligible:
        return 0, [], "no_source_matched_pre_kickoff_ai_panel"
    eligible.sort(key=lambda row:(utc(row["captured_at_utc"]),row["_path"]))
    row=next((x for x in eligible if len(x["_valid_votes"])>=2),eligible[0])
    names=sorted(row["_valid_votes"])
    return len(names),names,("ready" if len(names)>=2 else
                             "fewer_than_two_real_ai_providers")


def audit(base_root=MATCH_ROOT, ai_root=AI_ROOT):
    base_root=Path(base_root);ai_root=Path(ai_root)
    status=load_json(ai_root/"status.json",{})
    evaluation=load_json(ai_root/"evaluation.json",{})
    performance=load_json(base_root/"performance.json",{})
    records=[]
    unique_vote_fixtures=set()
    for filepath in sorted((ai_root/"forecasts").glob("*.json")):
        row=load_json(filepath,{})
        if not isinstance(row,dict) or not row.get("research_only"):
            continue
        try:
            if (utc(row["captured_at_utc"])>=utc(row["kickoff_utc"])
                or not row.get("source_forecast_id")
                or not row.get("input_digest")
                or not row.get("home_team_id")
                or not row.get("away_team_id")):
                continue
            votes=valid_saved_votes(row.get("votes"))
            if not votes:
                continue
            entry={**row,"_valid_votes":votes,"_path":str(filepath)}
            records.append(entry)
            for name in votes:
                unique_vote_fixtures.add((str(row["provider"]),
                    str(row["fixture_id"]),str(row["source_forecast_id"]),name))
        except (TypeError,KeyError,ValueError):
            continue
    provider_counts=Counter(name for *_,name in unique_vote_fixtures)
    sets=[];complete=0;partial=0
    reasons=Counter()
    for filepath in sorted((base_root/"doubles").glob("*.json")):
        row=load_json(filepath,{})
        if not isinstance(row,dict) or row.get("mode")!="prospective" or row.get("status")!="frozen_pre_kickoff":
            continue
        try:
            legs=[row["first"],row["second"]]
            frozen=utc(row["captured_at_utc"])
            if frozen>=min(utc(leg["kickoff_utc"]) for leg in legs):
                continue
            left=_leg_audit(legs[0],records,base_root)
            right=_leg_audit(legs[1],records,base_root)
            counts=[left[0],right[0]]
            valid=all(count>=2 for count in counts)
            complete+=int(valid)
            partial+=int(any(count>0 for count in counts))
            reasons.update([left[2],right[2]])
            sets.append({
                "sydney_day":row.get("sydney_date"),
                "actual_provider_votes_per_leg":counts,
                "actual_provider_names_per_leg":[left[1],right[1]],
                "shared_provider_names":sorted(set(left[1])&set(right[1])),
                "leg_reasons":[left[2],right[2]],
                "both_legs_have_at_least_two_providers":valid,
                "exact_frozen_source_and_digest_validated":True})
        except (TypeError,KeyError,ValueError):
            continue
    doubles=performance.get("five_doubles") or {}
    return {
        "kind":"read_only_v16_source_verified_ai_double_audit",
        "configuration_is_not_successful_votes":True,
        "configured_provider_names":status.get("configured_providers") or [],
        "configured_provider_count":len(status.get("configured_providers") or []),
        "latest_scheduled_attempted":status.get("attempted"),
        "latest_scheduled_captured":status.get("captured"),
        "saved_valid_individual_ai_fixture_records":len(records),
        "actual_successful_provider_counts_by_name":dict(provider_counts),
        "unique_providers_with_valid_saved_votes":len(provider_counts),
        "frozen_original_double_sets_checked":len(sets),
        "original_sets_with_any_individual_ai_evidence":partial,
        "original_sets_with_at_least_two_ai_votes_on_both_legs":complete,
        "source_match_abstention_counts":dict(reasons),
        "complete_five_ai_panels_with_verified_results":
            evaluation.get("complete_panels_with_verified_results",0),
        "complete_ten_ai_panels_with_verified_results":
            evaluation.get("ten_provider_complete_verified",0),
        "actually_verified_original_five_doubles":doubles.get("verified_pairs",0),
        "actually_verified_original_five_double_wins":doubles.get("wins",0),
        "original_five_doubles_pending":doubles.get("awaiting_both_results",0),
        "v15_real_ai_double_accuracy":None,
        "v15_approval_for_production":False,
        "explanation":"Counts demand matching original forecast IDs, source digests, timestamps and both-leg valid votes. The independent V15 settlement report supplies future real win rates.",
        "source_rows":sets,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root",default=str(MATCH_ROOT))
    parser.add_argument("--ai-root",default=str(AI_ROOT))
    parser.add_argument("--output",default="work/v15_real_source_audit.json")
    args=parser.parse_args()
    result=audit(Path(args.base_root),Path(args.ai_root))
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("===== STRICT PRE-MATCH FIVE-DOUBLE SOURCE AUDIT =====")
    print(json.dumps({k:v for k,v in result.items() if k!="source_rows"},indent=2))


if __name__=="__main__":
    main()
