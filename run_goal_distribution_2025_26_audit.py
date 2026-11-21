"""Independent later-season calendar/offset replay, no model promotion.

2025-26 English PL results come from pinned CC0 OpenFootball sources.
The regular and detailed releases must agree on all 380 event identities
and regulation scores; 27 compressed 0-0 score arrays are corroborated
with explicit full-time scores. No future results enter model features.
2025-26 is retrospective and not a preregistered prospective benchmark.
"""
import hashlib
import json
import re
from datetime import datetime,timezone,timedelta
from urllib.request import urlopen

from backtest_dynamic_dixon_coles import evaluate
from run_dynamic_dc_pinned_replay import load_matches
from run_goal_distribution_study import compare_phase,select_model, METHODS

PIN="e6744429ee395bc86f247348c6184bb08d4eb361"
BLOBS={
    "2025-26/en.1.json":"41e223ee5954e6b634beb4bc70d7188b3183c665",
    "2025-26/en.1-full.json":"0f7c699cba701a9714be4e88089c4024fdcfd701",
}
EXPECTED_SEASON_EVENTS=380
EXPECTED_COMPRESSED_SCORE_ROWS=27
EXPECTED_PRIOR_SELECTION="engine_baseline"

def normalize_team(name):
    """Identity aliases are limited to source-observed club prefixes/suffixes."""
    return re.sub(r"^afc\s+|\s+(?:fc|afc)$","",
                  " ".join(str(name).strip().casefold().split())).strip()

def key(row):
    return (row["date"],normalize_team(row["team1"]),normalize_team(row["team2"]))

def parse_source_utc(date,source_time):
    m=re.fullmatch(r"(\d{2}:\d{2}) UTC([+-]\d{1,2})",source_time)
    if not m:
        raise ValueError("missing_or_unsupported_source_utc_offset")
    dt=datetime.strptime(date+" "+m.group(1),"%Y-%m-%d %H:%M")
    offset=int(m.group(2))
    if not -12<=offset<=14:
        raise ValueError("invalid_utc_offset")
    return dt.replace(tzinfo=timezone(timedelta(hours=offset))).astimezone(timezone.utc).isoformat()

def verify_blob(data,sha):
    actual=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
    if actual!=sha:
        raise ValueError("pinned_git_blob_integrity_failure")

def fetch_pinned(path):
    url=f"https://raw.githubusercontent.com/openfootball/football.json/{PIN}/{path}"
    with urlopen(url,timeout=30) as response:
        data=response.read()
    verify_blob(data,BLOBS[path])
    return json.loads(data)["matches"]

def join_season(compact,detailed):
    if len(compact)!=EXPECTED_SEASON_EVENTS or len(detailed)!=EXPECTED_SEASON_EVENTS:
        raise ValueError("incomplete_source_season")
    lookup={}
    for item in detailed:
        k=key(item)
        if k in lookup:
            raise ValueError("ambiguous_detailed_event_identity")
        lookup[k]=item
    observed=set()
    compressed=0
    matches=[]
    for i,item in enumerate(compact):
        k=key(item)
        if k in observed or k not in lookup:
            raise ValueError("missing_or_duplicate_source_fixture_identity")
        observed.add(k)
        full=lookup[k]
        compact_score=item.get("score")
        if isinstance(compact_score,list):
            compressed+=1
        else:
            compact_score=compact_score.get("ft") if isinstance(compact_score,dict) else None
        full_score=(full.get("score") or {}).get("ft")
        valid=lambda s:(type(s) is list and len(s)==2
                         and all(type(x) is int and 0<=x<=20 for x in s))
        if not valid(compact_score) or not valid(full_score) or compact_score!=full_score:
            raise ValueError("source_full_time_disagreement")
        matches.append({
            "id":f"2025-26:{i}","provider":"openfootball-calendar-replay",
            "competition":{"id":"en.1"},
            "utcDate":parse_source_utc(full["date"],full["time"]),
            "status":"FINISHED",
            "homeTeam":{"id":k[1]},"awayTeam":{"id":k[2]},
            "score":{"fullTime":{"home":full_score[0],"away":full_score[1]}}
        })
    if len(observed)!=len(lookup) or compressed!=EXPECTED_COMPRESSED_SCORE_ROWS:
        raise ValueError("source_schema_or_fixture_completeness_changed")
    if len(set(m["id"] for m in matches))!=len(matches):
        raise ValueError("duplicate_match_id")
    return matches,compressed

def main():
    history,missing=load_matches()
    history=[m for m in history if m["utcDate"][:10] not in missing]
    # Normalize historic club spellings so promoted teams match by identity.
    for m in history:
        for field in ("homeTeam","awayTeam"):
            m[field]["id"]=normalize_team(m[field]["id"])
    season,compressed=join_season(
        fetch_pinned("2025-26/en.1.json"),
        fetch_pinned("2025-26/en.1-full.json"))
    combined=history+season
    reference=evaluate(combined,start="2023-08-01")
    earlier=compare_phase(combined,reference,"2023-08-01","2024-08-01")
    selection=select_model(earlier)
    if selection!=EXPECTED_PRIOR_SELECTION:
        raise ValueError("predeclared_challenger_selection_changed")
    later=compare_phase(combined,reference,"2025-08-01","2026-08-01")
    print(json.dumps({
        "source_commit":PIN,"source_git_blob_shas":BLOBS,
        "complete_2025_26_events":len(season),
        "cross_verified_draw_score_arrays":compressed,
        "source_kickoff_offsets_used_without_independent_attestation":True,
        "prior_selection_source":"2023-24_only",
        "selected_method":selection,
        "new_cohort":"2025-26_English_Premier_League",
        "models_compared":METHODS,
        "post_season_retrospective_not_prospective":True,
        "source_hash_and_identity_verified":True,
        "model_promotion":False,
        "results":later},indent=2))

if __name__=="__main__":
    main()
