"""V23 optional two-window pre-match bookmaker capture (at most two requests/leg).

No silent API spending: requires V23_EARLY_ODDS_ENABLED=true plus API key.
Keeps the original V8 forecasts and all later provider packets immutable.
Records are only written when genuinely time-checked 1X2 + totals odds exist.
"""
import hashlib
import json
import os
from datetime import datetime,timedelta,timezone
from pathlib import Path

from v10_prospective import ROOT as V8_ROOT,utc
from v16_pair_research import original_model
from v21_capture_sources import source_packet_path
from v21_provider_evidence import valid_snapshot
from v22_evidence_models import parse_markets
from persist_prematch import load_json,write_json

ROOT=Path("data/v21_sources")
START_HOURS=6
END_HOURS=36
MAX_REQUESTS_PER_RUN=2

def _window(hours_until_kickoff):
    """Provider odds often exist earlier than the old 6–10h-only window.

    A later retry is allowed only when the earlier window had no still-fresh
    qualified data. Each stage is attempted at most once per fixture.
    """
    if 18 <= hours_until_kickoff <= 36:
        return "early"
    if 6 <= hours_until_kickoff <= 12:
        return "late"
    return None


def eligible_legs(root, now):
    clock=utc(now)
    selected={}
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        pair=load_json(path,{})
        try:
            if pair.get("status")!="frozen_pre_kickoff" or utc(pair["captured_at_utc"])>clock:
                continue
            for leg in (pair["first"],pair["second"]):
                kickoff=utc(leg["kickoff_utc"])
                hours=(kickoff-clock).total_seconds()/3600
                if _window(hours) is None:
                    continue
                model=original_model(root,leg)
                if not model or model.get("input_digest")!=leg.get("input_digest"):
                    continue
                key=(str(model["provider"]),str(model["fixture_id"]))
                selected.setdefault(key,(kickoff,model))
        except (TypeError,ValueError,KeyError):
            continue
    # Do not repeatedly reselect only the earliest two after their attempts
    # have been spent; the capture loop itself caps ACTUAL provider calls.
    return [x[1] for x in sorted(selected.values(),key=lambda t:(t[0],str(t[1]["fixture_id"])))]


def capture(*,now=None,enabled=None,api_key=None,source_root=V8_ROOT,
            root=ROOT,request=None):
    from api_football import _request
    clock=utc(now or datetime.now(timezone.utc))
    active=(os.environ.get("V23_EARLY_ODDS_ENABLED")=="true"
            if enabled is None else bool(enabled))
    key=api_key if api_key is not None else os.environ.get("API_FOOTBALL_KEY","")
    status={"at_utc":clock.isoformat(),"enabled":active,
            "queries":0,"new_packets":[],"skipped":[]}
    if not active or not key:
        status["reason"]="not_opted_in_or_api_key_missing"
        return status
    cb=request or _request
    from v21_capture_sources import best_packet
    for leg in eligible_legs(source_root,clock):
        if status["queries"] >= MAX_REQUESTS_PER_RUN:
            break
        try:
            canonical=source_packet_path(root,leg)
            hours=(utc(leg["kickoff_utc"])-clock).total_seconds()/3600
            phase=_window(hours)
            if phase is None:
                continue
            marker=Path(root)/"early_attempts"/(canonical.stem+"_"+phase+".json")
            legacy_marker=Path(root)/"early_attempts"/(canonical.stem+".json")
            if marker.exists() or (phase=="early" and legacy_marker.exists()):
                status["skipped"].append(str(leg["fixture_id"])+":"+phase+"_already_attempted")
                continue
            # A separate frozen packet may already have fully qualifying
            # 1X2 + goal-total prices; do not spend another API request.
            saved=best_packet(root,leg,now=clock,priority="market")
            saved_markets=((saved or {}).get("provider_snapshot") or {}).get("multi_market") or {}
            if ("1x2" in saved_markets and
                    ("over_2_5" in saved_markets or "over_3_5" in saved_markets)):
                current=True
                for name in ("1x2","over_2_5","over_3_5"):
                    if name not in saved_markets:
                        continue
                    try:
                        updated=utc(saved_markets[name]["last_source_update_utc"])
                        if updated>clock or clock-updated>timedelta(hours=12):
                            current=False
                    except (KeyError,TypeError,ValueError):
                        current=False
                if current:
                    status["skipped"].append(str(leg["fixture_id"])+":current_market_packet_already_saved")
                    continue
            marker.parent.mkdir(parents=True,exist_ok=True)
            try:
                with marker.open("x",encoding="utf-8") as file:
                    file.write(json.dumps({"attempted_at_utc":clock.isoformat(),
                                           "fixture_id":leg["fixture_id"],
                                           "policy":"one_request_each_early_and_late_window",
                                           "stage":phase})+"\n")
            except FileExistsError:
                status["skipped"].append(str(leg["fixture_id"])+":concurrent_attempt")
                continue
            status["queries"]+=1
            # Bookmaker prices are useful earlier than confirmed starting
            # lineups; no expensive historical stats or AI calls here.
            if request is None:
                response=cb(key,"/odds",{"fixture":int(leg["fixture_id"])},
                            with_metadata=True)
                meta=response.get("metadata") or {}
                if meta.get("stale") or not meta.get("fetched_at_utc"):
                    status["skipped"].append(str(leg["fixture_id"])+":stale_odds_response")
                    continue
                fetched_at=meta["fetched_at_utc"]
                raw=response.get("data")
            else:
                # Injected requests must supply bookmaker update times;
                # never infer provenance from an unaudited callback.
                raw=cb(key,"/odds",{"fixture":int(leg["fixture_id"])})
                fetched_at=None
            markets=parse_markets(raw,observed_at=clock,
                                 verified_provider_fetched_at=fetched_at)
            if "1x2" not in markets or not ({"over_2_5","over_3_5"} & set(markets)):
                status["skipped"].append(str(leg["fixture_id"])+":complete_odds_unavailable")
                continue
            packet={"schema":1,"status":"captured_pre_kickoff",
                    "provider":leg["provider"],"fixture_id":str(leg["fixture_id"]),
                    "home_team_id":str(leg["home_team_id"]),
                    "away_team_id":str(leg["away_team_id"]),
                    "kickoff_utc":leg["kickoff_utc"],"captured_at_utc":clock.isoformat(),
                    "market":None,"multi_market":markets,
                    "verified_historical_fixture_xg":[],
                    "confirmed_starting_xi":[],"reported_injuries":[],
                    "available_sources":{
                        "timestamped_multi_market":True,
                        "lineups":False,"historical_xg":False},
                    "source_limits":{"calls_for_this_packet":1,
                                     "provider_fetched_at_utc":fetched_at,
                                     "not_a_bookmaker_update_timestamp":True}}
            packet["snapshot_sha256"]=hashlib.sha256(
                json.dumps(packet,sort_keys=True,separators=(",",":"),
                           allow_nan=False).encode()).hexdigest()
            fixture={"provider":leg["provider"],"id":leg["fixture_id"],
                     "homeTeam":{"id":leg["home_team_id"]},
                     "awayTeam":{"id":leg["away_team_id"]},
                     "utcDate":leg["kickoff_utc"]}
            if not valid_snapshot(packet,fixture,now=clock):
                status["skipped"].append(str(leg["fixture_id"])+":identity_validation_failed")
                continue
            record={"schema":1,
                    "source_forecast":{k:leg[k] for k in (
                        "provider","fixture_id","forecast_id","input_digest",
                        "home_team_id","away_team_id","kickoff_utc")},
                    "provider_snapshot":packet,
                    "supplemental_history_digest":None,
                    "not_baseline_forecast_digest":True,
                    "source_scope":"one_checked_pre_match_API_Football_odds_response"}
            path=Path(root)/"versions"/(canonical.stem+"_"+
                 packet["snapshot_sha256"][:16]+".json")
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                with path.open("x",encoding="utf-8") as file:
                    file.write(json.dumps(record,sort_keys=True,indent=2,
                                          allow_nan=False)+"\n")
                status["new_packets"].append(str(leg["fixture_id"]))
            except FileExistsError:
                status["skipped"].append(str(leg["fixture_id"])+":identical_snapshot_preserved")
        except (KeyError,TypeError,ValueError,RuntimeError,OSError) as exc:
            status["skipped"].append(str(leg.get("fixture_id"))+":"+type(exc).__name__)
    write_json(Path(root)/"market_early_status.json",status)
    return status


if __name__=="__main__":
    print(json.dumps(capture(),indent=2))
