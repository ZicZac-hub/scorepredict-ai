"""Independent, quota-capped historical Sportmonks genuine-xG source capture.

API-Football fixture IDs and Sportmonks fixture/team IDs are never conflated.
We match only a uniquely identified *future* fixture, then use Sportmonks team
IDs for historical matches from that same source. No new subscription is bought,
no production prediction is changed, and results are pre-match append-only.
"""
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from persist_prematch import load_json, write_json
from source_capture_clock import capture_clock
from v10_prospective import ROOT as ORIGINAL_ROOT, read_latest, utc
from sportmonks_probability_research import (
    API_ROOT, _get, _kickoff, _name, match_fixture,
)

ROOT = Path("data/sportmonks_xg")
DEFAULT_DAILY_REQUESTS = 4
MAX_DAILY_REQUESTS = 12
MAX_PER_RUN = 4
MAX_FORECASTS_PER_RUN = 1
MAX_HISTORY_DAYS = 60
EXPECTED_GOALS_TYPE = 5304  # Sportmonks EXPECTED_GOALS, not xGOT or xPTS.


class RequestBudgetExhausted(Exception):
    pass


def eligible_originals(rows, now):
    """Only genuinely frozen v8 forecasts 45 minutes to 48 hours ahead."""
    chosen = {}
    for row in rows.values() if isinstance(rows, dict) else rows:
        try:
            if (row.get("provider") != "API-Football" or
                row.get("model_key") != "v8" or
                not row.get("forecast_id") or not row.get("input_digest") or
                not row.get("home_team") or not row.get("away_team")):
                continue
            target = utc(row["kickoff_utc"])
            frozen = utc(row["captured_at_utc"])
            if not frozen <= now < target or not (
                    timedelta(minutes=45) < target-now <= timedelta(hours=48)):
                continue
            key = str(row["fixture_id"])
            prior = chosen.get(key)
            if prior is None or frozen < utc(prior["captured_at_utc"]):
                chosen[key] = row
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(chosen.values(), key=lambda r: (
        utc(r["kickoff_utc"]), str(r["fixture_id"])))


def _matched_future(forecast, rows):
    """Match a unique future fixture and its two independently sourced team IDs."""
    fid = match_fixture(forecast, rows)
    if fid is None:
        return None
    matches = [r for r in rows if str(r.get("id")) == str(fid)]
    if len(matches) != 1:
        return None
    item = matches[0]
    target = utc(forecast["kickoff_utc"])
    at = _kickoff(item)
    if at is None or abs((at-target).total_seconds()) > 300:
        return None
    teams = {}
    for team in item.get("participants") or []:
        side = (team.get("meta") or {}).get("location")
        if side not in ("home", "away"):
            continue
        try:
            tid = int(team["id"])
        except (KeyError, ValueError, TypeError):
            return None
        name = team.get("name")
        if not name or _name(name) != _name(forecast.get(side+"_team")):
            return None
        if side in teams and teams[side] != tid:
            return None
        teams[side] = tid
    if len(teams) != 2 or teams["home"] == teams["away"]:
        return None
    return {"fixture_id": int(fid), "home_team_id": teams["home"],
            "away_team_id": teams["away"]}


def _xg_pair(item):
    """Accept only unambiguous 5304 xG for both Sportmonks team IDs."""
    try:
        fid = str(int(item["id"]))
    except (KeyError, TypeError, ValueError):
        return None
    participants = {}
    for part in item.get("participants") or []:
        location = (part.get("meta") or {}).get("location")
        if location not in ("home", "away") or location in participants:
            return None
        try:
            participants[location] = str(int(part["id"]))
        except (KeyError, ValueError, TypeError):
            return None
    if len(participants) != 2 or participants["home"] == participants["away"]:
        return None
    entries = item.get("xgfixture") or item.get("expected") or []
    found = {}
    for entry in entries:
        if not isinstance(entry, dict) or str(entry.get("type_id")) != str(EXPECTED_GOALS_TYPE):
            continue
        if str(entry.get("fixture_id", fid)) != fid:
            return None
        side = entry.get("location")
        tid = str(entry.get("participant_id", ""))
        if side not in participants or participants[side] != tid or side in found:
            return None
        value = (entry.get("data") or {}).get("value")
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (ValueError, TypeError):
            return None
        if not math.isfinite(number) or number < 0 or number > 15:
            return None
        found[side] = number
    if set(found) != {"home", "away"}:
        return None
    return {"home_xg": found["home"], "away_xg": found["away"],
            "home_team_id": participants["home"],
            "away_team_id": participants["away"]}


def parse_history(items, team_id, as_of, target_kickoff):
    """Keep only pre-cutoff, completed 90-minute historical genuine xG pairs."""
    found = {}
    team_id = str(team_id)
    for item in items:
        if not isinstance(item, dict):
            continue
        at = _kickoff(item)
        if (at is None or at+timedelta(hours=3) > as_of or
                at >= target_kickoff):
            continue
        try:
            # Sportmonks state 5 = FT (regular time), exclude abandoned/AET.
            if int(item.get("state_id", (item.get("state") or {}).get("id", -1))) != 5:
                continue
        except (ValueError, TypeError):
            continue
        pair = _xg_pair(item)
        if pair is None or team_id not in (pair["home_team_id"],
                                          pair["away_team_id"]):
            continue
        source_fixture_id = str(item["id"])
        result = {
            "sportmonks_fixture_id": source_fixture_id,
            "kickoff_utc": at.isoformat(), "source": "Sportmonks",
            "type_id": EXPECTED_GOALS_TYPE,
            "home_team_id": pair["home_team_id"],
            "away_team_id": pair["away_team_id"],
            "home_xg": pair["home_xg"], "away_xg": pair["away_xg"],
            "available_before_target_kickoff": True,
        }
        previous = found.get(source_fixture_id)
        if previous is not None and previous != result:
            found[source_fixture_id] = None
        elif source_fixture_id not in found:
            found[source_fixture_id] = result
    valid = [r for r in found.values() if r is not None]
    return sorted(valid, key=lambda r:(r["kickoff_utc"],
                 r["sportmonks_fixture_id"]), reverse=True)[:6]


def capture(*, now=None, environ=None, root=ROOT, original_root=ORIGINAL_ROOT,
            request=None, clock=None, originals=None,
            daily_limit=None, max_per_run=MAX_PER_RUN):
    """One immutable multi-provider xG research packet, at most 4 requests/day.

    The bound is for Sportmonks calls. We do not make API-Football HTTP calls.
    """
    started, observed = capture_clock(now, clock)
    env = os.environ if environ is None else environ
    root = Path(root)
    token = env.get("THIRD_FOOTBALL_API_KEY") or env.get("SPORTMONKS_API_TOKEN")
    active = str(env.get("SPORTMONKS_XG_ENABLED", "")).lower() == "true"
    if daily_limit is None:
        try:
            daily_limit = int(env.get("SPORTMONKS_XG_DAILY_REQUESTS") or
                              DEFAULT_DAILY_REQUESTS)
        except (TypeError, ValueError):
            daily_limit = DEFAULT_DAILY_REQUESTS
    daily_limit = max(0, min(MAX_DAILY_REQUESTS, int(daily_limit)))
    day_path = root/"requests"/started.date().isoformat()
    used = len(list(day_path.glob("*.json")))
    status = {
        "at_utc": started.isoformat(), "enabled": active,
        "token_present": bool(token), "provider": "Sportmonks",
        "scope": "independent_historical_xg_shadow_only",
        "daily_request_limit": daily_limit, "daily_requests_used": used,
        "requests_this_run": 0, "eligible_frozen_forecasts": 0,
        "matched_future_fixtures": 0, "matched_historical_xg_fixtures": 0,
        "captured": 0, "reason": "", "subscription_confirmed": False,
        "no_production_model_changes": True,
    }

    def finish(reason):
        status["reason"] = reason
        status["daily_requests_used"] = used
        write_json(root/"status.json", status)
        return status

    if not active:
        return finish("disabled")
    if not token:
        return finish("sportmonks_token_missing")
    candidates = eligible_originals(originals if originals is not None
                                    else read_latest(original_root), started)
    status["eligible_frozen_forecasts"] = len(candidates)
    if not candidates:
        return finish("no_frozen_forecasts_within_48_hours")
    selected = None
    for item in candidates:
        path = root/"forecasts"/(
            "API-Football_"+str(item["fixture_id"])+"_"+str(item["forecast_id"])+".json")
        attempt = root/"attempts"/(
            "API-Football_"+str(item["fixture_id"])+"_"+str(item["forecast_id"])+".json")
        if not path.exists() and not attempt.exists():
            selected = (item, path, attempt)
            break
    if selected is None:
        return finish("eligible_forecasts_already_checked")
    if used >= daily_limit or used >= MAX_DAILY_REQUESTS:
        return finish("daily_request_budget_exhausted")
    forecast, destination, attempt_path = selected
    per_run = min(MAX_PER_RUN, max(0, int(max_per_run)))

    def call(url, params):
        nonlocal used
        if used >= daily_limit or status["requests_this_run"] >= per_run:
            raise RequestBudgetExhausted
        # A call reservation survives failed requests and scheduled reruns.
        day_path.mkdir(parents=True, exist_ok=True)
        marker = day_path/(f"{used:03d}_sportmonks.json")
        with marker.open("x", encoding="utf-8") as output:
            output.write(json.dumps({"reserved_at_utc":observed().isoformat(),
                                     "endpoint":url.replace(API_ROOT, "")})+"\n")
        used += 1
        status["requests_this_run"] += 1
        return _get(token, url, params=params, request=request)

    # One attempted forecast per day unless a previous one is already marked.
    # Checkpoint before provider requests so subscription errors don't retry
    # every fifteen minutes on an account without the xG add-on.
    write_json(attempt_path, {
        "forecast_id":forecast["forecast_id"],
        "api_football_fixture_id":str(forecast["fixture_id"]),
        "attempted_at_utc":started.isoformat()})
    try:
        target_day=utc(forecast["kickoff_utc"]).date().isoformat()
        fixtures=[]
        match=None
        # Search the exact UTC match date, not all worldwide matches over
        # four days; pagination avoids silently losing fixtures after page 1.
        for page in (1,2):
            response=call(f"{API_ROOT}/fixtures/date/{target_day}",
                          {"include":"participants","per_page":50,"page":page})
            observed_items=response.get("data") or []
            if not isinstance(observed_items,list):
                return finish("invalid_discovery_response")
            fixtures.extend(observed_items)
            match=_matched_future(forecast, fixtures)
            if match is not None:
                break
            if not (response.get("pagination") or {}).get("has_more"):
                break
        if match is None:
            return finish("no_unique_cross_provider_fixture_match_within_page_budget")
        status["matched_future_fixtures"] = 1
        cutoff = observed()
        if cutoff >= utc(forecast["kickoff_utc"]):
            return finish("past_kickoff_not_valid_for_research")
        since=(cutoff-timedelta(days=MAX_HISTORY_DAYS)).date().isoformat()
        until=(cutoff-timedelta(days=1)).date().isoformat()
        evidence = {}
        for side in ("home", "away"):
            tid = match[side+"_team_id"]
            payload=call(f"{API_ROOT}/fixtures/between/{since}/{until}/{tid}",
                         {"include":"participants;state;xGFixture",
                          "per_page":50,"page":1,"order":"desc"})
            items=payload.get("data") or []
            if not isinstance(items,list):
                return finish("invalid_historical_response")
            evidence[side]=parse_history(items,tid,observed(),
                                         utc(forecast["kickoff_utc"]))
        finished=observed()
        if finished >= utc(forecast["kickoff_utc"]):
            return finish("collection_finished_after_kickoff")
        merged={r["sportmonks_fixture_id"]:r
                for side in ("home","away") for r in evidence[side]}
        status["matched_historical_xg_fixtures"] = len(merged)
        if not merged:
            return finish("genuine_xg_unavailable_subscription_or_coverage")
        status["subscription_confirmed"] = True  # Valid xG, not a plan entitlement query.
        record={
            "schema": 1, "provider": "Sportmonks", "research_only": True,
            "forecast_provider": "API-Football",
            "source_forecast_id":forecast["forecast_id"],
            "source_input_digest":forecast["input_digest"],
            "api_football_fixture_id":str(forecast["fixture_id"]),
            "sportmonks_fixture_id":str(match["fixture_id"]),
            "sportmonks_home_team_id":str(match["home_team_id"]),
            "sportmonks_away_team_id":str(match["away_team_id"]),
            "home_team":forecast["home_team"],
            "away_team":forecast["away_team"],
            "kickoff_utc":forecast["kickoff_utc"],
            "captured_at_utc":finished.isoformat(),
            "xg_type_id":EXPECTED_GOALS_TYPE,
            "historical_xg":list(merged.values()),
            "historical_xg_home_count":len(evidence["home"]),
            "historical_xg_away_count":len(evidence["away"]),
            "both_teams_six_xg":min(len(evidence["home"]),
                                   len(evidence["away"])) >= 6,
            "source_identity_policy":"exact_normalized_teams_and_kickoff_within_5min",
            "historical_source":"Sportmonks_fixture_5304_both_participants",
            "not_api_football_xg":True,
            "not_automatically_promoted":True,
        }
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open("x",encoding="utf-8") as output:
            output.write(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n")
        status["captured"]=1
        return finish("genuine_sportmonks_xg_captured")
    except RequestBudgetExhausted:
        return finish("request_budget_exhausted_mid_research")
    except requests.RequestException as exc:
        status["error_type"]=type(exc).__name__
        return finish("provider_access_subscription_or_network_error")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        status["error_type"]=type(exc).__name__
        return finish("source_error")
    except Exception as exc:
        # Missing xG add-on, subscription, rate limit and network failures
        # remain observable but must never reveal auth tokens/response bodies.
        status["error_type"]=type(exc).__name__
        return finish("provider_access_subscription_or_network_error")


if __name__ == "__main__":
    print(json.dumps(capture(),indent=2))
