"""Independent, quota-capped historical Sportmonks genuine-xG source capture.

API-Football fixture IDs and Sportmonks fixture/team IDs are never conflated.
We match only a uniquely identified *future* fixture, then use Sportmonks team
IDs for historical matches from that same source. No new subscription is bought,
no production prediction is changed, and results are pre-match append-only.
"""
import json
import math
import os
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests

from persist_prematch import load_json, write_json
from source_capture_clock import capture_clock
from v10_prospective import ROOT as ORIGINAL_ROOT, read_latest, utc
from sportmonks_probability_research import (
    API_ROOT, _get, _kickoff, _name,
)

ROOT = Path("data/sportmonks_xg")
DEFAULT_DAILY_REQUESTS = 4
MAX_DAILY_REQUESTS = 12
MAX_PER_RUN = 4
MAX_FORECASTS_PER_RUN = 1
MAX_HISTORY_DAYS = 60
MAX_DISCOVERY_PAGES = 8  # Hard stop on pagination; incomplete scans never certify a match.
DISCOVERY_CACHE_HOURS = 36
MAX_ATTEMPTS_PER_FORECAST = 3
RETRY_MINUTES = 45
EXPECTED_GOALS_TYPE = 5304  # Sportmonks EXPECTED_GOALS, not xGOT or xPTS.


class RequestBudgetExhausted(Exception):
    pass


def _http_status_category(exc):
    """Redacted HTTP error category, never persist response text or headers."""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if type(code) is not int:
        return "not_reported"
    if code in (401, 403):
        return "authentication_or_permission"
    if code == 429:
        return "rate_limited"
    if 400 <= code < 500:
        return "other_client_error"
    if 500 <= code < 600:
        return "provider_server_error"
    return "other_http_status"


def _safe_provider_observation(status):
    """Keep only allowlisted, non-sensitive descriptors after a real request."""
    schema = status.get("fixture_discovery_response_schema")
    schema = schema if isinstance(schema, dict) else {}
    allowed_kinds = ("object", "list", "dict", "str", "int", "bool",
                     "NoneType", "float", "tuple")
    allowed_categories = (
        "absent", "verified_empty_message", "provider_access_or_subscription",
        "provider_quota_or_rate_limit", "provider_invalid_request",
        "unclassified_provider_message")
    classification = schema.get("message_class")
    return {
        "at_utc": status["at_utc"],
        "reason": status["reason"],
        "requests_this_run": status["requests_this_run"],
        "response_shape": {
            "data_kind": schema.get("data_kind") if schema.get("data_kind") in allowed_kinds else "unknown",
            "row_count": schema.get("row_count") if type(schema.get("row_count")) is int else None,
            "pagination_kind": schema.get("pagination_kind") if schema.get("pagination_kind") in allowed_kinds else "unknown",
            "message_class": classification if classification in allowed_categories else "not_recorded",
        },
        "http_status_category": status.get("http_status_category", "not_reported"),
        "matched_future_fixtures": status["matched_future_fixtures"],
        "matched_historical_xg_fixtures": status["matched_historical_xg_fixtures"],
        "captured": status["captured"],
    }


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


# Small, reviewed *whole-name* alias groups; never delete junior,
# women's, reserve or geographic qualifiers or use fuzzy edit distances.
_ALIAS_GROUPS = (
    ("coritiba", "coritiba fc", "coritiba foot ball club"),
    ("santos", "santos fc", "santos futebol clube"),
    ("corinthians", "corinthians paulista", "sc corinthians paulista"),
    ("cruzeiro", "cruzeiro ec", "cruzeiro esporte clube"),
    ("sao paulo", "sao paulo fc", "sao paulo futebol clube"),
    ("manchester city", "manchester city fc", "man city"),
    ("athletic club", "athletic club bilbao", "athletic bilbao"),
    ("palmeiras", "sociedade esportiva palmeiras", "se palmeiras"),
    ("fluminense", "fluminense fc", "fluminense football club"),
    ("vasco da gama", "vasco da gama rj", "cr vasco da gama"),
    ("botafogo", "botafogo fr", "botafogo de futebol e regatas"),
)
_ALIASES = { _name(variant): _name(group[0])
             for group in _ALIAS_GROUPS for variant in group }


def _team_key(value):
    """Case/accent-insensitive *exact* names or expressly reviewed aliases."""
    name = _name(value)
    return _ALIASES.get(name, name)


def _fixture_candidates(forecast, rows):
    target = utc(forecast["kickoff_utc"])
    matches = {}
    conflicts = set()
    for row in rows:
        if not isinstance(row,dict):
            continue
        at = _kickoff(row)
        if at is None or abs((at-target).total_seconds()) > 300:
            continue
        expected = {side: _team_key(forecast.get(side+"_team"))
                    for side in ("home","away")}
        if not all(expected.values()):
            continue
        sides = {}
        for part in row.get("participants") or []:
            if not isinstance(part,dict):
                continue
            side = (part.get("meta") or {}).get("location")
            if side not in ("home","away"):
                continue
            name = _team_key(part.get("name"))
            try:
                tid = int(part["id"])
            except (KeyError,ValueError,TypeError):
                continue
            if side in sides and sides[side] != (name,tid):
                sides[side] = None
            elif side not in sides:
                sides[side] = (name,tid)
        if any(sides.get(side) is None or
               sides[side][0] != expected[side] for side in ("home","away")):
            continue
        if sides["home"][1] == sides["away"][1]:
            continue
        try:
            fid = int(row["id"])
        except (KeyError,TypeError,ValueError):
            continue
        candidate={"fixture_id":fid,
                   "home_team_id":sides["home"][1],
                   "away_team_id":sides["away"][1]}
        if fid in matches and matches[fid]!=candidate:
            conflicts.add(fid)
        else:
            matches[fid]=candidate
    return [row for fid,row in matches.items() if fid not in conflicts],bool(conflicts)


def _matched_future(forecast, rows):
    """Only one unambiguous source fixture, preserving Sportmonks team IDs."""
    matches,conflict=_fixture_candidates(forecast,rows)
    return matches[0] if len(matches)==1 and not conflict else None


def _safe_message_class(message):
    """Only a fixed category leaves this function; never retain provider text."""
    if not isinstance(message, str) or not message.strip():
        return "absent"
    msg = " ".join(message.casefold().split())
    # Order matters: "no data ... subscription denied" is NOT a valid empty page.
    if any(fragment in msg for fragment in (
        "unauthorized", "unauthorised", "not authorized", "not authorised",
        "forbidden", "access denied", "not allowed", "subscription",
        "not subscribed", "your plan", "not included in your plan",
        "missing permission", "invalid token", "invalid api key",
        "authentication failed", "auth failed", "entitlement",
    )):
        return "provider_access_or_subscription"
    if any(fragment in msg for fragment in (
        "rate limit", "too many requests", "quota exceeded",
        "request limit", "credits exhausted",
    )):
        return "provider_quota_or_rate_limit"
    if any(fragment in msg for fragment in (
        "invalid filter", "unsupported filter", "invalid parameter",
        "invalid date", "malformed request", "validation failed",
    )):
        return "provider_invalid_request"
    if any(fragment in msg for fragment in (
        "no results", "no fixtures", "no data found", "no data available",
        "no matches found", "no records found", "nothing found",
    )):
        return "verified_empty_message"
    return "unclassified_provider_message"


def _schema_diagnostic(body):
    """Allowlisted shape descriptors ONLY: no names, credentials, URLs or cursors."""
    if not isinstance(body, dict):
        return {"body_kind":type(body).__name__,"reason":"non_object_response"}
    rows=body.get("data")
    pagination=body.get("pagination")
    pag=pagination if isinstance(pagination,dict) else {}
    return {
        "body_kind":"object",
        "data_kind":type(rows).__name__,
        "row_count":len(rows) if isinstance(rows,list) else None,
        "pagination_kind":type(pagination).__name__,
        "has_more_kind":type(pag.get("has_more")).__name__,
        "next_cursor_present":bool(isinstance(pag.get("next_cursor"),str)
                                   and pag["next_cursor"]),
        "next_page_present":bool(pag.get("next_page")),
        "message_present":isinstance(body.get("message"),str),
        "message_class":_safe_message_class(body.get("message")),
        "error_present":bool(body.get("error") or body.get("errors")),
    }


def _next_page_number(raw, current):
    """Parse only a number from a Sportmonks next_page URL; never follow it."""
    if not isinstance(raw,str) or len(raw)>2048:
        return None
    target=urlparse(raw)
    if target.scheme!="https" or target.hostname!="api.sportmonks.com":
        return None
    try:
        pages=parse_qs(target.query).get("page",[])
        number=int(pages[0]) if len(pages)==1 else None
        if number is not None and number==current+1:
            return number
    except (ValueError,TypeError):
        pass
    return None


def _decode_page(body, current_page):
    """Parse *documented* Sportmonks v3 cursor / legacy page pagination.

    Do not infer completion from a short, nonempty list or silently assume
    that missing pagination means has_more=False.
    """
    diag=_schema_diagnostic(body)
    if not isinstance(body,dict) or not isinstance(body.get("data"),list):
        return None,None,"invalid_discovery_response",diag
    rows=body["data"]
    if body.get("error") or body.get("errors"):
        return None,None,"provider_reported_error",diag
    pagination=body.get("pagination")
    if not isinstance(pagination,dict):
        # Sportmonks sometimes documents an explicit "no results" message.
        # This permits only a truly empty page, not a potentially truncated list.
        classification=diag["message_class"]
        if not rows and classification=="verified_empty_message":
            return rows,{"mode":"terminal","more":False},None,diag
        if not rows and classification in (
            "provider_access_or_subscription",
            "provider_quota_or_rate_limit",
            "provider_invalid_request",
            "unclassified_provider_message",
        ):
            return None,None,classification,diag
        return None,None,"invalid_discovery_response",diag
    flag=pagination.get("has_more")
    if flag is not None and type(flag) is not bool:
        return None,None,"invalid_discovery_response",diag
    cursor=pagination.get("next_cursor")
    cursor=cursor if isinstance(cursor,str) and 0<len(cursor)<=2048 else None
    next_page=_next_page_number(pagination.get("next_page"),current_page)
    if flag is False:
        if cursor or next_page:
            return None,None,"invalid_discovery_response",diag
        return rows,{"mode":"terminal","more":False},None,diag
    if flag is None and not (cursor or next_page):
        return None,None,"invalid_discovery_response",diag
    if cursor:
        return rows,{"mode":"cursor","more":True,"cursor":cursor},None,diag
    if next_page is not None:
        return rows,{"mode":"page","more":True,"page":next_page},None,diag
    # Legacy v3 page mode documents has_more=true and page=N+1. Cursor
    # mode should supply next_cursor, so no undocumented token is invented.
    if flag is True:
        return rows,{"mode":"page","more":True,"page":current_page+1},None,diag
    return None,None,"invalid_discovery_response",diag


def _query_pages(forecast, now, root, call):
    """Use verified Sportmonks cursor/legacy page modes, bounded and resumable.

    The last page must be observed before confirming unique fixture identity.
    Persist only allowlisted shape diagnostics and cached fixture observations,
    NEVER API credentials, raw next_page URLs or cursors in public status.
    """
    day=utc(forecast["kickoff_utc"]).date().isoformat()
    url=f"{API_ROOT}/fixtures/date/{day}"
    phases=(("team", {"filters":"participantSearch:"+str(forecast["home_team"])}),
            ("date", {}))
    summary={"pages_read":0,"cached_pages_used":0,"scope":None,
             "candidates_seen":0,"response_schema":None}
    for scope,extra in phases:
        fixtures=[]
        marker=hashlib.sha256((day+"|"+scope+"|"+
               extra.get("filters","")).encode()).hexdigest()[:16]
        folder=Path(root)/"discovery"/day/marker
        cursor=None
        next_page=1
        for count in range(1,MAX_DISCOVERY_PAGES+1):
            # Cursor pagination MUST NOT be mixed with per_page or page.
            request_cursor_hash=(hashlib.sha256(cursor.encode()).hexdigest()
                                 if cursor else None)
            key=folder/("page_"+str(count)+".json")
            prior=load_json(key,{})
            captured=prior.get("fetched_at_utc")
            try:
                fresh=bool(captured and 0 <= (now-utc(captured)).total_seconds()
                           <= DISCOVERY_CACHE_HOURS*3600 and
                           prior.get("request_cursor_sha256")==request_cursor_hash
                           and isinstance(prior.get("pagination_state"),dict))
            except (ValueError,TypeError):
                fresh=False
            if fresh:
                rows=prior["data"]
                state=prior["pagination_state"]
                diag=prior.get("schema")
                summary["cached_pages_used"]+=1
            else:
                params={"include":"participants",**extra}
                if cursor is not None:
                    params["cursor"]=cursor
                else:
                    params.update({"per_page":50,"page":next_page})
                try:
                    response=call(url,params)
                except RequestBudgetExhausted:
                    return None,"discovery_request_budget_exhausted",summary
                rows,state,error,diag=_decode_page(response,next_page)
                summary["response_schema"]=diag
                summary["scope"]=scope
                if error:
                    return None,error,summary
                # Opaque cursors are only stored in private workflow cache files
                # for bounded replay; never in status.json or application UI.
                write_json(key,{"fetched_at_utc":now.isoformat(),
                                "request_cursor_sha256":request_cursor_hash,
                                "data":rows,"pagination_state":state,
                                "schema":diag,"scope":scope})
            if not isinstance(rows,list) or not isinstance(state,dict):
                return None,"invalid_cached_discovery",summary
            summary["response_schema"]=diag
            summary["pages_read"]+=1
            summary["scope"]=scope
            fixtures.extend(rows)
            if not state.get("more"):
                matches,conflict=_fixture_candidates(forecast,fixtures)
                summary["candidates_seen"]=len(matches)
                if conflict or len(matches)>1:
                    return None,"ambiguous_cross_provider_fixture",summary
                if len(matches)==1:
                    return matches[0],"source_identity_matched",summary
                break
            if state.get("mode")=="cursor":
                cursor=state.get("cursor")
                if not isinstance(cursor,str) or not cursor:
                    return None,"invalid_cached_discovery",summary
            elif state.get("mode")=="page":
                cursor=None
                try:
                    following=int(state["page"])
                except (ValueError,KeyError,TypeError):
                    return None,"invalid_cached_discovery",summary
                if following<=next_page:
                    return None,"invalid_cached_discovery",summary
                next_page=following
            else:
                return None,"invalid_cached_discovery",summary
        else:
            return None,"discovery_pagination_incomplete",summary
    return None,"no_unique_cross_provider_fixture_match",summary


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
        "fixture_discovery_pages":0, "fixture_discovery_cached_pages":0,
        "fixture_discovery_scope":None,"attempt_number":0,
    }
    attempt_path=None
    attempt_number=0

    def finish(reason):
        status["reason"] = reason
        status["daily_requests_used"] = used
        if attempt_path is not None:
            delay=(24*60 if reason in (
                "provider_access_subscription_or_network_error",
                "source_error", "invalid_discovery_response")
                   else RETRY_MINUTES)
            terminal=(reason=="genuine_sportmonks_xg_captured" or
                      reason=="ambiguous_cross_provider_fixture" or
                      reason=="past_kickoff_not_valid_for_research" or
                      attempt_number>=MAX_ATTEMPTS_PER_FORECAST)
            write_json(attempt_path, {
                "forecast_id":forecast["forecast_id"],
                "api_football_fixture_id":str(forecast["fixture_id"]),
                "attempted_at_utc":started.isoformat(),
                "attempt_count":attempt_number,
                "last_reason":reason,
                "next_retry_at_utc":(
                    started+timedelta(minutes=delay)).isoformat(),
                "terminal":terminal,
            })
        # Keep the most recent ACTUAL provider-request diagnosis even when
        # 15-minute no-call jobs overwrite current-run status with quota/idle.
        # Never persist provider response text, URLs, headers, tokens or cursors.
        if status["requests_this_run"] > 0:
            observation = _safe_provider_observation(status)
            write_json(root/"last_provider_observation.json", observation)
        else:
            observation = load_json(root/"last_provider_observation.json", {})
        status["last_provider_observation"] = observation or None
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
        if path.exists():
            continue
        previous=load_json(attempt,{})
        try:
            count=int(previous.get("attempt_count",1)) if previous else 0
            retry_at=previous.get("next_retry_at_utc") or (
                (utc(previous["attempted_at_utc"])+
                 timedelta(minutes=RETRY_MINUTES)).isoformat()
                if previous.get("attempted_at_utc") else None)
            ready=retry_at is None or utc(retry_at)<=started
        except (TypeError,ValueError,KeyError):
            # Corrupt checkpoints never authorize unbounded retry traffic.
            continue
        if (not previous or (not previous.get("terminal") and
            count<MAX_ATTEMPTS_PER_FORECAST and ready)):
            selected = (item, path, attempt, count)
            break
    if selected is None:
        return finish("eligible_forecasts_already_checked")
    if used >= daily_limit or used >= MAX_DAILY_REQUESTS:
        return finish("daily_request_budget_exhausted")
    forecast, destination, attempt_path, previous_attempts = selected
    attempt_number=previous_attempts+1
    status["attempt_number"]=attempt_number
    status["selected_forecast_fixture_id"]=str(forecast["fixture_id"])
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

    # Reservation persists across workflow runs, including failed requests.
    # A prior run's marker is compatible with the new bounded retry counter.
    write_json(attempt_path, {
        "forecast_id":forecast["forecast_id"],
        "api_football_fixture_id":str(forecast["fixture_id"]),
        "attempted_at_utc":started.isoformat(),
        "attempt_count":attempt_number,
        "last_reason":"attempt_reserved",
        "next_retry_at_utc":(
            started+timedelta(minutes=RETRY_MINUTES)).isoformat(),
        "terminal":attempt_number>=MAX_ATTEMPTS_PER_FORECAST,
    })
    try:
        match,discovery_reason,discovery_meta=_query_pages(
            forecast,started,root,call)
        status["fixture_discovery_pages"]=discovery_meta["pages_read"]
        status["fixture_discovery_cached_pages"]=discovery_meta[
            "cached_pages_used"]
        status["fixture_discovery_scope"]=discovery_meta["scope"]
        status["fixture_discovery_candidates"]=discovery_meta[
            "candidates_seen"]
        # Safe schema-only metadata: do not write response bodies, cursors,
        # raw URLs, provider messages, fixture names or API credentials.
        status["fixture_discovery_response_schema"]=discovery_meta.get(
            "response_schema")
        if match is None:
            return finish(discovery_reason)
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
            "source_identity_policy":"reviewed_whole_name_aliases_and_kickoff_within_5min",
            "source_discovery_scope":status["fixture_discovery_scope"],
            "source_discovery_pages":status["fixture_discovery_pages"],
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
        # Discovery pages already fetched remain cached for controlled retry.
        return finish("request_budget_exhausted_mid_research")
    except requests.RequestException as exc:
        status["error_type"]=type(exc).__name__
        status["http_status_category"]=_http_status_category(exc)
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
