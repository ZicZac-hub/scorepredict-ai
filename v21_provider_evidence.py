"""V21 optional, quota-bounded API-Football pre-match evidence.

Uses documented provider endpoints only; never scrapes arbitrary sites or
fabricates xG, XI, injury availability, or odds. This module makes HTTP calls
ONLY when an authenticated owner explicitly requests them or an opt-in Actions
variable permits scheduled capture. Missing or stale evidence is unavailable.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math

from v10_prospective import utc
from source_capture_clock import capture_clock


def _timestamp(value):
    if value is None:
        return None
    try:
        return utc(value)
    except (TypeError, ValueError, AttributeError):
        return None


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def _identity(fixture):
    try:
        assert fixture.get("provider") == "API-Football"
        return (str(fixture["id"]), str(fixture["homeTeam"]["id"]),
                str(fixture["awayTeam"]["id"]), utc(fixture["utcDate"]))
    except (KeyError, TypeError, AssertionError, ValueError):
        return None


def _fetch(request, key, endpoint, args, diagnostics, *, with_fetch_metadata=False):
    diagnostics["queries"] += 1
    try:
        # Only the unmodified, first-party request helper supports certified
        # API-response timestamps. An injected request with no metadata may
        # still supply bookmaker timestamps, but never a guessed fetch time.
        from api_football import _request
        if with_fetch_metadata and request is _request:
            envelope=request(key,endpoint,args,with_metadata=True)
            meta=envelope.get("metadata") or {}
            if meta.get("stale") or not meta.get("fetched_at_utc"):
                diagnostics["unavailable"][endpoint]="stale_or_undated_api_response"
                return None
            diagnostics.setdefault("fetch_metadata",{})[endpoint]={
                "fetched_at_utc":meta["fetched_at_utc"],
                "cache_status":meta.get("cache_status")}
            result=envelope.get("data")
        else:
            result = request(key, endpoint, args)
        if isinstance(result, (dict, list)):
            return result
        raise ValueError("Malformed API response")
    except Exception as exc:
        diagnostics["unavailable"][endpoint] = type(exc).__name__
        return None


def _goals_statistics(raw, home_id, away_id):
    """Only genuine API-Football 'Expected Goals' fixture statistics."""
    found = {}
    for row in raw if isinstance(raw, list) else []:
        tid = str((row.get("team") or {}).get("id"))
        if tid not in (str(home_id), str(away_id)):
            continue
        for item in row.get("statistics") or []:
            label = str(item.get("type") or "").strip().casefold().replace("_", " ")
            if label == "expected goals":
                value = _number(item.get("value"))
                if value is not None:
                    found[tid] = value
    return found if len(found) == 2 else None



# Descriptive, provider-reported match counts, NOT a reconstructed xG metric.
_STATISTIC_KEYS = {
    "shots on goal": "shots_on_target",
    "shots off goal": "shots_off_target",
    "shots insidebox": "shots_inside_box",
    "shots outsidebox": "shots_outside_box",
    "total shots": "total_shots",
    "blocked shots": "blocked_shots",
    "corner kicks": "corners",
    "offsides": "offsides",
    "yellow cards": "yellow_cards",
    "red cards": "red_cards",
    "ball possession": "possession_pct",
    "fouls": "fouls",
    "goalkeeper saves": "goalkeeper_saves",
    "total passes": "total_passes",
    "passes accurate": "accurate_passes",
    "passes %": "pass_accuracy_pct",
}


def _historical_match_stats(raw, home_id, away_id):
    """Only historical match statistics actually returned for BOTH teams."""
    found = {}
    for row in raw if isinstance(raw, list) else []:
        if not isinstance(row, dict):
            continue
        tid = str((row.get("team") or {}).get("id"))
        if tid not in (str(home_id), str(away_id)):
            continue
        values = {}
        for item in row.get("statistics") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("type") or "").strip().casefold().replace("_"," ")
            kind = _STATISTIC_KEYS.get(label)
            if kind is None:
                continue
            raw_value = item.get("value")
            if kind in ("possession_pct", "pass_accuracy_pct") and isinstance(raw_value, str):
                raw_value = raw_value.strip().rstrip("%")
            value = _number(raw_value)
            percentage=kind in ("possession_pct", "pass_accuracy_pct")
            if value is not None and (percentage or value == int(value)):
                if not percentage or value <= 100:
                    values[kind] = int(value) if value == int(value) else value
        if values:
            found[tid] = values
    return found if len(found) == 2 else None


def _provider_prediction(raw, fixture_id, home_id, away_id):
    """Source-identical independent API algorithm opinion; never a final result."""
    for row in raw if isinstance(raw,list) else []:
        if not isinstance(row,dict):
            continue
        fixture = row.get("fixture") or {}
        if fixture.get("id") is not None and str(fixture["id"]) != str(fixture_id):
            continue
        teams = row.get("teams") or {}
        if (str((teams.get("home") or {}).get("id")) != str(home_id) or
                str((teams.get("away") or {}).get("id")) != str(away_id)):
            continue
        prediction = row.get("predictions") or {}
        raw_p = prediction.get("percent") or {}
        percents = {}
        for key in ("home", "draw", "away"):
            number = _number(str(raw_p.get(key,"")).strip().rstrip("%"))
            if number is None or number > 100:
                percents = {}
                break
            percents[key] = number/100
        if percents and not (0.98 <= sum(percents.values()) <= 1.02):
            percents = {}
        winner = prediction.get("winner") or {}
        winner_id = str(winner.get("id") or "")
        if winner_id not in (str(home_id),str(away_id)):
            winner_id = None
        result = {
            "source": "API_Football_independent_predictions",
            "probabilities_1x2": percents if percents else None,
            "winner_team_id": winner_id,
            "provider_advice": str(prediction.get("advice") or "")[:200],
            "not_ground_truth": True,
            "not_production_model_weight": True,
        }
        if result["probabilities_1x2"] or result["winner_team_id"] or result["provider_advice"]:
            return result
    return None

def _market(raw, now):
    """Independent one-bookmaker 1X2 with a timestamp, no fabricated consensus."""
    choices = []
    if not isinstance(raw, list):
        return None
    for fixture in raw:
        for bookmaker in (fixture.get("bookmakers") or []):
            name = str(bookmaker.get("name") or "").strip()[:80]
            if not name:
                continue
            for bet in bookmaker.get("bets") or []:
                if str(bet.get("name") or "").casefold() not in ("match winner", "1x2"):
                    continue
                values = {}
                for v in bet.get("values") or []:
                    key = str(v.get("value") or "").strip().casefold()
                    target = {"home": "home", "draw": "draw", "away": "away"}.get(key)
                    amount = _number(v.get("odd"))
                    if target and amount and amount > 1.001:
                        values[target] = amount
                if set(values) != {"home", "draw", "away"}:
                    continue
                update = _timestamp(bet.get("update") or bookmaker.get("update") or
                                    fixture.get("update"))
                if not update or update > now or now-update > timedelta(hours=12):
                    continue
                inverse = {k: 1.0/v for k, v in values.items()}
                overround = sum(inverse.values())
                if not 1.0 <= overround <= 1.30:
                    continue
                choices.append({"bookmaker": name, "updated_at_utc": update.isoformat(),
                                "observed_at_utc": now.isoformat(),
                                "fair_1x2": {k: round(v/overround, 7)
                                            for k, v in inverse.items()},
                                "overround": round(overround-1, 5)})
    return max(choices, key=lambda row: row["updated_at_utc"]) if choices else None


def capture(fixture, history, api_key, *, now=None, request=None, max_history_per_team=2,
            clock=None, include_provider_prediction=False):
    """Capture genuine fixture-level data and 2*N historical statistics requests.

    Caller must enforce owner consent or an opt-in scheduled policy. Historical
    matches must be provider/fixture qualified and genuinely finished before
    the snapshot. No post-kickoff snapshot may be labelled as pre-match.
    """
    from api_football import _request
    started, observation_time = capture_clock(now, clock)
    clock = started
    ident = _identity(fixture)
    if not ident or not api_key:
        return {"status": "unavailable", "reason": "matching_api_key_or_fixture_required"}
    fid, home, away, kickoff = ident
    if not clock < kickoff - timedelta(minutes=5):
        return {"status": "unavailable", "reason": "not_before_kickoff"}
    request = request or _request
    limits = max(0, min(12, int(max_history_per_team)))
    diag = {"queries": 0, "unavailable": {}}
    # All HTTP calls occur only after pre-match identity/consent gates.
    lineup_raw = _fetch(request, api_key, "/fixtures/lineups", {"fixture": fid}, diag)
    if observation_time() >= kickoff:
        return {"status": "unavailable", "reason": "capture_completed_after_kickoff",
                "diagnostics": diag}
    injury_raw = _fetch(request, api_key, "/injuries", {"fixture": fid}, diag)
    if observation_time() >= kickoff:
        return {"status": "unavailable", "reason": "capture_completed_after_kickoff",
                "diagnostics": diag}
    odds_raw = _fetch(request, api_key, "/odds", {"fixture": fid}, diag,
                      with_fetch_metadata=True)
    provider_prediction_raw = None
    if include_provider_prediction and observation_time() < kickoff:
        provider_prediction_raw = _fetch(request, api_key, "/predictions",
                                         {"fixture": fid}, diag)
    if observation_time() >= kickoff:
        return {"status": "unavailable", "reason": "capture_completed_after_kickoff",
                "diagnostics": diag}
    lineups = []
    for data in lineup_raw if isinstance(lineup_raw, list) else []:
        tid = str((data.get("team") or {}).get("id"))
        if tid not in (home, away):
            continue
        starters = [str((p.get("player") or {}).get("name") or "")[:100]
                    for p in data.get("startXI") or []]
        starters = [n for n in starters if n]
        if len(starters) == 11:
            lineups.append({"team_id": tid, "formation": data.get("formation"),
                            "starting_xi": starters, "provider_reported": True})
    injuries = []
    for data in injury_raw if isinstance(injury_raw, list) else []:
        tid = str((data.get("team") or {}).get("id"))
        if tid not in (home, away):
            continue
        player = data.get("player") or {}
        if player.get("name"):
            injuries.append({"team_id": tid, "player": str(player["name"])[:100],
                             "reason": str(player.get("reason") or "unreported")[:100]})
    # Past xG is only accepted if two team figures were returned for the
    # historical fixture; never treat one team's missing value as zero.
    history_rows = []
    seen = set()
    for match in history or []:
        try:
            if match.get("status") != "FINISHED" or match.get("provider") != "API-Football":
                continue
            match_id = str(match["id"])
            at = utc(match["utcDate"])
            if at + timedelta(hours=3) > clock or at >= kickoff or match_id == fid:
                continue
            h, a = str(match["homeTeam"]["id"]), str(match["awayTeam"]["id"])
            if (h not in (home, away) and a not in (home, away)) or match_id in seen:
                continue
            seen.add(match_id)
            history_rows.append((at, match_id, h, a))
        except (KeyError, TypeError, ValueError):
            continue
    history_rows.sort(reverse=True)
    selected = []
    counts = {home: 0, away: 0}
    for at, match_id, h, a in history_rows:
        matched = [tid for tid in counts if tid in (h, a) and counts[tid] < limits]
        if not matched:
            continue
        selected.append((at, match_id, h, a))
        for tid in matched:
            counts[tid] += 1
        if all(v >= limits for v in counts.values()):
            break
    verified_xg = []
    verified_stats = []
    for at, match_id, h, a in selected:
        if observation_time() >= kickoff:
            return {"status": "unavailable", "reason": "capture_completed_after_kickoff",
                    "diagnostics": diag}
        raw = _fetch(request, api_key, "/fixtures/statistics",
                     {"fixture": match_id}, diag)
        stats = _goals_statistics(raw, h, a)
        numbers = _historical_match_stats(raw, h, a)
        if numbers is not None:
            verified_stats.append({"fixture_id": match_id,
                                   "kickoff_utc": at.isoformat(),
                                   "home_team_id": h, "away_team_id": a,
                                   "home": numbers[h], "away": numbers[a],
                                   "source": "API_Football_past_fixture_statistics",
                                   "not_genuine_xg": True})
        if stats is not None:
            verified_xg.append({"fixture_id": match_id, "kickoff_utc": at.isoformat(),
                                "home_team_id": h, "away_team_id": a,
                                "home_xg": stats[h], "away_xg": stats[a],
                                "type": "api_football_reported_expected_goals"})
    # A source is observed only once its response has arrived. Do not stamp
    # later/live information with the earlier click or job-start timestamp.
    clock = observation_time()
    if clock >= kickoff:
        return {"status": "unavailable", "reason": "capture_completed_after_kickoff",
                "diagnostics": diag}
    market = _market(odds_raw, clock)
    independent = _provider_prediction(provider_prediction_raw, fid, home, away)
    diag["historical_stats_fixtures_queried"] = len(selected)
    diag["historical_stats_fixtures_returned"] = len(verified_stats)
    diag["genuine_xg_fixtures_returned"] = len(verified_xg)
    if verified_stats and not verified_xg:
        diag["xg_unavailable_reason"] = "provider_returned_stats_without_genuine_xg"
    elif not verified_xg:
        diag["xg_unavailable_reason"] = "genuine_xg_not_reported_for_eligible_history"
    # V22: extract complete, source-timestamped totals and BTTS prices
    # from the SAME already-approved /odds call. No extra paid endpoint.
    from v22_evidence_models import parse_markets
    odds_time=(diag.get("fetch_metadata",{}).get("/odds",{}).get("fetched_at_utc"))
    multi_market = parse_markets(odds_raw,observed_at=clock,
                                 verified_provider_fetched_at=odds_time)
    payload = {"schema": 1, "status": "captured_pre_kickoff", "provider": "API-Football",
               "fixture_id": fid, "home_team_id": home, "away_team_id": away,
               "kickoff_utc": kickoff.isoformat(), "captured_at_utc": clock.isoformat(),
               "verified_historical_fixture_xg": verified_xg,
               "verified_historical_match_statistics": verified_stats,
               "provider_pre_match_opinion": independent,
               "confirmed_starting_xi": lineups, "reported_injuries": injuries,
               "market": market, "multi_market": multi_market, "available_sources": {
                   "lineups": bool(lineups), "injuries_reported": bool(injuries),
                   "historical_xg": bool(verified_xg),
                   "historical_match_statistics": bool(verified_stats),
                   "independent_provider_prediction": independent is not None,
                   "timestamped_1x2": market is not None,
                   "timestamped_multi_market":len(multi_market)>=2},
               "diagnostics": diag, "source_limits": {
                   "max_historical_fixtures_per_team": limits,
                   "historical_fixture_statistics_max_http_requests": limits*2,
                   "xg_not_a_predicted_goal_adjustment": True,
                   "shot_counts_are_not_xg": True,
                   "provider_prediction_is_research_shadow_only": True,
                   "injury_not_a_validated_player_coefficient": True,
                   "market_not_a_verified_three_book_consensus": True}}
    digest_body = json.dumps(payload, sort_keys=True, separators=(",",":"), allow_nan=False)
    payload["snapshot_sha256"] = hashlib.sha256(digest_body.encode()).hexdigest()
    return payload


def valid_snapshot(packet, fixture, *, now=None):
    """Do not allow a later/source-mismatched packet into AI or scoring."""
    ident = _identity(fixture)
    if not ident or not isinstance(packet, dict):
        return False
    try:
        fid, home, away, kickoff = ident
        captured = utc(packet["captured_at_utc"])
        if (packet.get("status") != "captured_pre_kickoff" or
            packet.get("provider") != "API-Football" or
            str(packet["fixture_id"]) != fid or
            str(packet["home_team_id"]) != home or
            str(packet["away_team_id"]) != away or
            utc(packet["kickoff_utc"]) != kickoff or captured >= kickoff):
            return False
        if now is not None and captured > utc(now):
            return False
        sha = packet.get("snapshot_sha256")
        body = {k:v for k,v in packet.items() if k != "snapshot_sha256"}
        digest = hashlib.sha256(json.dumps(body,sort_keys=True,
                                separators=(",",":"),allow_nan=False).encode()).hexdigest()
        return sha == digest
    except (KeyError, TypeError, ValueError):
        return False
