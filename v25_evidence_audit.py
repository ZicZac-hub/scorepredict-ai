"""Read-only, identity-verified V25 evidence delivery audit.

This does NOT add a provider request, modify frozen predictions or feed
untrained xG/odds/lineup coefficients into V25. Missing is unknown, not zero.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from v10_prospective import utc
from v21_capture_sources import qualified_packet

SOURCE_ROOT = Path("data/v21_sources")
MARKET_MAX_AGE = timedelta(hours=12)


def _matching_source(row):
    """Only exact stored source-forecast identities can access source packets."""
    fixture = row.get("fixture") or {}
    source = row.get("source_forecast") or {}
    required = ("provider", "fixture_id", "forecast_id", "input_digest",
                "home_team_id", "away_team_id", "kickoff_utc")
    if not all(source.get(key) is not None for key in required):
        return None
    try:
        if (str(source["provider"]) != str(fixture["provider"]) or
            str(source["fixture_id"]) != str(fixture["id"]) or
            str(source["home_team_id"]) != str(fixture["homeTeam"]["id"]) or
            str(source["away_team_id"]) != str(fixture["awayTeam"]["id"]) or
            utc(source["kickoff_utc"]) != utc(fixture["utcDate"])):
            return None
        return source
    except (KeyError, TypeError, ValueError):
        return None


def _fresh_market(packet, now, kickoff):
    markets = packet.get("multi_market") or {}
    if isinstance(markets, dict) and "1x2" in markets and (
            "over_2_5" in markets or "over_3_5" in markets):
        keys = [k for k in ("1x2", "over_2_5", "over_3_5") if k in markets]
        try:
            stamps = [utc(markets[k]["last_source_update_utc"]) for k in keys]
            return bool(stamps) and all(
                stamp <= now and stamp < kickoff and now-stamp <= MARKET_MAX_AGE
                for stamp in stamps)
        except (KeyError, TypeError, ValueError):
            return False
    quote = packet.get("market")
    if isinstance(quote, dict) and quote.get("fair_1x2"):
        try:
            stamp = utc(quote["updated_at_utc"])
            return stamp <= now and stamp < kickoff and now-stamp <= MARKET_MAX_AGE
        except (KeyError, TypeError, ValueError):
            pass
    return False


def trace_leg(candidate, now, *, root=SOURCE_ROOT, packet_loader=None):
    """Check real delivery to one V25 leg, without inferring absent source data.

    The result is descriptive only. It must never be mistaken for proof that
    V25 mathematically incorporated independent evidence.
    """
    clock = utc(now)
    row = candidate.get("row", candidate)
    fixture = row.get("fixture") or {}
    info = {
        "fixture_id": str(fixture.get("id") or ""),
        "fixture": (str((fixture.get("homeTeam") or {}).get("name") or "Home") +
                    " vs " + str((fixture.get("awayTeam") or {}).get("name") or "Away")),
        "source_identity_verified": False,
        "matching_provider_packet": False,
        "verified_pre_match_market": False,
        "verified_historical_xg_both_teams": False,
        "verified_historical_xg_fixtures_per_team": [0, 0],
        "confirmed_starting_xi_both_teams": False,
        "reported_injury_entries": None,
        "independent_context_applied_to_v25_model": False,
        "reason": "missing_original_source_identity",
    }
    source = _matching_source(row)
    if source is None:
        return info
    try:
        kickoff = utc(source["kickoff_utc"])
        if not clock < kickoff:
            info["reason"] = "kickoff_already_started"
            return info
        info["source_identity_verified"] = True
        if packet_loader is None:
            from v21_capture_sources import best_packet
            packet_loader = best_packet
        record = packet_loader(root, source, now=clock, priority="research")
        if not qualified_packet(record, source, now=clock):
            info["reason"] = "no_matching_timestamp_verified_provider_packet"
            return info
        packet = record["provider_snapshot"]
        info["matching_provider_packet"] = True
        info["provider_snapshot_at_utc"] = packet.get("captured_at_utc")
        home, away = str(source["home_team_id"]), str(source["away_team_id"])
        historical = packet.get("verified_historical_fixture_xg") or []
        counts = [
            len({str(item.get("fixture_id")) for item in historical
                 if item.get("type") == "api_football_reported_expected_goals"
                 and tid in (str(item.get("home_team_id")),
                             str(item.get("away_team_id")))})
            for tid in (home, away)
        ]
        info["verified_historical_xg_fixtures_per_team"] = counts
        info["verified_historical_xg_both_teams"] = min(counts) >= 1
        lineups = packet.get("confirmed_starting_xi") or []
        info["confirmed_starting_xi_both_teams"] = all(
            any(str(item.get("team_id")) == tid and
                item.get("provider_reported") is True and
                len(item.get("starting_xi") or []) == 11 for item in lineups)
            for tid in (home, away))
        info["reported_injury_entries"] = len(packet.get("reported_injuries") or [])
        info["verified_pre_match_market"] = _fresh_market(packet, clock, kickoff)
        info["reason"] = ("verified_market_available_not_model_weighted"
                          if info["verified_pre_match_market"] else
                          "verified_snapshot_without_fresh_market")
    except (KeyError, ValueError, TypeError, OSError):
        info["reason"] = "unreadable_or_invalid_source_packet"
    return info


def trace_pair(first, second, now, **kwargs):
    return [trace_leg(first, now, **kwargs), trace_leg(second, now, **kwargs)]
