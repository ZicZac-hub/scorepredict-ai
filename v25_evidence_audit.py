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
    """Require a fresh, source-dated 1X2 quote AND a fresh totals quote.

    Older research packets may contain only a 1X2 line. That is useful
    context, but must never satisfy the two-market evidence quality gate.
    An optional stale O/U 3.5 quote must not invalidate a good O/U 2.5.
    """
    markets = packet.get("multi_market") or {}
    if not isinstance(markets, dict):
        return False
    try:
        captured = utc(packet["captured_at_utc"])
    except (KeyError, TypeError, ValueError):
        return False

    def fresh(kind):
        try:
            market = markets[kind]
            stamp = utc(market["last_source_update_utc"])
            return (stamp <= captured <= now < kickoff and
                    now - stamp <= MARKET_MAX_AGE)
        except (KeyError, TypeError, ValueError, AttributeError):
            return False

    return fresh("1x2") and (fresh("over_2_5") or fresh("over_3_5"))


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
        # A history-rich packet may contain stale odds even when another
        # separately frozen, identity-matched packet contains current prices.
        # Inspect that version independently; do not combine or rewrite hashes.
        market_packet = packet if _fresh_market(packet, clock, kickoff) else None
        if market_packet is None:
            market_record = packet_loader(root, source, now=clock, priority="market")
            if qualified_packet(market_record, source, now=clock):
                contender = market_record["provider_snapshot"]
                if _fresh_market(contender, clock, kickoff):
                    market_packet = contender
        info["verified_pre_match_market"] = market_packet is not None
        if market_packet is not None:
            info["market_snapshot_at_utc"] = market_packet["captured_at_utc"]
        info["reason"] = ("verified_market_available_not_model_weighted"
                          if info["verified_pre_match_market"] else
                          "verified_snapshot_without_fresh_market")
    except (KeyError, ValueError, TypeError, OSError):
        info["reason"] = "unreadable_or_invalid_source_packet"
    return info


def trace_pair(first, second, now, **kwargs):
    return [trace_leg(first, now, **kwargs), trace_leg(second, now, **kwargs)]
