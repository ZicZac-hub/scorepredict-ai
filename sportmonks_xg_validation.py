"""Read-only, strict Sportmonks historical xG evidence verification.

Packets are not forecasts. This validator never accesses live APIs, alters the
original frozen predictions, or assumes extra predictive skill from xG alone.
"""
from collections import Counter
from datetime import timedelta
import math

from v10_prospective import utc

XG_TYPE = 5304


def _positive_id(value):
    if isinstance(value, bool):
        return None
    raw = str(value)
    return raw if raw.isdecimal() and int(raw) > 0 else None


def _xg_number(value):
    if type(value) not in (int, float):
        return False
    return math.isfinite(value) and 0 <= value <= 15


def _fail(reason):
    return {"valid": False, "reason": reason, "home_history": 0,
            "away_history": 0, "both_teams_have_xg": False}


def validate_packet(packet, originals):
    """Require original immutable identity and fully historical two-team xG.

    Rows without a unique matching original, source-specific participant IDs,
    completed-match timing, or correct type 5304 must never be called valid.
    """
    if not isinstance(packet, dict) or not isinstance(originals, (list, tuple)):
        return _fail("invalid_packet")
    if (packet.get("provider") != "Sportmonks" or
        packet.get("forecast_provider") != "API-Football" or
        packet.get("research_only") is not True or
        packet.get("not_automatically_promoted") is not True or
        type(packet.get("xg_type_id")) is not int or
        packet["xg_type_id"] != XG_TYPE):
        return _fail("unverified_origin")
    source_id = _positive_id(packet.get("api_football_fixture_id"))
    future_id = _positive_id(packet.get("sportmonks_fixture_id"))
    home_id = _positive_id(packet.get("sportmonks_home_team_id"))
    away_id = _positive_id(packet.get("sportmonks_away_team_id"))
    if not (source_id and future_id and home_id and away_id) or home_id == away_id:
        return _fail("invalid_source_ids")
    matches = [record for record in originals if isinstance(record, dict)
        and record.get("provider") == "API-Football"
        and record.get("model_key") == "v8"
        and str(record.get("fixture_id")) == source_id
        and record.get("forecast_id") == packet.get("source_forecast_id")
        and record.get("input_digest") == packet.get("source_input_digest")
        and packet.get("source_input_digest")
    ]
    if len(matches) != 1:
        return _fail("original_identity_not_verified")
    original = matches[0]
    if (str(packet.get("home_team")) != str(original.get("home_team")) or
        str(packet.get("away_team")) != str(original.get("away_team"))):
        return _fail("team_identity_mismatch")
    try:
        capture = utc(packet["captured_at_utc"])
        original_time = utc(original["captured_at_utc"])
        kickoff = utc(original["kickoff_utc"])
        if (utc(packet["kickoff_utc"]) != kickoff or
            not original_time <= capture < kickoff):
            return _fail("invalid_pre_match_timing")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _fail("invalid_pre_match_timing")
    history = packet.get("historical_xg")
    if not isinstance(history, list) or not history:
        return _fail("no_historical_xg")
    seen = set()
    sides = {"home": set(), "away": set()}
    for item in history:
        if not isinstance(item, dict):
            return _fail("invalid_history_item")
        fid = _positive_id(item.get("sportmonks_fixture_id"))
        hist_home = _positive_id(item.get("home_team_id"))
        hist_away = _positive_id(item.get("away_team_id"))
        if (not fid or fid in seen or not hist_home or not hist_away
                or hist_home == hist_away):
            return _fail("invalid_historical_ids")
        seen.add(fid)
        if (item.get("source") != "Sportmonks" or
            str(item.get("type_id")) != str(XG_TYPE) or
            item.get("available_before_target_kickoff") is not True):
            return _fail("unverified_historical_xg_source")
        if not (_xg_number(item.get("home_xg")) and
                _xg_number(item.get("away_xg"))):
            return _fail("invalid_historical_xg_values")
        try:
            history_kickoff = utc(item["kickoff_utc"])
            if not (history_kickoff + timedelta(hours=3) <= capture and
                    history_kickoff < kickoff):
                return _fail("historical_match_not_finished_by_capture")
        except (KeyError, TypeError, ValueError, OverflowError):
            return _fail("invalid_historical_timestamp")
        participants = {hist_home, hist_away}
        if home_id in participants:
            sides["home"].add(fid)
        if away_id in participants:
            sides["away"].add(fid)
    counts = (len(sides["home"]), len(sides["away"]))
    if not any(counts):
        return _fail("history_unrelated_to_selected_teams")
    if (type(packet.get("historical_xg_home_count")) is not int or
        type(packet.get("historical_xg_away_count")) is not int or
        packet["historical_xg_home_count"] != counts[0] or
        packet["historical_xg_away_count"] != counts[1]):
        return _fail("history_counts_disagree")
    return {"valid": True, "reason": "verified_historical_source_packet",
            "home_history": counts[0], "away_history": counts[1],
            "both_teams_have_xg": min(counts) >= 1}


def verify_packet_collection(packets, originals):
    """Safe aggregate diagnostics for a read-only quality dashboard."""
    validated = [validate_packet(p, originals) for p in packets]
    rejected = Counter(x["reason"] for x in validated if not x["valid"])
    return {
        "checked": len(packets),
        "source_validated": sum(x["valid"] for x in validated),
        "with_both_teams": sum(x["valid"] and x["both_teams_have_xg"]
                               for x in validated),
        "with_six_per_team": sum(x["valid"] and min(x["home_history"],
                               x["away_history"]) >= 6 for x in validated),
        "rejected_by_reason": dict(rejected),
        "no_proven_incremental_score_accuracy": True,
    }
