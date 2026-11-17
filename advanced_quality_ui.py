"""Read-only Advanced Quality Report for GitHub's saved, frozen evidence.

This page makes no paid provider calls, does not change forecasting weights,
and never treats missing verification as either a hit or a miss.
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent / "data"
QUALITY_URL = ("https://github.com/ZicZac-hub/scorepredict-ai/blob/main/"
               "data/v22_prospective/quality_report.json")
WORKFLOW_URL = ("https://github.com/ZicZac-hub/scorepredict-ai/actions/"
                "workflows/prematch-evidence.yml")


def _read_json(path):
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        return content if isinstance(content, dict) else {}
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}


def _count(value):
    try:
        return max(0, int(value))
    except (ValueError, TypeError, OverflowError):
        return 0


def summary(report, market=None):
    """Safely summarise *recorded* coverage; never infer missing source status."""
    market = market or {}
    source = report.get("source_completeness") or {}
    wide = market.get("coverage_health") or {}
    v22 = (report.get("frozen_v22_vs_v8") or {}).get("v22_research") or {}
    sm = report.get("sportmonks_independent_xg") or {}
    total = _count(source.get("packets_checked"))
    market_total = _count(wide.get("eligible_frozen_forecasts"))
    market_found = _count(wide.get("matched_market_forecasts"))
    return {
        "as_of": report.get("checked_at_utc"),
        "source_packets": total,
        "historical_xg": _count(source.get("with_historical_xg")),
        "six_xg_per_team": _count(source.get("with_6_per_team")),
        "verified_match_stats": _count(source.get("with_verified_historical_match_statistics")),
        "stats_without_xg": _count(source.get("with_stats_but_no_genuine_xg")),
        "independent_opinions": _count(source.get("with_independent_provider_prediction")),
        "reported_injuries": _count(source.get("with_reported_injuries")),
        "both_lineups": _count(source.get("with_both_starting_XIs")),
        "two_market_types": _count(source.get("with_2_market_types")),
        "market_forecasts": market_total,
        "market_matched": market_found,
        "market_percent": (100.0 * market_found / market_total
                           if market_total else None),
        "v22_verified_pairs": _count(v22.get("verified")),
        "v22_hits": _count(v22.get("hits")),
        "sportmonks_enabled": sm.get("enabled"),
        "sportmonks_token_present": sm.get("token_present"),
        "sportmonks_reason": sm.get("reason"),
        "sportmonks_last_observation": sm.get("last_provider_observation"),
        "sportmonks_last_run": sm.get("last_run_utc"),
        "sportmonks_matched_fixtures": _count(sm.get("matched_future_fixtures_last_run")),
        "sportmonks_saved": _count(sm.get("frozen_evidence_packets")),
        "sportmonks_six_xg": _count(sm.get("six_xg_for_both_teams_packets")),
        "sportmonks_used": sm.get("requests_used_today"),
        "sportmonks_budget": sm.get("requests_limit_per_day"),
        "sportmonks_discovery_scope": sm.get("discovery_scope"),
        "sportmonks_discovery_response_schema": sm.get("discovery_response_schema"),
        "sportmonks_discovery_pages": _count(sm.get("discovery_pages_examined")),
        "sportmonks_cached_pages": _count(sm.get("discovery_cached_pages")),
        "sportmonks_attempt_number": sm.get("attempt_number"),
        "source_capture_enabled": report.get("source_capture_enabled"),
        "source_capture_last_run": report.get("source_capture_last_run"),
        "capture_reason": report.get("source_capture_reason"),
        "capture_eligible": report.get("source_capture_eligible_legs"),
        "capture_attempted": report.get("source_capture_attempted_legs"),
        "capture_saved": _count(report.get("source_capture_saved_packets")),
        "capture_skipped": report.get("source_capture_skipped_reasons") or [],
        "capture_daily_used": report.get("source_capture_daily_attempts_used"),
        "capture_daily_limit": report.get("source_capture_daily_attempts_limit"),
        "early_odds_enabled": report.get("v23_early_market_capture_enabled"),
        "wide_odds_enabled": ((report.get("v24_broad_market_collection") or {})
                              .get("enabled")),
    }


def _local_time(raw):
    if not raw:
        return "Not recorded"
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return "Timestamp missing timezone"
        return parsed.astimezone(ZoneInfo("Australia/Sydney")).strftime(
            "%d %b %Y, %I:%M %p Sydney")
    except (TypeError, ValueError):
        return "Invalid timestamp"


def _state(value):
    if value is True:
        return "Enabled in saved status"
    if value is False:
        return "Disabled in saved status"
    return "Not confirmed by saved status"


def render(st, data_root=None):
    """Display checked-in research status even when football API keys are absent."""
    data_root = Path(data_root) if data_root is not None else ROOT
    report = _read_json(data_root / "v22_prospective" / "quality_report.json")
    market = _read_json(data_root / "v24_market_catalog" / "status.json")
    st.subheader("🧪 Advanced Quality Report")
    st.caption("Based on saved GitHub evidence, not a new live API request. "
               "Missing observations are not scored as misses.")
    if not report:
        st.warning("The saved quality report is missing or unreadable. "
                   "Check the GitHub evidence workflow.")
        st.link_button("Open collection workflow", WORKFLOW_URL)
        return
    s = summary(report, market)
    st.caption("Last generated: " + _local_time(s["as_of"]))
    cols = st.columns(4)
    cols[0].metric("Source packets", s["source_packets"])
    cols[1].metric("With genuine xG", s["historical_xg"])
    cols[2].metric("With both lineups", s["both_lineups"])
    cols[3].metric("With two market types", s["two_market_types"])
    st.markdown("**Advanced source completeness**")
    st.dataframe([
        {"Evidence": "Genuine historical shot-based xG", "Available": s["historical_xg"], "Packets checked": s["source_packets"]},
        {"Evidence": "At least six historical xG fixtures per team", "Available": s["six_xg_per_team"], "Packets checked": s["source_packets"]},
        {"Evidence": "Provider-reported historical shot and match statistics", "Available": s["verified_match_stats"], "Packets checked": s["source_packets"]},
        {"Evidence": "Independent API-Football pre-match opinion (shadow only)", "Available": s["independent_opinions"], "Packets checked": s["source_packets"]},
        {"Evidence": "Reported unavailable/injured players", "Available": s["reported_injuries"], "Packets checked": s["source_packets"]},
        {"Evidence": "Both confirmed starting XIs", "Available": s["both_lineups"], "Packets checked": s["source_packets"]},
        {"Evidence": "At least two bookmaker market types", "Available": s["two_market_types"], "Packets checked": s["source_packets"]},
    ], hide_index=True, use_container_width=True)
    if s["market_forecasts"]:
        st.metric("Frozen forecasts with matched odds",
                  f'{s["market_matched"]} / {s["market_forecasts"]}',
                  help="Identity-matched saved odds. This does not imply improved prediction accuracy.")
        st.caption(f'Bookmaker market coverage: {s["market_percent"]:.1f}% '
                   "of the eligible frozen forecasts in the saved catalog.")
    else:
        st.info("No comparable frozen-forecast market coverage has been recorded.")
    st.markdown("**Collection status**")
    st.write("Historical xG / lineups / injuries: " + _state(s["source_capture_enabled"]))
    st.write("V21 last capture: " + _local_time(s["source_capture_last_run"]))
    if s["capture_reason"]:
        st.write("Capture result: " + str(s["capture_reason"]).replace("_", " "))
    if s["capture_daily_used"] is not None and s["capture_daily_limit"] is not None:
        st.write(f'Advanced capture attempts today: {s["capture_daily_used"]} / '
                 f'{s["capture_daily_limit"]} (safety budget)')
    if s["capture_eligible"] is not None:
        st.write(f'Eligible frozen legs: {s["capture_eligible"]} · '
                 f'API attempts: {s["capture_attempted"]} · '
                 f'New source packets: {s["capture_saved"]}')
    if s["capture_skipped"]:
        st.caption("Not collected, with reasons: " +
                   " · ".join(map(str, s["capture_skipped"][:8])))
    st.write("Earlier bookmaker odds: " + _state(s["early_odds_enabled"]))
    st.write("Broad market collector: " + _state(s["wide_odds_enabled"]))
    if s["stats_without_xg"]:
        st.caption(f'{s["stats_without_xg"]} captured packets have historical '
                   'match statistics but NO verified xG. Shot counts are not '
                   'being presented as expected goals.')
    if s["historical_xg"] == 0 or s["both_lineups"] == 0:
        st.warning("Advanced research is incomplete: genuine historical xG "
                   "and/or confirmed starting lineups have not been recorded "
                   "in these saved packets. Do not label forecasts as "
                   "fully source-qualified.")
    st.markdown("**Sportmonks independent genuine xG**")
    st.caption("Separate Sportmonks evidence, only for a unique cross-provider "
               "match and completed Sportmonks historical fixtures. It is not "
               "currently an input to the V22 prediction model.")
    st.write("Collector: " + _state(s["sportmonks_enabled"]))
    if s["sportmonks_last_run"]:
        st.write("Last Sportmonks check: " + _local_time(s["sportmonks_last_run"]))
    if s["sportmonks_reason"]:
        st.write("Latest workflow result: " +
                 str(s["sportmonks_reason"]).replace("_"," "))
    last=s["sportmonks_last_observation"]
    if isinstance(last,dict) and last:
        st.write("Last actual Sportmonks request: " +
                 _local_time(last.get("at_utc")))
        st.caption("Actual provider-request outcome: " +
                   str(last.get("reason", "not_recorded")).replace("_", " ") +
                   " · response category: " +
                   str((last.get("response_shape") or {}).get("message_class",
                       "not_recorded")) +
                   " · HTTP category: " +
                   str(last.get("http_status_category", "not_reported")) +
                   ". Saved without raw provider messages or credentials.")
    if isinstance(s["sportmonks_discovery_response_schema"],dict):
        details=s["sportmonks_discovery_response_schema"]
        st.caption("Safe response shape: data=" +
                   str(details.get("data_kind")) +
                   ", rows=" + str(details.get("row_count")) +
                   ", pagination=" + str(details.get("pagination_kind")) +
                   ", has_more=" + str(details.get("has_more_kind")) +
                   ", next_cursor=" + str(bool(details.get("next_cursor_present"))) +
                   ", message_category=" + str(details.get("message_class", "not_recorded")) +
                   ". No API tokens, provider messages or raw cursor values are displayed.")
    if s["sportmonks_discovery_pages"]:
        st.caption("Fixture identity scan: " +
                   str(s["sportmonks_discovery_pages"]) + " pages checked (" +
                   str(s["sportmonks_cached_pages"]) + " reused from cache), scope: " +
                   str(s["sportmonks_discovery_scope"] or "unknown") +
                   ", attempt: " + str(s["sportmonks_attempt_number"] or "n/a") +
                   ". An incomplete scan is never claimed as a verified match.")
    cols = st.columns(2)
    cols[0].metric("Verified Sportmonks xG packets",s["sportmonks_saved"])
    cols[1].metric("Both teams with six historical xG",s["sportmonks_six_xg"])
    if s["sportmonks_used"] is not None and s["sportmonks_budget"] is not None:
        st.caption(f'Sportmonks calls used today: {s["sportmonks_used"]} / '
                   f'{s["sportmonks_budget"]}. The source add-on is not assumed '
                   'active merely because an API key exists.')
    if s["sportmonks_saved"] == 0:
        st.info("No qualifying cross-provider historical xG packet has been "
                "verified yet. Check Sportmonks access and fixture matching; "
                "do not interpret this as zero expected goals.")
    st.markdown("**Verified V22 five-double results**")
    if s["v22_verified_pairs"]:
        st.metric("Verified hits / verified pairs",
                  f'{s["v22_hits"]} / {s["v22_verified_pairs"]}')
    else:
        st.info("No independently verified V22 five-double pairs yet. "
                "No hit-rate claim can be made.")
    st.caption("Scheduled scouting starts up to 48 hours before kickoff for "
               "previously frozen fixtures, then makes one guarded near-kickoff "
               "refresh to check announced lineups. Historical shot counts, "
               "true xG and independent provider opinions are labelled separately. "
               "Source coverage, quota and fixture eligibility limit collection.")
    st.link_button("Open saved quality report", QUALITY_URL)
    st.link_button("Open collection workflow", WORKFLOW_URL)
    with st.expander("View saved evidence fields"):
        st.json(report)
