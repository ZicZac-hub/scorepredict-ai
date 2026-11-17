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
    total = _count(source.get("packets_checked"))
    market_total = _count(wide.get("eligible_frozen_forecasts"))
    market_found = _count(wide.get("matched_market_forecasts"))
    return {
        "as_of": report.get("checked_at_utc"),
        "source_packets": total,
        "historical_xg": _count(source.get("with_historical_xg")),
        "six_xg_per_team": _count(source.get("with_6_per_team")),
        "both_lineups": _count(source.get("with_both_starting_XIs")),
        "two_market_types": _count(source.get("with_2_market_types")),
        "market_forecasts": market_total,
        "market_matched": market_found,
        "market_percent": (100.0 * market_found / market_total
                           if market_total else None),
        "v22_verified_pairs": _count(v22.get("verified")),
        "v22_hits": _count(v22.get("hits")),
        "source_capture_enabled": report.get("source_capture_enabled"),
        "source_capture_last_run": report.get("source_capture_last_run"),
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
    st.write("Earlier bookmaker odds: " + _state(s["early_odds_enabled"]))
    st.write("Broad market collector: " + _state(s["wide_odds_enabled"]))
    if s["historical_xg"] == 0 or s["both_lineups"] == 0:
        st.warning("Advanced research is incomplete: genuine historical xG "
                   "and/or confirmed starting lineups have not been recorded "
                   "in these saved packets. Do not label forecasts as "
                   "fully source-qualified.")
    st.markdown("**Verified V22 five-double results**")
    if s["v22_verified_pairs"]:
        st.metric("Verified hits / verified pairs",
                  f'{s["v22_hits"]} / {s["v22_verified_pairs"]}')
    else:
        st.info("No independently verified V22 five-double pairs yet. "
                "No hit-rate claim can be made.")
    st.caption("The existing GitHub workflow regenerates the report on its "
               "scheduled runs. Additional V21/V22 source collection is a "
               "separate owner opt-in that may use paid football API quota.")
    st.link_button("Open saved quality report", QUALITY_URL)
    st.link_button("Open collection workflow", WORKFLOW_URL)
    with st.expander("View saved evidence fields"):
        st.json(report)
