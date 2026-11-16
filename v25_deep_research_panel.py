"""Mobile-first, read-only V25 Professional Analyst panel for Deep Research.

V25 is a *two-fixture* research-only shadow experiment. It is not a
single-match prediction method, cannot be launched by the purple button,
and must not be represented as proven accuracy or live bookmaker advice.
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STATUS = Path("data/pro_analyst_shadow/status.json")
SETS = Path("data/pro_analyst_shadow/sets")
ACTIONS_URL = (
    "https://github.com/ZicZac-hub/scorepredict-ai/actions/"
    "workflows/prematch-evidence.yml"
)
STATUS_URL = (
    "https://github.com/ZicZac-hub/scorepredict-ai/blob/main/"
    "data/pro_analyst_shadow/status.json"
)
SYDNEY = ZoneInfo("Australia/Sydney")


def sydney_time(value):
    """Format source timestamp, not a locally invented run or kickoff."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(SYDNEY).strftime("%d %b %Y, %I:%M %p").lstrip("0") + " Sydney"
    except (ValueError,TypeError,OverflowError):
        return None


def diagnostic_reasons(report):
    """Validated source scan counts, never falsely interpreted as matches lost."""
    coverage=report.get("candidate_coverage") or {}
    if not isinstance(coverage,dict):
        return {}
    excluded=coverage.get("excluded") or {}
    if not isinstance(excluded,dict):
        return {}
    return {str(k):v for k,v in excluded.items()
            if type(v) is int and v >= 0}


def load_status(path=STATUS):
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(report, dict):
        return None
    names = ("total_frozen_sets", "verified_sets", "pending_sets",
             "robust_wins", "baseline_wins_on_same_pairs")
    if any(type(report.get(key)) is not int or report[key] < 0
           for key in names):
        return None
    count = report["total_frozen_sets"]
    verified = report["verified_sets"]
    pending = report["pending_sets"]
    if verified + pending != count:
        return None
    if (report["robust_wins"] > verified or
            report["baseline_wins_on_same_pairs"] > verified):
        return None
    return report


def latest_frozen_set(directory=SETS):
    """Only present an original archived, write-once five-double set."""
    for path in sorted(Path(directory).glob("*.json"), reverse=True):
        try:
            row=json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row,dict):
                continue
            if row.get("decision")!="FROZEN_RESEARCH_ONLY":
                continue
            day=row.get("sydney_date")
            if not isinstance(day,str) or path.stem != day:
                continue
            picks=row.get("five_doubles")
            if (not isinstance(picks,list) or len(picks)!=5 or
                    any(not isinstance(p,dict) or
                        not isinstance(p.get("score1"),str) or
                        not isinstance(p.get("score2"),str)
                        for p in picks)):
                continue
            first,second=row.get("first") or {},row.get("second") or {}
            if not all(k in first and k in second for k in
                       ("home_team","away_team","forecast_id")):
                continue
            return row
        except (OSError, ValueError, TypeError, UnicodeError):
            continue
    return None


def render(st, *, report_path=STATUS, sets_dir=SETS):
    """Always show the research section, including before its first report."""
    with st.expander("🧮 V25 Professional Analyst · Five Doubles", expanded=True):
        st.markdown("**Want five V25 score doubles for upcoming matches?**")
        st.link_button("⚽ Generate V25 Five Doubles now",
                       "https://ziczacscorepredict-ai.streamlit.app/?view=ai_five",
                       use_container_width=True)
        st.caption("The generator supports manual pairs, automatic selection "
                   "and every eligible same-day upcoming pair, one page at a time.")
        st.caption("This panel below is the automatic pre-match V25 experiment. "
                   "Its saved results are separate from predictions you create "
                   "on demand. Existing V8 picks stay unchanged.")
        report=load_status(report_path)
        if report is None:
            st.info("V25 is installed, but no saved research run is available "
                    "in this deployment yet. GitHub will evaluate eligible "
                    "pairs on its next scheduled evidence run. No missing "
                    "report is counted as a loss or a winning result.")
            st.link_button("See automatic V25 collection runs", ACTIONS_URL)
            return
        evaluated=sydney_time(report.get("generated_at_utc"))
        if evaluated:
            st.caption("Last V25 collection: " + evaluated)
        total,verified,pending = (report[k] for k in
                                  ("total_frozen_sets","verified_sets",
                                   "pending_sets"))
        robust=report["robust_wins"]
        baseline=report["baseline_wins_on_same_pairs"]
        st.write(f"Frozen research pairs: **{total}** · "
                 f"Verified: **{verified}** · Awaiting results: **{pending}**")
        st.write(f"V25 five-double hits: **{robust}/{verified}** · "
                 f"Original V8 on identical pairs: **{baseline}/{verified}**"
                 if verified else
                 "No finished, independently verified V25 pairs yet. "
                 "Real five-double accuracy is not measurable.")
        st.caption("This is a strict pre-kickoff experiment comparing four "
                   "related goal-distribution assumptions. Model-estimated "
                   "coverage is not observed success, and 70% accuracy "
                   "has not been demonstrated.")
        if total == 0:
            st.info("V25 collection has run successfully. No two-match "
                    "research set has qualified yet; this is not a prediction "
                    "loss or an application error.")
            reasons=diagnostic_reasons(report)
            waiting=reasons.get("outside_research_freeze_window",0)
            if waiting:
                st.write(f"**{waiting}** saved source forecasts are waiting "
                         "for the 24-hour pre-match research window.")
            invalid=reasons.get("invalid_source_time",0)
            if invalid:
                st.caption(f"{invalid} source records failed strict timestamp "
                           "checks and are excluded, not silently repaired.")
            other=reasons.get("not_original_v8_model",0)
            if other:
                st.caption(f"{other} other-model records were excluded because "
                           "V25 compares only genuine original V8-family sources.")
            metadata=report.get("candidate_coverage") or {}
            next_window=(sydney_time(metadata.get("next_possible_pair_window_utc"))
                         if isinstance(metadata,dict) else None)
            if next_window:
                st.write("Earliest *possible* next V25 two-match research "
                         "window: **" + next_window + "**")
                st.caption("An opening window does not guarantee a prediction. "
                           "V25 can still abstain if the evidence or model "
                           "agreement is insufficient.")
            st.caption("Archived genuine pre-match forecasts older than 72 hours "
                       "may be used only in the research comparison, labelled "
                       "as early snapshots. They are never described as fresh "
                       "lineups, injury news or live bookmaker information.")
        latest=latest_frozen_set(sets_dir)
        if latest:
            one,two=latest["first"],latest["second"]
            st.write(f"Latest frozen Sydney date: **{latest['sydney_date']}**")
            st.caption(f"Match 1: {one['home_team']} vs {one['away_team']}")
            st.caption(f"Match 2: {two['home_team']} vs {two['away_team']}")
            for index,item in enumerate(latest["five_doubles"],1):
                st.write(f"{index}. **{item['score1']}** / "
                         f"**{item['score2']}**")
            st.caption("Saved before kickoff as research-only five doubles. "
                       "Selections are not updated after the match.")
        else:
            st.caption("No eligible saved five-double pairs to display yet. "
                       "The model may abstain when evidence is insufficient.")
        st.link_button("Open verified V25 status on GitHub", STATUS_URL)
        st.link_button("Open automatic collection history", ACTIONS_URL)
