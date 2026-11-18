"""Display source-authenticated future comparisons without promoting a model."""
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


LABELS = {
    "baseline_v8": "Original V8 control",
    "market_only": "Market only",
    "xg_only": "Historical xG only",
    "xg_market": "Historical xG + market",
}


def _evaluate(**kwargs):
    from professional_paired_study import evaluate
    return evaluate(**kwargs)


def _rate(value):
    if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
        return f"{value:.2%}"
    return "Unavailable"


def _count(value):
    return value if type(value) is int and value >= 0 else 0


def render(st, root=Path("data"), *, now=None):
    """Re-evaluate immutable records; never display performance.json as proof."""
    base = Path(root)
    study_root = base/"professional_paired_study"
    protocol_path = study_root/"protocol.json"
    with st.expander("Prospective five-double evidence · same original pair"):
        st.caption("Original V8 control, market and historical xG variants use the same "
                   "two fixtures and a common decision before either kickoff. "
                   "All match dates use Australia/Sydney.")
        if not protocol_path.is_file():
            st.info("The matched evidence study is waiting for its fixed protocol "
                    "and first eligible pair.")
            return None
        try:
            report = _evaluate(root=study_root,
                               pair_root=base/"professional_double_shadow",
                               source_root=base/"v10_prospective",
                               provider_root=base/"v21_sources",
                               prematch_root=base/"prematch", now=now)
            methods = report["methods"]
            if not isinstance(methods, dict):
                raise ValueError("invalid_verified_methods")
            rows = []
            for name, stats in methods.items():
                if not isinstance(stats, dict):
                    raise ValueError("invalid_verified_method")
                settled = _count(stats.get("verified"))
                hits = _count(stats.get("hits"))
                if hits > settled:
                    raise ValueError("invalid_verified_counts")
                interval = stats.get("wilson_95")
                interval_label = "Unavailable"
                if settled and isinstance(interval, (list, tuple)) and len(interval) == 2:
                    interval_label = f"{_rate(interval[0])}–{_rate(interval[1])}"
                rows.append({
                    "Model": LABELS.get(name, name),
                    "Issued": _count(stats.get("issued")),
                    "Missing evidence / abstained": _count(stats.get("abstained")),
                    "Pending": _count(stats.get("pending")),
                    "Verified sets": settled,
                    "Winning sets": hits,
                    "Observed five-double hit rate": _rate(hits/settled) if settled else "Unavailable",
                    "Descriptive 95% interval": interval_label,
                    "Single-score log loss": stats.get("single_score_log_loss"),
                    "Single-outcome Brier": stats.get("outcome_brier"),
                })
            rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
            protocol_bytes = protocol_path.read_bytes()
        except (ImportError, OSError, ValueError, TypeError, KeyError, AttributeError):
            st.warning("The matched study could not authenticate its saved evidence. "
                       "No accuracy result is displayed.")
            return None
        if not any(row["Verified sets"] for row in rows):
            st.info("No settled matched results yet. Observed accuracy is unavailable; "
                    "pending games and missing inputs are not losses.")
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        checked_at=report.get("evaluated_at_utc")
        if checked_at:
            try:
                checked=datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
                if checked.tzinfo is not None:
                    st.caption("Evidence rechecked " + checked.astimezone(
                        ZoneInfo("Australia/Sydney")).strftime("%d %b %Y, %H:%M %Z") + ".")
            except (TypeError, ValueError):
                pass
        dates=[]
        for row in report.get("rows") or []:
            if not isinstance(row, dict):
                continue
            actual=row.get("actual_scores") or [None, None]
            if not isinstance(actual, (list, tuple)) or len(actual)!=2:
                continue
            def score(value):
                if (isinstance(value, (list, tuple)) and len(value)==2
                        and all(type(n) is int and n>=0 for n in value)):
                    return f"{value[0]}–{value[1]}"
                return "Pending verification"
            dates.append({"Sydney match date": row.get("sydney_date"),
                          "Set status": row.get("status"),
                          "First verified final score": score(actual[0]),
                          "Second verified final score": score(actual[1])})
        if dates:
            st.dataframe(dates, hide_index=True, use_container_width=True)
        st.caption("A winning set needs one complete row to match both final regulation "
                   "scores. Available-arm samples can differ; use the matched comparisons "
                   "below. Set Brier scores concern different selected outcome sets and "
                   "are descriptive. Single-score comparisons use the same actual results.")
        st.caption("Lineups are contextual evidence with zero numerical weight. "
                   "A 70% verified hit rate is not established. Betting ROI is unavailable "
                   "without accepted double prices, stakes and returns.")
        comparisons = report.get("comparisons") or report.get("paired_comparisons")
        if comparisons:
            st.write("Matched comparisons on the same available-arm cohort")
            st.json(comparisons)
        rejected = report.get("rejections") or report.get("source_rejections")
        if rejected:
            st.warning("Some saved records failed source validation and were excluded.")
            st.json(rejected)
        st.download_button("Download authenticated prospective comparison", rendered,
                           "professional-paired-comparison.json", "application/json",
                           key="professional_paired_comparison_download")
        st.download_button("Download fixed study protocol", protocol_bytes,
                           "professional-paired-protocol.json", "application/json",
                           key="professional_paired_protocol_download")
        return report
