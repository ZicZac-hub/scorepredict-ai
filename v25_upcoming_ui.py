"""Interactive V25 upcoming two-match five-double generator.

Generated scores are explicit on-demand research, not automatic prospective
shadow winners. Pagination exposes all eligible pairs without API spending.
"""
import csv
from datetime import datetime, timezone
from io import StringIO
import math

from ai_five_double_engine import identity
from v25_upcoming import (
    MAX_PAGE_PAIRS, available, pair_catalog, generate, best_pair,
    rows_for_page, VERSION,
)


def _rows_csv(items):
    output=StringIO()
    columns=("Sydney date","Match 1","Match 2","Double","Score 1","Score 2",
             "Source age warning")
    writer=csv.DictWriter(output,fieldnames=columns,extrasaction="ignore")
    writer.writeheader()
    for item in items:
        writer.writerow(item)
    return output.getvalue().encode("utf-8")


def _save_explicit_selection(st,report):
    """Persist only the single user's chosen five-double set if possible."""
    from ziczac_interactive_capture import pair as audit_pair,store_ui
    try:
        packet=audit_pair(
            report["first_fixture"],report["second_fixture"],
            report["scores"],
            method="v25_professional_analyst",
            now=datetime.fromisoformat(report["generated_at_utc"]))
        store_ui(st,packet)
    except (KeyError,ValueError,TypeError) as error:
        st.warning("V25 research is visible, but permanent verified-result "
                   "tracking could not save this pre-match selection: "+
                   type(error).__name__)


def _show_report(st,report):
    if report.get("status")!="ready":
        st.warning(report.get("reason") or
                   "Cannot produce five verified-source doubles for this pair.")
        return
    st.markdown("### 🧮 V25 Professional Analyst · Five Exact-Score Doubles")
    st.write("**Match 1:** "+report["first"])
    st.write("**Match 2:** "+report["second"])
    st.caption("Model version: "+report["method"]+
               " · Generated before kickoff at "+report["generated_at_utc"])
    left,right=st.columns(2)
    left.metric("Original V8 best-five model mass",
                f"{report['v8_optimal_five_estimated_mass']:.2%}")
    right.metric("V25 robustness criterion",
                 f"{report['v25_lower_envelope_criterion']:.2%}")
    st.caption("Neither number is a verified hit rate or real winning "
               "probability. The V25 value is an uncalibrated lower-envelope "
               "score for ranking candidates, not a true probability bound.")
    if any(report["early_frozen_source"]):
        st.warning("At least one original score model was saved more than 72 "
                   "hours ago. These are early frozen forecast estimates, "
                   "NOT updated lineups, injuries, xG or bookmaker odds.")
    st.caption("Original source timestamps (UTC): "+
               " | ".join(report["source_times_utc"]))
    table=[]
    for i,item in enumerate(report["scores"],1):
        table.append({"Double":i,
                      "Match 1 score":item["score1"],
                      "Match 2 score":item["score2"]})
    st.dataframe(table,use_container_width=True,hide_index=True)
    with st.expander("Compare with original V8 predictions",expanded=False):
        st.dataframe([{"Double":i,"Match 1 score":x["score1"],
                       "Match 2 score":x["score2"]}
                      for i,x in enumerate(report["v8_same_pair_scores"],1)],
                     use_container_width=True,hide_index=True)
    flat=[{"Sydney date":report["sydney_date"],
           "Match 1":report["first"],"Match 2":report["second"],
           "Double":i,"Score 1":x["score1"],"Score 2":x["score2"],
           "Source age warning":any(report["early_frozen_source"])}
          for i,x in enumerate(report["scores"],1)]
    st.download_button("⬇️ Download these V25 five doubles (.csv)",
                       _rows_csv(flat),"ziczac_v25_five_doubles.csv","text/csv",
                       key="zz_v25_one_csv",use_container_width=True)
    st.caption("Predictions are experimental. Five correct-score doubles "
               "have not demonstrated a 70% hit rate. A real result is "
               "counted only after both matches finish and are verified.")


def render(st,rows,histories=None):
    """Fast V25 entrypoint for all saved or freshly modelled upcoming fixtures."""
    from ai_five_double_ui import options,_minute_archived_candidates
    from ai_five_double_engine import pair_check
    import os
    clock=datetime.now(timezone.utc)
    minute=int(clock.timestamp()//60)
    joined=options(rows or [],now=clock,
                   archived=_minute_archived_candidates(minute))
    available_by_day,excluded=available(joined,clock)
    st.markdown("### 🧮 V25 · Predict five doubles for upcoming matches")
    st.caption("Select any qualifying two-match pair, auto-select a researched "
               "pair, or generate five V25 combinations for every eligible "
               "pair, one manageable page at a time. Uses only existing "
               "full pre-kickoff score distributions; no paid AI calls.")
    total=sum(len(items) for items in available_by_day.values())
    st.write(f"**{total}** complete-grid upcoming matches across "
             f"**{len(available_by_day)}** Sydney matchdays.")
    if excluded:
        st.caption("Forecasts excluded from V25 (not match losses): "+
                   ", ".join(f"{name.replace('_',' ')}: {count}"
                             for name,count in sorted(excluded.items())))
    groups={day:items for day,items in available_by_day.items()
            if len(items)>=2}
    if not groups:
        st.info("No date has two eligible, distinct matches with complete "
                "score probability grids yet. This is not a losing prediction. "
                "Try the 'Load fresh live fixtures' switch at the top of the "
                "AI Five-Double page, or run Search & model for Today/Tomorrow "
                "using available authentic API-Football data.")
        return
    day=st.selectbox("📅 Choose upcoming match date (Sydney)",
                     list(groups),key="zz_v25_matchday")
    st.caption("Showing EVERY available future matchday with at least two "
               "complete-grid models, not only the next 24 hours.")
    search=st.text_input("🔎 Filter pairs by team (optional)",
                         key="zz_v25_team_search",placeholder="Team name")
    catalog=pair_catalog(groups[day],clock,search=search)
    if not catalog:
        st.warning("No four-team match pair on this date matches the search.")
        return
    st.write(f"**{len(catalog)}** eligible two-match pairs on {day}. "
             "All pair combinations are browsable.")
    mode=st.radio("Choose V25 prediction mode",[
        "Manual · pick any pair",
        "Automatic · find strongest model-agreement pair",
        "All pairs · generate five doubles in pages"],
        key="zz_v25_mode")
    cache=st.session_state.setdefault("zz_v25_generated",{})
    key=(day,search,mode)
    if mode.startswith("Manual"):
        names=[
            f"{i+1}. {left['title']} / {right['title']} "
            f"({left['key'][1]}, {right['key'][1]})"
            for i,(left,right) in enumerate(catalog)]
        selected=st.selectbox("Choose two upcoming matches",range(len(names)),
                              format_func=lambda i:names[i],key="zz_v25_pair")
        pair=catalog[selected]
        if st.button("🧮 Predict five V25 doubles for these matches",
                     key="zz_v25_manual_run",type="primary",
                     use_container_width=True):
            latest=datetime.now(timezone.utc)
            report=generate(*pair,latest)
            cache[key]={"report":report,"pair_ids":(
                pair[0]["key"],pair[1]["key"])}
            if report.get("status")=="ready":
                _save_explicit_selection(st,report)
        saved=cache.get(key)
        if saved and saved.get("pair_ids")==(
                pair[0]["key"],pair[1]["key"]):
            _show_report(st,saved["report"])
    elif mode.startswith("Automatic"):
        st.caption("Compare goal-model agreement for eligible four-team "
                   "pairs. On very busy dates the first 150 pairs by the "
                   "original V8 coverage proxy are screened; no global "
                   "optimum is claimed for excluded candidates.")
        if st.button("✨ V25 Auto Select + Predict",key="zz_v25_auto_run",
                     type="primary",use_container_width=True):
            latest=datetime.now(timezone.utc)
            chosen=best_pair(catalog,latest)
            cache[key]={"auto":chosen}
            if chosen.get("status")=="ready":
                _save_explicit_selection(st,chosen["report"])
        saved=cache.get(key)
        if saved:
            outcome=saved.get("auto") or {}
            if outcome.get("status")=="ready":
                st.caption(f"Screened {outcome['screened']} of "+
                           str(outcome["total_pairs"])+" eligible pairs."+
                           (" This is a shortlist, not all pairs." if
                            outcome.get("shortlist_only") else ""))
                _show_report(st,outcome["report"])
            else:
                st.warning("No suitable V25 pair could be selected with "
                           "the available model evidence.")
    else:
        pages=math.ceil(len(catalog)/MAX_PAGE_PAIRS)
        page=st.selectbox("Page of eligible match pairs",
                          list(range(1,pages+1)),
                          format_func=lambda n:f"Page {n} of {pages} "
                                               f"(up to {MAX_PAGE_PAIRS} pairs)",
                          key="zz_v25_page")
        st.caption("Each page covers up to 15 different two-match combinations "
                   "(five score doubles per pair). Select later pages to "
                   "generate the rest; no pair is hidden or fabricated.")
        if st.button("📋 Generate five V25 doubles for this page",
                     key="zz_v25_batch_run",type="primary",
                     use_container_width=True):
            latest=datetime.now(timezone.utc)
            result=rows_for_page(catalog,page-1,latest)
            cache[key]={"page":page,"result":result}
        saved=cache.get(key)
        if saved and saved.get("page")==page:
            result=saved["result"]
            if result["rows"]:
                st.write(f"**{len(result['rows'])}** generated double "
                         f"combinations, from up to {result['shown_pairs']} "
                         "distinct match pairs.")
                st.dataframe(result["rows"],use_container_width=True,
                             hide_index=True)
                st.download_button(
                    "⬇️ Download this page of V25 doubles (.csv)",
                    _rows_csv(result["rows"]),
                    f"ziczac_v25_pairs_{day}_page{page}.csv","text/csv",
                    key="zz_v25_batch_csv",use_container_width=True)
            else:
                st.info("No pair on this page passed the current evidence "
                        "and future-kickoff check.")
            st.caption("Batch results are for research and export. They are "
                       "not 15 automatically placed bets, and multiple "
                       "overlapping pairs must not inflate verified hit-rate statistics.")
    st.caption("V25 is experimental; model-robust score ranking has not been "
               "independently proven more accurate than V8. "
               "Each pair remains subject to the same Sydney date, four "
               "distinct teams and pre-kickoff rules. Historical forecasts "
               "are never rewritten.")
