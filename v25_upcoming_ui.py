"""Interactive V25 upcoming two-match five-double generator.

Generated scores are explicit on-demand research, not automatic prospective
shadow winners. Pagination exposes all eligible pairs without API spending.
"""
import csv
from datetime import datetime, timezone, timedelta, date
from io import StringIO
from sydney_display import sydney_timestamp
import math

from ai_five_double_engine import identity
from v25_upcoming import (
    MAX_PAGE_PAIRS, available, pair_catalog, generate, best_pair,
    rows_for_page, VERSION, AUDIT_METHOD, automatic_quality,
)
from v25_goal_validation import historical_reality_check
from v25_date_discovery import (
    SYDNEY, visible_days, date_label, merge_modelled, fixture_table,
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
            method=AUDIT_METHOD,
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
    st.caption("Kickoff: "+sydney_timestamp(report["first_fixture"].get("utcDate")))
    st.write("**Match 2:** "+report["second"])
    st.caption("Kickoff: "+sydney_timestamp(report["second_fixture"].get("utcDate")))
    st.caption("Model version: "+report["method"]+
               " · Generated before kickoff at "+sydney_timestamp(report["generated_at_utc"]))
    left,right=st.columns(2)
    left.metric("Original V8 best-five model mass",
                f"{report['v8_optimal_five_estimated_mass']:.2%}")
    right.metric("V25 robustness criterion",
                 f"{report['v25_lower_envelope_criterion']:.2%}")
    # Compare the SAME V8 distribution for both selections. The robustness
    # criterion is a different uncalibrated measure, not a winning chance.
    st.metric("V25 selected-five mass under original V8 model",
              f"{report['v25_five_original_v8_mass']:.2%}")
    st.caption("The two V8 model-mass figures are directly comparable; "
               "the V25 robustness criterion is not. None is a verified hit "
               "rate or a real winning probability. The robustness criterion "
               "is an uncalibrated model-ranking score, not a probability bound.")
    frontier = report.get("five_double_seventy_percent_model_feasibility") or {}
    if frontier:
        needed = frontier.get("minimum_distinct_combinations_for_70pct_model_mass")
        st.markdown("#### 70% five-double feasibility check")
        st.write({
            "Maximum model-estimated coverage using five V8-score combinations":
                f"{frontier['max_five_model_mass_from_original_v8']:.2%}",
            "Combinations estimated for 70% model coverage":
                needed if needed is not None else "Not established",
            "Five combinations meet 70% model coverage":
                frontier["five_original_v8_choices_reach_70pct_mass"],
        })
        st.caption(
            "This is a mathematical comparison within the original frozen "
            "V8 distribution, not the observed V25 win rate. Increasing "
            "the number of combinations changes the five-double format. "
            "The five proposed V25 predictions are unchanged."
        )
    screening=automatic_quality(report)
    if not screening["passed"]:
        st.warning("Cautious auto-selection would abstain on this pair: "+
                   ", ".join(reason.replace("_"," ") for reason in screening["reasons"])+
                   ". These five scores are exploratory only.")
    if any(report["early_frozen_source"]):
        st.warning("At least one original score model was saved more than 72 "
                   "hours ago. These are early frozen forecast estimates, "
                   "NOT updated lineups, injuries, xG or bookmaker odds.")
    st.caption("Original source timestamps (Sydney time): "+
               " | ".join(sydney_timestamp(value) for value in report["source_times_utc"]))
    with st.expander("🔬 Verified evidence delivery · both matches",expanded=False):
        st.caption("This audits original fixture IDs, forecast digests, source "
                   "timestamps, odds, expected-goal history and reported starting XIs. "
                   "Missing means unverified, not zero or a clean injury report.")
        for idx,leg in enumerate(report.get("source_evidence_audit") or [],1):
            st.markdown("**Match "+str(idx)+": "+str(leg.get("fixture") or "Unknown")+"**")
            st.write({
                "Matched provider evidence":leg.get("matching_provider_packet") is True,
                "Fresh timestamped bookmaker market":leg.get("verified_pre_match_market") is True,
                "Historical source xG for both teams":leg.get("verified_historical_xg_both_teams") is True,
                "Verified xG fixture counts (home/away)":leg.get("verified_historical_xg_fixtures_per_team"),
                "Both provider-reported starting XIs":leg.get("confirmed_starting_xi_both_teams") is True,
                "Evidence status":str(leg.get("reason") or "not checked").replace("_"," "),
            })
        spot=report.get("live_market_evidence") or {}
        if spot:
            st.write({"Separately checked live API odds":spot.get("both_verified") is True,
                      "Exact fixture-ID / kickoff matching results":spot.get("legs") or [],
                      "Not applied to score probabilities":True})
        st.caption("None of these external signals is numerically weighted in V25 "
                   "until a separate pre-kickoff-trained challenger shows independent "
                   "improvement. V25 scores still come from the V8 goal grid.")
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
    from ai_five_double_ui import options, _minute_archived_candidates
    from ai_five_live_finder import research_day
    import os

    clock=datetime.now(timezone.utc)
    local_today=clock.astimezone(SYDNEY).date()
    minute=int(clock.timestamp()//60)
    research=st.session_state.setdefault("zz_v25_discovered_by_day",{})
    all_session_rows=[
        result for day,stored in research.items()
        if day>=local_today.isoformat()
        for result in stored.get("rows",[])
    ]

    def merged_forecasts():
        return options(list(rows or [])+all_session_rows,now=clock,
                       archived=_minute_archived_candidates(minute))

    joined=merged_forecasts()
    available_by_day,excluded=available(joined,clock)
    st.markdown("### 🧮 V25 · Predict five doubles for upcoming matches")
    with st.expander("📉 Historical model limits (read-only)",expanded=False):
        history=historical_reality_check()
        if history.get("status")!="ready":
            st.caption("No complete validated historical replay is available. "
                       "Do not infer a five-double hit rate.")
        else:
            st.write("**Retrospective single-match sample:** "+
                     str(history["retrospective_single_fixtures"])+" fixtures")
            st.write("**Observed 4+ goal share:** "+
                     f'{history["observed_high_goal_share"]:.1%}' +
                     " · **Model-predicted:** "+
                     f'{history["predicted_high_goal_mass"]:.1%}')
            st.write("**V8 exact-score doubles in historical replay:** "+
                     str(history["retrospective_five_double_wins"])+"/"+
                     str(history["retrospective_five_double_issued"]))
            st.caption("This is development/backtest evidence, NOT independent "
                       "future accuracy. The aggregate high-goal data do not "
                       "justify an arbitrary high-score boost. Goal-model "
                       "coefficients and frozen predictions remain unchanged.")
    st.caption("Choose **Today**, **Tomorrow**, or any upcoming Sydney matchday. "
               "All provider-confirmed future fixtures can be listed even when "
               "they lack sufficient historical data to predict exact scores. "
               "V25 does not invent missing score probability grids.")

    total=sum(len(items) for items in available_by_day.values())
    st.write(f"**{total}** complete-grid upcoming matches across "
             f"**{len(available_by_day)}** Sydney matchdays.")
    if excluded:
        st.caption("Source forecasts excluded from V25 (not match losses): "+
                   ", ".join(f"{name.replace('_',' ')}: {count}"
                             for name,count in sorted(excluded.items())))
    days=visible_days(clock,available_by_day)
    def _go_to_date(day_value):
        st.session_state["zz_v25_matchday"]=day_value

    today_button,tomorrow_button=st.columns(2)
    today_button.button("📍 Today · "+local_today.strftime("%d %b"),
                        on_click=_go_to_date,args=(local_today.isoformat(),),
                        key="zz_v25_go_today",use_container_width=True)
    tomorrow=local_today+timedelta(days=1)
    tomorrow_button.button("🗓 Tomorrow · "+tomorrow.strftime("%d %b"),
                           on_click=_go_to_date,args=(tomorrow.isoformat(),),
                           key="zz_v25_go_tomorrow",use_container_width=True)

    day=st.selectbox(
        "📅 Choose upcoming match date (Sydney)",
        days, key="zz_v25_matchday",
        format_func=lambda value:date_label(
            value,clock,
            eligible_count=len(available_by_day.get(value,[])),
            discovered_count=len((research.get(value) or {}).get("fixtures") or [])),
        help="Always starts with today's Sydney date. A day with no V25 "
             "score model is still available for fixture discovery.")

    st.caption("Fixture discovery uses API-Football's Sydney-timezone date "
               "filter. Tap below to look up the actual fixtures for the "
               "selected date; the provider's available matches may differ "
               "from the already-modelled forecast archive.")
    batch_size=st.select_slider(
        "New fixture model attempts per search (uses provider API allowance)",
        options=[4,8,12],value=4,key="zz_v25_discovery_batch",
        help="Four remains the default. Larger batches can use much more API-Football quota; "
             "a fixture without sufficient real history stays unmodelled.")
    requested=st.button(
        "🔎 Find & model matches · "+date.fromisoformat(day).strftime("%d %b"),
        key="zz_v25_discover_day",type="primary",use_container_width=True,
        help="Makes one date-scoped API-Football fixture request and up to "
             "the selected number of bounded model attempts. History and "
             "team-context calls may use API quota; no paid AI calls.")
    if requested:
        try:
            key=str(st.secrets.get("API_FOOTBALL_KEY") or
                    os.environ.get("API_FOOTBALL_KEY") or "").strip()
        except (AttributeError,OSError,KeyError,RuntimeError):
            key=str(os.environ.get("API_FOOTBALL_KEY") or "").strip()
        if not key:
            st.error("API_FOOTBALL_KEY is unavailable in Streamlit Secrets. "
                     "Cannot confirm missing today's fixtures; saved forecast "
                     "models remain available.")
        else:
            previous=research.get(day) or {}
            if (previous.get("last") or {}).get("quota_limited"):
                st.warning("The last provider search hit a rate limit or quota. "
                           "Additional searches for this date are paused for "
                           "this session to avoid repeated API costs.")
            else:
                with st.spinner("Checking real upcoming fixtures and modelling "
                                f"up to {batch_size} previously unchecked matches..."):
                    outcome=research_day(
                        key,date.fromisoformat(day),
                        known_ids=previous.get("checked_ids") or [],
                        limit=batch_size,now=clock)
                new_rows=merge_modelled(previous.get("rows") or [],
                                       outcome.get("modeled") or [])
                checked=list(dict.fromkeys(
                    list(previous.get("checked_ids") or [])+
                    list(outcome.get("checked_ids") or [])))
                # An empty or quota-limited search never erases an earlier
                # verified fixture list for this same date.
                fixtures=outcome.get("fixtures") or previous.get("fixtures") or []
                research[day]={
                    "rows":new_rows,"checked_ids":checked,"fixtures":fixtures,
                    "last":{k:v for k,v in outcome.items()
                            if k not in ("modeled","checked_ids","fixtures")},
                }
                # The newly modelled rows must take effect on the SAME click.
                all_session_rows[:]=[
                    row for date_key,record in research.items()
                    if date_key>=local_today.isoformat()
                    for row in record.get("rows",[])
                ]
                joined=merged_forecasts()
                available_by_day,excluded=available(joined,clock)

    saved=research.get(day) or {}
    last=saved.get("last") or {}
    if last:
        if last.get("quota_limited"):
            st.error("API-Football quota or rate limit prevented a complete "
                     "search. Zero returned fixtures does NOT mean today's "
                     "football schedule is empty.")
        else:
            st.caption("Source search "+day+": "+
                       str(last.get("discovered",0))+" upcoming fixtures · "+
                       str(len(saved.get("checked_ids") or []))+" attempted "+
                       "match models · "+
                       str(len(saved.get("rows") or []))+" modelled with "+
                       "available statistical evidence · "+
                       str(last.get("remaining",0))+
                       " not yet attempted.")
        for message in (last.get("errors") or [])[:3]:
            st.caption("Provider/model limitation: "+str(message)[:190])
        if last.get("remaining",0):
            st.caption("Tap Find & model again to process another selected "
                       "batch of previously unchecked fixtures. Historical "
                       "evidence and API quota determine which become V25-ready.")

    chosen=available_by_day.get(day,[])
    all_fixtures=fixture_table(day,chosen,saved.get("fixtures") or [])
    st.write(f"**{len(all_fixtures)}** genuine upcoming matches identified "
             f"for {day} · **{len(chosen)}** with complete V25 score grids.")
    if all_fixtures:
        st.dataframe(all_fixtures,use_container_width=True,hide_index=True)
    elif day<=tomorrow.isoformat() and not last:
        st.info("The existing saved models do not yet cover this date. "
                "That does **not** mean no games are scheduled. Tap "
                "**Find & model matches** to discover today's or tomorrow's "
                "genuine fixtures using the connected provider.")

    if len(chosen)<2:
        st.info("No two eligible full-grid score forecasts on this selected "
                "Sydney date yet. Fixtures with incomplete historical evidence "
                "are listed above but cannot honestly be given five exact-score "
                "doubles. Choose another date or use the discovery button.")
        return

    search=st.text_input("🔎 Filter pairs by team (optional)",
                         key="zz_v25_team_search",placeholder="Team name")
    catalog=pair_catalog(chosen,clock,search=search)
    if not catalog:
        st.warning("No eligible two-match pair of four distinct teams "
                   "matches this date and search.")
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
        cautious=st.checkbox(
            "Cautious auto-select: abstain unless both legs have recent model "
            "sources, genuine fresh source-matched odds and at least 8% V8 "
            "five-score model mass (not a verified success probability)",
            value=True,key="zz_v25_cautious",help="An experimental screening "
            "policy; stricter coverage is not proven to improve actual hit rate. "
            "Uncheck for research-only best available pair.")
        key=(day,search,mode,cautious)
        st.caption("Compare goal-model agreement for eligible four-team "
                   "pairs. On very busy dates the first 150 pairs by the "
                   "original V8 coverage proxy are screened; no global "
                   "optimum is claimed for excluded candidates.")
        if st.button("✨ V25 Auto Select + Predict",key="zz_v25_auto_run",
                     type="primary",use_container_width=True):
            latest=datetime.now(timezone.utc)
            chosen=best_pair(catalog,latest,cautious=cautious)
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
                st.warning(outcome.get("reason") or
                           "No suitable V25 pair could be selected.")
                if outcome.get("quality_exclusions"):
                    st.write({"Cautious exclusion reasons (may overlap)":
                              {k.replace("_"," "):v for k,v in
                               outcome["quality_exclusions"].items()}})
                if outcome.get("highest_v8_estimated_mass") is not None:
                    st.caption("Highest V8 top-five mass checked: "+
                               f'{outcome["highest_v8_estimated_mass"]:.2%}' +
                               ". Model estimate only, not observed hit rate.")
                alternative=outcome.get("research_alternative")
                if alternative:
                    with st.expander("Strongest rejected pair · optional live market check",
                                     expanded=False):
                        st.warning("This pair failed the cautious checks: "+
                                   ", ".join(alternative["failed_checks"])+
                                   ". The five scorelines are research-only until "
                                   "the original checks genuinely pass.")
                        st.caption("Optional check: up to TWO API-Football odds "
                                   "requests to verify source fixture IDs, kickoffs, "
                                   "quote types and timestamps. These prices do not "
                                   "change V25 scores or prove higher accuracy.")
                        if st.button("🔎 Check current odds for both legs",
                                     key="zz_v25_check_live_odds"):
                            try:
                                api_key=str(st.secrets.get("API_FOOTBALL_KEY") or
                                            os.environ.get("API_FOOTBALL_KEY") or "").strip()
                            except (AttributeError,OSError,KeyError,RuntimeError):
                                api_key=str(os.environ.get("API_FOOTBALL_KEY") or "").strip()
                            if not api_key:
                                st.error("API_FOOTBALL_KEY is missing; cannot "
                                         "verify live bookmaker markets.")
                            else:
                                from v25_live_market_check import inspect_pair
                                now_check=datetime.now(timezone.utc)
                                live=inspect_pair(api_key,alternative["report"],
                                                  now=now_check)
                                checked=dict(alternative["report"])
                                checked["live_market_evidence"]=live
                                quality=automatic_quality(checked)
                                cache[key]["live_market_check"]={
                                    "report":checked,"quality":quality}
                                if quality["passed"]:
                                    _save_explicit_selection(st,checked)
                        checked=(cache.get(key) or {}).get("live_market_check")
                        if checked:
                            audit=checked["report"]["live_market_evidence"]
                            if checked["quality"]["passed"]:
                                st.success("Both genuine markets passed the source "
                                           "and model checks. This remains an "
                                           "experimental five-double research set, "
                                           "not a demonstrated winning rate.")
                            else:
                                st.warning("Source checks still prevent cautious "
                                           "selection: "+", ".join(
                                               checked["quality"]["reasons"]))
                            st.write({"Live market spot-check":audit.get("legs") or [],
                                      "Provider requests attempted":audit.get(
                                          "provider_requests_attempted"),
                                      "Odds NOT used to change score estimates":True})
                        _show_report(st,checked["report"] if checked
                                     else alternative["report"])
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
