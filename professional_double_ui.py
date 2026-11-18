"""Manual probability optimizer and evidence-gated premium selection."""
from datetime import datetime, timezone

from professional_double_optimizer import POLICY, manual, premium, compare_score_prices
from sydney_display import sydney_timestamp


def _source_rows(report, candidates):
    """Recover the exact selected fixture rows, rather than the first pair shown."""
    from v10_prospective import utc
    selected=[]
    for fixture in (report["first_fixture"],report["second_fixture"]):
        key=(str(fixture.get("provider") or "football-data.org"),str(fixture["id"]),
             utc(fixture["utcDate"]))
        match=next((item["row"] for item in candidates if (
            str(item["row"]["fixture"].get("provider") or "football-data.org"),
            str(item["row"]["fixture"]["id"]),
            utc(item["row"]["fixture"]["utcDate"]))==key),None)
        if match is None:
            raise ValueError("Selected fixture no longer matches the current source rows")
        selected.append(match)
    return selected


def _display_report(st, report, source_rows, histories=None, *, heading=None):
    """Render and attempt an authenticated durable save for every ready mode."""
    from ai_five_double_ui import label, repository_provider_packet
    from v10_prospective import utc
    try:
        started=any(datetime.now(timezone.utc)>=utc(fixture["utcDate"])
                    for fixture in (report["first_fixture"],report["second_fixture"]))
    except (ValueError,TypeError,KeyError):
        st.warning("The selected pair has invalid kickoff provenance; no prediction "
                   "was displayed or saved.")
        return
    if started:
        st.warning("A selected match has started; no new prediction was displayed or saved.")
        return
    if heading:
        st.markdown("#### "+heading)
    first,second=source_rows
    st.caption("Match 1: "+label(first))
    st.caption("Match 2: "+label(second))
    st.metric("Uncalibrated model coverage", f"{report['estimated_coverage']:.2%}")
    if report.get("coverage_is_calibrated"):
        estimate=(report.get("calibration") or {}).get("calibrated_coverage_estimate")
        if estimate is not None:
            st.metric("Calibrated five-combination coverage estimate",f"{estimate:.2%}")
            st.caption("Calibration applies to the combined set; individual exact-score "
                       "probabilities below remain model estimates.")
    st.caption("Estimated probability is separate from observed accuracy, bookmaker value "
               "and ROI. A 70% verified hit rate is not established.")
    st.caption("Generated: "+sydney_timestamp(report["generated_at_utc"]))
    st.dataframe([{"Double": n, "Match 1": p["score1"], "Match 2": p["score2"],
                   "Joint model probability": f"{p['joint_probability']:.3%}"}
                  for n, p in enumerate(report["choices"], 1)], hide_index=True,
                 use_container_width=True)
    quotes=[]
    for row in source_rows:
        try:
            packet=repository_provider_packet(row["fixture"],now=report["generated_at_utc"],
                                              priority="research")
            observed_quotes=(packet or {}).get("correct_score_market_quotes")
            # Current source packets save the quote list directly. Older
            # parser envelopes may still wrap it under ``quotes``.
            if isinstance(observed_quotes,dict):
                observed_quotes=observed_quotes.get("quotes")
            if isinstance(observed_quotes,list):
                quotes.extend(quote for quote in observed_quotes if isinstance(quote,dict))
        except (KeyError,ValueError,TypeError,OSError):
            continue
    prices=compare_score_prices(report,quotes,now=report["generated_at_utc"])
    if prices["available"]:
        with st.expander("Source-backed score prices · illustrative value comparison"):
            st.caption("Single-leg price products are not verified executable double quotes. "
                       "Expected return uses uncalibrated per-score probabilities; actual ROI "
                       "is unavailable.")
            st.dataframe(prices["comparisons"],hide_index=True,use_container_width=True)
    else:
        st.caption("Matched, fresh correct-score prices are unavailable; expected value "
                   "and realized betting ROI cannot be verified.")
    from ziczac_interactive_capture import pair, store_ui
    try:
        if any(datetime.now(timezone.utc)>=utc(fixture["utcDate"])
               for fixture in (report["first_fixture"],report["second_fixture"])):
            raise ValueError("Durable capture cannot begin after kickoff")
        metadata={k:report[k] for k in
                  ("estimated_coverage","model_versions","source_times_utc","selection_scope")}
        if report.get("source_forecast_refs") is not None:
            metadata["source_forecast_refs"]=report["source_forecast_refs"]
        saved=pair(report["first_fixture"],report["second_fixture"],report["choices"],
                   method=POLICY,now=report["generated_at_utc"],policy_metadata=metadata)
        store_ui(st,saved)
    except (ValueError,TypeError,KeyError,RuntimeError):
        st.warning("The forecast is displayed but was not saved: its prediction or "
                   "provenance failed validation. No durable save is claimed.")
    for row in source_rows:
        try:
            from professional_match_evidence import build_dossier, feature_rows
            dossier=build_dossier(row["fixture"],row.get("past") or
                                  (histories or {}).get(row.get("league"),[]),
                                  as_of=report["generated_at_utc"],statistical=row["result"])
            with st.expander("Evidence: "+label(row)):
                st.dataframe(feature_rows(dossier),hide_index=True,use_container_width=True)
        except (ValueError,TypeError,KeyError):
            st.caption("Dated source evidence is incomplete for this fixture.")


def render(st, rows, histories=None):
    from ai_five_double_ui import options, label
    from v25_upcoming import available, pair_catalog
    clock = datetime.now(timezone.utc)
    found = options(rows, now=clock)
    days, excluded = available(found, clock)
    st.markdown("### Professional five-double probability optimizer")
    st.caption("Five distinct combinations maximize the supplied score models' covered "
               "probability under the independent-match assumption. This method keeps "
               "the original models and saved predictions intact.")
    if not days:
        st.info("No upcoming complete score distributions are available. Find and model "
                "fixtures in the existing All Upcoming Matches engine first.")
        return
    day = st.selectbox("Sydney matchday", list(days), key="professional_pair_day")
    candidates = days[day]
    mode = st.radio("Selection mode", ["All upcoming · manual", "Premium AI Auto Select"],
                    key="professional_pair_mode")
    if mode.startswith("Premium"):
        st.caption("Premium selection requires recent source timestamps and sufficient "
                   "prior verified automatic trials of this exact policy showing calibration "
                   "skill. Manually chosen pairs, V25 sensitivity results and unrelated "
                   "models cannot qualify it.")
        trials=[]
        from on_demand_ai import private_gate_ready
        if private_gate_ready(st):
            import os
            from ziczac_interactive_capture import load
            from ziczac_unified_audit import _collect_finals
            from professional_double_optimizer import trials_from_verified_ledger
            try:
                address=st.secrets.get("DATABASE_URL") or os.environ.get("DATABASE_URL","")
            except (OSError,KeyError,RuntimeError,AttributeError):
                address=os.environ.get("DATABASE_URL","")
            if address:
                records,state=load(address)
                if state=="connected":
                    from pathlib import Path
                    finals,_=_collect_finals(Path("data/v10_prospective"),Path("data/prematch"))
                    trials=trials_from_verified_ledger(records,finals)
        try:
            from professional_double_shadow import load_verified_trials
            trials.extend(load_verified_trials() or [])
        except (ImportError,OSError,ValueError,TypeError,KeyError):
            st.caption("Public automatic research trials are not yet available for calibration.")
        outcome=premium([item["row"] for item in candidates],
                        now=datetime.now(timezone.utc),verified_trials=trials)
        if outcome["status"] != "ready":
            st.warning("Premium Auto Select abstained: validated calibration evidence "
                       "for this policy is insufficient.")
            st.write({"same_day_pairs_checked": outcome["eligible_pairs_compared"],
                      "exclusions": outcome["rejections"]})
            if not st.button("Generate automatic research forecast for validation",
                             key="professional_auto_research_generate"):
                return
            from professional_double_optimizer import auto_shadow
            research=auto_shadow([item["row"] for item in candidates],
                                 now=datetime.now(timezone.utc))
            if research["status"] != "ready":
                st.warning("Automatic research abstained: no eligible supported pair is available.")
                return
            report=research["selection"]
            st.caption("Automatic research only: this uncalibrated forecast collects future "
                       "validation evidence. It is not a qualified Premium pick.")
            heading="Automatic research forecast for validation"
        else:
            report=outcome["selection"]
            heading="Premium automatic selection"
        try:
            selected=_source_rows(report,candidates)
        except (ValueError,TypeError,KeyError):
            st.warning("The selected pair could not be matched to its source fixtures; "
                       "no prediction was displayed or saved.")
            return
        _display_report(st,report,selected,histories,heading=heading)
        return
    catalog = pair_catalog(candidates, clock)
    if not catalog:
        st.info("This day has no eligible pair with four distinct teams.")
        return
    i = st.selectbox("Choose any supported two-match pair", range(len(catalog)),
                     format_func=lambda n: label(catalog[n][0]["row"])+" / "+
                                           label(catalog[n][1]["row"]),
                     key="professional_pair_choice")
    if not st.button("Generate maximum-coverage five doubles", key="professional_pair_generate"):
        return
    first, second = (item["row"] for item in catalog[i])
    report = manual(first, second, now=datetime.now(timezone.utc))
    if report["status"] != "ready":
        st.warning(report["reason"])
        return
    _display_report(st,report,(first,second),histories)
