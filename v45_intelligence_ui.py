"""V45 verification dashboard backed by the repository evidence archives."""
from datetime import datetime, timezone
from functools import lru_cache

from sydney_display import sydney_timestamp
from v45_data_intelligence import evaluate


@lru_cache(maxsize=3)
def _report(minute):
    return evaluate(now=datetime.fromtimestamp(minute * 60, timezone.utc))


def _label(status):
    return str(status or "unavailable").replace("_", " ").capitalize()


def _impact_rows(comparisons):
    sources = {"market": "Bookmaker markets", "historical_xg": "Historical shot xG"}
    cohorts = {"single_fixtures": "Single matches at T−15 min",
               "early_five_doubles": "Early five-double replay",
               "late_five_doubles": "Late five-double replay",
               "archived_v9_fixtures": "Older V9 single matches"}
    rows = []
    for source, results in comparisons.items():
        for cohort, report in results.items():
            models = report["models"]
            loss_key = "score_log_loss" if cohort.endswith("five_doubles") else "exact_log_loss"
            rows.append({"Source": sources[source], "Cohort": cohorts[cohort],
                         "Qualified": report["qualified_inputs"], "Comparable": report["matched"],
                         "Log loss without": models["baseline"][loss_key],
                         "Log loss with": models["with_source"][loss_key],
                         "Hits without": models["baseline"]["hits"] if report["matched"] else None,
                         "Hits with": models["with_source"]["hits"] if report["matched"] else None,
                         "Status": _label(report["status"])})
    return rows


def render(st):
    clock = datetime.now(timezone.utc)
    report = _report(int(clock.timestamp() // 60))
    st.caption(f"Archive checked {sydney_timestamp(report['as_of_utc'])} · Kickoffs in the next 24 hours")
    st.subheader("Match information quality")
    st.caption("Points reflect evidence available at the check time. History 35, historical shot xG 25, bookmaker data 20, cross-checked starting XIs 20. Lineup collection opens 60 minutes before kickoff.")
    fixtures = report["fixtures"]
    if fixtures:
        st.dataframe([
            {"Fixture": f"{row['home']} vs {row['away']}", "Competition": row["competition"],
             "Kickoff": sydney_timestamp(row["kickoff_utc"]), "Quality / 100": row["score"],
             "History": _label(row["statuses"]["history"]), "Historical xG": _label(row["statuses"]["xg"]),
             "Markets": _label(row["statuses"]["market"]), "Starting XIs": _label(row["statuses"]["lineup"])}
            for row in fixtures
        ], hide_index=True, use_container_width=True)
    else:
        st.info("No eligible archived forecast in the next 24 hours.")
    if report["conflicting_fixture_identities"]:
        st.warning(f"Excluded {report['conflicting_fixture_identities']} conflicting fixture identities.")

    st.subheader("Collection and provider access")
    collection = report["collection"]
    capture = collection["capture"]
    st.write(f"Published pre-match collection · {_label(capture['status'])}")
    if capture["timestamp_utc"]:
        st.caption(sydney_timestamp(capture["timestamp_utc"]))
    observation = (report["sportmonks"].get("last_provider_observation") or {})
    if observation.get("reason"):
        st.write(f"Sportmonks historical xG · {_label(observation['reason'])}")
        if observation.get("at_utc"):
            st.caption(f"Last archived provider response · {sydney_timestamp(observation['at_utc'])}")
    probe = report["sportmonks_probe"]
    if probe:
        with st.expander("Sportmonks fixture and xG coverage probe"):
            st.json(probe)
    else:
        st.caption("A separate Sportmonks fixture and xG probe is awaiting configuration or its first published run.")
    lineup = report["lineup_collection"]
    if lineup:
        with st.expander("Latest confirmed-lineup collection run"):
            st.json(lineup)
    else:
        st.caption("No V45 lineup collection run has been published.")

    with st.expander("Historical coverage and source impact"):
        st.caption("Recheck archived history and compare forecasts with and without each numerical source on the same verified matches.")
        if st.button("Audit history and source impact", key="v45_audit"):
            from v45_history import evaluate as history_audit
            from v45_source_impact import evaluate as impact_audit

            with st.spinner("Authenticating receipts and comparing matched forecasts…"):
                st.session_state["v45_audits"] = (history_audit(now=clock), impact_audit(now=clock))
        audits = st.session_state.get("v45_audits")
        if audits:
            history, impact = audits
            st.metric("Recovered historical results", history["recovered_historical_fixtures"])
            st.dataframe([{"Original pair evidence": _label(status), "Count": count}
                          for status, count in history["original_pair_legs"].items()],
                         hide_index=True, use_container_width=True)
            st.write("Matched forecast comparisons")
            st.dataframe(_impact_rows(impact["comparisons"]), hide_index=True, use_container_width=True)
            st.caption("Starting XIs and AI reviews are contextual. Their effect on forecast accuracy remains unmeasured.")
