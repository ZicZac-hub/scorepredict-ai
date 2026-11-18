"""Auditable, automatically refreshed same-fixture benchmark UI."""
import json

from model_comparison_audit import benchmark, paired_benchmark


def render(st):
    st.header("📐 Verified model accuracy comparison")
    from professional_research_ui import render as render_research
    render_research(st)
    st.caption(
        "Original frozen forecasts, matched against independent regulation-time "
        "final scores. Incomplete games and retrospective rescores are never "
        "counted. This page does not change the live prediction model."
    )
    st.success("Model Benchmark · v2.5 · automatic multi-ledger verification")

    from benchmark_archive_loader import load_archive

    @st.cache_data(ttl=300, show_spinner=False)
    def read_verified_v25_records():
        return load_archive(".")

    if st.button("🔄 Refresh verified benchmark", key="zz_benchmark_refresh",
                 use_container_width=True):
        read_verified_v25_records.clear()

    with st.expander("Optional: import a separate verified benchmark JSON export"):
        uploaded = st.file_uploader(
            "Upload a JSON list of verified fixture forecasts",
            type=["json"], key="zz_benchmark_json")
        st.code(
            '[{"fixture_id":"api:123","kickoff":"2026-11-20T19:00:00Z",'
            '"actual":[1,0],"forecasts":{"baseline":{"created_at":'
            '"2026-11-20T12:00:00Z","scores":[[1,0,0.15],[0,0,0.12]]}}}]',
            language="json")

    if uploaded is None:
        try:
            with st.spinner("Reading verified results and saved pre-match forecasts…"):
                rows, diagnostics = read_verified_v25_records()
        except Exception as exc:
            st.error("Automatic benchmark scan failed (" + type(exc).__name__ + ").")
            return
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Saved fixtures", diagnostics["archived_fixture_ids"])
        m2.metric("Verified results", diagnostics["verified_fixtures"])
        m3.metric("Benchmarked fixtures", diagnostics["compatible_records"])
        m4.metric("Awaiting results", diagnostics["awaiting_verified_results"])
        st.caption(
            "Automatically reads the V10 and prospective forecast ledgers, "
            "both verified-result files and verified ledger settlements. "
            "Refreshed on page opening and cached for up to five minutes."
        )
        if diagnostics["conflicting_result_fixtures"]:
            st.error(
                str(diagnostics["conflicting_result_fixtures"]) +
                " fixture(s) have conflicting independently saved final scores "
                "and were excluded pending review."
            )
        if diagnostics["truncated"]:
            st.warning("Safety scan limit reached. This is a partial benchmark.")
        if diagnostics["unreadable_files"] or diagnostics["unreadable_results"]:
            st.warning("Some archived records could not be read. Check diagnostics.")
        with st.expander("Archive diagnostics"):
            st.json(diagnostics)
        if not rows:
            st.info(
                "No verified pre-kickoff forecasts are eligible yet. "
                "Fixtures awaiting a verified final score remain pending; "
                "they are not classified as prediction losses."
            )
            return
    else:
        try:
            rows = json.load(uploaded)
            if not isinstance(rows, list) or len(rows) > 100000:
                raise ValueError("Expected a JSON array of at most 100,000 records")
        except (ValueError, TypeError, UnicodeError) as exc:
            st.error("Invalid benchmark input: " + str(exc)[:160])
            return
        st.info("Showing your imported JSON data, not the automatic live archive.")

    report = benchmark(rows)
    if not report["models"]:
        st.warning("No valid settled, pre-kickoff model predictions found.")
        st.json(report["rejected"])
        return

    paired = paired_benchmark(rows)
    st.subheader("Same-fixture model comparison")
    if paired["models"]:
        st.dataframe(
            [{"Model": name, **stats} for name, stats in paired["models"].items()],
            use_container_width=True, hide_index=True,
        )
    else:
        st.warning(
            "Models do not yet share enough valid finished fixtures for an "
            "all-model head-to-head comparison. Individual rates are not "
            "directly comparable."
        )
    st.caption("All-model shared finished fixtures: " + str(paired["shared_fixtures"]))

    st.subheader("Individual model diagnostic (unpaired)")
    st.dataframe(
        [{"Model": name, **stats} for name, stats in report["models"].items()],
        use_container_width=True, hide_index=True,
    )
    st.caption("Rejected benchmark entries: " + str(report["rejected"]))
    st.warning(
        "No model is automatically promoted. Five-double success requires "
        "separately verified, frozen two-fixture bet sets; single-score "
        "accuracy does not measure double success."
    )
