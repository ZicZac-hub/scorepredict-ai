"""Auditable model comparison UI; never fabricate a benchmark sample."""
import json
from model_comparison_audit import benchmark

def render(st):
    st.header("📐 Verified model accuracy comparison")
    st.caption("Compare frozen forecasts for identical settled fixtures. "
               "Only forecasts timestamped before kickoff count. "
               "This page does not alter or promote the live prediction model.")
    st.success("Model Benchmark is open · v2.3")
    st.write("This page compares verified pre-match forecasts. You can scan the archive or upload a JSON export.")
    scan=st.button("🔎 Scan saved prediction archive",key="zz_benchmark_scan",use_container_width=True)
    uploaded=st.file_uploader("Upload a JSON list of verified fixture forecasts",
                              type=["json"],key="zz_benchmark_json")
    with st.expander("Required JSON record format"):
        st.code('[{"fixture_id":"api:123","kickoff":"2026-11-20T19:00:00Z",'
                '"actual":[1,0],"forecasts":{"baseline":{"created_at":'
                '"2026-11-20T12:00:00Z","scores":[[1,0,0.15],[0,0,0.12]]}}}]',
                language="json")
    from benchmark_archive_loader import load_archive
    if uploaded is None and not scan and not st.session_state.get("zz_benchmark_scan_done"):
        st.info("Press Scan saved prediction archive to begin. The scan may take time on large archives.")
        return
    if uploaded is None:
        st.session_state["zz_benchmark_scan_done"]=True
        @st.cache_data(ttl=1800,show_spinner=False)
        def read_local_records():
            return load_archive(".")
        try:
            with st.spinner("Reading compatible saved forecast records…"):
                rows,diagnostics=read_local_records()
        except Exception as exc:
            st.error("Archive scan failed ("+type(exc).__name__+"). You can still upload JSON.")
            return
        st.caption("Read-only local archive scan: "+str(diagnostics))
        if not rows:
            st.info("No archive records use the benchmark's verified fixture schema. "
                    "The existing forecast archive may require a format-specific "
                    "adapter. No results or timestamps are inferred.")
            return
    else:
        try:
            rows=json.load(uploaded)
            if not isinstance(rows,list) or len(rows)>100000:
                raise ValueError("Expected a JSON array with at most 100,000 records")
        except (ValueError,TypeError,KeyError,UnicodeError) as exc:
            st.error("Invalid benchmark input: "+str(exc)[:160])
            return
    report=benchmark(rows)
    if not report["models"]:
        st.warning("No valid settled, pre-kickoff model predictions found.")
        st.json(report["rejected"])
        return
    # Paired cohorts are mandatory for a fair cross-model leaderboard.
    from model_comparison_audit import paired_benchmark
    paired=paired_benchmark(rows)
    st.subheader("Same-fixture head-to-head comparison")
    if paired["models"]:
        st.dataframe([{"Model":name,**stats} for name,stats in paired["models"].items()],
                     use_container_width=True,hide_index=True)
    else:
        st.warning("No shared valid fixture cohort across models. "
                   "Individual model rates below are NOT directly comparable.")
    st.caption("All-model shared fixtures: "+str(paired["shared_fixtures"]))
    st.subheader("Individual model diagnostic (unpaired)")
    st.dataframe([{"Model":name,**stats} for name,stats in report["models"].items()],
                 use_container_width=True,hide_index=True)
    st.caption("Rejections: "+str(report["rejected"]))
    st.warning("No model is automatically promoted. Five-double accuracy "
               "requires separately verified paired-fixture bet-set outcomes.")
