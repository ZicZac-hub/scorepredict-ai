"""Compact real-result board for all saved ZicZac forecast sources."""
from sydney_display import sydney_timestamp
import csv
import io
import os
from pathlib import Path

from persist_prematch import load_json
from ziczac_unified_audit import build_with_private


def _label(record):
    names=[]
    for leg in record.get("legs") or []:
        home=leg.get("home_team") or "Home"
        away=leg.get("away_team") or "Away"
        names.append(home+" vs "+away)
    return " + ".join(names)


def _export(records):
    out=io.StringIO()
    writer=csv.writer(out)
    writer.writerow(["Source","Type","Match","Kickoff (Sydney)","Saved (Sydney)","Kickoff UTC","Saved UTC",
                     "Original exact score(s)","Verified final score(s)","Status"])
    for row in records:
        writer.writerow([row.get("source"),row.get("kind"),_label(row),
                         sydney_timestamp(row.get("scheduled_kickoff_utc")),
                         sydney_timestamp(row.get("captured_at_utc")),
                         row.get("scheduled_kickoff_utc"),row.get("captured_at_utc"),
                         " / ".join(" + ".join(c) for c in row.get("choices") or []),
                         " + ".join(row.get("actual_scores") or []),
                         row.get("status")])
    return out.getvalue().encode("utf-8")


def render(st,database_url=""):
    st.markdown("### 🧾 All Predictions · Verified HIT / MISS Tracker")
    st.caption("One original pre-match record for every eligible model, "
               "Deep Research run, individual AI provider vote and five-double set. "
               "Only provider-verified regulation-time scores are graded.")
    public=load_json(Path("data/tracking/status.json"),{})
    if not isinstance(public,dict):
        public={}
    from on_demand_ai import private_gate_ready
    from ziczac_unified_audit import build as build_public
    access=private_gate_ready(st)
    if not access:
        # A public visitor must never see another owner's private DB picks,
        # including a report left cached by a previous authenticated session.
        st.session_state.pop("ziczac_unified_audit_status",None)
        st.caption("Private website selections are hidden until the owner "
                   "signs in through Private Access. This public summary "
                   "contains only previously published GitHub forecasts.")
    previous=st.session_state.get("ziczac_unified_audit_status") or {}
    if st.button("🔄 Check all saved prediction results",key="ziczac_unified_refresh",
                 use_container_width=True):
        with st.spinner("Comparing preserved predictions with identity-verified finals..."):
            # Public visitors can refresh the GitHub side, but only an
            # authenticated owner may read the private PostgreSQL archive.
            result=(build_with_private(database_url) if access
                    else build_public(private_connection="private_access_required"))
            if access:
                st.session_state["ziczac_unified_audit_status"]=result
            previous=result
    report=previous if previous else public
    if not report:
        st.info("The scheduled consolidated report has not been published yet. "
                "Press Check all saved prediction results to evaluate current "
                "GitHub evidence and your private database.")
        return
    if previous:
        state=previous.get("private_database_status")
        if state!="connected":
            st.warning("Private on-demand forecast database: "+
                       str(state or "unavailable")+
                       ". The displayed results may omit website-only AI or "
                       "manual five-double picks.")
    else:
        st.info("Showing the scheduled GitHub-only audit. Press the refresh "
                "button to include private website predictions and saved doubles "
                "from your configured central database.")
    singles=report.get("singles") or {}
    doubles=report.get("five_doubles") or {}
    a,b,c=st.columns(3)
    a.metric("Singles graded",str(singles.get("verified",0)))
    b.metric("Exact-score hits",str(singles.get("hit",0)))
    c.metric("Singles awaiting finals",str(singles.get("pending",0)))
    x,y,z=st.columns(3)
    x.metric("Five-double sets graded",str(doubles.get("verified",0)))
    y.metric("Five-double hits",str(doubles.get("hit",0)))
    z.metric("Double sets awaiting finals",str(doubles.get("pending",0)))
    st.caption("A miss requires a real verified score that differs. A pending "
               "match is NOT a loss. Each five-double set is a hit only if one "
               "of its five alternatives matches BOTH finished scores.")
    kinds=report.get("by_source") or {}
    if kinds:
        with st.expander("📊 Performance by model, Deep Research and AI provider",
                         expanded=True):
            for name,data in sorted(kinds.items()):
                st.write("**"+name.replace("_"," ").title()+"** · "+
                         str(data.get("hit",0))+" hit · "+
                         str(data.get("miss",0))+" miss · "+
                         str(data.get("pending",0))+" pending")
    options=["All","HIT","MISS","PENDING"]
    status=st.selectbox("Show forecast results",options,key="ziczac_unified_filter")
    sources=["All sources"]+sorted(kinds)
    source=st.selectbox("Method or AI provider",sources,key="ziczac_unified_source")
    filtered=[row for row in report.get("records") or []
              if (status=="All" or row.get("status")==status.lower()) and
                 (source=="All sources" or row.get("source")==source)]
    st.caption(str(len(filtered))+" matching preserved forecast records.")
    for row in filtered[:60]:
        symbol={"hit":"🟢 HIT","miss":"🔴 MISS","pending":"⏳ PENDING"}.get(
            row.get("status"),"⚪ UNVERIFIED")
        with st.expander(symbol+" · "+_label(row)+" · "+
                         row.get("source","unknown").replace("_"," ").title()):
            st.caption("Original saved: "+sydney_timestamp(row.get("captured_at_utc"))+
                       " · Model: "+str(row.get("model") or "not recorded"))
            for index,choice in enumerate(row.get("choices") or [],1):
                combined=" + ".join(choice)
                picked=(row.get("option_hits") or [])
                mark=(" · ✅ MATCHED" if len(picked)>=index and picked[index-1]
                      else "")
                st.write("**"+str(index)+".** "+combined+mark)
            actual=row.get("actual_scores") or []
            if actual:
                st.caption("Verified full-time 90-minute score: "+
                           " + ".join(actual))
            else:
                st.caption("No independently verified matching final score "
                           "is available yet.")
    if len(filtered)>60:
        st.caption("Showing the newest 60 of "+str(len(filtered))+
                   " matching records. Download CSV for the complete audit.")
    st.download_button("⬇️ Export original predictions and results (.csv)",
                       _export(filtered),"ziczac_verified_prediction_audit.csv",
                       "text/csv",key="ziczac_unified_export",
                       use_container_width=True)
    unverified=report.get("legacy_records_without_verifiable_identity",0)
    if unverified:
        st.warning(str(unverified)+" older archived records are still preserved "
                   "but lack enough original fixture identity to grade fairly. "
                   "They have NOT been counted as losses.")
    if report.get("identity_conflicts"):
        st.error("Conflicting independent full-time results require review: "+
                 str(report["identity_conflicts"])+". Affected fixtures "
                 "have not been graded.")
    st.caption("Latest audit: "+sydney_timestamp(report.get("checked_at_utc"))+
               ". Historical replay and post-kickoff predictions are excluded.")
