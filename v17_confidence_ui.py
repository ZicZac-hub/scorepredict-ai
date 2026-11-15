"""Read-only evidence and independently verified V17 outcome research display."""
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from persist_prematch import load_json
from v17_confidence import ROOT, VERSION, evidence_dashboard, wilson
from v10_prospective import utc

SYDNEY=ZoneInfo("Australia/Sydney")


def render(st,root=ROOT):
    st.subheader("🛡️ V17 High-Confidence Outcomes · Research")
    st.caption("Separate from the original five exact-score doubles. Predictions "
               "are made ONLY from frozen V8 score distributions. No paid API "
               "requests. 70% is a MODEL ESTIMATE threshold, NOT proven accuracy.")
    current=evidence_dashboard()
    if (Path(root)/"evidence_status.json").exists():
        current=load_json(Path(root)/"evidence_status.json",current)
    p=load_json(Path(root)/"performance.json",{})
    cols=st.columns(3)
    cols[0].metric("V8 verified matches",str(current.get("verified_v8",0)))
    cols[1].metric("Completed exact doubles",str(current.get("verified_five_double_sets",0)))
    cols[2].metric("Valid paired AI sets",str(current.get("source_matched_two_match_ai_sets",0)))
    with st.expander("🔎 Evidence readiness and missing inputs",expanded=True):
        st.write({
            "Archived V8 research forecasts":current.get("frozen_v8",0),
            "Real shot-quality xG feed":"Not verified / not integrated" if not current.get("real_xg_feed") else "Available",
            "Validated lineup/injury score adjustment":"Not available" if not current.get("verified_lineup_numeric_adjustment") else "Available",
            "Market-odds calibration benchmark":"Not independently validated" if not current.get("timestamped_market_benchmark_validated") else "Validated",
            "AI provider credentials configured":current.get("configured_ai_providers",0),
            "Verified AI evidence for both frozen matches":current.get("source_matched_two_match_ai_sets",0)
        })
        st.caption("Configured keys, historical scores, estimated goals and "
                   "LLM opinions cannot be substituted for verified xG, "
                   "confirmed lineups or independent market probabilities.")
    from v18_accuracy_ui import render as render_v18_accuracy
    render_v18_accuracy(st)
    st.divider()
    from v19_match_intelligence_ui import render as render_v19_match
    render_v19_match(st)
    st.divider()
    single=(p.get("single") or {})
    pair=(p.get("two_match_double_chance") or {})
    a,b=st.columns(2)
    for col,label,record in ((a,"Single-match double chance",single),
                             (b,"Two-match double chance",pair)):
        n=int(record.get("verified") or 0)
        wins=int(record.get("wins") or 0)
        col.metric(label+" verified",f"{wins}/{n}" if n else "No sample")
        if n:
            interval=record.get("wilson_95") or wilson(wins,n)
            col.caption(f"Observed: {wins/n:.1%} · 95% Wilson interval: "
                        f"{interval[0]:.1%}–{interval[1]:.1%}")
        else:
            col.caption("No verified performance yet; pending is not a failure.")
        if record.get("pending"):
            col.caption(f"{record['pending']} still awaiting verified results.")
    if p.get("meets_70_percent_claim_gate"):
        st.success("The prespecified 70% verification gate was met for the "
                   "experimental TWO-match double-chance category; check full "
                   "sample selection and independence before making claims.")
    else:
        st.info("70% observed two-match accuracy is NOT established. The "
                "verification gate requires 100+ genuinely future frozen pairs "
                "and a 95% lower confidence bound of at least 70%.")
    now=datetime.now(timezone.utc)
    upcoming=[]
    for path in sorted((Path(root)/"forecasts").glob("*.json")):
        row=load_json(path,{})
        if row.get("version")!=VERSION:
            continue
        slots=[row.get("single")]
        if row.get("pair"):
            slots.extend((row["pair"]["first"],row["pair"]["second"]))
        try:
            if any(x and utc(x["kickoff_utc"])>now for x in slots):
                upcoming.append(row)
        except (TypeError,KeyError,ValueError):
            continue
    if not upcoming:
        st.info("No upcoming qualifying V17 frozen selections yet. The "
                "automatic collector evaluates fixtures 45 minutes to 24 hours "
                "before kickoff and may abstain.")
        return
    row=upcoming[0]
    st.markdown("#### Next eligible Sydney date: "+str(row["sydney_date"]))
    st.caption("Frozen "+utc(row["captured_at_utc"]).astimezone(SYDNEY).strftime("%d %b %Y, %I:%M %p")+" Sydney")
    def show(label,leg):
        st.markdown("**"+label+"**")
        st.write(str(leg["home_team"])+" vs "+str(leg["away_team"]))
        st.caption("Kickoff "+utc(leg["kickoff_utc"]).astimezone(SYDNEY).strftime("%d %b, %I:%M %p")+
                   " Sydney · original forecast "+str(leg["forecast_id"])[:12])
        st.metric("Double chance: "+str(leg["selection"]),
                  f"{float(leg['model_estimated_probability']):.1%}",
                  help="Estimated probability only, not calibrated or observed accuracy.")
    if row.get("single"):
        show("Single-match research",row["single"])
    else:
        st.warning("No single-match estimate met the predeclared 70% threshold.")
    if row.get("pair"):
        st.divider()
        st.markdown("#### Experimental paired double chance")
        show("Leg 1",row["pair"]["first"])
        show("Leg 2",row["pair"]["second"])
        st.metric("Model-estimated joint coverage",
                  f"{float(row['pair']['joint_model_estimate']):.1%}")
        st.caption("Independence is assumed. This estimated figure is NOT "
                   "empirically calibrated or a guarantee of winning.")
    else:
        st.caption("Pair abstained: "+str(row.get("pair_abstention") or "insufficient evidence"))
    st.caption("All V17 selections are research-only, write-once and "
               "separately verified after full-time. The V8 five exact-score "
               "doubles have not been altered.")
