"""Read-only V18 observed model reliability and evidence quality dashboard."""
from pathlib import Path

from persist_prematch import load_json
from v18_accuracy_audit import ROOT


def render(st,root=ROOT):
    st.markdown("#### 📈 V18 Real-Accuracy & Calibration Audit")
    st.caption("This tracks genuinely saved pre-kickoff V8 predictions with "
               "verified match results. It never substitutes historical "
               "backtest scores, model confidence or unit tests for real wins.")
    path=Path(root)/"accuracy_report.json"
    result=load_json(path,{})
    if not result:
        st.info("The scheduled no-cost accuracy audit has not published its "
                "first report yet. The original forecasts remain unchanged.")
        return
    evidence=result.get("evidence") or {}
    baseline=result.get("baseline_double_chance") or {}
    n=int(baseline.get("verified") or 0)
    pending=int(evidence.get("awaiting_or_unverified") or 0)
    c1,c2,c3=st.columns(3)
    c1.metric("Source-verified matches",str(n))
    c2.metric("Pending/unverified results",str(pending))
    wins=int(baseline.get("hits") or 0)
    c3.metric("Observed double-chance hits",f"{wins}/{n}" if n else "Not available")
    if n:
        observed=baseline.get("observed_win_rate")
        estimate=baseline.get("mean_model_estimate")
        interval=baseline.get("wilson_95")
        st.caption("Observed success: "+f"{observed:.1%}"+
                   " · Average original forecast estimate: "+f"{estimate:.1%}"+
                   " · 95% Wilson interval: "+f"{interval[0]:.1%}–{interval[1]:.1%}")
        st.caption("Brier error (lower better): "+
                   f"{float(baseline['brier']):.4f}"+
                   " · Observed calibration gap: "+
                   f"{float(baseline['expected_calibration_error']):.2%}")
    else:
        st.warning("No verified original V8 forecast pairs/singles for this "
                   "audit yet. 70% accuracy cannot be claimed or inferred.")
    with st.expander("Reliability by model-estimated confidence band",expanded=False):
        st.caption("Each row is one ORIGINAL V8 single-match double-chance "
                   "prediction, not a five-exact-score-double result.")
        records=[]
        for entry in baseline.get("bins") or []:
            records.append({"Model estimate band":entry["estimated_band"],
                            "Verified":entry["verified"],
                            "Hits":entry["hits"],
                            "Mean model estimate":(
                                f"{entry['mean_estimated']:.1%}"
                                if entry["mean_estimated"] is not None else "—"),
                            "Actual accuracy":(
                                f"{entry['observed_rate']:.1%}"
                                if entry["observed_rate"] is not None else "Insufficient")})
        if records:
            st.dataframe(records,hide_index=True,use_container_width=True)
    calibrator=result.get("time_safe_shrinkage_research") or {}
    tested=int(calibrator.get("walk_forward_verified_test_matches") or 0)
    with st.expander("Walk-forward calibration validation",expanded=False):
        st.write({
            "Future-tested original matches":tested,
            "Minimum earlier verified training fixtures":calibrator.get("min_global_previous_verified",100),
            "Minimum earlier verified same-band fixtures":calibrator.get("min_prior_verified_in_same_band",20),
            "Raw Brier":calibrator.get("raw_brier"),
            "Time-safe calibrated Brier":calibrator.get("shrunk_calibrated_brier"),
            "Calibrator promoted":"No",
            "Accuracy gain proven":"No"
        })
        st.caption("The diagnostic uses only scores independently verified "
                   "BEFORE the relevant forecast timestamp. It never edits "
                   "the production forecast or predicts games retrospectively.")
    with st.expander("Evidence gaps found in saved forecasts",expanded=False):
        counts=evidence.get("missing_feature_counts") or {}
        st.write({
            "No genuine shot-level xG / xGA":counts.get("shot_based_xg_not_verified",0),
            "No tested numerical lineup adjustment":counts.get("lineup_adjustment_not_validated",0),
            "No independently source-backed market quote":counts.get("independent_market_snapshot_unavailable",0),
            "Insufficient last-10 form":counts.get("short_last_ten_sample",0),
            "Other source integrity rejections":evidence.get("source_rejections") or {}
        })
        st.caption("A missing feature is an evidence gap, not automatically "
                   "proof it would improve the model if purchased.")
    st.caption("A 70% claim requires independent, adequately sampled and "
               "calibrated future outcomes. Original exact-score five-double "
               "accuracy remains a separate, far harder task.")
