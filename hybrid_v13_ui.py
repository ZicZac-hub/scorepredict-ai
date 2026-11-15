"""V13 Hybrid Five Doubles: manual and bounded-auto on-demand challenger UI.

Run V6, V7, V8 and qualified V12 over the same pre-kickoff evidence.
Show top-probability, balanced and fully diversified FIVE exact-score doubles
without replacing or retrospectively marking original forecasts as hits.
"""
from datetime import datetime, timezone
import pandas as pd
from history_pipeline import build_evidence
from hybrid_five_v13 import predict, pair, best_pairs, VERSION
from smart_double_v8 import select_auto_research_pool


def research(rows, histories, *, now=None, limit=12):
    as_of=now or datetime.now(timezone.utc)
    output=[];reasons={}
    seen=set()
    for row in rows:
        if len(output)>=limit:
            break
        fixture=row.get("fixture") or {}
        key=(str(fixture.get("provider")),str(fixture.get("id")))
        if not fixture.get("id") or key in seen:
            continue
        seen.add(key)
        prior=histories.get(row.get("league")) or row.get("past") or []
        if not prior:
            reasons[str(fixture.get("id"))]="no completed league evidence"
            continue
        try:
            packet=build_evidence(fixture,prior,as_of)
            candidate=predict(packet["fixture"],packet["history"],
                              league_history=packet["league_history"],
                              as_of=as_of)
        except (ValueError,KeyError,TypeError,OverflowError,
                ZeroDivisionError,ArithmeticError) as exc:
            reasons[str(fixture.get("id"))]=type(exc).__name__
            continue
        if not candidate:
            reasons[str(fixture.get("id"))]="requires genuine v8 league evidence"
            continue
        output.append({**row,"fixture":packet["fixture"],"result":candidate,
                       "evidence_digest":packet["input_digest"]})
    return output,reasons


def _display(st,candidate,labeler):
    first,second=candidate["first"],candidate["second"]
    st.markdown("#### V13 Hybrid · Five exact-score doubles")
    st.caption("Match 1: "+labeler(first))
    st.caption("Match 2: "+labeler(second))
    st.metric("Estimated model coverage",f"{candidate['estimated_joint_coverage']:.2%}")
    st.caption("Independent-match model estimate, NOT historical hit rate. "
               "Both exact scores in the SAME row must match.")
    for i,choice in enumerate(candidate["doubles"],1):
        with st.container(border=True):
            st.markdown("**Double "+str(i)+"**")
            c1,c2=st.columns(2)
            c1.metric("Match 1",choice["Score 1"])
            c2.metric("Match 2",choice["Score 2"])
            st.caption("Estimated joint probability "+f"{choice['Joint']:.2%}")
    weights=first["result"].get("component_weights") or {}
    weights2=second["result"].get("component_weights") or {}
    with st.expander("🔬 Exactly which numerical models were combined"):
        st.write({"First match":{key:f"{value:.1%}" for key,value in weights.items()},
                  "Second match":{key:f"{value:.1%}" for key,value in weights2.items()}})
        st.caption("V8 itself contains overlapping V6/V7-type evidence. "
                   "These weights were declared before this test, not "
                   "statistically fitted. Missing V12/V7 data are skipped, "
                   "never invented.")
        st.caption("Model modes available: highest probability (best total "
                   "estimated coverage when model probabilities are right), "
                   "balanced and fully diversified (experimental).")
    data=pd.DataFrame([{
        "double":i,"match_1":labeler(first),"score_1":x["Score 1"],
        "match_2":labeler(second),"score_2":x["Score 2"],
        "model_joint_estimate":x["Joint"],
        "method":candidate["selection_mode"],"model":VERSION}
        for i,x in enumerate(candidate["doubles"],1)])
    st.download_button("Download V13 five doubles CSV",
        data.to_csv(index=False).encode("utf-8"),
        "ziczac_v13_hybrid_five_doubles.csv","text/csv",
        key="v13_download_five",use_container_width=True)
    st.caption("Experimental challenger only. Existing original predictions, "
               "verified results and automatic V8 selections are untouched.")


def render(st,rows,histories,labeler):
    st.markdown("### 🧠 V13 Hybrid AI · Combined Five Doubles")
    st.caption("Combines pre-match V8 + fitted V7 + original V6 + optional "
               "league-calibrated V12. External AI/market odds are NOT included "
               "unless independently verified; no fake votes.")
    st.warning("V13 is a new research challenger, NOT proven more accurate. "
               "In the previous 20-pair test, V7 diversified won 2/20 and "
               "many methods won 0/20. This does not prove future performance.")
    eligible=[r for r in rows if r.get("fixture") and r.get("result")]
    if len(eligible)<2:
        st.info("Load at least two upcoming researched fixtures.")
        return
    mode_label=st.radio("Hybrid match selection",
        ["AI Auto Select · Hybrid", "Manual Select · Hybrid"],
        horizontal=True,key="v13_mode")
    strategy=st.selectbox("Five-double selection strategy",[
        "Top five by joint probability (maximum model coverage)",
        "Balanced (experimental)",
        "Fully diversified (experimental)"],key="v13_strategy")
    mode=("top5" if strategy.startswith("Top") else
          "balanced" if strategy.startswith("Balanced") else "diversified")
    now=datetime.now(timezone.utc)
    identity=(mode_label,strategy,
              tuple((str(r["fixture"].get("provider")),
                     str(r["fixture"].get("id"))) for r in eligible))
    if mode_label.startswith("AI"):
        st.caption("Scans at most 12 already loaded fixtures across the next "
                   "three Sydney matchdays; selects the best model-estimated "
                   "five-double pair among valid four-team same-day matches.")
        clicked=st.button("🧠 Calculate V13 Auto Five Doubles",
            key="v13_calculate_auto",type="primary",use_container_width=True)
        selection=None
    else:
        options={labeler(row):row for row in eligible}
        first=st.selectbox("Hybrid · Match 1",list(options),key="v13_first")
        others=[name for name in options if name!=first]
        second=st.selectbox("Hybrid · Match 2",others,key="v13_second")
        identity=identity+(first,second)
        clicked=st.button("🧠 Calculate V13 Manual Five Doubles",
            key="v13_calculate_manual",type="primary",use_container_width=True)
        selection=(options[first],options[second])
    if clicked:
        with st.spinner("Building independent pre-kickoff evidence for each match..."):
            if selection is None:
                pool=select_auto_research_pool(eligible,max_fixtures=12,max_days=3,
                                               as_of=now)
            else:
                pool=list(selection)
            researched,reasons=research(pool,histories,now=now,limit=12)
            candidates=best_pairs(researched,mode=mode,now=now,limit=1)
            st.session_state["v13_hybrid_result"]={
                "identity":identity,"candidate":candidates[0] if candidates else None,
                "searched":len(pool),"researched":len(researched),
                "reasons":reasons}
    saved=st.session_state.get("v13_hybrid_result") or {}
    if saved.get("identity")!=identity:
        st.info("Choose your matches and strategy, then run the V13 challenger.")
        return
    if not saved.get("candidate"):
        st.warning("Not enough independent pre-kickoff evidence for a "
                   "four-team same-day hybrid pair. Try Manual Select or "
                   "load more verified league history.")
        st.caption("Matches studied: "+str(saved.get("researched",0))+
                   " / "+str(saved.get("searched",0))+". "+
                   "No automatic fallback or invented scorelines.")
        return
    _display(st,saved["candidate"],labeler)
