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
from robust_double_v14 import rank_robust_pairs


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
    st.markdown("#### Five exact-score doubles · "+candidate.get("model_choice","V13 combined"))
    st.caption("Match 1: "+labeler(first))
    st.caption("Match 2: "+labeler(second))
    robust=bool(candidate.get("robust_lower_envelope"))
    if robust:
        st.metric("V7/V8 conservative model envelope (not a win rate)",
                  f"{candidate['estimated_joint_coverage']:.2%}")
        st.caption("V7 coverage for these five combinations: "+
                   f"{candidate['model_coverage_v7']:.2%}"+
                   " · V8 coverage: "+f"{candidate['model_coverage_v8']:.2%}"+
                   ". The envelope is the minimum of two statistical "
                   "estimates, not a calibrated probability or guarantee.")
    else:
        st.metric("Estimated model coverage",f"{candidate['estimated_joint_coverage']:.2%}")
    st.caption("Both exact scores in the SAME double must match. "
               "Selection method: "+str(candidate.get("selection_mode","unknown"))+
               ". Independent-match assumption; not a verified win rate.")
    for i,choice in enumerate(candidate["doubles"],1):
        with st.container(border=True):
            st.markdown("**Double "+str(i)+"**")
            c1,c2=st.columns(2)
            c1.metric("Match 1",choice["Score 1"])
            c2.metric("Match 2",choice["Score 2"])
            if robust:
                st.caption("V7 model joint "+f"{choice['V7 joint']:.2%}"+
                           " · V8 model joint "+f"{choice['V8 joint']:.2%}"+
                           " · lower envelope "+f"{choice['Joint']:.2%}")
            else:
                st.caption("Estimated model joint probability "+
                           f"{choice['Joint']:.2%}")
    weights=first["result"].get("component_weights") or {}
    weights2=second["result"].get("component_weights") or {}
    with st.expander("🔬 Model evidence and agreement"):
        if robust:
            st.write({"V7 vs V8 score-grid disagreement · match 1":
                      f"{candidate['agreement_total_variation']['first']:.2%}",
                      "V7 vs V8 score-grid disagreement · match 2":
                      f"{candidate['agreement_total_variation']['second']:.2%}"})
            st.caption("Total variation describes disagreement between two "
                       "correlated statistical models, NOT a calibrated "
                       "prediction confidence. V14 evaluates every exact "
                       "score-pair cell, not just top-ten guesses.")
        elif candidate.get("model_choice","V13 combined").startswith("V13"):
            st.write({"First match":{key:f"{value:.1%}" for key,value in weights.items()},
                      "Second match":{key:f"{value:.1%}" for key,value in weights2.items()}})
        else:
            st.caption("These score probabilities are from the selected "
                       "individual V7 or V8 statistical component, not the "
                       "V13 mixture. Weights below show the separately "
                       "available V13 mixture for comparison, NOT the "
                       "weights applied to these five doubles.")
            st.write({"V13 reference weights · first":weights,
                      "V13 reference weights · second":weights2})
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
        "v7_joint_estimate":x.get("V7 joint"),
        "v8_joint_estimate":x.get("V8 joint"),
        "method":candidate["selection_mode"],
        "model":candidate.get("model_version",VERSION)}
        for i,x in enumerate(candidate["doubles"],1)])
    st.download_button("Download V13 five doubles CSV",
        data.to_csv(index=False).encode("utf-8"),
        "ziczac_research_five_doubles.csv","text/csv",
        key="v13_download_five",use_container_width=True)
    st.caption("Experimental challenger only. Original forecasts, verified "
               "results and automatic V8 selections are untouched. V14 "
               "agreement is not yet proven to improve accuracy.")


def render(st,rows,histories,labeler):
    st.markdown("### 🧠 V13 Hybrid AI · Combined Five Doubles")
    st.caption("Compare V8, fitted V7, V13 blend, and the new V14 robust "
               "five-double selector, which checks V7 and V8 agreement over "
               "complete score grids. None are guaranteed. External AI and "
               "bookmaker odds are not fabricated.")
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
    model_choice=st.selectbox("Statistical model for the five doubles",[
        "V13 combined model (new experiment)",
        "V8 evidence ensemble (strongest tested exact-score model)",
        "V7 fitted Dixon–Coles (best diversified result in 20 historical pairs)",
        "V14 robust agreement · V7 and V8 full-grid consensus (experimental)"],
        key="v13_model_choice")
    if model_choice.startswith("V14"):
        st.caption("V14 uses its own exact, conservative top-five selection "
                   "over the full V7/V8 score grids and is not the same "
                   "as arbitrary score diversification.")
        strategy="V14 robust V7/V8 lower envelope"
        mode="v14_robust"
    else:
        strategy=st.selectbox("Five-double selection strategy",[
            "Top five by joint probability (maximum model coverage)",
            "Balanced (experimental)",
            "Fully diversified (experimental)"],key="v13_strategy")
        mode=("top5" if strategy.startswith("Top") else
              "balanced" if strategy.startswith("Balanced") else "diversified")
    now=datetime.now(timezone.utc)
    identity=(mode_label,strategy,model_choice,
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
            # Use the REAL same-evidence full-score component chosen by the
            # user; a model name is not a replacement for numerical evidence.
            chosen_key=("v7" if model_choice.startswith("V7") else
                        "v8" if model_choice.startswith("V8") else None)
            candidates_pool=[]
            for row in researched:
                original=row["result"]
                if chosen_key is None:
                    candidates_pool.append(row)
                else:
                    scores=original.get("component_score_distributions",{}).get(chosen_key)
                    version=original.get("component_versions",{}).get(chosen_key)
                    if scores and version:
                        candidates_pool.append({**row,"result":{
                            **original,"scores":scores,"model_version":version}})
            if mode=="v14_robust":
                candidates=rank_robust_pairs(candidates_pool,now=now,limit=1)
            else:
                candidates=best_pairs(candidates_pool,mode=mode,now=now,limit=1)
            if candidates:
                candidates[0]["model_choice"]=model_choice
            st.session_state["v13_hybrid_result"]={
                "identity":identity,"candidate":candidates[0] if candidates else None,
                "searched":len(pool),"researched":len(researched),
                "reasons":reasons}
    saved=st.session_state.get("v13_hybrid_result") or {}
    if saved.get("identity")!=identity:
        st.info("Choose your two matches, model and strategy, then run the challenger.")
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
