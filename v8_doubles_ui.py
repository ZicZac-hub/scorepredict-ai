"""On-demand five-double v8 challenger screen.

Never alters the historical predictions, manual original selections, or live
smart-double output. The stored result is session-only and not a verified pick.
"""
from datetime import datetime, timezone
import pandas as pd
from smart_double_ai import five_doubles
from smart_double_v8 import research_fixture_rows, rank_research_pairs, manually_selected


def _label(row, labeler):
    return labeler(row)


def _show(st, candidate, labeler):
    first, second = candidate["first"], candidate["second"]
    st.markdown("**Match 1:** " + _label(first,labeler))
    st.markdown("**Match 2:** " + _label(second,labeler))
    st.metric("Estimated coverage of five exact-score doubles", f"{candidate['coverage']:.2%}")
    st.caption("This is a MODEL ESTIMATE, not historical hit rate, with independent-match assumption. Exact-score doubles require BOTH games to finish as selected.")
    old_first = {**first,"result":first["result"].get("baseline_reference") or first.get("original_result")}
    # The v8 predictor retains old baseline modal estimates as metadata. Re-run
    # no network calls to recover full original distributions from caller rows.
    if candidate.get("baseline"):
        st.caption("Preserved v6 same-pair estimated coverage: "+
                   f"{sum(x['Joint'] for x in candidate['baseline']):.2%}"+
                   " · Differences are not evidence that v8 wins more matches.")
    for position, item in enumerate(candidate["doubles"][:5],1):
        with st.container(border=True):
            st.markdown("**Double "+str(position)+"**")
            c1,c2=st.columns(2)
            c1.metric("Match 1 score",item["Score 1"])
            c2.metric("Match 2 score",item["Score 2"])
            st.caption("Match 1 "+f"{item['P1']:.1%}"+
                       " · Match 2 "+f"{item['P2']:.1%}"+
                       " · Joint "+f"{item['Joint']:.2%}")
    with st.expander("Full five-double probability table"):
        st.dataframe(pd.DataFrame([{"Double":n,"Match 1":x["Score 1"],
                                   "Match 2":x["Score 2"],
                                   "Joint probability":x["Joint"]}
                                  for n,x in enumerate(candidate["doubles"][:5],1)]),
                     hide_index=True,use_container_width=True)
    export = pd.DataFrame([{"match_1":_label(first,labeler),"score_1":x["Score 1"],
                            "match_2":_label(second,labeler),"score_2":x["Score 2"],
                            "model_joint_probability":x["Joint"]}
                           for x in candidate["doubles"][:5]])
    st.download_button("Download v8 challenger doubles CSV",
        export.to_csv(index=False).encode("utf-8"),"ziczac_v8_challenger_doubles.csv",
        "text/csv",key="v8_double_export")
    st.caption("Candidate only · Does not change saved original selections, verified results, or the production six-hour workflow.")


def render(st, rows, histories, labeler):
    st.markdown("### 🧪 Deep Research AI v8 · five-double challenger")
    st.caption("Preserves your existing doubles. This experimental model combines venue-form, fitted opponent strengths, Dixon–Coles and historical Elo. It uses no external ChatGPT/Claude/Gemini/Grok/DeepSeek API calls.")
    eligible = [r for r in rows if r.get("result") and r.get("fixture")]
    if len(eligible)<2:
        st.info("At least two researched upcoming fixtures are required.")
        return
    mode=st.radio("Challenger selection",["AI Auto Select (v8)","Manual Select (v8)"],
                  horizontal=True,key="v8_double_mode")
    identifiers = tuple((str(r["fixture"].get("provider")),str(r["fixture"].get("id")))
                        for r in eligible)
    identity=("v8",mode,identifiers)
    if mode.startswith("AI"):
        st.caption("Scans up to 18 upcoming researched fixtures, requiring two distinct same-Sydney-date matches with four teams.")
        if st.button("🔬 Run v8 Auto Select",key="v8_run_auto",type="primary",use_container_width=True):
            with st.spinner("Comparing verified match history and score distributions..."):
                researched = research_fixture_rows(eligible,histories,max_fixtures=18)
                best = rank_research_pairs(researched,limit=1)
            if best:
                st.session_state["v8_double_report"] = {"identity":identity,"candidate":best[0]}
            else:
                st.session_state["v8_double_report"] = {"identity":identity,
                    "reason":"Not enough same-day, independently researched v8 fixtures. Try extending the date window or selecting a manual pair."}
    else:
        options = {_label(row,labeler):row for row in eligible}
        first = st.selectbox("V8 match 1",list(options),key="v8_manual_first")
        others=[x for x in options if x!=first]
        second = st.selectbox("V8 match 2",others,key="v8_manual_second")
        identity += (first,second)
        if st.button("🧠 Research my two matches with v8",key="v8_run_manual",
                     type="primary",use_container_width=True):
            with st.spinner("Running opponent research for both matches..."):
                candidate=manually_selected(options[first],options[second],histories)
            if candidate.get("available"):
                preserved=five_doubles(options[first],options[second])
                candidate["baseline"]=preserved
                st.session_state["v8_double_report"]={"identity":identity,"candidate":candidate}
            else:
                st.session_state["v8_double_report"]={"identity":identity,
                    "reason":candidate.get("reason","Verified match evidence is insufficient.")}
    saved=st.session_state.get("v8_double_report") or {}
    if saved.get("identity")!=identity:
        st.info("Choose your mode, then press its research button to generate five experimental doubles.")
        return
    if saved.get("reason"):
        st.warning(saved["reason"])
        return
    _show(st,saved["candidate"],labeler)
