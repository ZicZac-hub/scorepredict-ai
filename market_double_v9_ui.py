"""Optional two-fixture market-assisted exact-score doubles, on demand only.

Uses no odds endpoint unless user presses Run. Existing v8 Auto Select,
v6 forecasts and the frozen pre-kickoff prediction ledger are not changed.
"""
from datetime import datetime, timezone
import pandas as pd
from market_score_v9 import select_market_response, predict
from smart_double_ai import five_doubles, pair_issue
from smart_double_v8 import _kick

def _retrieve(key,fixture):
    """One current, provider-documented /odds request per selected fixture."""
    from api_football import _request
    packet=_request(key,"/odds",{"fixture":fixture["id"]},with_metadata=True)
    meta=packet.get("metadata") or {}
    if meta.get("stale") or meta.get("freshness_known") is False:
        return None,"Provider odds response was stale or its fetch time unverified."
    found=select_market_response(packet.get("data") or [],fixture)
    if not found:
        return None,"No recent complete verified 90-minute odds were supplied."
    return found,None


def render(st,rows,histories,labeler,api_key):
    with st.expander("📊 Market-Informed v9 · Manual five-double comparison", expanded=False):
        st.caption("Optional: choose two distinct same-Sydney-date fixtures. v9 fetches real pre-match 1X2 / Over-Under prices for ONLY these two games, then compares five exact-score combinations. Original models and saved predictions remain untouched.")
        ready=[r for r in rows if r.get("result") and (r.get("fixture") or {}).get("provider")=="API-Football"
               and (r.get("fixture") or {}).get("id") and _kick(r) and _kick(r)>datetime.now(timezone.utc)]
        if len(ready)<2 or not api_key:
            st.info("Two upcoming researched API-Football fixtures and an API_FOOTBALL_KEY are required. The v8 manual/auto doubles above still work without odds.")
            return
        entries={labeler(r):r for r in ready}
        first=st.selectbox("Market double · Match 1",list(entries),key="v9_market_first")
        choices=[x for x in entries if x!=first]
        second=st.selectbox("Market double · Match 2",choices,key="v9_market_second")
        sig=(first,second)
        selected = (entries[first], entries[second])
        issue = pair_issue(*selected, now=datetime.now(timezone.utc))
        if issue:
            st.warning(issue)
            return
        if st.button("🔬 Calculate market-informed five doubles",key="run_v9_manual_double",type="primary",use_container_width=True):
            forecast=[]
            evidence=[]
            for label in (first,second):
                issue = pair_issue(*selected, now=datetime.now(timezone.utc))
                if issue:
                    st.session_state["v9_market_double"]={"signature":sig,"error":issue}
                    break
                row=entries[label]
                fixture=row["fixture"]
                past=row.get("past") or histories.get(row.get("league"),[])
                league=histories.get(row.get("league"),past)
                market,warning=None,None
                try:
                    market,warning=_retrieve(api_key,fixture)
                except (ValueError,TypeError,RuntimeError) as exc:
                    warning="Odds provider was unavailable: "+type(exc).__name__
                observed = datetime.now(timezone.utc)
                issue = pair_issue(*selected, now=observed)
                if issue:
                    st.session_state["v9_market_double"]={"signature":sig,"error":issue}
                    break
                try:
                    result=predict(fixture,past,league_history=league,market=market,as_of=observed)
                except ValueError:
                    if market is None:
                        raise
                    market=None
                    warning="The odds failed pre-match verification."
                    result=predict(fixture,past,league_history=league,as_of=observed)
                if not result:
                    st.session_state["v9_market_double"]={"signature":sig,
                        "error":"Insufficient verified history for "+label}
                    break
                forecast.append({**row,"result":result})
                evidence.append({"fixture":label,"bookmaker":(market or {}).get("bookmaker"),
                                 "snapshot_utc":(market or {}).get("snapshot_utc"),
                                 "market_used":bool(market),
                                 "note":warning or "Valid pre-match prices used."})
            if len(forecast)==2:
                issue = pair_issue(*forecast, now=datetime.now(timezone.utc))
                doubles = [] if issue else five_doubles(*forecast)
                if len(doubles)==5:
                    st.session_state["v9_market_double"]={"signature":sig,"doubles":doubles,
                                                          "evidence":evidence}
                else:
                    st.session_state["v9_market_double"]={"signature":sig,
                        "error":issue or "Not enough distinct valid score pairs."}
        issue = pair_issue(*selected, now=datetime.now(timezone.utc))
        if issue:
            st.info(issue)
            return
        output=st.session_state.get("v9_market_double") or {}
        if output.get("signature")!=sig:
            st.caption("Press Calculate to run the v9 check for your selected pair.")
            return
        if output.get("error"):
            st.warning(output["error"])
            return
        if not output:
            return
        for row in output["evidence"]:
            if row["market_used"]:
                st.success(str(row["fixture"])+" · "+str(row["bookmaker"])+" · "+str(row["snapshot_utc"]))
            else:
                st.warning(str(row["fixture"])+" · Historical v8 fallback. "+str(row["note"]))
        combined=sum(x["Joint"] for x in output["doubles"])
        st.metric("Estimated joint five-score coverage",f"{combined:.2%}")
        for i,x in enumerate(output["doubles"],1):
            with st.container(border=True):
                st.markdown("**Double "+str(i)+"**")
                a,b=st.columns(2)
                a.metric("Match 1",x["Score 1"])
                b.metric("Match 2",x["Score 2"])
                st.caption("Estimated independent-match probability: "+f"{x['Joint']:.2%}")
        st.caption("Experimental only. Coverage is a model probability estimate, not observed success or guaranteed outcome. No picks were saved to your live performance history.")
        table=pd.DataFrame(output["doubles"])
        st.download_button("Download market-informed double comparison CSV",
                           table.to_csv(index=False).encode(),"ziczac_market_v9_doubles.csv",
                           mime="text/csv",key="v9_double_download")
