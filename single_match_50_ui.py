"""Single-match long-odds research. Decimal FAIR odds, never bookmaker quotes."""
from datetime import datetime, timezone
import math

def candidates(matrix, threshold=50.0):
    """Return actual model-grid outcomes with fair decimal odds >= threshold."""
    output=[]
    if matrix is None:
        return output
    for h, line in enumerate(matrix):
        for a, value in enumerate(line):
            try:
                p=float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(p) or p<=0 or p>1:
                continue
            fair=1.0/p
            if fair>=threshold:
                output.append({"Score":f"{h}-{a}", "Model probability":round(100*p,3),
                               "Estimated fair odds":round(fair,2)})
    return sorted(output,key=lambda x:x["Estimated fair odds"])[:25]

def render(st, rows, histories, sydney_kickoff):
    st.header("🎯 Single Match · 50+ Odds Research")
    st.caption("Experimental research: estimates FAIR decimal odds from a score-probability model. "
               "These are NOT live bookmaker prices or betting offers. Odds of 50 imply "
               "only a 2% break-even chance before bookmaker margin.")
    now=datetime.now(timezone.utc)
    available=[]
    for row in rows or []:
        fixture=row.get("fixture") or {}
        result=row.get("result") or {}
        try:
            kickoff=datetime.fromisoformat(str(row["kickoff"]).replace("Z","+00:00"))
            if kickoff.tzinfo is None:
                kickoff=kickoff.replace(tzinfo=timezone.utc)
            if kickoff<=now or not fixture.get("id") or result.get("matrix") is None:
                continue
            home=(fixture.get("homeTeam") or {}).get("name","Home")
            away=(fixture.get("awayTeam") or {}).get("name","Away")
            available.append((f"{home} vs {away} · {sydney_kickoff(row['kickoff'])}",row))
        except (KeyError,ValueError,TypeError):
            continue
    if not available:
        st.info("No eligible pre-kickoff fixtures with a full score probability grid. "
                "No odds have been invented.")
        return
    search=st.text_input("Find a team",key="zz_single50_search")
    options={label:row for label,row in available if search.casefold() in label.casefold()}
    if not options:
        st.info("No matching eligible fixture.")
        return
    selected=st.selectbox("Select one upcoming match",list(options),key="zz_single50_match")
    row=options[selected]
    result=row["result"]
    fixture=row["fixture"]
    st.caption("Model: "+str(result.get("model_version","version not provided"))+
               " · "+str(row.get("league",""))+
               " · "+sydney_kickoff(row["kickoff"]))
    threshold=st.number_input("Minimum estimated fair decimal odds",min_value=50.0,
                              max_value=1000.0,value=50.0,step=10.0,key="zz_single50_threshold")
    st.subheader("Deep research evidence")
    from research_ui import _games, _stats
    home_id=(fixture.get("homeTeam") or {}).get("id")
    away_id=(fixture.get("awayTeam") or {}).get("id")
    competition=str(row.get("league") or "")
    historical=histories.get(competition,[]) if isinstance(histories,dict) else []
    home=_games(fixture,historical,home_id,limit=10)
    away=_games(fixture,historical,away_id,limit=10)
    a,b=st.columns(2)
    with a:
        st.markdown("**Home team · last 10 verified**")
        st.json(_stats(home))
    with b:
        st.markdown("**Away team · last 10 verified**")
        st.json(_stats(away))
    if not home or not away:
        st.warning("Verified recent-history coverage is incomplete. Treat all model odds cautiously.")
    st.subheader("Single-match exact-score outcomes")
    options=candidates(result.get("matrix"),threshold)
    if options:
        st.dataframe(options,use_container_width=True,hide_index=True)
        st.caption("The list contains low-probability score outcomes, not high-confidence picks. "
                   "A larger estimated payout does not imply a higher chance of winning. "
                   "No real bookmaker odds or value advantage has been verified.")
    else:
        st.info("No model-grid exact score meets this threshold. No qualifying selection.")
