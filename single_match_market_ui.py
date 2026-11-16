"""Verified bookmaker-market explorer; never manufactures props or combined odds."""
from datetime import datetime, timezone
from api_football import _request

KEYWORDS={
    "Corners":("corner",),
    "Penalties":("penalt",),
    "Goals":("goal","score","both teams"),
    "Free kicks":("free kick",),
    "Offsides":("offside",),
    "Cards":("card","booking"),
    "Shots":("shot",),
    "Other":(),
}

def flatten_markets(payload, fixture_id, minimum=1.01):
    """Read provider odds for exactly one fixture, with explicit source labels."""
    output=[]
    for item in payload if isinstance(payload,list) else []:
        raw=(item.get("fixture") or {})
        if str(raw.get("id"))!=str(fixture_id):
            continue
        for bookmaker in item.get("bookmakers") or []:
            bookmaker_name=str(bookmaker.get("name") or "Unknown bookmaker")
            for market in bookmaker.get("bets") or []:
                market_name=str(market.get("name") or "")
                lower=market_name.casefold()
                category=next((name for name,terms in KEYWORDS.items()
                               if terms and any(term in lower for term in terms)),"Other")
                for option in market.get("values") or []:
                    try:
                        odd=float(option.get("odd"))
                    except (TypeError,ValueError):
                        continue
                    if not (minimum<=odd<100000):
                        continue
                    output.append({"Category":category,"Market":market_name,
                                   "Selection":str(option.get("value") or ""),
                                   "Bookmaker":bookmaker_name,
                                   "Quoted decimal odds":odd})
    return sorted(output,key=lambda x:(x["Quoted decimal odds"],x["Category"],
                                      x["Market"],x["Selection"],x["Bookmaker"]))

def render(st, api_key, event):
    st.subheader("📡 Real bookmaker markets · single match")
    st.caption("Provider-listed odds only. Corners, penalties, goals, free kicks, "
               "offsides, cards and shots appear only if a bookmaker supplies them. "
               "A market may not be available on your plan or for this fixture.")
    minimum=st.number_input("Minimum individual quoted decimal odds",min_value=1.01,
                            max_value=10000.0,value=1.20,step=0.5,
                            key="zz_real_market_minimum")
    if st.button("🔎 Search actual markets for this match",key="zz_real_market_search",
                 use_container_width=True):
        try:
            data=_request(api_key,"/odds",{"fixture":event["id"]})
            rows=flatten_markets(data,event["id"],1.01)
            st.session_state["zz_real_market_result"]={
                "fixture":str(event["id"]),"rows":rows,"error":None,
                "checked":datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            st.session_state["zz_real_market_result"]={
                "fixture":str(event["id"]),"rows":[],"error":type(exc).__name__}
    record=st.session_state.get("zz_real_market_result") or {}
    if record.get("fixture")!=str(event["id"]):
        st.info("Press Search actual markets to check this selected fixture.")
        return
    if record.get("error"):
        st.warning("Bookmaker market feed unavailable ("+record["error"]+
                   "). No market or price is inferred.")
        return
    rows=[x for x in record.get("rows",[]) if x["Quoted decimal odds"]>=minimum]
    st.caption("Last queried: "+str(record.get("checked","unknown"))+" UTC")
    if not rows:
        st.info("No verified bookmaker selections at this threshold for this match. "
                "This does not mean those outcomes are impossible.")
        return
    selected=st.multiselect("Filter markets",list(KEYWORDS),
                            default=list(KEYWORDS),key="zz_real_market_categories")
    filtered=[x for x in rows if x["Category"] in selected]
    st.caption(str(len(filtered))+" qualifying bookmaker quotes. "
               "Displayed quotes are not guaranteed to remain available.")
    st.dataframe(filtered,use_container_width=True,hide_index=True)
    from same_game_builder_ui import render as render_builders
    render_builders(st,record.get("rows",[]),target=80.0)
    st.warning("Do not multiply odds from correlated same-game markets to claim an "
               "80+ combined price. A valid same-game combination requires a "
               "bookmaker-quoted combined market and independently validated probabilities.")
