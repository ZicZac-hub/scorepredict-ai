"""Fast, date-scoped single-match research independent of primary competition filters."""
from datetime import datetime, timedelta, timezone
from worldwide import SYDNEY
from global_fixtures import global_fixtures
from worldwide_model import model_worldwide
from single_match_50_ui import candidates

def render(st, api_key):
    st.header("🔎 Single Match · Deep Research & 80+ Odds")
    st.success("Market Explorer v2.2 · Corners / Penalties / Goals / Offsides / Cards / Shots")
    st.caption("Search genuine upcoming API-Football fixtures by team. Research runs only "
               "after selecting a match; no artificial bookmaker odds or guaranteed picks.")
    if not api_key:
        st.error("Team search cannot start: API_FOOTBALL_KEY is missing from Streamlit Secrets. "
                 "Add your existing provider key there (never paste it into the website).")
        st.link_button("Open Streamlit Cloud settings","https://share.streamlit.io/")
        return
    date_choice=st.selectbox("Sydney match dates",
        ["Next 7 days","Today and tomorrow","Next 14 days"],key="single50_dates")
    days={"Today and tomorrow":2,"Next 7 days":7,"Next 14 days":14}[date_choice]
    @st.cache_data(ttl=1800,show_spinner=False,max_entries=8)
    def search_fixtures(start,span,credential):
        return global_fixtures(credential,start,span)
    with st.spinner("Searching real fixtures by Sydney date…"):
        try:
            events,errors=search_fixtures(datetime.now(SYDNEY).date(),days,api_key)
        except Exception as exc:
            st.error("Fixture search failed ("+type(exc).__name__+"). Try refreshing the page or check provider availability.")
            return
    if errors:
        st.warning("Provider search limitations: "+"; ".join(errors[:2]))
    search=st.text_input("🔎 Search team or league",key="single50_team_search",
                         placeholder="e.g. Arsenal, Real Madrid, Premier League")
    filtered=[e for e in events if search.casefold().strip() in
              (e["home"]+" "+e["away"]+" "+e["league"]).casefold()]
    st.caption(f"{len(filtered)} matching fixtures out of {len(events)} provider fixtures.")
    if events:
        st.caption("Choose a fixture from the list below. The search box filters the returned team and league names.")
    if not filtered:
        st.info("No matching upcoming fixtures returned by API-Football. Try another team or expand the date range. If the total provider fixture count is zero, check the configured API key, subscription access, and provider warnings above.")
        return
    labels={str(e["id"])+" · "+e["home"]+" vs "+e["away"]+" · "+
            e["local"].strftime("%d %b %I:%M %p")+" Sydney · "+e["league"]:e
            for e in filtered}
    label=st.selectbox("Select the match",list(labels),key="single50_selected_fixture")
    event=labels[label]
    st.markdown("**"+event["home"]+" vs "+event["away"]+"**")
    st.caption("Kickoff: "+event["local"].strftime("%d %b %Y %I:%M %p")+
               " Sydney · Provider fixture ID: "+str(event["id"]))
    st.markdown("### ⚽ All available bookmaker markets for this match")
    from single_match_market_ui import render as render_bookmaker_markets
    render_bookmaker_markets(st,api_key,event)
    st.divider()
    st.subheader("Experimental statistical exact-score model")
    threshold=st.number_input("Minimum estimated fair odds",min_value=80.0,
                               max_value=1000.0,value=80.0,step=10.0,
                               key="single50_min_odds")
    if st.button("🧠 Run deep research for selected match",type="primary",
                 use_container_width=True,key="single50_run"):
        if event["kickoff"]<=datetime.now(timezone.utc):
            st.error("Kickoff has passed. Pre-match analysis is unavailable.")
            return
        with st.spinner("Checking historical evidence and calculating score probabilities…"):
            try:
                row,error=model_worldwide(api_key,event,model_variant="research_v8")
            except Exception as exc:
                row,error=None,"Research failed ("+type(exc).__name__+"). Provider data may be unavailable; no forecast has been invented."
        st.session_state["single50_report"]={
            "fixture_id":str(event["id"]),"kickoff":event["kickoff"].isoformat(),
            "row":row,"error":error}
    report=st.session_state.get("single50_report") or {}
    if report.get("fixture_id")!=str(event["id"]) or report.get("kickoff")!=event["kickoff"].isoformat():
        st.info("Choose a fixture above and press Run deep research.")
        return
    if report.get("error"):
        st.warning(str(report["error"])[:300])
    row=report.get("row") or {}
    result=row.get("result") or {}
    if not result:
        st.warning("No valid prediction was produced from available evidence.")
        return
    st.subheader("Verified research summary")
    st.caption("Model: "+str(result.get("model_version","unversioned"))+
               " · Estimated home goals: "+str(round(float(result.get("home_xg",0)),2))+
               " · Estimated away goals: "+str(round(float(result.get("away_xg",0)),2)))
    counts=row.get("history_counts") or {}
    if counts:
        st.write("Verified history coverage:",counts)
    details=row.get("research_context") or {}
    if details:
        with st.expander("Provider research: team form, standings, injuries and lineups"):
            st.json(details)
    for field in ("research_details","research_warnings","sources","warnings","evidence_warnings"):
        data=row.get(field)
        if data:
            st.write(field.replace("_"," ").title()+":",data)
    st.subheader("Five single-match exact-score options · 80+ estimated fair odds")
    picks=candidates(result.get("matrix"),threshold)
    if picks:
        st.dataframe(picks[:5],hide_index=True,use_container_width=True)
        st.caption(f"{min(5,len(picks))} qualifying distinct exact-score outcomes displayed. "
                   "Fewer than five means insufficient qualifying model outcomes; no filler picks.")
    else:
        st.warning("No qualifying model-derived exact score. No odds invented.")
    st.caption("These are five alternatives for ONE match, not five independent bets. "
               "Fair odds = 1 / model probability; NOT verified bookmaker prices. "
               "Outcomes at 80+ are individually very unlikely and no profitable edge is demonstrated.")
