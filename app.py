"""ZicZac Predict AI Worldwide: free Streamlit UI, live API, walk-forward validation."""
from datetime import datetime, timedelta, timezone
from itertools import combinations
import os
import json
import pandas as pd
import streamlit as st
from data import competitions, fixtures, history
from api_football import get_fixtures as secondary_fixtures, get_history as secondary_history, supported_codes
from worldwide import worldwide, SYDNEY
from global_fixtures import global_fixtures
from sportmonks import sportmonks_fixtures
from engine import predict
from worldwide_model import model_worldwide
from advanced import backtest, summarize_backtest, ranked_double_chance, five_two_leg_doubles, context_review
from history_store import connect, save_predictions, reconcile, load

st.set_page_config(page_title='ZicZac Predict AI',page_icon='app/static/icon.svg',layout='wide',initial_sidebar_state='collapsed')
st.markdown("""<style>
:root{color-scheme:dark}
.block-container{max-width:1200px;padding-top:1rem;padding-bottom:5rem}
h1,h2,h3{letter-spacing:-.035em}
[data-testid="stMetric"]{background:linear-gradient(135deg,#101c39,#1d1940);padding:15px;border:1px solid #36486b;border-radius:18px}
.stButton button,.stDownloadButton button{border-radius:14px;min-height:44px}
[data-testid="stTabs"] button{white-space:nowrap}
@media(max-width:680px){
.block-container{padding-left:.75rem;padding-right:.75rem;padding-top:.7rem}
h1{font-size:1.7rem!important}
h2{font-size:1.3rem!important}
[data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:.5rem}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:135px;flex:1 1 45%}
[data-testid="stMetric"]{padding:12px}
[data-testid="stMetricValue"]{font-size:1.45rem}
[data-testid="stTabs"] [role="tablist"]{overflow-x:auto}
.stDataFrame{max-width:100%;overflow-x:auto}
}
/* Illustrated football stadium, floodlights and pitch behind the dashboard. */
[data-testid="stAppViewContainer"]{
 background-image:linear-gradient(180deg,rgba(4,10,25,.63),rgba(5,10,22,.84) 52%,rgba(5,10,22,.91)),url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%201440%201000%22%20preserveAspectRatio%3D%22xMidYMid%20slice%22%3E%0A%3Cdefs%3E%0A%3ClinearGradient%20id%3D%22sky%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20stop-color%3D%22%23071226%22%2F%3E%3Cstop%20offset%3D%22.68%22%20stop-color%3D%22%230b1930%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%23071422%22%2F%3E%3C%2FlinearGradient%3E%0A%3CradialGradient%20id%3D%22light%22%3E%3Cstop%20stop-color%3D%22%23d5ecff%22%20stop-opacity%3D%22.75%22%2F%3E%3Cstop%20offset%3D%22.15%22%20stop-color%3D%22%236bb8ff%22%20stop-opacity%3D%22.23%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%234b9dff%22%20stop-opacity%3D%220%22%2F%3E%3C%2FradialGradient%3E%0A%3ClinearGradient%20id%3D%22pitch%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20stop-color%3D%22%230a3b3e%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%23071c2c%22%2F%3E%3C%2FlinearGradient%3E%0A%3C%2Fdefs%3E%0A%3Crect%20width%3D%221440%22%20height%3D%221000%22%20fill%3D%22url(%23sky)%22%2F%3E%0A%3Cpath%20d%3D%22M0%20390%20Q720%20110%201440%20390%20L1440%20750%20Q720%20570%200%20750Z%22%20fill%3D%22%230c1b35%22%20stroke%3D%22%2328466b%22%20stroke-width%3D%228%22%2F%3E%0A%3Cpath%20d%3D%22M0%20475%20Q720%20225%201440%20475%20M0%20545%20Q720%20305%201440%20545%20M0%20615%20Q720%20395%201440%20615%22%20fill%3D%22none%22%20stroke%3D%22%235476a1%22%20stroke-opacity%3D%22.24%22%20stroke-width%3D%2215%22%20stroke-dasharray%3D%228%2016%22%2F%3E%0A%3Cpath%20d%3D%22M0%20710%20Q720%20580%201440%20710%20L1440%201000%20H0Z%22%20fill%3D%22url(%23pitch)%22%2F%3E%0A%3Cpath%20d%3D%22M0%20800%20Q720%20690%201440%20800%20M720%20745%20V1000%20M510%20840%20Q720%20780%20930%20840%20Q1000%20885%20930%20930%20Q720%20980%20510%20930%20Q440%20885%20510%20840%22%20fill%3D%22none%22%20stroke%3D%22%238ee7dd%22%20stroke-opacity%3D%22.23%22%20stroke-width%3D%225%22%2F%3E%0A%3Cpath%20d%3D%22M0%20980%20Q720%20835%201440%20980%22%20fill%3D%22none%22%20stroke%3D%22%238ee7dd%22%20stroke-opacity%3D%22.16%22%20stroke-width%3D%226%22%2F%3E%0A%3Ccircle%20cx%3D%22125%22%20cy%3D%22205%22%20r%3D%22350%22%20fill%3D%22url(%23light)%22%2F%3E%3Ccircle%20cx%3D%221315%22%20cy%3D%22205%22%20r%3D%22350%22%20fill%3D%22url(%23light)%22%2F%3E%0A%3Cg%20fill%3D%22%23d9efff%22%20opacity%3D%22.55%22%3E%3Crect%20x%3D%2270%22%20y%3D%22165%22%20width%3D%22110%22%20height%3D%2217%22%20rx%3D%225%22%2F%3E%3Crect%20x%3D%221260%22%20y%3D%22165%22%20width%3D%22110%22%20height%3D%2217%22%20rx%3D%225%22%2F%3E%3C%2Fg%3E%0A%3Cg%20stroke%3D%22%2392c9ff%22%20stroke-opacity%3D%22.13%22%20stroke-width%3D%223%22%3E%3Cpath%20d%3D%22M110%20185%20L0%20770%20M150%20185%20L460%20840%20M1280%20185%20L980%20840%20M1340%20185%20L1440%20770%22%2F%3E%3C%2Fg%3E%0A%3C%2Fsvg%3E")!important;
 background-position:center top!important;
 background-size:cover!important;
 background-repeat:no-repeat!important;
 background-attachment:fixed!important;
}
[data-testid="stAppViewContainer"]::before{
 background:linear-gradient(90deg,transparent,rgba(45,108,170,.045) 50%,transparent)!important;
}
[data-testid="stMainBlockContainer"]{position:relative;z-index:1}
@media(max-width:680px){
 [data-testid="stAppViewContainer"]{background-position:center top!important;background-attachment:scroll!important;background-size:auto 100vh!important}
}

/* Football stadium atmosphere: CSS-only, fast and readable on phones. */
[data-testid="stAppViewContainer"]{
 background-color:#070e1e;
 background-image:
 radial-gradient(ellipse 68% 25% at 50% 0%,rgba(40,126,255,.28),transparent 85%),
 radial-gradient(ellipse 42% 30% at 95% 24%,rgba(131,56,232,.16),transparent 90%),
 radial-gradient(ellipse 44% 30% at 0% 55%,rgba(13,157,170,.10),transparent 90%),
 linear-gradient(180deg,#08152d 0%,#080d1b 48%,#090e1d 100%);
 background-attachment:fixed;
}
[data-testid="stAppViewContainer"]::before{
 content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
 background:repeating-linear-gradient(110deg,transparent 0 110px,rgba(100,167,255,.025) 111px,transparent 113px);
}
[data-testid="stHeader"]{background:rgba(7,14,30,.78);backdrop-filter:blur(14px)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101c36,#101326)}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px}
[data-testid="stMetric"]{
 background:linear-gradient(140deg,rgba(22,46,86,.95),rgba(29,20,62,.95))!important;
 box-shadow:0 8px 26px rgba(0,0,0,.18),inset 0 1px rgba(152,191,255,.12);
 border:1px solid rgba(107,140,213,.30)!important;
}
div[data-testid="stExpander"],div[data-testid="stForm"]{
 border-radius:18px;border-color:rgba(102,139,210,.3);
 background:rgba(11,21,43,.55);
}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:#b7a2ff}
div[data-testid="stTabs"] [role="tablist"]{border-bottom-color:rgba(123,151,216,.25)}
.stButton button[kind="primary"],.stDownloadButton button{
 background:linear-gradient(120deg,#5145d9,#7d37d7);
 border:1px solid rgba(176,160,255,.35);color:white;
 box-shadow:0 5px 18px rgba(88,68,214,.18);
}
[data-testid="stMarkdownContainer"] a{color:#8fcaff}
@media(max-width:680px){
 [data-testid="stAppViewContainer"]{background-attachment:scroll}
 [data-testid="stMetric"]{box-shadow:none}
}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}

</style>""",unsafe_allow_html=True)
st.markdown("""<style>
/* ZicZac Predict AI — compact, mobile-first dashboard */
.zz-head{background:linear-gradient(145deg,#0a1634,#101b41 58%,#181340);border:1px solid #3e4c84;border-radius:22px;padding:18px 16px 16px;text-align:center;box-shadow:0 12px 30px #0005;margin-bottom:16px}
.zz-logo{font-size:37px;line-height:1.15;filter:drop-shadow(0 0 14px #5f82ff70)}
.zz-name{font-weight:900;font-size:clamp(25px,5vw,36px);line-height:1.25;color:#f9fbff;letter-spacing:-.045em}
.zz-name span{color:#ff547f}
.zz-sub{color:#b6c6e8;font-size:13px;margin-top:4px}
.zz-by{color:#9fd6ff;font-weight:700;font-size:12px;letter-spacing:.08em;margin-top:8px}
.zz-nav{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;margin-top:18px}
.zz-nav-item{background:linear-gradient(150deg,#172449,#101a38);border:1px solid #344573;border-radius:13px;padding:11px 3px 9px;color:#f1f5ff;font-size:11px;font-weight:700}
.zz-nav-item:first-child{background:linear-gradient(130deg,#5744f2,#8737d6);box-shadow:0 0 16px #7551ff55}
.zz-nav-item .zz-ico{display:block;font-size:21px;margin-bottom:4px}
.zz-stadium{position:relative;overflow:hidden;border-radius:19px;min-height:165px;margin:14px 0 0;background:radial-gradient(ellipse at 15% 18%,#90d8ff70,transparent 18%),radial-gradient(ellipse at 85% 18%,#90d8ff70,transparent 18%),linear-gradient(180deg,#0a1640 0%,#152f60 51%,#075039 88%,#093322);border:1px solid #425da0}
.zz-stadium:before{content:"";position:absolute;inset:42% 0 15%;background:repeating-linear-gradient(0deg,#182a4c 0 7px,#30466a 8px 10px);clip-path:polygon(0 25%,50% 0,100% 25%,100% 100%,0 100%);opacity:.75}
.zz-stadium:after{content:"";position:absolute;bottom:-40px;left:-8%;width:116%;height:85px;border:2px solid #c4f9d060;border-radius:50%;background:repeating-linear-gradient(90deg,#116044 0 16%,#0e533e 16% 32%)}
.zz-stadium-copy{position:relative;z-index:1;padding:20px 10px 35px;text-align:center;text-shadow:0 3px 12px #000}
.zz-stadium-title{font-size:clamp(22px,5vw,30px);font-weight:900;color:white;line-height:1.25}
.zz-stadium-desc{color:#e6eeff;font-size:12px;margin-top:7px}
.zz-feature-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:12px 0 18px}
.zz-feature{background:linear-gradient(135deg,#192750,#111b38);border:1px solid #364d86;border-radius:15px;padding:14px 12px;color:#fff;min-height:74px;display:flex;align-items:center;gap:10px}
.zz-feature-icon{font-size:26px}
.zz-feature strong{display:block;font-size:14px}.zz-feature small{color:#b6c7e6;font-size:11px}
.zz-section-title{font-size:22px;font-weight:850;color:#f8faff;margin:15px 0 9px}
.zz-match-card{border:1px solid #42538a;border-radius:17px;background:linear-gradient(140deg,#19234a,#111a35);padding:15px;margin-bottom:11px;color:#f6f7ff}
.zz-match-top{font-size:12px;color:#c0d2ff;margin-bottom:10px}
.zz-match-teams{font-weight:800;font-size:17px;line-height:1.45;text-align:center}
.zz-match-score{display:inline-block;background:linear-gradient(135deg,#0f756a,#1a9b70);border-radius:10px;padding:6px 14px;font-size:20px;margin:9px auto;color:white}
.zz-match-meta{text-align:center;font-size:12px;color:#aabde5}
@media(max-width:600px){.zz-head{padding:14px 10px}.zz-nav{gap:5px}.zz-nav-item{font-size:10px;padding:9px 2px}.zz-nav-item .zz-ico{font-size:19px}.zz-stadium{min-height:155px}.zz-feature{padding:11px 9px;gap:7px}.zz-feature strong{font-size:12px}.zz-feature-icon{font-size:21px}}
/* Readability upgrade for iPhone: stronger hierarchy, contrast and spacing. */
.zz-head{background:linear-gradient(155deg,#13254c 0%,#14254c 58%,#211c53 100%);border:2px solid #6e87cb;box-shadow:0 16px 38px #0009}
.zz-logo{font-size:46px}
.zz-name{font-size:clamp(29px,6.5vw,40px);text-shadow:0 2px 8px #0008}
.zz-sub{font-size:clamp(15px,3.5vw,18px);color:#f0f5ff;font-weight:600}
.zz-by{font-size:14px;color:#c7e7ff;letter-spacing:.09em}
.zz-nav{gap:9px}
.zz-nav-item{background:#1d315c;border:2px solid #6481bc;color:#fff;font-size:13px;padding:15px 4px 13px;box-shadow:inset 0 1px #ffffff1a}
.zz-nav-item:first-child{background:linear-gradient(130deg,#6549ef,#9944dd);border-color:#ac8bff}
.zz-nav-item .zz-ico{font-size:27px}
.zz-stadium{border:2px solid #7999dc;background:linear-gradient(180deg,#132c59 0%,#193b6c 58%,#086244 100%);min-height:205px}
.zz-stadium-copy{padding:24px 12px 48px;background:linear-gradient(180deg,#07142b65,#07142b20)}
.zz-stadium-title{font-size:clamp(26px,6vw,37px);text-shadow:0 3px 7px #000,0 0 14px #0009}
.zz-stadium-desc{font-size:clamp(14px,3.2vw,17px);font-weight:650;color:#fff;text-shadow:0 2px 7px #000}
.zz-section-title{font-size:clamp(24px,5vw,30px)}
.zz-feature{background:#1a2c55;border-color:#6c86bd}
.zz-feature strong{font-size:16px}.zz-feature small{font-size:13px;color:#e0eaff}
.zz-match-card{background:#192c55;border-color:#6f8dcb}
.zz-match-top,.zz-match-meta{font-size:14px;color:#e2ecff}
.zz-match-teams{font-size:20px}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{line-height:1.65}
[data-testid="stWidgetLabel"] p{font-size:16px!important;font-weight:650;color:#f2f6ff!important}
[data-testid="stExpander"]{background:#142443!important;border:1px solid #536fa5!important}
[data-baseweb="select"]>div,[data-testid="stTextInput"] input{background:#1a2b4b!important;border-color:#6786b8!important;color:#fff!important}
@media(max-width:680px){
.zz-head{padding:18px 12px}
.zz-nav{gap:6px}
.zz-nav-item{font-size:12px;padding:13px 2px;min-height:86px}
.zz-nav-item .zz-ico{font-size:24px;margin-bottom:8px}
.zz-stadium{min-height:190px}
.zz-feature strong{font-size:14px}
.zz-feature-icon{font-size:25px}
[data-testid="stMarkdownContainer"] p{font-size:15px}
}
</style>
<div class="zz-head"><div class="zz-logo">⚽</div><div class="zz-name">ZicZac Predict <span>AI</span></div><div class="zz-sub">AI Football Predictions &amp; Exact Scores</div><div class="zz-by">PREDICTION BY ZICZAC</div>
<div class="zz-nav"><a class="zz-nav-item" href="?view=home"><span class="zz-ico">⌂</span>Home</a><a class="zz-nav-item" href="?view=predict"><span class="zz-ico">🎯</span>Predict</a><a class="zz-nav-item" href="?view=doubles"><span class="zz-ico">🔗</span>Doubles</a><a class="zz-nav-item" href="?view=archive"><span class="zz-ico">📊</span>Archive</a><a class="zz-nav-item" href="?view=more"><span class="zz-ico">☰</span>More</a></div>
<div class="zz-stadium"><div class="zz-stadium-copy"><div class="zz-stadium-title">Today's Football<br>Predictions</div><div class="zz-stadium-desc">⚡ Data-driven analysis · Exact scores · Worldwide matches</div><div style="font-size:28px;margin-top:9px">⚽</div></div></div></div>
""",unsafe_allow_html=True)

try:key=st.secrets.get('FOOTBALL_DATA_API_KEY','')
except Exception:key=''
key=key or os.environ.get('FOOTBALL_DATA_API_KEY','')
if not key:
    st.warning('Add a football-data.org API key in Streamlit Secrets: `FOOTBALL_DATA_API_KEY = "your_key"`. No fabricated fixtures or scores are shown.')
    st.link_button('Get a free API key','https://www.football-data.org/client/register')
    st.stop()
with st.sidebar:
    st.header('Model settings')
    days=st.slider('Upcoming days',1,14,14)
    st.caption('Fixtures are sorted by kickoff, earliest first. Today and tomorrow are included.')
    max_games=st.slider('Display fixtures',5,80,30,5)
    min_prob=st.slider('Minimum double-chance probability',0.50,0.95,0.65,0.01)
    use_secondary=st.checkbox('Use API-Football backup',value=True,help='Uses a separate free daily quota. Requests are cached for six hours.')
    if st.button('Refresh football data'):
        st.rerun()
    st.caption('Data cached for 1 hour. Refresh reruns the page without bypassing API cooldowns. Saved snapshots may be older.')
try:secondary_key=st.secrets.get('API_FOOTBALL_KEY','') or os.environ.get('API_FOOTBALL_KEY','')
except Exception:secondary_key=os.environ.get('API_FOOTBALL_KEY','')
try:sportmonks_key=st.secrets.get('THIRD_FOOTBALL_API_KEY','') or os.environ.get('THIRD_FOOTBALL_API_KEY','')
except Exception:sportmonks_key=os.environ.get('THIRD_FOOTBALL_API_KEY','')
st.subheader('🌍 Worldwide fixtures by Sydney date')
st.caption('Date-first discovery combines API-Football, TheSportsDB and Sportmonks (when configured). Each provider is limited to its subscription coverage.')
choice=st.radio('Fixture date', ['Today','Tomorrow','Next 3 days','Next 7 days','Next 14 days'],horizontal=True)
today=datetime.now(SYDNEY).date()
start_day=today+timedelta(days=1) if choice=='Tomorrow' else today
span=3 if choice=='Next 3 days' else 7 if choice=='Next 7 days' else 14 if choice=='Next 14 days' else 1
with st.expander('Worldwide match finder',expanded=True):
    try:
        global_matches,global_errors=worldwide(start_day,span)
        if use_secondary and secondary_key:
            api_matches,api_errors=global_fixtures(secondary_key,start_day,span)
            global_errors+=api_errors
            seen={(m['kickoff'].strftime('%Y-%m-%dT%H:%M'),m['home'].casefold(),m['away'].casefold()) for m in api_matches}
            global_matches=api_matches+[m for m in global_matches if (m['kickoff'].strftime('%Y-%m-%dT%H:%M'),m['home'].casefold(),m['away'].casefold()) not in seen]
            global_matches.sort(key=lambda m:m['kickoff'])
        if sportmonks_key:
            sm_matches,sm_errors=sportmonks_fixtures(sportmonks_key,start_day,span)
            global_errors+=sm_errors
            known={(m['kickoff'].strftime('%Y-%m-%dT%H:%M'),m['home'].casefold(),m['away'].casefold()) for m in global_matches}
            global_matches += [m for m in sm_matches if (m['kickoff'].strftime('%Y-%m-%dT%H:%M'),m['home'].casefold(),m['away'].casefold()) not in known]
            global_matches.sort(key=lambda m:m['kickoff'])
        country_options=sorted({m['country'] for m in global_matches})
        country=st.selectbox('Country', ['All countries']+country_options)
        term=st.text_input('Search team or league',placeholder='Type a team or competition')
        filtered=[m for m in global_matches if m['kickoff']>datetime.now(timezone.utc) and (country=='All countries' or m['country']==country)
                  and (not term or term.lower() in (m['home']+' '+m['away']+' '+m['league']).lower())]
        st.caption(f'{len(filtered)} returned matches · Sydney time · Results depend on free-plan coverage and API quotas.')
        st.caption('Tick any match to select it for analysis below. Matches without sufficient history can still be selected.')
        picked_ids=set(st.session_state.get('worldwide_picked_ids', []))
        for m in filtered:
            event_id=str(m['source'])+':'+str(m['id'])
            selected_now=st.checkbox(f"{m['home']} vs {m['away']} · {m['local']:%a %d %b, %I:%M %p} Sydney · {m['league']} ({m['country']})", value=event_id in picked_ids, key='finder_'+event_id)
            if selected_now:picked_ids.add(event_id)
            else:picked_ids.discard(event_id)
        st.session_state['worldwide_picked_ids']=list(picked_ids)
        if not filtered:st.info('No matches returned for these filters. This does not mean no games are scheduled.')
        if global_errors:st.warning('Some date requests failed: '+'; '.join(global_errors[:3]))
    except Exception as exc:
        st.warning(f'Worldwide free fixture source unavailable: {exc}')
st.divider()
try:comps=competitions(key)
except Exception as e:st.error(f'Unable to retrieve competitions: {e}');st.stop()
names={f"{c.get('area',{}).get('name','World')} · {c['name']} ({c['code']})":c['code'] for c in comps}
selected=st.multiselect('Competitions to monitor',list(names),default=[n for n in names if any(token in n for token in ('Premier League (PL)','Primera Division (PD)','Champions League (CL)'))][:3] or list(names)[:min(3,len(names))])
if not selected:st.info('Choose one or more competitions for statistical modelling. Worldwide fixture discovery above remains available.');st.stop()
if use_secondary and not secondary_key:st.warning('API-Football backup is enabled but API_FOOTBALL_KEY is missing from Secrets.')
if len(selected)>8:st.warning('Select up to eight competitions to protect free-tier API limits.');st.stop()
now=datetime.now(timezone.utc)
rows=[];histories={};errors=[];fixture_counts=[]
with st.spinner('Collecting fixtures and completed results...'):
    for name in selected:
        code=names[name]
        upcoming=[];past=[];source='football-data.org'
        try:
            upcoming=fixtures(key,code,(now.date()-timedelta(days=1)).isoformat(),(now.date()+timedelta(days=15)).isoformat())
        except Exception as e:
            errors.append(f'{name}: primary fixtures unavailable ({e})')
        try:
            past=history(key,code)
        except Exception as e:
            errors.append(f'{name}: primary historical data unavailable ({e})')
        # The backup is queried only when primary coverage is missing.
        if use_secondary and secondary_key and code in supported_codes() and (not upcoming or not past):
            try:
                if not upcoming:
                    upcoming=secondary_fixtures(secondary_key,code,now,14)
                    if upcoming:source='API-Football backup'
                if not past:
                    past=secondary_history(secondary_key,code,now)
            except Exception as e:
                errors.append(f'{name}: API-Football backup unavailable ({e})')
        histories[name]=past
        visible=0
        for m in upcoming:
            kickoff=m.get('utcDate','')
            if not kickoff or kickoff < now.isoformat().replace('+00:00','Z'):continue
            if kickoff > (now+timedelta(days=days)).isoformat().replace('+00:00','Z'):continue
            visible+=1
            has_teams=bool((m.get('homeTeam') or {}).get('id') and (m.get('awayTeam') or {}).get('id'))
            matching_past=[x for x in past if x.get('provider','football-data.org')==m.get('provider','football-data.org')]
            try:
                result=predict(m,matching_past) if has_teams and matching_past else None
            except Exception:
                result=None
            rows.append({'league':name,'fixture':m,'result':result,'kickoff':kickoff})
        fixture_counts.append({'Competition':name,'Source':source,'API scheduled fixtures':len(upcoming),'Within selected time window':visible})
rows.sort(key=lambda r:r['kickoff'])
# Combine date-first discovery with statistically modelled fixtures.
# Do not equate team IDs across providers or invent model results.
worldwide_rows=[]
try:
    worldwide_rows=global_matches
except NameError:
    pass
def normalized_team(name):
    import re
    return re.sub(r'[^a-z0-9]+','',str(name).casefold())
model_index={}
for r in rows:
    m=r['fixture']
    try:
        utc_day=r['kickoff'][:10]
        identity=(utc_day,normalized_team(m['homeTeam']['name']),normalized_team(m['awayTeam']['name']))
        model_index[identity]=r
    except (KeyError,TypeError):
        continue
discoverable=[]
for m in worldwide_rows:
    identity=(m['kickoff'].strftime('%Y-%m-%d'),normalized_team(m['home']),normalized_team(m['away']))
    match=model_index.get(identity)
    if match is None:
        # Also match by Sydney calendar date to avoid UTC midnight boundary mismatches.
        for r in rows:
            f=r['fixture']
            if (normalized_team(f['homeTeam']['name'])==normalized_team(m['home']) and
                normalized_team(f['awayTeam']['name'])==normalized_team(m['away']) and
                abs((datetime.fromisoformat(r['kickoff'].replace('Z','+00:00'))-m['kickoff']).total_seconds())<7200):
                match=r
                break
    discoverable.append({'name':f"{m['home']} vs {m['away']} · {m['local']:%d %b %I:%M %p} Sydney · {m['league']}",
                         'model':match,'event':m})

# Allow the user to select fixtures explicitly; no match is invented.
eligible=[r for r in rows if r.get('result')]
def fixture_label(r):
    m=r['fixture']
    return f"{m['homeTeam'].get('name','Home')} vs {m['awayTeam'].get('name','Away')} · {r['kickoff'][:16]} UTC"
labels={f"{r['fixture'].get('id')} · {fixture_label(r)}":r for r in eligible}
st.subheader('🌐 Choose matches from worldwide discovery')
st.caption('Free worldwide discovery and modelled fixtures are combined below. A discovered match without verified historical data cannot receive a statistical exact-score prediction.')
worldwide_options={item['name']:item for item in discoverable}
picked_ids=set(st.session_state.get('worldwide_picked_ids', []))
preselected=[name for name,item in worldwide_options.items() if str(item['event']['source'])+':'+str(item['event']['id']) in picked_ids]
selected_worldwide=st.multiselect('Search and select worldwide fixtures',list(worldwide_options),default=preselected,help='Matches selected in the finder appear here. Exact-score doubles require verified model history.')
# Model selected API-Football worldwide fixtures on demand to conserve quota.
if st.session_state.get('worldwide_model_version') != 'date-range-v3':
    st.session_state['worldwide_models']={}
    st.session_state['worldwide_model_version']='date-range-v3'
if 'worldwide_models' not in st.session_state:
    st.session_state['worldwide_models']={}
if selected_worldwide:
    if st.button('🔎 Analyse selected worldwide matches',help='Fetch verified recent results for selected API-Football matches; uses your API quota.'):
        st.session_state['worldwide_models']={k:v for k,v in st.session_state['worldwide_models'].items() if k not in selected_worldwide}
        for name in selected_worldwide[:8]:
            item=worldwide_options[name]
            if item['model'] and item['model'].get('result'):
                continue
            row,error=model_worldwide(secondary_key,item['event'])
            st.session_state['worldwide_models'][name]={'row':row,'error':error}
    for name in selected_worldwide:
        cached=st.session_state['worldwide_models'].get(name)
        if cached and 'Last parameter' in str(cached.get('error','')):
            st.session_state['worldwide_models'].pop(name,None)
            cached=None
        if cached:
            if cached['row']:
                worldwide_options[name]['model']=cached['row']
                histories[name]=cached['row']['past']
                st.success(f"Prediction ready: {name}")
            elif cached['error']:
                st.info(f"{name}: {cached['error']}")
for name in selected_worldwide:
    candidate=worldwide_options[name]['model']
    if candidate and candidate.get('result') and not any(str(r['fixture'].get('provider'))+':'+str(r['fixture'].get('id'))==str(candidate['fixture'].get('provider'))+':'+str(candidate['fixture'].get('id')) for r in rows):
        rows.append(candidate)
rows.sort(key=lambda r:r['kickoff'])
unmodelled=[worldwide_options[name] for name in selected_worldwide if not worldwide_options[name]['model'] or not worldwide_options[name]['model'].get('result')]
if unmodelled:
    st.info(f"{len(unmodelled)} selected worldwide match(es) do not have sufficient compatible historical data for exact-score predictions. They are displayed for discovery only.")
st.subheader('🎯 Choose your matches for detailed analysis')
chosen_labels=st.multiselect('Select fixtures to analyse and combine',list(labels),default=list(labels)[:min(8,len(labels))],help='Select at least two fixtures for exact-score doubles. The available fixtures depend on your API plan and chosen competitions.')
chosen_rows=[labels[k] for k in chosen_labels]
for name in selected_worldwide:
    candidate=worldwide_options[name]['model']
    if candidate and candidate.get('result') and candidate not in chosen_rows:
        chosen_rows.append(candidate)
if selected_worldwide and not any(worldwide_options[name]['model'] and worldwide_options[name]['model'].get('result') for name in selected_worldwide):
    st.caption('To calculate doubles, choose two fixtures with verified model data from the detailed-analysis selector.')

for e in errors:st.warning(e)
with st.expander('Why are some matches missing? · API coverage diagnostics'):
    st.caption('API-Football backup is used when primary fixture/history data is missing. Both providers have free-plan limits. Backup snapshots may be outdated.')
    if fixture_counts:st.dataframe(pd.DataFrame(fixture_counts),hide_index=True,use_container_width=True)
    st.caption('Times are shown in UTC. Matches without sufficient history appear without predictions. Provider team IDs are never mixed in score modelling.')
conn=None
try:
    conn=connect()
    save_predictions(conn,[r for r in rows if r['result']])
    for past in histories.values():reconcile(conn,past)
except Exception as e:st.warning(f'Local prediction snapshot storage unavailable: {e}')

c1,c2,c3,c4=st.columns(4)
c1.metric('Upcoming fixtures',len(rows));c2.metric('Modelled fixtures',sum(bool(r['result']) for r in rows))
c3.metric('Competitions loaded',len(histories));c4.metric('Saved forecasts',len(load(conn)) if conn else 'N/A')
st.markdown("""<div class="zz-feature-grid">
<div class="zz-feature"><div class="zz-feature-icon">🎯</div><div><strong>Exact Scores</strong><small>Model probabilities</small></div></div>
<div class="zz-feature"><div class="zz-feature-icon">🔗</div><div><strong>Five Doubles</strong><small>Two-leg combinations</small></div></div>
<div class="zz-feature"><div class="zz-feature-icon">📈</div><div><strong>Live Form</strong><small>Team stats &amp; H2H</small></div></div>
<div class="zz-feature"><div class="zz-feature-icon">🏆</div><div><strong>Accuracy</strong><small>Backtesting &amp; archive</small></div></div>
</div><div class="zz-section-title">⚽ Upcoming match previews</div>""",unsafe_allow_html=True)
dashboard_rows=[r for r in rows if r.get('result') and r.get('kickoff','') >= datetime.now(timezone.utc).isoformat()[:10]][:3]
if not dashboard_rows:
    st.caption('No prediction-ready fixtures available right now. Check the worldwide finder or upcoming predictions below.')
for preview in dashboard_rows:
    match=preview['fixture']
    pred=preview['result']
    best=pred['scores'][0]
    home=match['homeTeam'].get('shortName') or match['homeTeam'].get('name','Home')
    away=match['awayTeam'].get('shortName') or match['awayTeam'].get('name','Away')
    import html
    st.markdown('<div class="zz-match-card"><div class="zz-match-top">🏟️ '+html.escape(str(preview.get('league','Football')))+' · '+html.escape(str(preview.get('kickoff',''))[:16])+' UTC</div><div class="zz-match-teams">'+html.escape(home)+' &nbsp; VS &nbsp; '+html.escape(away)+'</div><div style="text-align:center"><span class="zz-match-score">'+str(best[0])+' – '+str(best[1])+'</span></div><div class="zz-match-meta">Top exact score: '+f'{best[2]:.1%}'+' · Home '+f"{pred['home_win']:.0%}"+' · Draw '+f"{pred['draw']:.0%}"+' · Away '+f"{pred['away_win']:.0%}"+'</div></div>',unsafe_allow_html=True)
st.caption('Model probabilities are estimates, not guaranteed results. Use the tabs below for full analysis.')
st.caption('New analysis: last 10 overall, last 10 home/away, available H2H, and two mathematical goal estimators. These changes require walk-forward backtesting before any improvement in accuracy can be claimed.')
# The header links change the URL view; only the corresponding sections are rendered.
view=st.query_params.get('view','home')
if view not in ('home','predict','doubles','archive','more'):
    view='home'
st.markdown("""<style>.zz-nav-item{text-decoration:none!important;display:block;cursor:pointer;transition:background .15s,border-color .15s}.zz-nav-item:hover{background:#31477b;border-color:#a7c5ff}.zz-nav-item:first-child{background:#1d315c;border-color:#6481bc}.zz-nav-item[href="?view="""+view+""""]{background:linear-gradient(130deg,#6549ef,#9944dd)!important;border-color:#c1aaff!important;box-shadow:0 0 18px #7856ff77}</style>""",unsafe_allow_html=True)
st.caption({'home':'🏠 Home · Upcoming matches','predict':'🎯 Predict · Match analysis','doubles':'🔗 Doubles · Combined predictions','archive':'📊 Archive · Results and backtesting','more':'☰ More · Model information'}[view])
tabs=[st.container() for _ in range(7)]

def match_name(m):return f"{m['homeTeam'].get('shortName') or m['homeTeam'].get('name','Home')} vs {m['awayTeam'].get('shortName') or m['awayTeam'].get('name','Away')}"
if view in ['home','predict']:
    with tabs[0]:if view in ['home','predict']:
    with tabs[0]:
    st.caption('Prediction dates use Sydney local time. Worldwide selected matches appear here after successful history analysis.')
    pred_window=st.radio('Show predictions for', ['Today','Tomorrow','Next 3 days','Next 7 days','Next 14 days'],horizontal=True,key='prediction_window')
    pred_start=today+timedelta(days=1) if pred_window=='Tomorrow' else today
    pred_span={'Today':1,'Tomorrow':1,'Next 3 days':3,'Next 7 days':7,'Next 14 days':14}[pred_window]
    def in_prediction_window(r):
        try:
            dt=datetime.fromisoformat(r['kickoff'].replace('Z','+00:00')).astimezone(SYDNEY)
            return pred_start<=dt.date()<pred_start+timedelta(days=pred_span)
        except (ValueError,KeyError,TypeError):return False
    visible_predictions=[r for r in rows if r.get('result') and in_prediction_window(r)]
    visible_fixtures=[r for r in rows if in_prediction_window(r)]
    st.caption(f'{len(visible_fixtures)} fixtures in monitored/modelled competitions · {len(visible_predictions)} prediction-ready matches for {pred_window.lower()}.')
    if visible_fixtures and not visible_predictions:
        st.warning('Fixtures exist for this date, but the available historical data is insufficient for the prediction model.')
    if not visible_predictions:st.info('No prediction-ready matches for this period from the available API history. Try another date or analyse a worldwide fixture above; this does not mean no football is scheduled.')
    if visible_fixtures:
        with st.expander('Show scheduled fixtures without predictions'):
            for r in visible_fixtures:
                if not r.get('result'):
                    st.write(f"{fixture_label(r)} — insufficient verified history")
    for r in visible_predictions[:max_games]:
        m,p=r['fixture'],r['result']
        with st.container(border=True):
            st.caption(f"{r['league']} · {r['kickoff']} UTC")
            st.subheader(match_name(m))
            if p is None:st.info('Insufficient historical results for this fixture.');continue
            best=p['scores'][0]
            cols=st.columns(5)
            for col,label,val in zip(cols,['Top exact score','Home win','Draw','Away win','Expected goals'],
                                     [f'{best[0]}–{best[1]} ({best[2]:.1%})',f"{p['home_win']:.1%}",f"{p['draw']:.1%}",f"{p['away_win']:.1%}",f"{p['home_xg']:.2f} / {p['away_xg']:.2f}"]):col.metric(label,val)
            with st.expander('Top 10 exact scores · human-style reasoning · recent form'):
                st.dataframe(pd.DataFrame([{'Score':f'{h}-{a}','Probability %':round(q*100,2)} for h,a,q in p['scores']]),hide_index=True,use_container_width=True)
                for note in context_review(m,p,r.get('past',histories.get(r['league'],[]))):st.write('• '+note)
                st.dataframe(pd.DataFrame([{'Team':'Home · last 10 overall',**p['home_stats']},{'Team':'Away · last 10 overall',**p['away_stats']},{'Team':'Home · last 10 at home',**(p.get('home_venue_stats') or {})},{'Team':'Away · last 10 away',**(p.get('away_venue_stats') or {})}]),hide_index=True,use_container_width=True)
                st.caption('Last 10 is the maximum available per team and venue. Results are shown newest first. Venue averages are shrunk toward overall form for smaller samples.')
                col_h,col_a=st.columns(2)
                with col_h:
                    st.markdown('**Home team — previous 10 matches**')
                    st.dataframe(pd.DataFrame(p.get('home_last10',[])),hide_index=True,use_container_width=True)
                    st.markdown('**Home team — previous 10 home matches**')
                    st.dataframe(pd.DataFrame(p.get('home_home10',[])),hide_index=True,use_container_width=True)
                with col_a:
                    st.markdown('**Away team — previous 10 matches**')
                    st.dataframe(pd.DataFrame(p.get('away_last10',[])),hide_index=True,use_container_width=True)
                    st.markdown('**Away team — previous 10 away matches**')
                    st.dataframe(pd.DataFrame(p.get('away_away10',[])),hide_index=True,use_container_width=True)
                st.markdown('**Last 10 head-to-head meetings (if available)**')
                if p.get('h2h_last10'):
                    st.dataframe(pd.DataFrame(p['h2h_last10']),hide_index=True,use_container_width=True)
                    st.caption('H2H goals are oriented to the upcoming home/away teams, even if prior meetings used the opposite venue.')
                else:
                    st.caption('No verified head-to-head matches in the accessible history.')
                st.markdown('**Mathematical model comparison**')
                methods=p.get('methods',{})
                if methods:
                    st.dataframe(pd.DataFrame([{'Estimator':name,'Expected home goals':round(vals[0],2),'Expected away goals':round(vals[1],2)} for name,vals in methods.items()]+[{'Estimator':'Blended forecast','Expected home goals':round(p['home_xg'],2),'Expected away goals':round(p['away_xg'],2)}]),hide_index=True,use_container_width=True)
                st.caption('Data quality: '+str(p.get('data_quality','unknown'))+' · Model: '+str(p.get('model_version','legacy'))+'. H2H is descriptive, not an extra goal multiplier. Injuries and lineups are not independently verified.')
if view in ['doubles']:
    with tabs[1]:if view in ['doubles']:
    with tabs[1]:
    st.subheader('Five highest-probability double-chance selections')
    st.caption('1X = home or draw · X2 = away or draw · 12 = either team wins. These are NOT exact-score doubles.')
    picks=ranked_double_chance(rows,min_prob)
    if not picks:st.info('Not enough eligible fixtures at the selected probability threshold. Reduce the threshold or extend the date range.')
    else:
        st.dataframe(pd.DataFrame([{'Fixture':match_name(x['fixture']),'Market':x['selection'],'Model probability':f"{x['probability']:.1%}",'Kickoff UTC':x['kickoff'],'Competition':x['league']} for x in picks]),hide_index=True,use_container_width=True)
        st.caption('Selections are ranked by the model, not verified bookmaker odds or guaranteed wins.')
if view in ['doubles']:
    with tabs[2]:if view in ['doubles']:
    with tabs[2]:
    st.subheader('Five two-leg double combinations')
    st.caption('Each double combines two different matches. The displayed joint probability multiplies model probabilities and assumes independence; real outcomes may be correlated.')
    pairs=five_two_leg_doubles(rows)
    if not pairs:st.info('At least two eligible upcoming fixtures are required.')
    else:
        st.dataframe(pd.DataFrame([{'Double':i+1,'Leg 1':f"{match_name(x['leg1']['fixture'])} · {x['leg1']['selection']}",
          'Leg 2':f"{match_name(x['leg2']['fixture'])} · {x['leg2']['selection']}",
          'Estimated joint probability':f"{x['joint_probability_independent']:.1%}"} for i,x in enumerate(pairs)]),hide_index=True,use_container_width=True)
        st.caption('The top combinations may reuse individual fixtures across different doubles. These are not five independent opportunities.')
if view in ['archive']:
    with tabs[3]:if view in ['archive']:
    with tabs[3]:
    st.subheader('Historical walk-forward validation')
    st.write('For each completed match, the model uses only results strictly earlier than kickoff. This avoids using future results to predict past games.')
    league=st.selectbox('Backtest competition',list(histories),key='bt_league') if histories else None
    if league and st.button('Run backtest'):
        with st.spinner('Running historical predictions...'):
            records=backtest(histories[league],min_prior=35,max_tests=120)
        summary=summarize_backtest(records)
        if not summary:st.warning('Not enough accessible completed matches with prior team history for a valid backtest.')
        else:
            st.caption(f"Tested {summary['matches']} historical matches from the accessible API history. Not an independent future guarantee.")
            cols=st.columns(4)
            cols[0].metric('Exact-score hit rate',f"{summary['exact_score_accuracy']:.1%}")
            cols[1].metric('1X2 result accuracy',f"{summary['result_accuracy']:.1%}")
            cols[2].metric('Brier score (lower better)',f"{summary['mean_brier']:.3f}")
            cols[3].metric('Goals MAE (lower better)',f"{summary['mean_absolute_goal_error']:.2f}")
            df=pd.DataFrame(records)
            st.dataframe(df,hide_index=True,use_container_width=True)
            st.download_button('Download backtest CSV',df.to_csv(index=False).encode(),'backtest.csv','text/csv')
if view in ['archive']:
    with tabs[4]:if view in ['archive']:
    with tabs[4]:
    st.subheader('📊 Prediction Accuracy Dashboard')
    st.caption('Only saved forecasts with verified final scores count. Pending matches are excluded. Local storage on free Streamlit hosting can reset.')
    if conn:
        archive=pd.DataFrame(load(conn))
        if archive.empty:
            st.info('No saved predictions yet.')
        else:
            def score_outcome(score):
                try:
                    h,a=map(int,str(score).split('-'))
                    return 'Home' if h>a else 'Away' if a>h else 'Draw'
                except (ValueError,TypeError):
                    return None
            completed=archive[archive['actual_score'].notna()].copy()
            completed=completed[completed['actual_score'].apply(score_outcome).notna()].copy()
            total=len(archive)
            resolved=len(completed)
            if resolved:
                completed['exact_hit']=completed['predicted_score']==completed['actual_score']
                completed['result_hit']=completed['predicted_score'].apply(score_outcome)==completed['actual_score'].apply(score_outcome)
                def top_five_hit(r):
                    try:
                        scores=json.loads(r['scores_json'] or '[]')
                        return r['actual_score'] in {f"{int(x[0])}-{int(x[1])}" for x in scores[:5]}
                    except (ValueError,TypeError,IndexError):
                        return False
                completed['top5_hit']=completed.apply(top_five_hit,axis=1)
                cols=st.columns(4)
                cols[0].metric('Completed matches',resolved)
                cols[1].metric('Exact-score hit rate',f"{completed['exact_hit'].mean():.1%}")
                cols[2].metric('1X2 result accuracy',f"{completed['result_hit'].mean():.1%}")
                cols[3].metric('Top-five score hit rate',f"{completed['top5_hit'].mean():.1%}")
                st.caption(f'{total-resolved} forecasts awaiting results. Rates are based only on {resolved} completed matches.')
                st.subheader('✅ Correct predictions · ❌ Wrong predictions')
                st.caption('Exact score means both teams’ predicted goals matched the verified final score. Match result (1X2) means only the winner/draw was correct.')
                accuracy_type=st.radio('What should count as correct?', ['Exact score','Match result (1X2)','Top 5 exact scores'],horizontal=True,key='accuracy_type')
                hit_column={'Exact score':'exact_hit','Match result (1X2)':'result_hit','Top 5 exact scores':'top5_hit'}[accuracy_type]
                correct_count=int(completed[hit_column].sum())
                wrong_count=resolved-correct_count
                c_ok,c_bad,c_pending=st.columns(3)
                c_ok.metric('✅ Correct',correct_count)
                c_bad.metric('❌ Wrong',wrong_count)
                c_pending.metric('⏳ Pending',total-resolved)
                accuracy_filter=st.radio('Show predictions', ['All completed','✅ Correct predictions','❌ Wrong predictions','⏳ Awaiting results'],horizontal=True,key='accuracy_filter')
                details=archive.copy() if accuracy_filter=='⏳ Awaiting results' else completed.copy()
                if accuracy_filter=='⏳ Awaiting results':
                    details=details[details['actual_score'].isna()].copy()
                elif accuracy_filter=='✅ Correct predictions':
                    details=details[details[hit_column]].copy()
                elif accuracy_filter=='❌ Wrong predictions':
                    details=details[~details[hit_column]].copy()
                if details.empty:
                    st.info('No predictions found for this filter.')
                else:
                    columns=['kickoff','league','home','away','predicted_score','actual_score']
                    if accuracy_filter!='⏳ Awaiting results':
                        columns+=['exact_hit','result_hit','top5_hit']
                    display=details[columns].copy()
                    display.columns=['Kickoff (UTC)','League','Home','Away','Predicted score','Final score']+(['Exact score correct','Result correct','Top-five correct'] if accuracy_filter!='⏳ Awaiting results' else [])
                    st.dataframe(display,hide_index=True,use_container_width=True)
                    st.download_button('Download selected predictions CSV',display.to_csv(index=False).encode(),'ziczac_prediction_results.csv','text/csv')
                st.caption('Results are automatically classified only after a verified final score is stored. Unverified or missing results remain pending.')
            else:
                st.info(f'{total} forecasts saved; none has a verified final result yet. Accuracy is not available.')
            st.caption('Exact-score doubles are not yet archived, so double accuracy is not included.')
            st.dataframe(archive.drop(columns=['scores_json']),hide_index=True,use_container_width=True)
            st.download_button('Download prediction archive CSV',archive.to_csv(index=False).encode(),'prediction_archive.csv','text/csv')
if view in ['more']:
    with tabs[5]:if view in ['more']:
    with tabs[5]:
    st.markdown('''**Mathematical model:** Recency-weighted goals scored/conceded, venue-specific shrinkage, league home/away baselines, regularized expected-goal estimates and a normalized independent Poisson score matrix.\n\n**Human-style analysis:** Transparent rule-based commentary on relative strengths, sample sizes and uncertainty. It does **not** use an LLM, live injuries, verified lineups, bookmaker odds or premium xG data.\n\n**Backtesting:** Walk-forward evaluation of exact scores, match outcomes, Brier score and goal error. In-sample tuning on these same matches would overstate performance.\n\n**Limitations:** Data availability varies by league, free plan and season. Small samples, lineup changes, correlated matches and model miscalibration can make predictions inaccurate. No score can be guaranteed.''')
st.divider()
st.caption('For information and research only. Probabilities are model estimates, not betting advice or guaranteed results.')

if view in ['doubles']:
    with tabs[6]:
        st.subheader('Choose the two matches for your exact-score double')
        st.caption('Choose two different matches with verified historical results. Worldwide fixtures without sufficient model data cannot be assigned invented probabilities.')
        if len(chosen_rows)>=2:
            pair_options={fixture_label(r):r for r in chosen_rows}
            pair_names=list(pair_options)
            first_name=st.selectbox('Match 1',pair_names,key='manual_double_first')
            second_names=[name for name in pair_names if name!=first_name]
            second_name=st.selectbox('Match 2',second_names,key='manual_double_second')
            first=pair_options[first_name]
            second=pair_options[second_name]
            manual_options=[]
            for h1,a1,p1 in first['result']['scores'][:10]:
                for h2,a2,p2 in second['result']['scores'][:10]:
                    manual_options.append({'Score 1':f'{h1}-{a1}','Score 2':f'{h2}-{a2}',
                        'P1':p1,'P2':p2,'Joint':p1*p2})
            manual_options.sort(key=lambda o:o['Joint'],reverse=True)
            st.markdown('**Top five exact-score doubles for your two chosen matches**')
            st.write(f"**Match 1:** {first_name}")
            st.write(f"**Match 2:** {second_name}")
            manual_df=pd.DataFrame([{'Rank':i+1,'Match 1 score':o['Score 1'],
                'Match 2 score':o['Score 2'],'Match 1 probability':f"{o['P1']:.2%}",
                'Match 2 probability':f"{o['P2']:.2%}",
                'Double probability':f"{o['Joint']:.3%}"} for i,o in enumerate(manual_options[:5])])
            st.dataframe(manual_df,hide_index=True,use_container_width=True)
            st.download_button('Download my chosen match doubles',manual_df.to_csv(index=False).encode(),
                               'my_selected_exact_score_doubles.csv','text/csv',key='manual_double_download')
        else:
            st.info('Select at least two fixtures in the match selector above to choose your own double.')
        st.divider()
        st.subheader('Five model-ranked exact-score doubles from your selected fixtures')
        st.caption('An exact-score double requires BOTH matches to finish at the specified scores. These probabilities are usually low; they are estimates, not guarantees.')
        if len(chosen_rows)<2:
            st.info('Select at least two upcoming modelled matches above. Increase Upcoming days or choose more competitions if necessary.')
        else:
            options=[]
            for a,b in combinations(chosen_rows,2):
                # Explore the top five scorelines for each match rather than assuming the most likely score is certain.
                for ha,aa,pa in a['result']['scores'][:5]:
                    for hb,ab,pb in b['result']['scores'][:5]:
                        options.append({'Match 1':fixture_label(a),'Score 1':f'{ha}-{aa}',
                                        'Match 2':fixture_label(b),'Score 2':f'{hb}-{ab}',
                                        'Match 1 probability':pa,'Match 2 probability':pb,
                                        'Joint probability':pa*pb})
            options.sort(key=lambda x:x['Joint probability'],reverse=True)
            # Prefer five distinct match pairs when available, not five variants of one pair.
            best=[];seen=set()
            for o in options:
                pair=(o['Match 1'],o['Match 2'])
                if pair not in seen:
                    best.append(o);seen.add(pair)
                if len(best)==5:break
            if len(best)<5:
                for o in options:
                    if o not in best:best.append(o)
                    if len(best)==5:break
            display=pd.DataFrame([{'Rank':i+1,'Match 1':o['Match 1'],'Exact score 1':o['Score 1'],
                'Match 2':o['Match 2'],'Exact score 2':o['Score 2'],
                'Score 1 probability':f"{o['Match 1 probability']:.2%}",
                'Score 2 probability':f"{o['Match 2 probability']:.2%}",
                'Combined probability':f"{o['Joint probability']:.3%}"} for i,o in enumerate(best)])
            st.dataframe(display,hide_index=True,use_container_width=True)
            st.download_button('Download my five score doubles',display.to_csv(index=False).encode(),'exact_score_doubles.csv','text/csv')
            st.caption('Joint probabilities multiply two modelled score probabilities assuming independence. Matches may be correlated, and the model may be miscalibrated.')
        st.subheader('Detailed home/away form and head-to-head')
        for r in chosen_rows:
            m,p=r['fixture'],r['result']
            past=r.get('past',histories.get(r['league'],[]))
            hid=m['homeTeam']['id'];aid=m['awayTeam']['id']
            def form(team_id,venue=None,n=10):
                games=[]
                for g in past:
                    if g.get('status')!='FINISHED':continue
                    h=g.get('homeTeam') or {};a=g.get('awayTeam') or {}
                    if team_id not in (h.get('id'),a.get('id')):continue
                    ishome=h.get('id')==team_id
                    if venue=='home' and not ishome:continue
                    if venue=='away' and ishome:continue
                    sc=(g.get('score') or {}).get('fullTime') or {}
                    x=sc.get('home');y=sc.get('away')
                    if not isinstance(x,int) or not isinstance(y,int):continue
                    gf,ga=(x,y) if ishome else (y,x)
                    games.append((g.get('utcDate',''),gf,ga))
                games.sort(reverse=True)
                games=games[:n]
                return {'Matches':len(games),'Wins':sum(x>y for _,x,y in games),
                        'Draws':sum(x==y for _,x,y in games),'Losses':sum(x<y for _,x,y in games),
                        'Goals for':sum(x for _,x,y in games),'Goals against':sum(y for _,x,y in games),
                        'Last results':', '.join(f'{x}-{y}' for _,x,y in games)}
            with st.expander(f"Research: {fixture_label(r)}"):
                st.write('**Last 10 matches and venue-specific form**')
                st.dataframe(pd.DataFrame([
                    {'Team':m['homeTeam'].get('name'),'Split':'Overall',**form(hid)},
                    {'Team':m['homeTeam'].get('name'),'Split':'Home',**form(hid,'home')},
                    {'Team':m['awayTeam'].get('name'),'Split':'Overall',**form(aid)},
                    {'Team':m['awayTeam'].get('name'),'Split':'Away',**form(aid,'away')}]),hide_index=True,use_container_width=True)
                h2h=[]
                for g in past:
                    gh=(g.get('homeTeam') or {}).get('id');ga=(g.get('awayTeam') or {}).get('id')
                    if {gh,ga}!={hid,aid} or g.get('status')!='FINISHED':continue
                    sc=(g.get('score') or {}).get('fullTime') or {}
                    if isinstance(sc.get('home'),int) and isinstance(sc.get('away'),int):
                        h2h.append({'Date':g.get('utcDate','')[:10],
                            'Home':g['homeTeam'].get('name'),'Away':g['awayTeam'].get('name'),
                            'Final score':f"{sc['home']}-{sc['away']}"})
                if h2h:st.dataframe(pd.DataFrame(sorted(h2h,key=lambda x:x['Date'],reverse=True)[:10]),hide_index=True,use_container_width=True)
                else:st.info('No verified head-to-head games in the accessible competition history.')
                st.write(f"**Modelled expected goals:** {p['home_xg']:.2f} home / {p['away_xg']:.2f} away")
                st.write(f"**1X2:** Home {p['home_win']:.1%} · Draw {p['draw']:.1%} · Away {p['away_win']:.1%}")
                st.dataframe(pd.DataFrame([{'Exact score':f'{h}-{a}','Estimated probability':f'{q:.2%}'} for h,a,q in p['scores'][:10]]),hide_index=True,use_container_width=True)
                for note in context_review(m,p,past):st.write('• '+note)
        st.caption('Home/away and H2H use only accessible historical match data. Injuries, lineups, weather, travel and external scouting are NOT verified by this feature.')
