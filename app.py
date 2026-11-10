"""ZicZac Predict AI Worldwide: free Streamlit UI, live API, walk-forward validation."""
from datetime import datetime, timedelta, timezone
from itertools import combinations
import os
import json
import pandas as pd
import pycountry
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
_hero_html="""<style>
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
/* ZicZac premium mobile fixture finder — visual-only enhancement */
:root{--zz-ink:#eaf2ff;--zz-muted:#b4c8e9;--zz-stroke:#5377b3}
[data-testid="stAppViewContainer"] .block-container{max-width:1080px}
[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3{
  color:#f7faff;letter-spacing:-.025em;line-height:1.25;
  text-wrap:balance;text-shadow:0 3px 18px #02081988
}
[data-testid="stMarkdownContainer"] h2{font-weight:850;margin-top:1.15rem}
[data-testid="stMarkdownContainer"] h3{font-weight:820}
[data-testid="stCaptionContainer"] p{color:#bfd0ed!important;line-height:1.5!important}
[data-testid="stWidgetLabel"] p{color:#f4f7ff!important;font-weight:750!important;letter-spacing:.005em}
[data-testid="stMultiSelect"] [data-baseweb="select"]>div,
[data-testid="stSelectbox"] [data-baseweb="select"]>div{
  background:linear-gradient(145deg,#192d55,#172446)!important;
  border:1px solid #6686c5!important;border-radius:15px!important;
  min-height:55px;box-shadow:0 7px 22px #0003,inset 0 1px #ffffff0b
}
[data-testid="stMultiSelect"] [data-baseweb="tag"]{
  background:linear-gradient(115deg,#825cf5,#5f4fe6)!important;
  border:1px solid #b7a1ff!important;border-radius:10px!important;
  box-shadow:0 3px 10px #2a1e6e55;
  padding:5px 7px;max-width:100%
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span{
  color:#fff!important;font-size:13px!important;font-weight:700!important;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg{fill:#fff}
[data-testid="stCheckbox"]{
  background:linear-gradient(130deg,#1b3157,#172646);
  border:1px solid #395a8b;border-radius:14px;
  padding:11px 12px;margin:4px 0 8px;
  box-shadow:0 5px 15px #0002;transition:border-color .15s,background .15s
}
[data-testid="stCheckbox"]:hover{border-color:#a0baff;background:#243b67}
[data-testid="stCheckbox"] label{align-items:flex-start;gap:9px}
[data-testid="stCheckbox"] label p{
  color:#f0f5ff!important;font-size:14px!important;font-weight:650;
  line-height:1.55!important;overflow-wrap:anywhere
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"]{margin-top:3px}
[data-testid="stExpander"]{
  background:linear-gradient(125deg,#1a3057,#172342)!important;
  border:1px solid #6888c2!important;border-radius:16px!important;
  box-shadow:0 8px 24px #0003
}
[data-testid="stExpander"] summary{padding:13px 15px!important}
[data-testid="stExpander"] summary p{font-weight:750!important;color:#eaf3ff!important}
[data-testid="stTextInput"] input{
  min-height:51px;border-radius:13px!important;
  background:#1b2e52!important;border:1px solid #6584b9!important
}
[data-testid="stAlert"]{border-radius:14px!important}
hr{border-color:#46618a!important;margin:1.5rem 0!important}
@media(max-width:680px){
  [data-testid="stAppViewContainer"] .block-container{padding-left:13px;padding-right:13px}
  [data-testid="stMarkdownContainer"] h2{font-size:clamp(22px,6vw,28px)!important}
  [data-testid="stMarkdownContainer"] h3{font-size:clamp(19px,5vw,24px)!important}
  [data-testid="stMultiSelect"] [data-baseweb="tag"]{max-width:calc(100vw - 105px)}
  [data-testid="stCheckbox"]{padding:11px 10px}
  [data-testid="stCheckbox"] label p{font-size:13px!important}
  [data-testid="stWidgetLabel"] p{font-size:15px!important}
}
</style>
<div class="zz-head"><div class="zz-logo">⚽</div><div class="zz-name">ZicZac Predict <span>AI</span></div><div class="zz-sub">AI Football Predictions &amp; Exact Scores</div><div class="zz-by">PREDICTION BY ZICZAC</div>

<div class="zz-stadium"><div class="zz-stadium-copy"><div class="zz-stadium-title">Today's Football<br>Predictions</div><div class="zz-stadium-desc">⚡ Data-driven analysis · Exact scores · Worldwide matches</div><div style="font-size:28px;margin-top:9px">⚽</div></div></div></div>
"""
_hero_slot=st.empty()
# Native button navigation: each tap changes a single session-state route.
# Buttons avoid the iPhone radio-label behaviour that looked selectable but did not navigate.
nav_items=[('🏠','Home','home'),('🎯','Predict','predict'),('🔗','Doubles','doubles'),('📊','Archive','archive'),('☰','More','more')]
if 'ziczac_active_page' not in st.session_state:
    initial=st.query_params.get('view','home')
    st.session_state['ziczac_active_page']=initial if initial in {x[2] for x in nav_items} else 'home'
def _navigate_to(page):
    st.session_state['ziczac_active_page']=page
st.markdown("""<style>
/* Large native buttons with clear selected state on mobile. */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    width:100%;min-height:54px;border-radius:13px;
    font-size:clamp(10px,2.5vw,14px);font-weight:750;
    padding:6px 2px;white-space:normal;
}
</style>""",unsafe_allow_html=True)
nav_cols=st.columns(5,gap='small')
for col,(emoji,label,page) in zip(nav_cols,nav_items):
    with col:
        active=st.session_state['ziczac_active_page']==page
        st.button(emoji+' '+label,key='zz_route_'+page,
                  type='primary' if active else 'secondary',
                  use_container_width=True,
                  on_click=_navigate_to,args=(page,))
view=st.session_state['ziczac_active_page']
# Render the selected page directly beneath navigation, not after the long
# worldwide finder and API settings. This is what makes taps feel responsive.
active_page_slot=st.container()
with active_page_slot:
    st.info({'home':'🏠 Home dashboard and today’s featured match','predict':'🎯 Upcoming predictions and match analysis','doubles':'🔗 Doubles and exact-score combinations','archive':'📊 Prediction accuracy and archive','more':'☰ Model methodology and settings'}[view])

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
    use_secondary=st.checkbox('Use API-Football Pro for fixtures and historical results',value=True,help='Prioritise your API-Football subscription for supported leagues; requests are cached.')
    if st.button('Refresh football data'):
        st.rerun()
    st.caption('Data cached for 1 hour. Refresh reruns the page without bypassing API cooldowns. Saved snapshots may be older.')
try:secondary_key=st.secrets.get('API_FOOTBALL_KEY','') or os.environ.get('API_FOOTBALL_KEY','')
except Exception:secondary_key=os.environ.get('API_FOOTBALL_KEY','')
try:sportmonks_key=st.secrets.get('THIRD_FOOTBALL_API_KEY','') or os.environ.get('THIRD_FOOTBALL_API_KEY','')
except Exception:sportmonks_key=os.environ.get('THIRD_FOOTBALL_API_KEY','')
st.subheader('🚀 ZicZac Pro · Tomorrow & worldwide predictions')
if secondary_key:
    st.info('API-Football key found. Pro subscription status is not verified here. Select Tomorrow, choose a match, then run Pro analysis to check actual data access.')
else:
    st.warning('API_FOOTBALL_KEY is missing. Configure the existing key in Streamlit Secrets.')
st.subheader('🌍 Worldwide fixtures by Sydney date')
st.caption('Date-first discovery combines API-Football, TheSportsDB and Sportmonks (when configured). Each provider is limited to its subscription coverage.')
choice=st.radio('Fixture date', ['Today','Tomorrow','Next 3 days','Next 7 days','Next 14 days'],horizontal=True,key='prediction_window')
today=datetime.now(SYDNEY).date()
start_day=today+timedelta(days=1) if choice=='Tomorrow' else today
span=3 if choice=='Next 3 days' else 7 if choice=='Next 7 days' else 14 if choice=='Next 14 days' else 1
with st.expander('🌍 Worldwide Match Finder · Tap to search',expanded=False):
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
        # Full ISO 3166 country/territory directory, independent of today's
        # fixture API coverage. Keep provider labels that are not ISO countries.
        # Include football-specific national associations and regional
        # competition categories often shown separately in sportsbook menus.
        football_regions={'England','Scotland','Wales','Northern Ireland',
                          'Kosovo','World','International','Europe','Asia',
                          'Africa','North America','South America','Oceania',
                          'CONCACAF','CONMEBOL','AFC','CAF','UEFA'}
        country_options=sorted({c.name for c in pycountry.countries} |
                               football_regions |
                               {str(m.get('country','')).strip() for m in global_matches if m.get('country')},
                               key=str.casefold)
        country=st.selectbox('Country', ['All countries']+country_options,
                             key='finder_country',help='Worldwide country/territory and football-association directory, similar to football competition country menus. This is not a live bet365 feed; fixtures depend on connected providers.')
        term=st.text_input('Search team or league',placeholder='Type a team or competition')
        def _country_matches(provider_country, selected_country):
            if selected_country=='All countries':return True
            if str(provider_country).casefold()==selected_country.casefold():return True
            try:
                a=pycountry.countries.lookup(str(provider_country))
                b=pycountry.countries.lookup(selected_country)
                return a.alpha_2==b.alpha_2
            except LookupError:
                return False
        filtered=[m for m in global_matches if m['kickoff']>datetime.now(timezone.utc)
                  and _country_matches(m.get('country',''),country)
                  and (not term or term.lower() in (m['home']+' '+m['away']+' '+m['league']).lower())]
        st.caption(f'🌍 {len(country_options)} countries, territories and football regions in directory. Fixture availability depends on data coverage.')
        if country!='All countries' and not any(_country_matches(m.get('country',''),country) for m in global_matches):
            st.caption('No fixtures from this country in the current provider response and date range. Try Next 7 days or Next 14 days.')
        st.caption(f'{len(filtered)} returned matches · Sydney time · Results depend on free-plan coverage and API quotas.')
        st.caption('Use Search matches to reveal the fixture list. Select any match to add it to analysis.')
        show_fixtures=st.toggle('🔎 Search matches · Show fixture list',value=False,key='finder_show_results')
        picked_ids=set(st.session_state.get('worldwide_picked_ids', []))
        if show_fixtures:
            if filtered:
                st.caption('Tap a fixture checkbox to select it. Narrow by country or team above.')
                for m in filtered:
                    event_id=str(m['source'])+':'+str(m['id'])
                    selected_now=st.checkbox(f"{m['home']} vs {m['away']} · {m['local']:%a %d %b, %I:%M %p} Sydney · {m['league']} ({m['country']})",value=event_id in picked_ids,key='finder_'+event_id)
                    if selected_now:picked_ids.add(event_id)
                    else:picked_ids.discard(event_id)
            else:
                st.info('No matches returned for these filters. Try another date or search term.')
        else:
            st.info('Match list hidden. Turn on Search matches to browse upcoming fixtures.')
        st.session_state['worldwide_picked_ids']=list(picked_ids)
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
        # Prefer API-Football Pro data over the older provider.
        if use_secondary and secondary_key and code in supported_codes():
            try:
                pro_upcoming=secondary_fixtures(secondary_key,code,now,14)
                pro_past=secondary_history(secondary_key,code,now)
                if pro_upcoming:
                    upcoming=pro_upcoming
                    source='API-Football Pro'
                if pro_past and (source=='API-Football Pro' or not past):
                    past=pro_past
            except Exception as e:
                errors.append(f'{name}: API-Football Pro unavailable ({e})')
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

# Automatically research all provider-compatible discovered fixtures for the
# selected date. Cached per fixture across reruns and sessions to limit API use.
@st.cache_data(ttl=21600,show_spinner=False)
def _cached_pro_worldwide(key, event):
    return model_worldwide(key,event)

_auto_cache=st.session_state.setdefault('worldwide_models',{})
_auto_pending=[item for item in discoverable
               if item['name'] not in _auto_cache
               and not (item.get('model') and item['model'].get('result'))
               and item['event'].get('source')=='API-Football']
if secondary_key and _auto_pending:
    # Avoid unbounded work on each Streamlit rerun. Continue automatically on
    # subsequent interactions; allow an explicit button for the remaining batch.
    _auto_limit=12
    with st.spinner(f'Automatically researching {min(len(_auto_pending),_auto_limit)} worldwide matches using Pro…'):
        for _item in _auto_pending[:_auto_limit]:
            _row,_error=_cached_pro_worldwide(secondary_key,_item['event'])
            _auto_cache[_item['name']]={'row':_row,'error':_error}
    if len(_auto_pending)>_auto_limit:
        st.caption(f'Pro research completed its first batch; {len(_auto_pending)-_auto_limit} fixtures remain for the next batch to protect response time and API quota.')
elif not secondary_key:
    st.warning('Automatic Pro research requires API_FOOTBALL_KEY in Streamlit Secrets.')
# Allow the user to select fixtures explicitly; no match is invented.
# Add every successfully researched worldwide fixture to the shared prediction pool.
# This makes Pro results available to prediction cards and doubles, not just the explorer.
for _item in discoverable:
    _name=_item['name']
    _cached=st.session_state.get('worldwide_models',{}).get(_name,{})
    _model=_cached.get('row') or _item.get('model')
    if _model and _model.get('result') and not any(
        x['fixture'].get('provider')==_model['fixture'].get('provider') and
        x['fixture'].get('id')==_model['fixture'].get('id') for x in rows):
        rows.append(_model)
rows.sort(key=lambda r:r['kickoff'])
eligible=[r for r in rows if r.get('result')]
# Populate the large stadium hero with tomorrow's highest-probability
# verified exact-score estimate. This is the same live data used throughout.
import html as _hero_escape
_tomorrow_date=today+timedelta(days=1)
_tomorrow_ready=[]
for _r in eligible:
    try:
        _local=datetime.fromisoformat(_r['kickoff'].replace('Z','+00:00')).astimezone(SYDNEY)
        if _local.date()==_tomorrow_date and _r['result'].get('scores'):
            _tomorrow_ready.append(_r)
    except (ValueError,TypeError,KeyError):
        pass
_hero_title="Tomorrow’s Game of the Day"
if _tomorrow_ready:
    _featured=max(_tomorrow_ready,key=lambda _r:(
        min(len(_r['result'].get('home_last10',[])),len(_r['result'].get('away_last10',[])))>=5,
        _r['result']['scores'][0][2]))
    _ff=_featured['fixture']
    _hh,_aa,_pp=_featured['result']['scores'][0]
    _home=_hero_escape.escape(str(_ff['homeTeam'].get('name','Home')))
    _away=_hero_escape.escape(str(_ff['awayTeam'].get('name','Away')))
    _hero_title=f"""Tomorrow’s Game of the Day
<div style="font-size:clamp(16px,4vw,25px);font-weight:750;line-height:1.3;margin:12px 0 8px">{_home} <span style="color:#d7a7ff">vs</span> {_away}</div>
<div style="font-size:clamp(32px,9vw,52px);font-weight:900;color:#fff;line-height:1.1">{_hh} – {_aa}</div>
<div style="font-size:15px;color:#e2d6ff;margin-top:8px">Exact-score estimate: {_pp:.1%} · {_hero_escape.escape(str(_featured.get('league','Football')))}</div>"""
else:
    _hero_title="Tomorrow’s Game of the Day<div style='font-size:17px;font-weight:500;margin-top:12px'>Awaiting verified Pro score analysis</div>"
_hero_slot.markdown(_hero_html.replace("Today's Football<br>Predictions",_hero_title),unsafe_allow_html=True)
def fixture_label(r):
    m=r['fixture']
    return f"{m['homeTeam'].get('name','Home')} vs {m['awayTeam'].get('name','Away')} · {r['kickoff'][:16]} UTC"
labels={f"{r['fixture'].get('id')} · {fixture_label(r)}":r for r in eligible}
st.subheader('📅 All available fixtures for the selected date')
st.caption('Includes every fixture returned by the connected worldwide feeds, even if no prediction is ready. API coverage is not the same as every football match in existence.')
if discoverable:
    # The compact score columns stay beside the teams, including on mobile.
    # Never show an invented score when verified research has not completed.
    _research_cache=st.session_state.setdefault('worldwide_models',{})
    _ready=0
    def _score_for_fixture(item):
        cached=_research_cache.get(item['name'],{})
        model=cached.get('row') or item.get('model')
        if model and model.get('result'):
            scores=model['result'].get('scores',[])
            if scores:
                h,a,p=scores[0]
                return f"{h}–{a}",f"{p:.1%}",'Model ready'
        return '—','—',('History unavailable' if cached.get('error') else 'Not researched')
    if secondary_key:
        st.caption('Pro research starts automatically for available API-Football fixtures. Use the button below to continue any remaining fixtures; provider coverage and quota may limit results.')
        _max_batch=st.select_slider('Fixtures per research batch',options=[5,10,20,40,80],value=10,
                                    key='pro_fixture_batch_size')
        if st.button('⚡ Research scores for available fixtures',key='pro_fixture_batch',type='primary'):
            _pending=[item for item in discoverable if item['name'] not in _research_cache
                      and not (item.get('model') and item['model'].get('result'))
                      and item['event'].get('source')=='API-Football']
            with st.spinner(f'Researching up to {min(_max_batch,len(_pending))} fixtures using Pro…'):
                for item in _pending[:_max_batch]:
                    row,error=_cached_pro_worldwide(secondary_key,item['event'])
                    _research_cache[item['name']]={'row':row,'error':error}
            if not _pending:
                st.info('No additional compatible unresearched fixtures in this date range.')
            else:
                st.success(f'Research attempted for {min(_max_batch,len(_pending))} fixtures. Scores appear only when sufficient verified history is available.')
    _all_fixtures=[]
    for item in discoverable:
        _score,_prob,_status=_score_for_fixture(item)
        if _status=='Model ready': _ready+=1
        _all_fixtures.append({'Kickoff (Sydney)':item['event']['local'].strftime('%d %b %I:%M %p'),
                    'Home':item['event']['home'],'Away':item['event']['away'],
                    'Predicted score':_score,'Score probability':_prob,
                    'Research':_status,
                    'League':item['event'].get('league',''),
                    'Country':item['event'].get('country','')})
    st.dataframe(pd.DataFrame(_all_fixtures),hide_index=True,use_container_width=True,
                 column_config={'Home':st.column_config.TextColumn(width='medium'),
                                'Away':st.column_config.TextColumn(width='medium'),
                                'Predicted score':st.column_config.TextColumn(width='small'),
                                'Score probability':st.column_config.TextColumn(width='small')},
                 height=min(560,80+len(_all_fixtures)*35))
    st.caption(f"{len(_all_fixtures)} available fixtures · {_ready} with verified score estimates. Scroll horizontally to see all columns.")
    st.caption('A dash means no modelled score is available yet, not a 0–0 prediction.')
else:
    st.info('No fixtures were returned for the selected date. Check the provider warnings and date range.')
st.subheader('🌍 Worldwide Match Explorer')
st.caption('Research checks available authorised football results feeds. When history is insufficient, the app offers public-web research links. External results are not silently treated as verified model data.')
worldwide_options={item['name']:item for item in discoverable}
# League-first explorer: choose a competition, then inspect its teams and fixtures.
# The date-first provider returns only competitions playing in its narrow date
# window. Keep a stable catalogue of major competitions even during breaks.
# Broad football competition catalogue inspired by the competition categories
# publicly displayed by bet365. This is NOT a live bet365 feed.
major_leagues={
    '🇪🇺 UEFA Champions League':['uefa champions league','champions league','champions league (cl)'],
    '🇪🇺 UEFA Europa League':['uefa europa league','europa league'],
    '🇪🇺 UEFA Conference League':['uefa conference league','conference league'],
    '🇪🇺 UEFA Nations League':['uefa nations league','nations league'],
    '🇪🇺 UEFA European Championship':['uefa euro','european championship','euros'],
    '🌍 FIFA World Cup':['fifa world cup','world cup'],
    '🌍 FIFA Club World Cup':['club world cup'],
    '🌍 World Cup Qualifiers':['world cup qualification','world cup qualifiers','wc qualification'],
    '🌍 Africa Cup of Nations':['africa cup of nations','afcon'],
    '🌎 Copa América':['copa america','copa américa'],
    '🌎 CONCACAF Gold Cup':['gold cup'],
    '🇪🇺 UEFA Women’s Champions League':['womens champions league',"women's champions league",'uwcl'],
    '🏴 English Premier League':['english premier league','premier league','premier league (pl)'],
    '🏴 English Championship':['efl championship','championship','championship (elc)'],
    '🏴 English League One':['league one','efl league one'],
    '🏴 English League Two':['league two','efl league two'],
    '🏴 FA Cup':['fa cup'],
    '🏴 EFL Cup':['efl cup','carabao cup','league cup'],
    '🏴 Women’s Super League':['womens super league',"women's super league",'wsl'],
    '🏴 Scottish Premiership':['scottish premiership','scottish premier league'],
    '🏴 Scottish Championship':['scottish championship'],
    '🏴 Scottish Cup':['scottish cup'],
    '🇪🇸 La Liga':['la liga','laliga','primera division','primera división'],
    '🇪🇸 Segunda División':['segunda division','segunda división','laliga 2','la liga 2'],
    '🇪🇸 Copa del Rey':['copa del rey'],
    '🇩🇪 Bundesliga':['bundesliga','bundesliga (bl1)'],
    '🇩🇪 2. Bundesliga':['2. bundesliga','bundesliga 2','2 bundesliga'],
    '🇩🇪 DFB-Pokal':['dfb pokal','dfb-pokal'],
    '🇮🇹 Serie A':['serie a','serie a (sa)'],
    '🇮🇹 Serie B':['serie b'],
    '🇮🇹 Coppa Italia':['coppa italia'],
    '🇫🇷 Ligue 1':['ligue 1','ligue 1 (fl1)'],
    '🇫🇷 Ligue 2':['ligue 2'],
    '🇫🇷 Coupe de France':['coupe de france'],
    '🇵🇹 Primeira Liga':['primeira liga','liga portugal'],
    '🇵🇹 Liga Portugal 2':['liga portugal 2','segunda liga portugal'],
    '🇳🇱 Eredivisie':['eredivisie'],
    '🇳🇱 Eerste Divisie':['eerste divisie'],
    '🇧🇪 Belgian Pro League':['belgian pro league','jupiler pro league'],
    '🇹🇷 Süper Lig':['super lig','süper lig'],
    '🇬🇷 Greek Super League':['super league greece','greek super league'],
    '🇨🇭 Swiss Super League':['swiss super league'],
    '🇦🇹 Austrian Bundesliga':['austrian bundesliga'],
    '🇩🇰 Danish Superliga':['danish superliga'],
    '🇳🇴 Eliteserien':['eliteserien'],
    '🇸🇪 Allsvenskan':['allsvenskan'],
    '🇵🇱 Ekstraklasa':['ekstraklasa'],
    '🇨🇿 Czech First League':['czech first league','czech liga'],
    '🇧🇷 Brasileirão Série A':['brasileiro serie a','brasileirão','brazil serie a','serie a brazil'],
    '🇧🇷 Brasileirão Série B':['brasileiro serie b','brazil serie b'],
    '🇧🇷 Copa do Brasil':['copa do brasil'],
    '🇦🇷 Argentine Primera División':['argentine primera','liga profesional argentina','primera division argentina'],
    '🇦🇷 Copa Argentina':['copa argentina'],
    '🌎 Copa Libertadores':['copa libertadores','libertadores'],
    '🌎 Copa Sudamericana':['copa sudamericana','sudamericana'],
    '🇺🇸 Major League Soccer':['major league soccer','mls'],
    '🇺🇸 US Open Cup':['us open cup'],
    '🇲🇽 Liga MX':['liga mx'],
    '🇲🇽 Liga de Expansión MX':['liga de expansion','liga de expansión'],
    '🇨🇴 Categoría Primera A':['primera a colombia','categoria primera a'],
    '🇨🇱 Chilean Primera División':['primera division chile','chilean primera'],
    '🇺🇾 Uruguayan Primera División':['primera division uruguay','uruguayan primera'],
    '🇸🇦 Saudi Pro League':['saudi pro league','pro league saudi','saudi professional league'],
    '🇦🇪 UAE Pro League':['uae pro league','arabian gulf league'],
    '🇶🇦 Qatar Stars League':['qatar stars league'],
    '🇯🇵 J1 League':['j1 league','j league'],
    '🇯🇵 J2 League':['j2 league'],
    '🇰🇷 K League 1':['k league 1','k-league 1'],
    '🇨🇳 Chinese Super League':['chinese super league'],
    '🇮🇳 Indian Super League':['indian super league'],
    '🇦🇺 A-League Men':['a-league men','a league men','a-league'],
    '🇦🇺 A-League Women':['a-league women','a league women'],
    '🌍 CAF Champions League':['caf champions league'],
    '🌏 AFC Champions League Elite':['afc champions league','afc champions league elite'],
    '🌎 CONCACAF Champions Cup':['concacaf champions cup','concacaf champions league'],
}

def _league_group(name):
    lower=str(name).casefold()
    # Prefer longer, specific names: 'Club World Cup' before 'World Cup',
    # '2. Bundesliga' before 'Bundesliga', and women's before men's.
    matches=[(len(alias),title) for title,aliases in major_leagues.items()
             for alias in aliases if alias in lower]
    if matches:
        return max(matches,key=lambda x:x[0])[1]
    return None
league_groups={title:[] for title in major_leagues}
other_groups={}
for item in discoverable:
    event=item['event']
    label=_league_group(event.get('league',''))
    if label:
        league_groups[label].append(item)
    else:
        other_label=f"{event.get('league','Unknown league')} ({event.get('country','World')})"
        other_groups.setdefault(other_label,[]).append(item)
# Add the separately monitored football-data.org fixtures to the explorer.
# They may be available even when worldwide date-first discovery has no games.
for r in rows:
    fixture=r['fixture']
    label=_league_group(r.get('league',''))
    if not label:
        continue
    try:
        kickoff_dt=datetime.fromisoformat(r['kickoff'].replace('Z','+00:00'))
        home=fixture['homeTeam'].get('name','Home')
        away=fixture['awayTeam'].get('name','Away')
        if any(normalized_team(x['event'].get('home'))==normalized_team(home)
               and normalized_team(x['event'].get('away'))==normalized_team(away)
               and abs((x['event']['kickoff']-kickoff_dt).total_seconds())<7200
               for x in league_groups[label]):
            continue
        event={'home':home,'away':away,'league':label,'country':'',
               'local':kickoff_dt.astimezone(SYDNEY),'kickoff':kickoff_dt,
               'source':'football-data.org','id':fixture.get('id')}
        league_groups[label].append({'name':f"{home} vs {away} · {event['local']:%d %b %I:%M %p} Sydney · {label}",
                                      'model':r,'event':event})
    except (KeyError,TypeError,ValueError,AttributeError):
        continue
league_groups.update(other_groups)
st.markdown('**🏆 Step 1 · Choose a league**')
st.caption('Broad competition directory inspired by bet365 football categories (not affiliated or synchronised). Fixtures and teams appear only where your connected football data providers return them.')
league_names=['All leagues']+list(major_leagues)+sorted(other_groups)
selected_league=st.selectbox('League / competition',league_names,key='worldwide_league_filter')
league_items=(discoverable if selected_league=='All leagues' else league_groups[selected_league])
if selected_league=='All leagues':
    league_items=list(discoverable)+[item for title in major_leagues for item in league_groups[title]
                                  if item['event'].get('source')=='football-data.org']
league_teams=sorted({str(team) for item in league_items
                     for team in (item['event'].get('home'),item['event'].get('away')) if team})
st.caption(f'{len(league_items)} upcoming fixtures · {len(league_teams)} teams in the available fixture feed')
st.markdown('**⚽ Step 2 · View teams and matches**')
team_filter=st.selectbox('Team',['All teams']+league_teams,key='worldwide_team_filter')
visible_worldwide=[item for item in league_items
                   if team_filter=='All teams' or team_filter in (item['event'].get('home'),item['event'].get('away'))]
if visible_worldwide:
    team_rows=[]
    for item in visible_worldwide:
        ev=item['event']
        team_rows.append({'Home team':ev.get('home',''), 'Away team':ev.get('away',''),
                          'Kickoff (Sydney)':ev['local'].strftime('%d %b, %I:%M %p'),
                          'League':ev.get('league',''),'Country':ev.get('country',''),
                          'Research status': 'Model ready' if item.get('model') and item['model'].get('result') else 'Select to research'})
    st.dataframe(pd.DataFrame(team_rows),hide_index=True,use_container_width=True)
    st.caption('Select a fixture below and tap Research. Historical results are checked against available authorised sources; unavailable records are never invented.')
else:
    st.info('No upcoming fixtures found for this league and team in the current date range. Try another league or date.')
picked_ids=set(st.session_state.get('worldwide_picked_ids', []))
preselected=[item['name'] for item in visible_worldwide
             if str(item['event']['source'])+':'+str(item['event']['id']) in picked_ids]
st.markdown('**🎯 Step 3 · Select fixtures to analyse**')
# Reset only the fixture multiselect when league/team changes, avoiding invalid stale options.
selector_scope=(selected_league,team_filter)
if st.session_state.get('worldwide_selector_scope')!=selector_scope:
    st.session_state['worldwide_selector_scope']=selector_scope
    st.session_state['worldwide_fixture_selection']=preselected
visible_options=[item['name'] for item in visible_worldwide]
st.session_state['worldwide_fixture_selection']=[name for name in st.session_state.get('worldwide_fixture_selection',[])
                                                  if name in visible_options]
selected_worldwide=st.multiselect('Matches in selected league',visible_options,
                                   key='worldwide_fixture_selection',
                                   help='Choose fixtures from this league to research with API-Football Pro when supported.')
# Model selected API-Football worldwide fixtures on demand to conserve quota.
if st.session_state.get('worldwide_model_version') != 'pro-auto-v6':
    st.session_state['worldwide_models']={}
    st.session_state['worldwide_model_version']='pro-auto-v6'
if 'worldwide_models' not in st.session_state:
    st.session_state['worldwide_models']={}
if selected_worldwide:
    st.caption('⚡ Automatic Pro analysis: selecting a match starts verified history research. Cached results avoid repeating requests.')
    if secondary_key:
        for name in selected_worldwide[:8]:
            if name not in st.session_state['worldwide_models']:
                item=worldwide_options[name]
                if item.get('model') and item['model'].get('result'):
                    continue
                with st.spinner('Pro analysing '+name+'…'):
                    row,error=_cached_pro_worldwide(secondary_key,item['event'])
                st.session_state['worldwide_models'][name]={'row':row,'error':error}
    else:
        st.warning('API_FOOTBALL_KEY is missing; Pro history cannot run.')
    if st.button('🔎 Run Pro analysis for selected matches',type='primary',help='Fetch completed team matches and H2H. Uses API requests.'):
        st.session_state['worldwide_models']={k:v for k,v in st.session_state['worldwide_models'].items() if k not in selected_worldwide}
        for name in selected_worldwide[:8]:
            item=worldwide_options[name]
            if item['model'] and item['model'].get('result'):
                continue
            row,error=_cached_pro_worldwide(secondary_key,item['event'])
            st.session_state['worldwide_models'][name]={'row':row,'error':error}
    for name in selected_worldwide:
        cached=st.session_state['worldwide_models'].get(name)
        if cached and 'Last parameter' in str(cached.get('error','')):
            st.session_state['worldwide_models'].pop(name,None)
            cached=None
        if cached:
            if cached.get('error'):
                st.error(f"Research unavailable for {name}: {cached['error']}")
            if cached['row']:
                worldwide_options[name]['model']=cached['row']
                histories[name]=cached['row']['past']
                st.success(f"Prediction ready: {name}")
                _top=cached['row']['result'].get('scores',[])
                if _top:
                    st.markdown('### 🎯 Pro predicted exact scores')
                    st.dataframe(pd.DataFrame([{'Score':f'{h}–{a}','Model probability':f'{p:.1%}'}
                                               for h,a,p in _top[:5]]),hide_index=True,use_container_width=True)
                    st.caption('Estimated score probabilities, not guaranteed outcomes.')
                st.caption('Verified research source: '+', '.join(cached['row'].get('research_sources',['Football match data provider'])))
                details=cached['row'].get('history_counts',{})
                if details:
                    st.caption(f"Pro research coverage: {details.get('home',0)} completed home-team games, {details.get('away',0)} completed away-team games. H2H and venue splits are displayed below when available.")
                with st.expander('🔍 Research requests and provenance'):
                    for detail in cached['row'].get('research_details',[]):
                        st.write('• '+detail)
                    st.caption('Only verified completed matches enter the mathematical model; Pro access does not guarantee a correct score.')
                research=cached['row']['result']
                with st.expander('📊 Full mathematical research · '+name,expanded=True):
                    st.markdown('**Home win / Draw / Away win — probability model**')
                    st.bar_chart(pd.DataFrame({'Probability (%)':[round(research['home_win']*100,2),round(research['draw']*100,2),round(research['away_win']*100,2)]},index=['Home win','Draw','Away win']))
                    st.markdown('**Expected goals (Poisson goal model)**')
                    st.bar_chart(pd.DataFrame({'Expected goals':[research['home_xg'],research['away_xg']]},index=['Home','Away']))
                    st.markdown('**Recent form: overall and venue-specific**')
                    stats_rows=[]
                    for title,key in [('Home · overall','home_stats'),('Home · home only','home_venue_stats'),('Away · overall','away_stats'),('Away · away only','away_venue_stats')]:
                        x=research.get(key)
                        if x:stats_rows.append({'Team / split':title,'Matches':x['played'],'Wins':x['wins'],'Draws':x['draws'],'Losses':x['losses'],'Goals scored / game':round(x['scored'],2),'Goals conceded / game':round(x['conceded'],2)})
                    if stats_rows:st.dataframe(pd.DataFrame(stats_rows),hide_index=True,use_container_width=True)
                    for title,key in [('Home · last 10 overall','home_last10'),('Home · last 10 at home','home_home10'),('Away · last 10 overall','away_last10'),('Away · last 10 away','away_away10')]:
                        games=research.get(key,[])
                        st.markdown('**'+title+' ('+str(len(games))+'/10 available)**')
                        if games:
                            st.line_chart(pd.DataFrame([{'Match':str(i+1),'Scored':g['gf'],'Conceded':g['ga']} for i,g in enumerate(reversed(games))]).set_index('Match'))
                        else:st.caption('No verified matches available for this split.')
                    h2h=research.get('h2h_last10',[])
                    st.markdown('**Head-to-head — '+str(len(h2h))+'/10 available**')
                    if h2h:st.dataframe(pd.DataFrame(h2h),hide_index=True,use_container_width=True)
                    else:st.caption('No verified previous meetings available.')
                    st.markdown('**Top exact-score probabilities**')
                    st.bar_chart(pd.DataFrame({'Probability (%)':[round(p*100,2) for h,a,p in research['scores']]},index=[f'{h}-{a}' for h,a,p in research['scores']]))
                    if min(len(research.get('home_last10',[])),len(research.get('away_last10',[])),len(research.get('home_home10',[])),len(research.get('away_away10',[])))<10:
                        st.warning('Fewer than 10 verified results exist in at least one split. This is a limited-sample estimate, not a fully researched 10-match comparison.')
                    st.caption('Mathematical conclusion: most likely result = '+max([('Home win',research['home_win']),('Draw',research['draw']),('Away win',research['away_win'])],key=lambda x:x[1])[0]+'. This is the highest model probability, not a guaranteed outcome.')
            elif cached['error']:
                st.warning(f"Verified statistical model not ready: {name}. {cached['error']}")
                from urllib.parse import quote_plus
                ev=worldwide_options[name]['event']
                query=quote_plus(f"{ev.get('home','')} {ev.get('away','')} football last 10 results head to head")
                st.markdown(f"[🔎 Research match history across public websites](https://www.google.com/search?q={query})")
                st.caption("External search results are references only, not automatically verified or used in the probability model.")
for name in selected_worldwide:
    candidate=worldwide_options[name]['model']
    if candidate and candidate.get('result') and not any(str(r['fixture'].get('provider'))+':'+str(r['fixture'].get('id'))==str(candidate['fixture'].get('provider'))+':'+str(candidate['fixture'].get('id')) for r in rows):
        rows.append(candidate)
rows.sort(key=lambda r:r['kickoff'])
unmodelled=[worldwide_options[name] for name in selected_worldwide if not worldwide_options[name]['model'] or not worldwide_options[name]['model'].get('result')]
if unmodelled:
    st.info(f"{len(unmodelled)} selected worldwide match(es) do not have sufficient compatible historical data for exact-score predictions. They are displayed for discovery only.")
st.subheader('🎯 Match Analysis Studio')
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

# Corner assistant: works with live loaded fixtures and modelled results.
# Optional OPENAI_API_KEY enables conversational responses; the local fallback
# remains functional without any additional paid service.
def _assistant_reply(question):
    """Evidence-grounded on-site research assistant; no invented results."""
    import re
    q=question.casefold().strip()
    now_utc=datetime.now(timezone.utc)
    today_local=datetime.now(SYDNEY).date()
    future=sorted(
        [r for r in rows if r.get('result') and
         datetime.fromisoformat(r['kickoff'].replace('Z','+00:00'))>=now_utc],
        key=lambda r:r['kickoff'])
    def name(r):
        f=r['fixture']
        return f"{f['homeTeam'].get('name','Home')} vs {f['awayTeam'].get('name','Away')}"
    def local(r):
        return datetime.fromisoformat(r['kickoff'].replace('Z','+00:00')).astimezone(SYDNEY)
    def evidence(r):
        p=r['result']
        past=r.get('past') or []
        home_id=r['fixture']['homeTeam'].get('id')
        away_id=r['fixture']['awayTeam'].get('id')
        def team_games(tid,venue=None):
            games=[g for g in past if g.get('status')=='FINISHED' and
                   (g.get('homeTeam',{}).get('id')==tid or g.get('awayTeam',{}).get('id')==tid)]
            if venue=='home':games=[g for g in games if g['homeTeam'].get('id')==tid]
            if venue=='away':games=[g for g in games if g['awayTeam'].get('id')==tid]
            return sorted(games,key=lambda g:g.get('utcDate',''),reverse=True)[:10]
        def summary(games,tid):
            w=d=l=gf=ga=0
            for g in games:
                sc=g.get('score',{}).get('fullTime',{})
                if not isinstance(sc.get('home'),int) or not isinstance(sc.get('away'),int):continue
                is_home=g['homeTeam'].get('id')==tid
                x,y=(sc['home'],sc['away']) if is_home else (sc['away'],sc['home'])
                gf+=x;ga+=y
                if x>y:w+=1
                elif x==y:d+=1
                else:l+=1
            return f"{len(games)} games · W{w} D{d} L{l} · GF {gf} GA {ga}"
        h_all=team_games(home_id);a_all=team_games(away_id)
        h_venue=team_games(home_id,'home');a_venue=team_games(away_id,'away')
        h2h=[g for g in past if
             {g.get('homeTeam',{}).get('id'),g.get('awayTeam',{}).get('id')}=={home_id,away_id}
             and g.get('status')=='FINISHED']
        h2h=sorted(h2h,key=lambda g:g.get('utcDate',''),reverse=True)[:10]
        scores=p.get('scores',[])[:5]
        top=', '.join(f"{h}-{a} ({prob:.1%})" for h,a,prob in scores) or 'unavailable'
        sources=', '.join(r.get('research_sources') or ['connected verified football data'])
        text=(f"**{name(r)}** · {local(r):%d %b %Y, %I:%M %p} Sydney · {r.get('league','')}\n\n"
              f"**Last 10 available:**\n\n"
              f"- Home team, overall: {summary(h_all,home_id)}\n"
              f"- Home team, home fixtures: {summary(h_venue,home_id)}\n"
              f"- Away team, overall: {summary(a_all,away_id)}\n"
              f"- Away team, away fixtures: {summary(a_venue,away_id)}\n"
              f"- Verified H2H: {len(h2h)} previous meetings in accessible history\n\n"
              f"**Model (1X2):** Home {p['home_win']:.1%} · Draw {p['draw']:.1%} · Away {p['away_win']:.1%}\n\n"
              f"**Expected goals:** Home {p['home_xg']:.2f}, Away {p['away_xg']:.2f}\n\n"
              f"**Five most likely exact scores:** {top}\n\n"
              f"**Source:** {sources}.")
        if len(h_all)<10 or len(a_all)<10 or len(h_venue)<10 or len(a_venue)<10:
            text+='\n\n⚠️ At least one form split has fewer than 10 verified matches. Treat this as limited-sample analysis.'
        return text
    # Respond to requests about the website itself.
    if any(x in q for x in ['how to','how do','website','navigation','where is','settings','accuracy','help']):
        return ('**ZicZac website guide:** Home shows match previews. Predict shows detailed '
                'statistical forecasts. Doubles lets you filter by league and team and choose two '
                'matches. Worldwide Match Finder lists discovery-only fixtures. Archive tracks '
                'saved predictions. I use loaded data, not unrestricted live internet browsing.')
    # A request for five exact-score doubles is different from five exact-score
    # outcomes for a single game, and different from double-chance picks.
    if any(x in q for x in ['double','accumulator','acca','parlay']):
        if len(future)<2:
            return ('I need at least two upcoming fixtures with verified model history to calculate '
                    'exact-score doubles. Expand monitored competitions and date range, or select '
                    'two matches under **Doubles**. I cannot responsibly invent missing data.')
        from itertools import combinations as _combinations
        candidates=[]
        for first,second in _combinations(future[:30],2):
            if not first['result'].get('scores') or not second['result'].get('scores'):continue
            h1,a1,p1=first['result']['scores'][0]
            h2,a2,p2=second['result']['scores'][0]
            candidates.append((p1*p2,first,second,(h1,a1,p1),(h2,a2,p2)))
        candidates.sort(key=lambda x:x[0],reverse=True)
        if not candidates:return 'Not enough verified exact-score model outputs for two-leg doubles.'
        lines=['**Top five two-match exact-score doubles (model-ranked, not guaranteed):**']
        for i,(joint,first,second,sc1,sc2) in enumerate(candidates[:5],1):
            lines.append(f"{i}. **{name(first)} {sc1[0]}-{sc1[1]}** ({sc1[2]:.1%}) + "
                         f"**{name(second)} {sc2[0]}-{sc2[1]}** ({sc2[2]:.1%}) "
                         f"· combined estimate **{joint:.2%}**")
        lines.append('**Research basis:** Verified historical scores and the website’s goal model. '
                     'Combined estimates multiply two model probabilities and assume independence. '
                     'The five suggestions may reuse fixtures; they are not independent bets. '
                     'No score is certain, and the model does not verify injuries or confirmed lineups.')
        lines.append('For your own teams and leagues, open **Doubles → Choose your league and teams**.')
        return '\n\n'.join(lines)
    if any(w in q for w in ['today','tomorrow','upcoming','fixture','schedule','next match']):
        day=today_local+timedelta(days=1) if 'tomorrow' in q else today_local if 'today' in q else None
        ready=[r for r in future if day is None or local(r).date()==day]
        if ready:
            return '**Upcoming prediction-ready games (Sydney time):**\n\n'+'\n\n'.join(
                f"- **{name(r)}** · {local(r):%d %b, %I:%M %p} · {r.get('league','')} · "
                f"top score {r['result']['scores'][0][0]}-{r['result']['scores'][0][1]} "
                f"({r['result']['scores'][0][2]:.1%})"
                for r in ready[:12])
        discovered=[x['event'] for x in discoverable if
                    x['event'].get('kickoff') and x['event']['kickoff']>=now_utc and
                    (day is None or x['event']['local'].date()==day)]
        if discovered:
            return ('**Upcoming discovered fixtures (Sydney time; no verified score forecast):**\n\n'
                    +'\n\n'.join(f"- {e['home']} vs {e['away']} · {e['local']:%d %b, %I:%M %p} · {e.get('league','')}"
                                     for e in discovered[:12])
                    +'\n\nThese fixtures are not automatically prediction-ready.')
        return 'No upcoming fixtures are currently loaded for that date. Try a wider fixture date range.'
    ignored={'score','scores','match','predict','prediction','about','details','analysis',
             'league','team','football','correct','research','deep','last','five','give','show'}
    tokens=[x for x in re.findall(r'[a-z]{4,}',q) if x not in ignored]
    matches=[r for r in future if any(token in (name(r)+' '+str(r.get('league',''))).casefold() for token in tokens)]
    if matches:
        return evidence(matches[0])+'\n\n**Conclusion:** These are statistical estimates, not a mathematically proven result or a guaranteed bet.'
    return ('Ask for **“Tomorrow’s matches”**, **“Five exact-score doubles”**, or '
            '**“Chelsea deep analysis”**. I can report verified last-ten form, venue splits, '
            'H2H, expected goals and score probabilities when those records are loaded. '
            'I cannot claim to have searched every website or guarantee a winning bet.')

with st.container(key='ziczac_assistant_corner'):
    with st.popover('🤖 Ask ZicZac AI',use_container_width=False):
        st.markdown('**⚽ ZicZac Football Assistant**')
        st.caption('Ask about upcoming games, teams, exact scores or five doubles. Answers use the website’s currently loaded match data.')
        if 'zz_chat_messages' not in st.session_state:
            st.session_state.zz_chat_messages=[]
        for msg in st.session_state.zz_chat_messages[-10:]:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
        question=st.chat_input('Ask about a match or five doubles…',key='zz_corner_chat')
        if question:
            st.session_state.zz_chat_messages.append({'role':'user','content':question})
            answer=_assistant_reply(question)
            st.session_state.zz_chat_messages.append({'role':'assistant','content':answer})
            with st.chat_message('user'):st.markdown(question)
            with st.chat_message('assistant'):st.markdown(answer)
        if st.button('Clear chat',key='zz_chat_clear'):
            st.session_state.zz_chat_messages=[]
            st.rerun()
st.markdown("""<style>
.st-key-ziczac_assistant_corner {
    position:fixed!important;bottom:88px;right:14px;z-index:999!important;
    width:auto!important;max-width:calc(100vw - 28px);
}
.st-key-ziczac_assistant_corner [data-testid="stPopover"] > button {
    border-radius:999px!important;background:linear-gradient(125deg,#6845e8,#963ee4)!important;
    border:2px solid #b39aff!important;color:white!important;
    box-shadow:0 8px 26px #0009!important;font-weight:800!important;
    padding:11px 16px!important;min-height:49px!important;
}
@media(max-width:600px){.st-key-ziczac_assistant_corner{right:12px;bottom:90px}}

/* Bright, accessible chat composer and larger expanded panel on iPhone. */
[data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) {
    width:min(540px,calc(100vw - 22px))!important;
    max-width:calc(100vw - 22px)!important;
    background:linear-gradient(160deg,#1b315d,#122044)!important;
    border:2px solid #8d75ff!important;
    border-radius:20px!important;
    box-shadow:0 16px 48px #000b,0 0 0 3px #a28aff25!important;
    padding:18px!important;
}
[data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"] {
    border:2px solid #9c82ff!important;
    border-radius:17px!important;
    background:#233a6d!important;
    min-height:76px!important;
    box-shadow:0 0 0 3px #9275ff28,0 8px 20px #0004!important;
    transition:border-color .15s,box-shadow .15s,background .15s;
}
[data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"]:focus-within {
    border:3px solid #e2b7ff!important;
    background:#2c437d!important;
    box-shadow:0 0 0 5px #ac83ff50,0 0 25px #9671ff70!important;
}
[data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"] textarea {
    min-height:54px!important;
    font-size:17px!important;
    line-height:1.5!important;
    color:#fff!important;
    caret-color:#fff!important;
    padding:15px 12px!important;
}
[data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"] textarea::placeholder {
    color:#d3dcff!important;
    opacity:1!important;
}
[data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"] button {
    background:#7655e9!important;
    border-radius:12px!important;
    min-width:44px!important;
    min-height:44px!important;
}
[data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) [data-testid="stChatMessage"] {
    background:#1b3159!important;
    border:1px solid #4d69a2!important;
    border-radius:14px!important;
    margin-bottom:8px;
}
@media(max-width:600px){
  [data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) {
    width:calc(100vw - 18px)!important;
    max-width:calc(100vw - 18px)!important;
    max-height:65dvh!important;
    overflow-y:auto!important;
    padding:14px!important;
  }
  [data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"] {
    min-height:84px!important;
  }
  [data-testid="stPopoverBody"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"] textarea {
    font-size:18px!important;
    min-height:60px!important;
  }
}

</style>""",unsafe_allow_html=True)

with active_page_slot:
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Upcoming fixtures',len(rows));c2.metric('Modelled fixtures',sum(bool(r['result']) for r in rows))
    c3.metric('Competitions loaded',len(histories));c4.metric('Saved forecasts',len(load(conn)) if conn else 'N/A')
    # Native navigation buttons: decorative HTML cards were not clickable.
    feature_cols=st.columns(2,gap='small')
    with feature_cols[0]:
        st.button('🎯 Exact Scores\\nModel probabilities',key='feature_exact',use_container_width=True,on_click=_navigate_to,args=('predict',))
        st.button('📈 Live Form\\nTeam stats & H2H',key='feature_form',use_container_width=True,on_click=_navigate_to,args=('predict',))
    with feature_cols[1]:
        st.button('🔗 Five Doubles\\nChoose league & teams',key='feature_doubles',use_container_width=True,on_click=_navigate_to,args=('doubles',))
        st.button('🏆 Accuracy\\nBacktesting & archive',key='feature_accuracy',use_container_width=True,on_click=_navigate_to,args=('archive',))
    st.markdown('<div class="zz-section-title">⚽ Upcoming match previews</div>',unsafe_allow_html=True)
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
    # Exactly one evidence-based daily feature, selected only from fixtures on today's Sydney date.
    if view == 'home':
        st.subheader('🏆 Tomorrow’s Game of the Day')
        st.caption('One match for tomorrow in Sydney time, ranked by model evidence and probability. Exact scores are estimates, not guaranteed results.')
        def _sydney_day(row):
            try:
                return datetime.fromisoformat(row['kickoff'].replace('Z','+00:00')).astimezone(SYDNEY).date()
            except (KeyError,ValueError,TypeError):
                return None
        daily_candidates=[r for r in rows if r.get('result') and _sydney_day(r)==today+timedelta(days=1)]
        def _daily_rank(row):
            p=row['result']
            venue=min(len(p.get('home_home10',[])),len(p.get('away_away10',[])))
            recent=min(len(p.get('home_last10',[])),len(p.get('away_last10',[])))
            return (min(venue,10)>=5 and min(recent,10)>=5,
                    p['scores'][0][2] if p.get('scores') else 0,
                    min(venue,10),min(recent,10),min(len(p.get('h2h_last10',[])),5),
                    str(row['fixture'].get('id','')))
        if daily_candidates:
            daily=max(daily_candidates,key=_daily_rank)
            dm,dp=daily['fixture'],daily['result']
            top=dp['scores'][0]
            st.markdown('### '+match_name(dm) if 'match_name' in globals() else '### '+str(dm['homeTeam'].get('name','Home'))+' vs '+str(dm['awayTeam'].get('name','Away')))
            st.caption(str(daily.get('league','Football'))+' · '+datetime.fromisoformat(daily['kickoff'].replace('Z','+00:00')).astimezone(SYDNEY).strftime('%d %b %Y, %I:%M %p Sydney'))
            dcols=st.columns(3)
            dcols[0].metric('Most likely exact score',f"{top[0]}–{top[1]}")
            dcols[1].metric('Exact-score probability',f"{top[2]:.1%}")
            dcols[2].metric('Historical data quality',dp.get('data_quality','unknown').title())
            st.markdown('**Full-time result probabilities**')
            st.bar_chart(pd.DataFrame({'Probability (%)':[round(dp['home_win']*100,1),round(dp['draw']*100,1),round(dp['away_win']*100,1)]},index=['Home win','Draw','Away win']))
            st.markdown('**Expected goals and model comparison**')
            st.bar_chart(pd.DataFrame({'Expected goals':[dp['home_xg'],dp['away_xg']]},index=['Home','Away']))
            methods=dp.get('methods',{})
            if methods:
                st.dataframe(pd.DataFrame([{'Estimator':label,'Home expected goals':round(values[0],2),'Away expected goals':round(values[1],2)} for label,values in methods.items()]),hide_index=True,use_container_width=True)
            st.markdown('**Last 10 matches: overall and venue-specific form**')
            summary=[]
            for title,stats in [('Home · overall',dp.get('home_stats')),('Home · at home',dp.get('home_venue_stats')),('Away · overall',dp.get('away_stats')),('Away · away',dp.get('away_venue_stats'))]:
                if stats:
                    summary.append({'Sample':title,'Matches':stats['played'],'Wins':stats['wins'],'Draws':stats['draws'],'Losses':stats['losses'],'Goals scored / match':round(stats['scored'],2),'Goals conceded / match':round(stats['conceded'],2)})
            if summary:st.dataframe(pd.DataFrame(summary),hide_index=True,use_container_width=True)
            with st.expander('📈 Recent goal trends, H2H and top 10 score probabilities',expanded=False):
                for title,games in [('Home · last 10 overall',dp.get('home_last10',[])),('Home · last 10 home',dp.get('home_home10',[])),('Away · last 10 overall',dp.get('away_last10',[])),('Away · last 10 away',dp.get('away_away10',[]))]:
                    st.markdown('**'+title+'**')
                    if games:
                        chart=pd.DataFrame([{'Game':str(i+1),'Scored':g['gf'],'Conceded':g['ga']} for i,g in enumerate(reversed(games))]).set_index('Game')
                        st.line_chart(chart)
                        st.caption('Oldest to newest · '+str(len(games))+' verified matches')
                    else:st.caption('No verified results available for this split.')
                meetings=dp.get('h2h_last10',[])
                st.markdown('**Head-to-head · available last 10**')
                if meetings:
                    st.dataframe(pd.DataFrame([{'Date':h['date'][:10],'Featured home team goals':h['home_goals'],'Featured away team goals':h['away_goals'],'Result':h['result']} for h in meetings]),hide_index=True,use_container_width=True)
                else:st.caption('No verified H2H meetings in available historical data.')
                st.markdown('**Top 10 exact-score probabilities**')
                st.bar_chart(pd.DataFrame({'Probability (%)':[round(q*100,2) for _,_,q in dp['scores']]},index=[f'{h}–{a}' for h,a,_ in dp['scores']]))
            st.caption('Selection prioritises venue-specific sample size, recent form and H2H coverage. This is not a guarantee of accuracy. Injuries and confirmed lineups are not verified.')
        else:
            st.info('No match with sufficient verified historical data is available for today (Sydney time). The daily pick will not be fabricated or replaced with a match from a later date.')
    st.caption('Model probabilities are estimates, not guaranteed results. Use the tabs below for full analysis.')
    st.caption('New analysis: last 10 overall, last 10 home/away, available H2H, and two mathematical goal estimators. These changes require walk-forward backtesting before any improvement in accuracy can be claimed.')
    st.caption({'home':'🏠 Home · Upcoming matches','predict':'🎯 Predict · Match analysis','doubles':'🔗 Doubles · Combined predictions','archive':'📊 Archive · Results and backtesting','more':'☰ More · Model information'}[view])
    tabs=[st.container() for _ in range(7)]

    def match_name(m):return f"{m['homeTeam'].get('shortName') or m['homeTeam'].get('name','Home')} vs {m['awayTeam'].get('shortName') or m['awayTeam'].get('name','Away')}"
    if view in ['home','predict']:
        with tabs[0]:
            st.caption('Prediction dates use Sydney local time. Worldwide selected matches appear here after successful history analysis.')
            pred_window=st.session_state.get('prediction_window','Tomorrow')
            st.caption('Date selection is shared with Worldwide Match Finder above.')
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
        with tabs[1]:
            st.subheader('Five highest-probability double-chance selections')
            st.caption('1X = home or draw · X2 = away or draw · 12 = either team wins. These are NOT exact-score doubles.')
            picks=ranked_double_chance(rows,min_prob)
            if not picks:st.info('Not enough eligible fixtures at the selected probability threshold. Reduce the threshold or extend the date range.')
            else:
                st.dataframe(pd.DataFrame([{'Fixture':match_name(x['fixture']),'Market':x['selection'],'Model probability':f"{x['probability']:.1%}",'Kickoff UTC':x['kickoff'],'Competition':x['league']} for x in picks]),hide_index=True,use_container_width=True)
                st.caption('Selections are ranked by the model, not verified bookmaker odds or guaranteed wins.')
    if view in ['doubles']:
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
        with tabs[5]:
            st.markdown('''**Mathematical model:** Recency-weighted goals scored/conceded, venue-specific shrinkage, league home/away baselines, regularized expected-goal estimates and a normalized independent Poisson score matrix.\n\n**Human-style analysis:** Transparent rule-based commentary on relative strengths, sample sizes and uncertainty. It does **not** use an LLM, live injuries, verified lineups, bookmaker odds or premium xG data.\n\n**Backtesting:** Walk-forward evaluation of exact scores, match outcomes, Brier score and goal error. In-sample tuning on these same matches would overstate performance.\n\n**Limitations:** Data availability varies by league, free plan and season. Small samples, lineup changes, correlated matches and model miscalibration can make predictions inaccurate. No score can be guaranteed.''')
        st.divider()
        st.caption('For information and research only. Probabilities are model estimates, not betting advice or guaranteed results.')

    if view in ['doubles']:
        with tabs[6]:
            st.subheader('Choose the two matches for your exact-score double')
            st.caption('Choose two different matches with verified historical results. Worldwide fixtures without sufficient model data cannot be assigned invented probabilities.')
            if True:
                # Show future fixtures in Sydney time, nearest first; do not default
                # to arbitrary previously selected games several weeks away.
                def _double_local_time(r):
                    try:
                        return datetime.fromisoformat(r['kickoff'].replace('Z','+00:00')).astimezone(SYDNEY)
                    except (ValueError,TypeError,KeyError):
                        return datetime.max.replace(tzinfo=SYDNEY)
                tomorrow=today+timedelta(days=1)
                st.markdown('**📅 Upcoming matches — Sydney time**')
                double_window=st.selectbox('Show matches for',[
                    'Tomorrow','Next 3 days','Next 7 days','Next 14 days','All upcoming'
                ],index=0,key='manual_double_window')
                window_days={'Tomorrow':1,'Next 3 days':3,'Next 7 days':7,'Next 14 days':14}.get(double_window)
                # Include all available prediction-ready fixtures, not just the
                # initial eight preselected by the general analysis selector.
                all_pair_rows={str(r['fixture'].get('provider',''))+':'+str(r['fixture'].get('id','')):r
                               for r in rows if r.get('result')}
                # Include ALL discovered worldwide fixtures, even before Pro modelling.
                for item in discoverable:
                    ev=item['event']
                    fixture=ev.get('fixture')
                    if not fixture or not fixture.get('utcDate'):
                        continue
                    row=st.session_state.get('worldwide_models',{}).get(item['name'],{}).get('row') or item.get('model')
                    if not row:
                        row={'fixture':fixture,'kickoff':fixture['utcDate'],
                             'league':ev.get('league','Unknown competition'),'result':None,
                             '_worldwide_event':ev,'_worldwide_name':item['name']}
                    key_id=str(fixture.get('provider',''))+':'+str(fixture.get('id',''))
                    all_pair_rows[key_id]=row
                pair_candidates=sorted(all_pair_rows.values(),key=_double_local_time)
                pair_candidates=[r for r in pair_candidates
                                 if _double_local_time(r).date()>=tomorrow
                                 and (window_days is None or _double_local_time(r).date()<tomorrow+timedelta(days=window_days))]
                st.markdown('**🏆 Choose your league and teams for the double**')
                leagues=sorted({str(r.get('league','Unknown competition')) for r in pair_candidates})
                selected_double_league=st.selectbox('League',['All leagues']+leagues,key='double_league_filter')
                if selected_double_league!='All leagues':
                    pair_candidates=[r for r in pair_candidates if str(r.get('league','Unknown competition'))==selected_double_league]
                def _double_team_names(r):
                    fixture=r['fixture']
                    return (str(fixture.get('homeTeam',{}).get('name','')),str(fixture.get('awayTeam',{}).get('name','')))
                teams=sorted({team for r in pair_candidates for team in _double_team_names(r) if team})
                selected_double_team=st.selectbox('Team',['All teams']+teams,key='double_team_filter')
                if selected_double_team!='All teams':
                    pair_candidates=[r for r in pair_candidates if selected_double_team in _double_team_names(r)]
                st.caption(f'{len(pair_candidates)} available fixtures match your filters · {sum(bool(r.get("result")) for r in pair_candidates)} prediction-ready. Select two to run Pro research.')
                if len(pair_candidates)<2:
                    st.info('To build a two-match double, choose All teams or All leagues if this filter has fewer than two eligible fixtures.')
                if len(pair_candidates)<2:
                    st.info('Fewer than two fixtures were returned for this range. Try a longer range.')
                pair_options={fixture_label(r):r for r in pair_candidates}
                pair_names=list(pair_options)
                first_name=st.selectbox('Match 1',pair_names,key='manual_double_first_'+str(selected_double_league)+'_'+str(selected_double_team),
                                        index=0 if pair_names else None,
                                        placeholder='Choose an upcoming match')
                second_names=[name for name in pair_names if name!=first_name]
                second_name=st.selectbox('Match 2',second_names,key='manual_double_second_'+str(selected_double_league)+'_'+str(selected_double_team),
                                         index=0 if second_names else None,
                                         placeholder='Choose a different match')
                if first_name is None or second_name is None:
                    st.caption('Select two available matches to calculate exact-score doubles.')
                else:
                    st.caption('Match 1: '+_double_local_time(pair_options[first_name]).strftime('%a %d %b, %I:%M %p')+' Sydney · Match 2: '+_double_local_time(pair_options[second_name]).strftime('%a %d %b, %I:%M %p')+' Sydney')
                # Model selected worldwide fixtures on demand; do not hide them beforehand.
                if first_name is not None and second_name is not None and secondary_key:
                    for selected_name in (first_name,second_name):
                        selected_row=pair_options[selected_name]
                        if selected_row.get('result') or not selected_row.get('_worldwide_event'):
                            continue
                        ev=selected_row['_worldwide_event']
                        if ev.get('source')!='API-Football':
                            continue
                        with st.spinner('Pro researching '+selected_name+'…'):
                            researched,error=model_worldwide(secondary_key,ev)
                        st.session_state.setdefault('worldwide_models',{})[selected_row['_worldwide_name']]={'row':researched,'error':error}
                        if researched:
                            pair_options[selected_name]=researched
                        else:
                            st.warning(f"{selected_name}: {error}")
                manual_options=[]
                if first_name is not None and second_name is not None and pair_options[first_name].get('result') and pair_options[second_name].get('result'):
                    first=pair_options[first_name]
                    second=pair_options[second_name]
                    for h1,a1,p1 in first['result']['scores'][:10]:
                        for h2,a2,p2 in second['result']['scores'][:10]:
                            manual_options.append({'Score 1':f'{h1}-{a1}','Score 2':f'{h2}-{a2}',
                                'P1':p1,'P2':p2,'Joint':p1*p2})
                if first_name is not None and second_name is not None and not manual_options:
                    st.info('Both matches must have verified history before five exact-score doubles can be calculated. Choose another match or check the Pro research warning above.')
                if manual_options:
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
