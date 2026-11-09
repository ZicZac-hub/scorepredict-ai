"""ScorePredict AI Worldwide: free Streamlit UI, live API, walk-forward validation."""
from datetime import datetime, timedelta, timezone
import os
import pandas as pd
import streamlit as st
from data import competitions, fixtures, history
from engine import predict
from advanced import backtest, summarize_backtest, ranked_double_chance, five_two_leg_doubles, context_review
from history_store import connect, save_predictions, reconcile, load

st.set_page_config(page_title='ScorePredict AI · Worldwide Pro',page_icon='⚽',layout='wide')
st.markdown('''<style>
.block-container{max-width:1450px;padding-top:1.6rem}
h1,h2,h3{letter-spacing:-.035em}
[data-testid="stMetric"]{background:rgba(110,130,160,.10);padding:16px;border:1px solid rgba(140,150,170,.15);border-radius:14px}
</style>''',unsafe_allow_html=True)
st.title('⚽ ScorePredict AI | Worldwide Pro')
st.caption('Live fixtures · Statistical score probabilities · Five double-chance picks · Two-leg doubles · Walk-forward backtesting')
try:key=st.secrets.get('FOOTBALL_DATA_API_KEY','')
except Exception:key=''
key=key or os.environ.get('FOOTBALL_DATA_API_KEY','')
if not key:
    st.warning('Add a football-data.org API key in Streamlit Secrets: `FOOTBALL_DATA_API_KEY = "your_key"`. No fabricated fixtures or scores are shown.')
    st.link_button('Get a free API key','https://www.football-data.org/client/register')
    st.stop()
with st.sidebar:
    st.header('Model settings')
    days=st.slider('Upcoming days',1,14,7)
    max_games=st.slider('Display fixtures',5,80,30,5)
    min_prob=st.slider('Minimum double-chance probability',0.50,0.95,0.65,0.01)
    if st.button('Refresh football data'):
        st.cache_data.clear();st.rerun()
    st.caption('Data cached for 1 hour. API limits and competition availability depend on your provider plan.')
try:comps=competitions(key)
except Exception as e:st.error(f'Unable to retrieve competitions: {e}');st.stop()
names={f"{c.get('area',{}).get('name','World')} · {c['name']} ({c['code']})":c['code'] for c in comps}
selected=st.multiselect('Competitions to monitor',list(names),default=list(names)[:min(3,len(names))])
if not selected:st.info('Choose one or more competitions.');st.stop()
if len(selected)>8:st.warning('Select up to eight competitions to protect free-tier API limits.');st.stop()
now=datetime.now(timezone.utc)
rows=[];histories={};errors=[]
with st.spinner('Collecting fixtures and completed results...'):
    for name in selected:
        code=names[name]
        try:
            upcoming=fixtures(key,code,now.date().isoformat(),(now.date()+timedelta(days=days)).isoformat())
            past=history(key,code)
            histories[name]=past
        except Exception as e:errors.append(f'{name}: {e}');continue
        for m in upcoming:
            if not m.get('homeTeam',{}).get('id') or not m.get('awayTeam',{}).get('id'):continue
            if m.get('utcDate','') < now.isoformat().replace('+00:00','Z'):continue
            rows.append({'league':name,'fixture':m,'result':predict(m,past),'kickoff':m.get('utcDate','')})
rows.sort(key=lambda r:r['kickoff'])
for e in errors:st.warning(e)
conn=None
try:
    conn=connect()
    save_predictions(conn,rows)
    for past in histories.values():reconcile(conn,past)
except Exception as e:st.warning(f'Local prediction snapshot storage unavailable: {e}')

c1,c2,c3,c4=st.columns(4)
c1.metric('Upcoming fixtures',len(rows));c2.metric('Modelled fixtures',sum(bool(r['result']) for r in rows))
c3.metric('Competitions loaded',len(histories));c4.metric('Saved forecasts',len(load(conn)) if conn else 'N/A')
tabs=st.tabs(['📅 Upcoming predictions','🛡️ Five double-chance picks','🔗 Five two-leg doubles','📈 Backtesting','🗂️ Prediction archive','🧠 Methodology'])

def match_name(m):return f"{m['homeTeam'].get('shortName') or m['homeTeam'].get('name','Home')} vs {m['awayTeam'].get('shortName') or m['awayTeam'].get('name','Away')}"
with tabs[0]:
    if not rows:st.info('No upcoming fixtures in the selected competitions/date range. Try other competitions.')
    for r in rows[:max_games]:
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
                for note in context_review(m,p,histories[r['league']]):st.write('• '+note)
                st.dataframe(pd.DataFrame([{'Team':'Home',**p['home_stats']},{'Team':'Away',**p['away_stats']}]),hide_index=True,use_container_width=True)
with tabs[1]:
    st.subheader('Five highest-probability double-chance selections')
    st.caption('1X = home or draw · X2 = away or draw · 12 = either team wins. These are NOT exact-score doubles.')
    picks=ranked_double_chance(rows,min_prob)
    if not picks:st.info('Not enough eligible fixtures at the selected probability threshold. Reduce the threshold or extend the date range.')
    else:
        st.dataframe(pd.DataFrame([{'Fixture':match_name(x['fixture']),'Market':x['selection'],'Model probability':f"{x['probability']:.1%}",'Kickoff UTC':x['kickoff'],'Competition':x['league']} for x in picks]),hide_index=True,use_container_width=True)
        st.caption('Selections are ranked by the model, not verified bookmaker odds or guaranteed wins.')
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
with tabs[4]:
    st.subheader('Saved forecasts versus final scores')
    st.caption('Forecasts are saved only once per fixture. Refreshes cannot rewrite an existing prediction. Local SQLite storage can reset on free cloud hosts; download the CSV regularly or connect a persistent database.')
    if conn:
        archive=pd.DataFrame(load(conn))
        if archive.empty:st.info('No predictions saved yet.')
        else:
            archive['exact_hit']=archive.apply(lambda r: r['predicted_score']==r['actual_score'] if pd.notna(r['actual_score']) else None,axis=1)
            st.dataframe(archive.drop(columns=['scores_json']),hide_index=True,use_container_width=True)
            st.download_button('Download prediction archive CSV',archive.to_csv(index=False).encode(),'prediction_archive.csv','text/csv')
with tabs[5]:
    st.markdown('''**Mathematical model:** Recency-weighted goals scored/conceded, venue-specific shrinkage, league home/away baselines, regularized expected-goal estimates and a normalized independent Poisson score matrix.\n\n**Human-style analysis:** Transparent rule-based commentary on relative strengths, sample sizes and uncertainty. It does **not** use an LLM, live injuries, verified lineups, bookmaker odds or premium xG data.\n\n**Backtesting:** Walk-forward evaluation of exact scores, match outcomes, Brier score and goal error. In-sample tuning on these same matches would overstate performance.\n\n**Limitations:** Data availability varies by league, free plan and season. Small samples, lineup changes, correlated matches and model miscalibration can make predictions inaccurate. No score can be guaranteed.''')
st.divider()
st.caption('For information and research only. Probabilities are model estimates, not betting advice or guaranteed results.')
