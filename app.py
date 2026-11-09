"""ScorePredict AI Worldwide: free Streamlit UI, live API, walk-forward validation."""
from datetime import datetime, timedelta, timezone
from itertools import combinations
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
    st.caption('Fixtures are sorted by kickoff, earliest first. Today and tomorrow are included.')
    max_games=st.slider('Display fixtures',5,80,30,5)
    min_prob=st.slider('Minimum double-chance probability',0.50,0.95,0.65,0.01)
    if st.button('Refresh football data'):
        st.rerun()
    st.caption('Data cached for 1 hour. Refresh reruns the page without bypassing API cooldowns. Saved snapshots may be older.')
try:comps=competitions(key)
except Exception as e:st.error(f'Unable to retrieve competitions: {e}');st.stop()
names={f"{c.get('area',{}).get('name','World')} · {c['name']} ({c['code']})":c['code'] for c in comps}
selected=st.multiselect('Competitions to monitor',list(names),default=list(names)[:min(3,len(names))])
if not selected:st.info('Choose one or more competitions.');st.stop()
if len(selected)>8:st.warning('Select up to eight competitions to protect free-tier API limits.');st.stop()
now=datetime.now(timezone.utc)
rows=[];histories={};errors=[];fixture_counts=[]
with st.spinner('Collecting fixtures and completed results...'):
    for name in selected:
        code=names[name]
        try:
            upcoming=fixtures(key,code,(now.date()-timedelta(days=1)).isoformat(),(now.date()+timedelta(days=15)).isoformat())
        except Exception as e:
            errors.append(f'{name}: fixture request failed: {e}')
            continue
        try:
            past=history(key,code)
        except Exception as e:
            past=[]
            errors.append(f'{name}: historical data unavailable ({e}); fixtures can still be displayed.')
        histories[name]=past
        visible=0
        for m in upcoming:
            kickoff=m.get('utcDate','')
            if not kickoff or kickoff < now.isoformat().replace('+00:00','Z'):continue
            if kickoff > (now+timedelta(days=days)).isoformat().replace('+00:00','Z'):continue
            visible+=1
            has_teams=bool((m.get('homeTeam') or {}).get('id') and (m.get('awayTeam') or {}).get('id'))
            try:
                result=predict(m,past) if has_teams and past else None
            except Exception:
                result=None
            rows.append({'league':name,'fixture':m,'result':result,'kickoff':kickoff})
        fixture_counts.append({'Competition':name,'API scheduled fixtures':len(upcoming),'Within selected time window':visible})
rows.sort(key=lambda r:r['kickoff'])
# Allow the user to select fixtures explicitly; no match is invented.
eligible=[r for r in rows if r.get('result')]
def fixture_label(r):
    m=r['fixture']
    return f"{m['homeTeam'].get('name','Home')} vs {m['awayTeam'].get('name','Away')} · {r['kickoff'][:16]} UTC"
labels={f"{r['fixture'].get('id')} · {fixture_label(r)}":r for r in eligible}
st.subheader('🎯 Choose your matches for detailed analysis')
chosen_labels=st.multiselect('Select fixtures to analyse and combine',list(labels),default=list(labels)[:min(8,len(labels))],help='Select at least two fixtures for exact-score doubles. The available fixtures depend on your API plan and chosen competitions.')
chosen_rows=[labels[k] for k in chosen_labels]

for e in errors:st.warning(e)
with st.expander('Why are some matches missing? · API coverage diagnostics'):
    st.caption('Only matches returned by football-data.org for the chosen competitions can appear. An API rate limit or missing competition cannot be fixed by changing the date slider.')
    if fixture_counts:st.dataframe(pd.DataFrame(fixture_counts),hide_index=True,use_container_width=True)
    st.caption('Times are shown in UTC in match labels. Some competitions are not covered by the free API. Matches with unavailable teams or historical data are displayed without a prediction.')
conn=None
try:
    conn=connect()
    save_predictions(conn,[r for r in rows if r['result']])
    for past in histories.values():reconcile(conn,past)
except Exception as e:st.warning(f'Local prediction snapshot storage unavailable: {e}')

c1,c2,c3,c4=st.columns(4)
c1.metric('Upcoming fixtures',len(rows));c2.metric('Modelled fixtures',sum(bool(r['result']) for r in rows))
c3.metric('Competitions loaded',len(histories));c4.metric('Saved forecasts',len(load(conn)) if conn else 'N/A')
tabs=st.tabs(['📅 Upcoming predictions','🛡️ Five double-chance picks','🔗 Five two-leg doubles','📈 Backtesting','🗂️ Prediction archive','🧠 Methodology','🎯 My five exact-score doubles'])

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

with tabs[6]:
    st.subheader('Choose the two matches for your exact-score double')
    st.caption('Choose any two different fixtures from your selected matches. The app will rank five exact-score combinations for that pair.')
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
        past=histories[r['league']]
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
