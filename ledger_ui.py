"""User-facing comparison of genuinely frozen forecasts and durable coverage."""
import json
import pandas as pd


def render_ledger(st, status, report):
    st.subheader('Fresh pre-match model comparison')
    if status.get('durable') and status.get('state') == 'complete':
        st.success('Central database connected. Original archives and paired forecasts are backed up.')
    elif status.get('durable'):
        st.warning('The central database is connected, but synchronization is incomplete. Originals remain preserved; inspect coverage before relying on the backup.')
    elif status.get('central_configured'):
        st.warning('The central database is unavailable. Records are retained in the local outbox for retry.')
    else:
        st.info('Local ledger active. Add DATABASE_URL in Streamlit Secrets to enable central backup.')
    st.caption('Both models use the same history and cutoff. Only forecasts actually frozen before kickoff enter this comparison; retrospective revisions stay separate.')
    a,b,c=st.columns(3)
    a.metric('Frozen comparison slots', report.get('common_frozen_slots',report.get('common_frozen_fixtures',0)))
    b.metric('Verified comparison slots',report.get('paired_verified_slots',report.get('paired_verified_matches',0)))
    c.metric('Unique verified matches',report.get('unique_verified_fixtures',0))
    rows=[]
    for key,label in (('v6','Preserved v6'),('v7','Fitted v7'),('league_poisson','League-average Poisson'),('always_1_1','Always 1–1')):
        metrics=(report.get('models') or {}).get(key) or {}
        rows.append({'Model':label,**metrics})
    if rows:
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    if not report.get('paired_verified_matches'):
        st.info('No finished paired forecasts are verified yet. Predictions will be graded after their real matches finish.')
    with st.expander('Evidence, coverage and uncertainty'):
        st.write({'storage':status.get('stats',{}),'archive_backup':status.get('archive',{}),
                  'paired_test':report.get('paired',{}),'exclusions':report.get('exclusions',{})})
        st.caption('Forecast horizons are separate groups. Several horizons for one match are not independent matches. A point forecast such as always 1–1 has no calibrated probability score.')
    st.download_button('Download fresh comparison report',json.dumps(report,indent=2).encode(),
                       'prospective_model_comparison.json','application/json',key='prospective_comparison_export')
