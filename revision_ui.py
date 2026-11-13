"""Display revisions separately from immutable original prediction accuracy."""
import json
from pathlib import Path

import pandas as pd


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return {}


def render_revision_panel(st, status=None, format_time=str, on_retry=None):
    st.subheader('Updated forecasts and five doubles')
    st.caption('Original forecasts remain in the accuracy archive. Upcoming revisions are new forecasts; finished-match reconstructions are retrospective comparisons using earlier information.')
    status = status or {}
    state = status.get('state', status.get('status', 'not_started'))
    progress = status.get('processed', status.get('completed', status.get('done', 0)))
    total = status.get('input_records', status.get('total', 0))
    if state == 'running':
        st.info(f'Updating the saved archive in the background: {progress} of {total} records checked. Reopen this page to see progress.')
    elif state == 'incomplete':
        st.warning('Some saved forecasts or doubles could not be revised. Their original records are preserved; inspect the skipped records below.')
    elif state == 'error':
        st.warning('The archive update could not finish. Original forecasts remain available. Retry when provider access is available.')
    elif state == 'complete':
        st.success('The saved-archive revision run has completed.')
    if on_retry and st.button('Update all saved forecasts and five doubles', key='revision_retry_all'):
        on_retry()
        st.info('Revision run requested. Original forecasts and results remain unchanged.')

    report = _load('data/prematch/rescores/latest_report.json')
    revisions = report.get('revisions') or []
    upcoming, retrospective, doubles = st.tabs(['Upcoming revisions', 'Finished retrospective comparisons', 'Revised five doubles'])
    for tab, mode in ((upcoming, 'upcoming'), (retrospective, 'retrospective')):
        with tab:
            rows = []
            for revision in revisions:
                if revision.get('comparison_context') or revision.get('mode', 'upcoming') != mode:
                    continue
                original, updated = revision.get('original_score'), revision.get('updated_score')
                rows.append({
                    'Match': str(revision.get('home_team', ''))+' vs '+str(revision.get('away_team', '')),
                    'Kickoff (Sydney)': format_time(revision.get('kickoff_utc', '')),
                    'Original score': '-'.join(map(str, original or [])),
                    'Revised score': '-'.join(map(str, updated or [])),
                    'Revised score probability': revision.get('updated_score_probability'),
                    'Evidence cutoff': format_time(revision.get('history_as_of_utc', '')),
                    'Model': revision.get('updated_model', ''),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info('No revisions of this type have been generated yet. The server checks its saved archive after loading the upgrade.')
            if mode == 'retrospective':
                st.caption('Retrospective results do not count as predictions made before the match and do not change the original hit rate.')
    with doubles:
        double_report = _load('data/prematch/double_rescores/latest_report.json')
        rows = []
        for revision in double_report.get('revisions') or []:
            for rank, pick in enumerate(revision.get('updated_five_doubles', revision.get('five_doubles', [])), 1):
                rows.append({
                    'Match 1': revision.get('first_match', ''),
                    'Match 2': revision.get('second_match', ''),
                    'Comparison': revision.get('mode', ''),
                    'Double': rank,
                    'Score 1': pick.get('score1', pick.get('Score 1', '')),
                    'Score 2': pick.get('score2', pick.get('Score 2', '')),
                    'Joint probability': pick.get('probability', pick.get('Joint')),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info('Revised doubles appear after both saved legs have enough verified earlier history.')
        st.caption('Each row requires both exact scores. Probabilities assume independent matches; alternatives for the same pair are mutually exclusive.')
        if double_report:
            with st.expander('Double revision coverage and unavailable pairs'):
                st.write({key:double_report.get(key) for key in ('saved_pairs','rescored_pairs','upcoming_pairs','retrospective_pairs')})
                if double_report.get('skipped'):
                    st.dataframe(pd.DataFrame(double_report['skipped']),hide_index=True,use_container_width=True)
            st.download_button('Download double revision report',json.dumps(double_report,indent=2).encode(),'double_revisions.json','application/json')
    if report:
        with st.expander('Revision coverage and unavailable records'):
            st.write(report.get('summary') or {'revised': len(revisions), 'skipped': len(report.get('skipped') or [])})
            if report.get('skipped'):
                st.dataframe(pd.DataFrame(report['skipped']), hide_index=True, use_container_width=True)
        st.download_button('Download revision report', json.dumps(report, indent=2).encode(), 'forecast_revisions.json', 'application/json')
