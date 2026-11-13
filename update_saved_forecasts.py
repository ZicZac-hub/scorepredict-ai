"""Run the same complete singles-and-doubles revision pass used by the server."""
import argparse
import json
import os
import time
from pathlib import Path

from rescore_predictions import _read_archive
from revision_jobs import start_or_resume_revision_job, get_revision_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-db', default='predictions.sqlite3')
    args = parser.parse_args()
    primary_key = os.environ.get('FOOTBALL_DATA_API_KEY', '')
    primary_get = None
    if primary_key:
        from data import get_json
        primary_get = lambda path: get_json(path, primary_key)
    status = start_or_resume_revision_job(
        api_key=os.environ.get('API_FOOTBALL_KEY', ''), primary_get=primary_get,
        archive_rows=_read_archive(args.archive_db), db_path=args.archive_db, force=True)
    while status.get('state') == 'running':
        time.sleep(1)
        status = get_revision_status()
    print(json.dumps(status, indent=2))
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with Path(summary).open('a',encoding='utf-8') as handle:
            handle.write('## Saved forecast revision coverage\n\n')
            handle.write(f"State: {status.get('state')}. Singles revised: {status.get('revised',0)}; skipped: {status.get('skipped',0)}. Double sets revised: {status.get('revised_pairs',0)}; skipped: {status.get('skipped_pairs',0)}.\n")
            if status.get('state')=='incomplete':
                handle.write('\nCoverage is partial. Inspect revision reports for unavailable histories or identities; originals remain unchanged.\n')
    if status.get('state') == 'error':
        raise SystemExit('Revision run failed; original forecasts remain unchanged.')


if __name__ == '__main__':
    main()
