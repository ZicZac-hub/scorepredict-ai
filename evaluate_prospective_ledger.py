"""Evaluate the new prospective ledger without changing original accuracy reports."""
import argparse
import json
import os
from pathlib import Path

from forecast_ledger import append_file_record, open_ledger, read_file_records
from prospective_ledger import _settlement_matches, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger-dir', type=Path, default=Path('data/prospective_ledger'))
    parser.add_argument('--results', type=Path, default=Path('data/prematch/verified_results.json'))
    parser.add_argument('--output', type=Path, default=Path('data/prospective_ledger/comparison.json'))
    parser.add_argument('--database', action='store_true', help='Use DATABASE_URL when configured (also enabled automatically)')
    parser.add_argument('--files-only', action='store_true', help='Explicitly evaluate only the repository file ledger')
    args = parser.parse_args()
    file_forecasts = read_file_records(args.ledger_dir, kind='forecasts')
    forecasts = list(file_forecasts)
    settlements = read_file_records(args.ledger_dir, kind='settlements')
    verified = []
    if args.results.exists():
        source = json.loads(args.results.read_text(encoding='utf-8'))
        verified = list(source.values() if isinstance(source, dict) else source)
    persistence = {'file_settlements_written': 0, 'file_settlement_errors': 0,
                   'database': {'state': 'not_configured'}, 'state': 'complete'}
    url = None if args.files_only else (os.environ.get('DATABASE_URL') or os.environ.get('FORECAST_DATABASE_URL'))
    if url:
        try:
            with open_ledger(database_url=url) as store:
                # Push scheduled forecasts into the same central ledger before
                # settling them. Private server captures already in PostgreSQL
                # are read for comparison, never reconstructed from old scores.
                inserted = store.put_forecasts(file_forecasts)
                forecasts.extend(store.iter_forecasts())
                accepted = [row for row in verified if any(_settlement_matches(record, row) for record in forecasts)]
                written = store.put_settlements(accepted)
                settlements.extend(store.get_settlements())
                persistence['database'] = {'state': 'connected', 'forecasts_written': inserted,
                                           'settlements_written': written}
        except Exception as exc:
            # Do not expose a driver exception or URI: database passwords can
            # be embedded in either. The file report still records partial work.
            persistence['database'] = {'state': 'unavailable', 'error_type': type(exc).__name__}
            persistence['state'] = 'partial'
    for actual in verified:
        if not any(_settlement_matches(record, actual) for record in file_forecasts):
            continue
        try:
            persistence['file_settlements_written'] += int(append_file_record(actual, root=args.ledger_dir, kind='settlements'))
        except Exception as exc:
            persistence['file_settlement_errors'] += 1
            persistence['file_error_type'] = type(exc).__name__
            persistence['state'] = 'partial'
    settlements.extend(verified)
    report = evaluate(forecasts, settlements)
    report['persistence'] = persistence
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in ('common_frozen_fixtures', 'paired_verified_matches', 'pending_common_forecasts', 'exclusions')}, indent=2))
    if persistence['state'] == 'partial':
        raise SystemExit('Prospective comparison saved with incomplete ledger persistence; inspect its sanitized status.')


if __name__ == '__main__':
    main()
