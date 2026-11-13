"""Append-only model revisions; original prediction evidence is never rewritten."""
import hashlib
import json
import os
import re
from pathlib import Path
from uuid import uuid4


def _publish_once(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Link is atomic and cannot replace an existing evidence file.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError('Revision already exists with different contents')
    finally:
        temporary.unlink(missing_ok=True)


def _write_once(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    _publish_once(path, raw)


def _preserve_pointer(path, root):
    if not path.exists():
        return
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    destination = root / 'previous_latest' / path.name / (digest + '.json')
    _publish_once(destination, raw)


def _latest(path, value, root):
    _preserve_pointer(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_revision_report(report, output_dir):
    """Save one immutable run and per-match records, plus archived latest pointers."""
    root = Path(output_dir)
    run_id = str(report['run_id'])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid revision run ID')
    _write_once(root / 'runs' / (run_id + '.json'), report)
    for revision in report.get('revisions', []):
        provider = re.sub(r'[^A-Za-z0-9_-]', '_', str(revision['provider']))
        fixture_id = str(int(revision['fixture_id']))
        revision_id = str(revision.get('revision_id') or run_id)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', revision_id):
            raise ValueError('Invalid per-match revision ID')
        _write_once(root / 'records' / provider / fixture_id / (revision_id + '.json'), revision)
        if revision.get('comparison_context'):
            # Pair-as-of reconstructions are not the latest single-match pick.
            continue
        _latest(root / (provider + '_' + fixture_id + '.json'), revision, root)
        # Existing UI readers use numeric API-Football filenames. Preserve that
        # pointer without allowing a different provider to take the same path.
        if revision['provider'] == 'API-Football':
            _latest(root / (fixture_id + '.json'), revision, root)
    _latest(root / 'latest_report.json', report, root)
    return report
