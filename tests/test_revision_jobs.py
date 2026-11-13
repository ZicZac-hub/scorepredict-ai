"""Background rescoring coverage, progress, deduplication and privacy."""
from pathlib import Path
import json
import threading
from datetime import datetime, timezone

import pytest
import revision_jobs as jobs


def record(index=1, model='old-model'):
    return {'provider': 'API-Football', 'fixture_id': index,
            'captured_at_utc': '2026-11-01T00:00:00Z',
            'kickoff_utc': '2026-11-03T12:00:00Z',
            'predicted_score': [1, 1], 'original_model': model}


@pytest.fixture
def configured(monkeypatch, tmp_path):
    if jobs._ACTIVE is not None:
        jobs._ACTIVE.result(timeout=5)
    monkeypatch.setattr(jobs, '_ACTIVE', None)
    monkeypatch.setattr(jobs, '_ACTIVE_DIR', None)
    monkeypatch.setattr(jobs, '_model_version', lambda: 'new-model')
    monkeypatch.setattr(jobs, '_source_records',
                        lambda archive, snapshots: list(archive) + list(snapshots or []))
    monkeypatch.setattr(jobs, '_saved_pairs', lambda *args: [])
    calls = []

    def singles(**kwargs):
        sources = kwargs['records']
        calls.append(kwargs)
        revisions = []
        for i, source in enumerate(sources, 1):
            revisions.append(dict(source, mode='retrospective'))
            kwargs['progress'](i, len(sources), dict(source, status='revised'))
        return {'revisions': revisions, 'skipped': [],
                'summary': {'input_records': len(sources), 'processed': len(sources),
                            'revised': len(sources), 'skipped': 0,
                            'upcoming_revisions': 0, 'retrospective_comparisons': len(sources)}}

    monkeypatch.setattr(jobs, '_run_single_rescoring', singles)
    monkeypatch.setattr(jobs, '_run_double_rescoring',
                        lambda pairs, *args: {'revisions': list(pairs), 'skipped': []})
    return tmp_path / 'rescores', calls


def finish(output_dir):
    if jobs._ACTIVE is not None:
        jobs._ACTIVE.result(timeout=5)
    return jobs.get_revision_status(output_dir)


def test_every_old_archive_record_is_processed_without_an_interactive_batch_limit(configured):
    directory, calls = configured
    originals = [record(i) for i in range(236)]
    jobs.start_or_resume_revision_job(archive_rows=originals, snapshot_records=[], output_dir=directory)
    status = finish(directory)
    assert status['state'] == 'complete'
    assert status['processed'] == status['revised'] == 236
    assert status['retrospective_comparisons'] == 236
    assert len(calls[0]['records']) == 236
    assert originals[0]['predicted_score'] == [1, 1]


def test_same_sources_and_model_are_not_repeated_on_page_reruns(configured):
    directory, calls = configured
    for _ in range(2):
        jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
        finish(directory)
    assert len(calls) == 1


def test_new_old_source_changes_coverage_and_launches_another_pass(configured):
    directory, calls = configured
    jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
    finish(directory)
    jobs.start_or_resume_revision_job(archive_rows=[record(), record(2)], snapshot_records=[], output_dir=directory)
    assert finish(directory)['input_records'] == 2
    assert len(calls) == 2


def test_new_current_model_forecasts_do_not_restart_automatic_upgrade(configured):
    directory, calls = configured
    jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
    finish(directory)
    jobs.start_or_resume_revision_job(archive_rows=[record(), record(2, 'new-model-fallback-v6')],
                                    snapshot_records=[], output_dir=directory)
    finish(directory)
    assert len(calls) == 1
    jobs.start_or_resume_revision_job(archive_rows=[record(), record(2, 'new-model')],
                                    snapshot_records=[], output_dir=directory, force=True)
    assert finish(directory)['input_records'] == 2


def test_running_job_is_shared_without_queuing_duplicate_provider_calls(configured, monkeypatch):
    directory, calls = configured
    entered, release = threading.Event(), threading.Event()
    original = jobs._run_single_rescoring

    def blocked(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(jobs, '_run_single_rescoring', blocked)
    try:
        first = jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
        assert entered.wait(5)
        second = jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
        assert first['source_fingerprint'] == second['source_fingerprint']
        assert second['state'] == 'running'
    finally:
        release.set()
    finish(directory)
    assert len(calls) == 1


def test_skips_are_incomplete_and_do_not_create_an_automatic_quota_retry_loop(configured, monkeypatch):
    directory, calls = configured

    def skipped(**kwargs):
        calls.append(kwargs)
        kwargs['progress'](1, 1, {'fixture_id': 1, 'provider': 'API-Football', 'status': 'skipped'})
        return {'revisions': [], 'skipped': [{'fixture_id': 1}],
                'summary': {'processed': 1, 'revised': 0, 'skipped': 1}}

    monkeypatch.setattr(jobs, '_run_single_rescoring', skipped)
    jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
    status = finish(directory)
    assert status['state'] == 'incomplete'
    assert status['processed'] == 1 and status['revised'] == 0 and status['failed'] == 1
    jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
    finish(directory)
    assert len(calls) == 1


def test_unrevised_saved_double_keeps_job_incomplete(configured, monkeypatch):
    directory, calls = configured
    monkeypatch.setattr(jobs, '_saved_pairs', lambda *args: [{'pair_id': 'old-pair'}])
    monkeypatch.setattr(jobs, '_run_double_rescoring',
                        lambda *args: {'revisions': [], 'skipped': [{'pair_id': 'old-pair'}]})
    jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
    status = finish(directory)
    assert status['state'] == 'incomplete'
    assert status['revised'] == 1 and status['skipped_pairs'] == 1


def test_progress_files_never_contain_provider_credentials_or_arbitrary_errors(configured, monkeypatch):
    directory, calls = configured
    secret = 'provider-secret-test-value'

    def failed(**kwargs):
        raise RuntimeError('Denied ' + secret)

    monkeypatch.setattr(jobs, '_run_single_rescoring', failed)
    jobs.start_or_resume_revision_job(api_key=secret, archive_rows=[record()],
                                    snapshot_records=[], output_dir=directory)
    status = finish(directory)
    assert status['state'] == 'error'
    assert status['error_type'] == 'RuntimeError'
    assert secret not in (directory / 'server_status.json').read_text()


def test_stale_running_status_after_process_restart_is_resumed(configured):
    directory, calls = configured
    directory.mkdir()
    (directory / 'server_status.json').write_text(json.dumps({'state': 'running'}))
    jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
    assert finish(directory)['state'] == 'complete'
    assert len(calls) == 1


def test_only_current_model_sources_need_no_provider_requests(configured):
    directory, calls = configured
    status = jobs.start_or_resume_revision_job(archive_rows=[record(model='new-model')],
                                             snapshot_records=[], output_dir=directory)
    assert status['state'] == 'complete'
    assert status['input_records'] == 0
    assert calls == []


def test_mixed_historical_pair_reconstructs_both_legs_at_shared_capture(configured, monkeypatch):
    directory, calls = configured
    first, second = record(1), record(2)
    first['kickoff_utc'] = '2026-11-02T12:00:00Z'
    second['kickoff_utc'] = '2026-11-08T12:00:00Z'
    pair = {'pair_key': 'sqlite:old-pair', 'captured_at_utc': '2026-11-01T18:00:00Z',
            'first_provider': 'API-Football', 'second_provider': 'API-Football',
            'first_fixture_id': 1, 'second_fixture_id': 2,
            'first_kickoff_utc': first['kickoff_utc'], 'second_kickoff_utc': second['kickoff_utc']}
    monkeypatch.setattr(jobs, '_saved_pairs', lambda *args: [pair])
    now = datetime(2026, 11, 4, tzinfo=timezone.utc)
    jobs.start_or_resume_revision_job(archive_rows=[first, second], snapshot_records=[],
                                    now=now, output_dir=directory)
    status = finish(directory)
    assert status['input_original_records'] == 2
    assert status['derived_pair_leg_records'] == 2
    derived = [r for r in calls[0]['records'] if r.get('force_retrospective')]
    assert len(derived) == 2
    assert all(r['requested_as_of_utc'] == '2026-11-01T18:00:00+00:00' for r in derived)
    assert all(r['comparison_context'] == pair['pair_key'] for r in derived)
    assert all(r['captured_at_utc'] == '2026-11-01T00:00:00Z' for r in derived)


def test_new_credential_availability_retries_an_inaccessible_source(configured):
    directory, calls = configured
    jobs.start_or_resume_revision_job(archive_rows=[record()], snapshot_records=[], output_dir=directory)
    finish(directory)
    jobs.start_or_resume_revision_job(api_key='now-available', archive_rows=[record()],
                                    snapshot_records=[], output_dir=directory)
    finish(directory)
    assert len(calls) == 2


def test_background_job_uses_real_revision_pipeline_without_rewriting_sources(configured, monkeypatch):
    from copy import deepcopy
    import rescore_predictions as rescore
    directory, calls = configured
    original = {'provider': 'API-Football', 'fixture_id': 12, 'competition': 'PL',
        'captured_at_utc': '2026-11-01T00:00:00Z', 'kickoff_utc': '2026-11-03T12:00:00Z',
        'home_team': 'A', 'away_team': 'B', 'home_team_id': 1, 'away_team_id': 2,
        'predicted_score': [1, 1], 'forecast_model': 'preserved-v6'}
    before = deepcopy(original)
    fixture = {'id': 12, 'provider': 'API-Football', 'competition': {'code': 'PL'},
        'utcDate': original['kickoff_utc'], 'status': 'FINISHED',
        'homeTeam': {'id': 1, 'name': 'A'}, 'awayTeam': {'id': 2, 'name': 'B'},
        'score': {'fullTime': {'home': 3, 'away': 0}}}

    def predictor(target, history, **kwargs):
        assert target['status'] == 'SCHEDULED'
        assert target['score']['fullTime'] == {'home': None, 'away': None}
        return {'scores': [(2, 0, .13)], 'home_xg': 2.2, 'away_xg': .7,
                'home_win': .7, 'draw': .2, 'away_win': .1,
                'model_version': 'reference-informed-v7'}

    monkeypatch.setattr(jobs, '_source_records',
        lambda archive, snapshots: rescore.normalize_records(snapshot_records=snapshots))
    monkeypatch.setattr(jobs, '_model_version', lambda: 'reference-informed-v7')
    monkeypatch.setattr(rescore, 'provider_loaders',
        lambda *args: (lambda *loader_args: [], lambda source: deepcopy(fixture)))
    monkeypatch.setattr(jobs, '_run_single_rescoring',
        lambda **kwargs: rescore.run_rescoring(predictor=predictor, **kwargs))
    jobs.start_or_resume_revision_job(snapshot_records=[original], output_dir=directory,
                                    now=datetime(2026, 11, 4, tzinfo=timezone.utc))
    status = finish(directory)
    assert status['state'] == 'complete'
    assert status['retrospective_comparisons'] == 1
    report = json.loads((directory / 'latest_report.json').read_text())
    assert report['revisions'][0]['original_score'] == [1, 1]
    assert report['revisions'][0]['updated_score'] == [2, 0]
    assert report['revisions'][0]['excluded_from_original_accuracy'] is True
    assert original == before
