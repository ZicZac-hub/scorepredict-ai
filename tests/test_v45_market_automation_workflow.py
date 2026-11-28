"""Protect the scheduled handoff, quota reservations and immutable publication."""
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / '.github/workflows'


def test_market_automation_runs_independently_and_uses_shared_quota():
    workflow = (WORKFLOWS / 'v45-market-automation.yml').read_text(encoding='utf-8')
    prematch = (WORKFLOWS / 'prematch-evidence.yml').read_text(encoding='utf-8')
    collect = workflow.index('run: python v45_market_capture.py')
    freeze = workflow.index('run: python v45_market_trial.py --freeze')
    publish = workflow.index('bash stage_v45_market_evidence.sh')
    assert collect < freeze < publish
    assert "vars.V45_MARKET_ENABLED != 'false'" in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "cron: '6,16,26,36,46,56 * * * *'" in workflow
    assert 'capture(daily_limit=broad_daily_limit())' in prematch
    assert prematch.index('run: python v10_settlement.py') < prematch.index('git stash pop')
    assert prematch.index('git stash pop') < prematch.index('Commit permanent research history')


def test_checkpoints_stage_new_market_records_and_reject_rewrites(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()

    git('init', '-q')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com',
        'commit', '--allow-empty', '-qm', 'Base')
    marker = 'data/v24_market_catalog/requests/2026-11-28/v45_target_pair_late_10.json'
    records = (marker, 'data/v45_market_capture/versions/receipt.json',
               'data/v45_market_trial/protocol.json', 'data/v45_market_trial/trials/2026-11-28.json')
    for name in (*records, 'data/v45_market_capture/unpublished.json'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n', encoding='utf-8')

    command = ['bash', str(REPO / 'stage_v45_market_evidence.sh')]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == set(records)
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'Receipts')
    (tmp_path / marker).write_text('{"rewritten": true}\n', encoding='utf-8')
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'SAFE STOP' in result.stdout
