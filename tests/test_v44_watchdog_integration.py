"""Protect the V44 scheduled handoff and immutable history publication."""
from pathlib import Path

def test_early_history_checkpoint_is_explicit_and_append_only():
    text=Path(".github/workflows/prematch-evidence.yml").read_text()
    checkpoint=text.split("      - name: Commit write-once source receipts early",1)[1].split(
        "      - name: Back up uncommitted source files",1)[0]
    assert "for folder in data/v10_prospective/forecasts data/prospective_ledger/forecasts; do" in checkpoint
    assert "for folder in data/v44_early_history/receipts data/v44_early_history/blobs; do" in checkpoint
    assert "--diff-filter=DMRTUXB -- data/v44_early_history/" in checkpoint
    assert "SAFE STOP: existing V44 historical evidence changed" in checkpoint
    backup=text.split("      - name: Back up uncommitted source files",1)[1].split(
        "      - name: Persist evidence and verify completed matches",1)[0]
    assert "data/v44_early_history/receipts/" in backup
    assert "data/v44_early_history/blobs/" in backup


def test_watchdog_default_on_with_explicit_rollback_and_publish_only_on_success():
    text=Path(".github/workflows/v42-research-watchdog.yml").read_text()
    trial=text.split("      - name: Freeze and verify V44 exact-history ratings trial",1)[1].split(
        "      - name: Publish only isolated research archive",1)[0]
    assert "vars.V44_RATINGS_TRIAL_ENABLED != 'false'" in trial
    assert 'OPENAI_API_KEY: ""' in trial
    assert 'API_FOOTBALL_KEY: ""' in trial
    assert "python v44_ratings_prospective.py --freeze" in trial
    assert 'if report["counts"].get("rejected"):' in trial
    assert '("capture_status",run["capture"])' in trial
    assert trial.count('python v44_ratings_prospective.py --freeze')==1
    assert "github.ref == 'refs/heads/main'" in text
    assert "cancel-in-progress: false" in text
    assert 'if [ "$V44_OUTCOME" = "success" ]; then' in text
    assert "--diff-filter=DMRTUXB -- data/v44_ratings_prospective/protocol.json data/v44_ratings_prospective/trials/" in text


def test_early_history_default_enabled_but_venue_experiment_stays_opt_in():
    text=Path(".github/workflows/prematch-evidence.yml").read_text()
    assert "vars.V44_EARLY_HISTORY_ENABLED != 'false'" in text
    assert 'V44_VENUE_COLLECTION_ENABLED: '+chr(36)+'{{ vars.V44_VENUE_COLLECTION_ENABLED }}' in text
