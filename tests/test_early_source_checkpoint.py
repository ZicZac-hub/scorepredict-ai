"""Protect authentic pre-kickoff evidence when downstream audits run late.

The first checkpoint is a durability improvement, not an accuracy result.
"""
from pathlib import Path


WORKFLOW = Path(".github/workflows/prematch-evidence.yml")


def workflow():
    return WORKFLOW.read_text(encoding="utf-8")


def section(name, until):
    content = workflow()
    return content.split("      - name: " + name, 1)[1].split(
        "      - name: " + until, 1)[0]


def test_capture_is_not_suppressed_by_an_independent_settlement_failure():
    s = workflow()
    result = s.index("run: python v10_settlement.py")
    source = s.index("run: python capture_prematch.py")
    persist = s.index("run: python persist_prematch.py --snapshots-only")
    checkpoint = s.index('git commit -m "data: checkpoint genuine pre-match originals [skip ci]"')
    downstream = s.index("run: python persist_prematch.py\n")
    assert result < source < persist < checkpoint < downstream
    capture = section("Capture verified pre-kickoff snapshots",
                      "Persist newly captured originals")
    assert "id: capture" in capture
    assert "if: ${{ !cancelled() }}" in capture
    saved = section("Persist newly captured originals",
                    "Commit write-once source receipts early before long research stages")
    assert "steps.capture.outcome == 'success'" in saved
    assert "id: persist_originals" in saved


def test_early_checkpoint_never_stages_results_or_rewrites_old_forecasts():
    s = workflow()
    publish = section(
        "Commit write-once source receipts early before long research stages",
        "Back up uncommitted source files only if early checkpoint fails")
    allowed = {
        "data/v10_prospective/forecasts",
        "data/prospective_ledger/forecasts",
    }
    assert "git add -- data/prospective_ledger/capture_coverage.json" in publish
    assert "for folder in data/v10_prospective/forecasts data/prospective_ledger/forecasts; do" in publish
    for folder in allowed:
        assert folder in publish
    assert "git add -- data/prematch/snapshots" not in publish
    assert "git add -- data/daily_picks" not in publish
    assert "data/v10_prospective/verified_results.json" not in publish
    assert "data/prematch/verified_results.json" not in publish
    assert "git rebase --autostash origin/main" in publish
    assert "python reconcile_capture_conflicts.py" in publish
    assert "if ! git push origin HEAD:main" not in publish
    assert 'if git push origin HEAD:main; then' in publish
    assert "id: early_checkpoint" in publish
    assert "steps.capture.outcome == 'success'" in publish
    assert "steps.persist_originals.outcome == 'success'" in publish
    assert "SAFE STOP: conflicting immutable source receipts" in publish


def test_pre_publish_artifact_is_early_and_separate_from_final_audit_archive():
    s = workflow()
    backup = section(
        "Back up uncommitted source files only if early checkpoint fails",
        "Persist evidence and verify completed matches")
    assert "if: always() && steps.early_checkpoint.outcome != 'success'" in backup
    assert "uses: actions/upload-artifact@v4" in backup
    assert "name: early-prematch-authenticated-source-checkpoint" in backup
    assert "data/v10_prospective/forecasts/" in backup
    assert "data/prospective_ledger/capture_coverage.json" in backup
    assert "retention-days: 30" in backup
    assert "      - name: Commit permanent research history" in s
    assert "      - name: Archive immutable snapshots" in s
    assert "data/v36_prospective_challengers/sets/" in s
    assert "run: python v28_independent_pair_trial.py" in s
    assert "run: python v35_continuous_double_trials.py" in s
    assert "run: python v36_prospective_double_trials.py" in s


def test_no_new_source_provider_calls_or_pipeline_bypass():
    s = workflow()
    assert s.count("run: python capture_prematch.py") == 1
    assert "API_FOOTBALL_KEY: ${{ secrets.API_FOOTBALL_KEY }}" in s
    assert "jobs:\n  capture:\n" in s
    assert "if: github.ref == 'refs/heads/main'" in s
    assert "cancel-in-progress: false" in s
    assert "timeout-minutes: 25" in s
