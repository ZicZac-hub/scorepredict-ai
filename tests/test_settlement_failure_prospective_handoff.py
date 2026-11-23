"""Prospective five-double issue must survive an independent settlement outage.

No new provider requests, retrospective tickets or unverifiable wins are
introduced by this workflow-only resilience fix.
"""
from pathlib import Path

FLOW = Path(".github/workflows/prematch-evidence.yml")


def _stage(name, next_name):
    workflow = FLOW.read_text(encoding="utf-8")
    return workflow.split("      - name: " + name, 1)[1].split(
        "      - name: " + next_name, 1)[0]


def test_result_settlement_failure_does_not_cancel_future_capture():
    settlement = _stage(
        "Independently verify recently completed exact-score and five-double fixtures",
        "Capture verified pre-kickoff snapshots")
    capture = _stage(
        "Capture verified pre-kickoff snapshots",
        "Persist newly captured originals")
    assert "id: initial_settlement" in settlement
    assert "continue-on-error: true" in settlement
    assert "run: python v10_settlement.py" in settlement
    assert "V10_RESULT_LIMIT:" in settlement
    assert "if: ${{ !cancelled() }}" in capture
    assert "run: python capture_prematch.py" in capture


def test_settlement_failure_still_fails_overall_job_without_claiming_results():
    workflow = FLOW.read_text(encoding="utf-8")
    failure = _stage(
        "Publish settlement failure after preserving genuine source evidence",
        "Archive immutable snapshots")
    assert "if: ${{ always() && steps.initial_settlement.outcome == 'failure' }}" in failure
    assert "exit 1" in failure
    assert "Independent result-settlement failed" in failure
    # Preserve archived evidence even after the terminal settlement error.
    archive = workflow.split("      - name: Archive immutable snapshots",1)[1]
    assert "if: always()" in archive
    assert "uses: actions/upload-artifact@v4" in archive
    assert workflow.index("run: python v36_prospective_double_trials.py") < workflow.index(
        "- name: Publish settlement failure after preserving genuine source evidence")
    assert workflow.index("- name: Commit permanent research history") < workflow.index(
        "- name: Publish settlement failure after preserving genuine source evidence")


def test_existing_pre_match_integrity_checks_and_budgets_are_unchanged():
    workflow=FLOW.read_text(encoding="utf-8")
    assert workflow.count("run: python v10_settlement.py") == 1
    assert workflow.count("run: python capture_prematch.py") == 1
    assert workflow.count("run: python v28_independent_pair_trial.py") == 1
    assert workflow.count("run: python v35_continuous_double_trials.py") == 1
    assert workflow.count("run: python v36_prospective_double_trials.py") == 1
    assert "V10_RESULT_LIMIT: ${{ steps.cadence.outputs.heavy == 'true' && '20' || '2' }}" in workflow
    assert "CAPTURE_MAX_API_CALLS" not in _stage(
        "Independently verify recently completed exact-score and five-double fixtures",
        "Capture verified pre-kickoff snapshots")
    assert "cancel-in-progress: false" in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "id: early_checkpoint" in workflow
