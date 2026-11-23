"""Protect the single-run V28 producer to V35/V36 prospective handoff.

This is an engineering contract test. It neither fabricates fixture evidence
nor suggests that a challenger has demonstrated better prediction accuracy.
"""
from pathlib import Path

FLOW = Path(".github/workflows/continuous-five-double-bot.yml")


def test_upstream_original_v28_is_issued_before_future_only_challengers():
    workflow = FLOW.read_text(encoding="utf-8")
    commands = (
        "run: python v28_independent_pair_trial.py",
        "run: python v34_score_error_research.py",
        "run: python v35_continuous_double_trials.py",
        "run: python v36_prospective_double_trials.py",
        "python continuous_accuracy_bot.py",
    )
    assert all(command in workflow for command in commands)
    assert [workflow.index(command) for command in commands] == sorted(
        workflow.index(command) for command in commands)
    assert "run: python v28_independent_pair_trial.py --verify-only" not in workflow


def test_new_v28_receipts_survive_commit_and_artifact_failure():
    workflow = FLOW.read_text(encoding="utf-8")
    assert "git add data/autobot/ data/v28_independent_pair_trial/" in workflow
    assert "data/v28_independent_pair_trial/decisions/*.json" in workflow
    assert "data/v28_independent_pair_trial/status.json" in workflow
    assert "assert not o['forecast_production_model_modified']" in workflow
    assert "assert not o['verified_comparison']['excluded_invalid_or_tampered_files']" in workflow
    assert "cancel-in-progress: false" in workflow
