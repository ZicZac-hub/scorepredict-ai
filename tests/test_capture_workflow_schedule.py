"""Safeguards: daily quota is for matches, not repeated code pushes."""
from pathlib import Path


def test_result_verification_precedes_new_provider_discovery():
    workflow=Path(".github/workflows/prematch-evidence.yml").read_text()
    first=workflow.index("Independently verify recently completed exact-score")
    capture=workflow.index("Capture verified pre-kickoff snapshots")
    assert first < capture
    assert "V10_RESULT_LIMIT:" in workflow
    assert "github.event_name == 'schedule'" in workflow


def test_scheduled_capture_not_triggered_by_every_source_code_push():
    workflow=Path(".github/workflows/prematch-evidence.yml").read_text()
    triggers=workflow.split("\npermissions:",1)[0]
    assert "workflow_dispatch:" in triggers
    assert "schedule:" in triggers
    assert "\n  push:" not in triggers
    assert "cron: '2,17,32,47 * * * *'" in triggers


def test_matched_study_uses_collected_sources_before_permanent_publication():
    workflow=Path(".github/workflows/prematch-evidence.yml").read_text()
    sources=workflow.index("run: python v21_capture_sources.py")
    study=workflow.index("run: python professional_paired_study.py")
    publish=workflow.index("- name: Commit permanent research history")
    assert sources < study < publish
    assert "data/professional_paired_study/" in workflow[publish:]
    assert "if: github.ref == 'refs/heads/main'" in workflow
    # Refreshes reuse the owner's existing bounded collector and secrets.
    study_step=workflow[workflow.rfind("- name:",0,study):study]
    assert "API_FOOTBALL_KEY" not in study_step
