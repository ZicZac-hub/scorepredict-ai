"""Guard against silently rewriting older prospective ablation protocols."""
from datetime import datetime, timezone
import json
from pathlib import Path

import professional_paired_study as ablation


def test_v1_v2_archives_remain_untouched_and_v3_registers_its_own_protocol(tmp_path):
    legacy = Path("data/professional_paired_study/protocol.json")
    original = legacy.read_bytes()
    assert json.loads(original)["protocol"]["version"] == "professional-fixed-pair-ablation-v1"
    legacy_v2 = Path("data/professional_paired_study_v2/protocol.json")
    original_v2 = legacy_v2.read_bytes()
    assert json.loads(original_v2)["protocol"]["version"] == "professional-fixed-pair-ablation-v2"
    assert ablation.VERSION == "professional-fixed-pair-ablation-v3"
    assert str(ablation.ROOT) == "data/professional_paired_study_v3"
    root = tmp_path / "new-protocol"
    at = datetime(2035, 1, 1, tzinfo=timezone.utc)
    registered = ablation.register_protocol(root=root, now=at)
    assert registered["protocol"]["version"] == ablation.VERSION
    assert registered["implementation_sha256"]["professional_paired_study.py"]
    assert ablation.register_protocol(root=root, now=at) == registered
    assert legacy.read_bytes() == original
    assert legacy_v2.read_bytes() == original_v2


def test_scheduled_audit_runs_after_capture_and_stages_new_archive():
    workflow = Path(".github/workflows/prematch-evidence.yml").read_text()
    assert workflow.index("run: python v21_capture_sources.py") < workflow.index(
        "run: python verified_xg_evidence_pipeline.py --output data/verified_xg_replay/latest_report.json")
    assert "if: always() && (github.event_name == 'schedule' || github.event_name == 'workflow_dispatch')" in workflow
    assert "data/professional_paired_study_v2/" in workflow
    assert "data/professional_paired_study_v3/" in workflow
    assert "data/verified_xg_replay/" in workflow
    assert "data/professional_paired_study/" in workflow
    assert "automatic_model_promotion" in json.loads(
        Path("data/professional_paired_study/protocol.json").read_text())["protocol"]
