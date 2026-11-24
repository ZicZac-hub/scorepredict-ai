"""Keep V38 capture in the original V28 issuance window, without hindsight."""
from pathlib import Path


def test_v38_uses_same_capture_job_and_archives_only_immutable_sets():
    workflow = Path(".github/workflows/prematch-evidence.yml").read_text()
    assert workflow.index("run: python v28_independent_pair_trial.py") < workflow.index(
        "run: python v38_prospective_double_trials.py") < workflow.index(
        "- name: Commit permanent research history")
    assert workflow.count("run: python v38_prospective_double_trials.py") == 1
    assert "if: always()\n        run: python v38_prospective_double_trials.py" in workflow
    assert "git add data/v38_prospective_challengers/sets/" in workflow
    assert "git add data/v38_prospective_challengers/status.json" not in workflow
    assert "data/v38_prospective_challengers/" in workflow.split(
        "- name: Archive immutable snapshots",1)[1]


def test_v38_remains_forward_only():
    source = Path("v38_prospective_double_trials.py").read_text()
    assert "verify_decision(file, root=root, clock=clock)" in source
    assert "_frozen_eligible(original, clock)" in source
    assert "_write_once(dest, packet)" in source
    assert "requests.get" not in source and "requests.post" not in source
