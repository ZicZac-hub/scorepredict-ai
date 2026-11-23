"""The collecting workflow must freeze V36 while new authenticated V28
decisions are still local and within their pre-kickoff issuance window.

No paid provider calls, history rewrites, or model promotions are introduced.
"""
from pathlib import Path


def test_prematch_freezes_v36_in_same_job_as_v28_and_v35():
    yml = Path(".github/workflows/prematch-evidence.yml").read_text(encoding="utf-8")
    v28 = yml.index("run: python v28_independent_pair_trial.py")
    v35 = yml.index("run: python v35_continuous_double_trials.py")
    v36 = yml.index("run: python v36_prospective_double_trials.py")
    assert v28 < v35 < v36
    assert "if: always()\n        run: python v36_prospective_double_trials.py" in yml
    assert yml.count("run: python v36_prospective_double_trials.py") == 1


def test_new_v36_packets_are_published_and_status_owned_by_bot():
    yml = Path(".github/workflows/prematch-evidence.yml").read_text(encoding="utf-8")
    assert "git add data/v36_prospective_challengers/sets/" in yml
    assert "data/v36_prospective_challengers/ data/v34_score_error_research/" in yml
    # A separate bot owns changing V36 status; the collector must stage
    # ONLY newly frozen immutable packets and not its local status report.
    assert "git add data/v36_prospective_challengers/status.json" not in yml
    assert "data/v36_prospective_challengers/" in yml.split(
        "      - name: Archive immutable snapshots", 1)[1]
    assert "github.ref == 'refs/heads/main'" in yml


def test_v36_source_only_no_network_and_no_retrofit():
    src = Path("v36_prospective_double_trials.py").read_text(encoding="utf-8")
    assert "verify_decision(file, root=root, clock=clock)" in src
    assert "_frozen_eligible(original, clock)" in src
    assert "_write_once(dest, packet)" in src
    assert "read_latest" not in src
    assert "requests.post" not in src and "requests.get" not in src
    assert "research_only" in src
