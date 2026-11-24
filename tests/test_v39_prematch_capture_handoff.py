from pathlib import Path


def test_v39_handoff_and_archival():
    capture = Path(".github/workflows/prematch-evidence.yml").read_text()
    bot = Path(".github/workflows/continuous-five-double-bot.yml").read_text()
    assert capture.index("run: python v28_independent_pair_trial.py") < capture.index(
        "run: python v39_prospective_double_trials.py") < capture.index(
        "- name: Commit permanent research history")
    assert "git add data/v39_prospective_challengers/sets/" in capture
    assert "git add data/v39_prospective_challengers/status.json" not in capture
    assert "data/v39_prospective_challengers/" in capture.split("- name: Archive immutable snapshots",1)[1]
    assert "run: python v39_prospective_double_trials.py" in bot
    assert "tests/test_v39_prospective_double_trials.py" in bot
    assert "data/v39_prospective_challengers/" in bot
