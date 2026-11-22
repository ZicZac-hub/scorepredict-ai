"""Guardrail: result-backfill must not race the dedicated bot's report writes."""
from pathlib import Path


def test_backfill_keeps_newly_verified_scores_but_not_bot_owned_status_files():
    text = Path(".github/workflows/backfill-verified-results.yml").read_text(encoding="utf-8")
    assert "run: python v34_score_error_research.py" in text
    assert "run: python v35_continuous_double_trials.py --verify-only" in text
    stage_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("git add ")]
    assert len(stage_lines) == 1
    stage = stage_lines[0]
    assert "data/prematch/verified_results.json" in stage
    assert "data/v30_full_slate_ranking/" in stage
    assert "data/v34_score_error_research/" not in stage
    assert "data/v35_continuous_challengers/" not in stage
