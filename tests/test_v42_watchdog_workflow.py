from pathlib import Path

WORKFLOW = Path(".github/workflows/v42-research-watchdog.yml")


def test_startup_smoke_is_scoped_and_has_no_paid_ai():
    workflow = WORKFLOW.read_text()
    triggers = workflow.split("permissions:", 1)[0]
    assert "push:" in triggers
    assert "branches: [main]" in triggers
    assert "'.github/workflows/v42-research-watchdog.yml'" in triggers
    assert "'v42_research_bot.py'" in triggers
    assert "'data/" not in triggers
    assert "cron: '*/5 * * * *'" in triggers
    assert 'python v42_research_bot.py --no-ai' in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "if: steps.refresh.outcome == 'success'" in workflow
    assert "if: always()" in workflow
