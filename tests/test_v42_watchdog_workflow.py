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
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "if: steps.refresh.outcome == 'success'" in workflow
    assert "if: always()" in workflow


def test_collector_completion_is_same_repository_main_success_and_no_ai():
    workflow = WORKFLOW.read_text()
    triggers = workflow.split("permissions:", 1)[0]
    assert "workflow_run:" in triggers
    assert "workflows: ['Capture verified pre-match evidence']" in triggers
    assert "types: [completed]" in triggers
    assert "branches: [main]" in triggers.split("workflow_run:", 1)[1]
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "github.event.workflow_run.head_branch == 'main'" in workflow
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in workflow
    assert '= "push" ] ||' in workflow
    assert '= "workflow_run" ]; then' in workflow
    assert 'python v42_research_bot.py --no-ai' in workflow
    assert 'group: v42-research-only' in workflow
    assert 'cancel-in-progress: false' in workflow
