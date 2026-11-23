"""Ensure authenticated V37 research reaches durable grading on every bot run.

The main pre-match collector persists original write-once forecasts; the
15-minute bot regrades them using independently verified finals, never
inventing new tickets or changing V28/V35/V36 method selection.
"""
from pathlib import Path

FLOW = Path(".github/workflows/continuous-five-double-bot.yml")
CAPTURE_FLOW = Path(".github/workflows/prematch-evidence.yml")


def step(flow, name, next_name):
    return flow.split("      - name: "+name,1)[1].split(
        "      - name: "+next_name,1)[0]


def test_v37_evaluation_precedes_bot_report_on_every_quarter_hour():
    flow=FLOW.read_text(encoding="utf-8")
    assert "9,24,39,54 * * * *" in flow
    assert flow.index("run: python v37_sparse_evaluation.py") > (
        flow.index("run: python v36_prospective_double_trials.py"))
    assert flow.index("run: python v37_sparse_evaluation.py") < (
        flow.index("run: python continuous_accuracy_bot.py"))
    assert "tests/test_v37_sparse_history_research.py" in flow
    assert "tests/test_v37_quarter_hour_verification.py" in flow
    assert "- 'v37_sparse_evaluation.py'" in flow


def test_v37_report_is_committed_without_writing_new_forecast_records():
    flow=FLOW.read_text(encoding="utf-8")
    assess=step(flow,
        "Independently grade saved V37 sparse UEFA forecasts without new provider calls",
        "Generate watchdog report and optional bounded OpenAI research memo")
    assert "run: python v37_sparse_evaluation.py" in assess
    assert "capture_prematch" not in assess
    assert "API_FOOTBALL_KEY" not in assess
    assert "OPENAI_API_KEY" not in assess
    assert "git add data/autobot/" in flow
    assert "data/v37_sparse_prospective/status.json" in flow
    assert "assert z['not_a_five_double_trial']" in flow
    assert "assert z['no_live_model_change']" in flow
    assert "assert not z['ready_for_model_promotion']" in flow
    assert "assert not z['invalid_or_tampered_excluded']" in flow
    assert "z['frozen_unique_physical_fixtures'] == (" in flow
    assert "data/v37_sparse_prospective/status.json" in step(
        flow,"Preserve evidence even when status publication fails","")


def test_collector_still_owns_only_original_v37_ticket_issuance():
    capture=CAPTURE_FLOW.read_text(encoding="utf-8")
    bot=FLOW.read_text(encoding="utf-8")
    assert "run: python capture_prematch.py" in capture
    assert "run: python v37_sparse_evaluation.py" in capture
    assert "run: python capture_prematch.py" not in bot
    assert "data/v10_prospective/forecasts/" not in bot
    assert "v37_sparse_research.py" not in bot
    assert "run: python v28_independent_pair_trial.py" in bot
    assert "run: python v35_continuous_double_trials.py" in bot
    assert "run: python v36_prospective_double_trials.py" in bot


def test_source_provenance_and_pending_not_loss_remain_enforced():
    code=Path("v37_sparse_evaluation.py").read_text(encoding="utf-8")
    assert "index_verified_results(root,prematch,clock)" in code
    assert 'row["v37_arm_digest_sha256"] != frozen_arm_digest(row)' in code
    assert "if final is None or not _qualified(row,final)" in code
    assert "pending += 1" in code
    assert '"ready_for_model_promotion":False' in code
    assert '"not_a_five_double_trial":True' in code
    assert "requests." not in code
