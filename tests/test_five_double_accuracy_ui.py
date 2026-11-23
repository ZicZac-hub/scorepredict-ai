"""Show observed set outcomes separately from estimated model coverage."""
from contextlib import nullcontext
import json

import pytest

from five_double_accuracy_ui import render


class Screen:
    def __init__(self):
        self.lines = []
        self.downloads = []

    def expander(self, title):
        self.lines.append(title)
        return nullcontext()

    def caption(self, value):
        self.lines.append(value)

    def write(self, value):
        self.lines.append(value)

    def download_button(self, label, data, *args, **kwargs):
        self.downloads.append(data)

    def dataframe(self, rows, *args, **kwargs):
        self.lines.append(f"research rows: {len(rows)}")


def report():
    return {"schema_version": 1, "evaluated_at_utc": "2035-01-02T01:00:00Z",
            "target": .70, "observed": {"verified_sets": 2, "hits": 1,
                "pending_sets": 6, "hit_rate": .70},
            "diagnosis": {"model_coverage_range": [.06, .08]},
            "sets": [{"sydney_date": "2035-01-01", "status": "miss",
                "verified_result_attribution": {"legs": [
                    {"actual_score": "3-1"}, {"actual_score": "0-0"}],
                    "actual_joint_score_rank_min": 31},
                "full_grid_diagnosis": {
                    "minimum_combinations_for_target_model_coverage": 80}}]}


def show(tmp_path, packet):
    path = tmp_path/"report.json"
    path.write_text(json.dumps(packet))
    screen = Screen()
    render(screen, path)
    return screen


def test_observed_accuracy_uses_completed_sets_and_discloses_small_sample(tmp_path):
    screen = show(tmp_path, report())
    text = "\n".join(screen.lines)
    assert "1 hits / 2 completed; 6 pending" in text
    assert "Observed set accuracy: 1/2 = 50.00%" in text
    assert "Pending sets are excluded" in text
    assert "Small sample: 2 completed sets" in text
    assert "6.00%–8.00%" in text
    assert "Estimated coverage is separate from observed accuracy" in text
    assert "70% exact-score-double accuracy has not been established" in text
    assert "80 combinations to cover 70% estimated mass" in text
    assert "actual 3-1 / 0-0" in text
    assert len(screen.downloads) == 1


def test_pending_only_report_never_shows_zero_or_seventy_percent_accuracy(tmp_path):
    packet = report()
    packet["observed"].update(verified_sets=0, hits=0, pending_sets=8)
    packet["sets"] = []
    screen = show(tmp_path, packet)
    text = "\n".join(screen.lines)
    assert "0 hits / 0 completed; 8 pending" in text
    assert "observed accuracy is unavailable" in text
    assert "Observed set accuracy:" not in text


@pytest.mark.parametrize("packet", [[], None, {"schema_version": 1},
    {"schema_version": 1, "observed": {}},
    {"schema_version": 1, "observed": {"verified_sets": 1, "hits": 2,
                                       "pending_sets": 0}}])
def test_incomplete_or_corrupt_generated_report_does_not_break_prediction_page(tmp_path, packet):
    screen = show(tmp_path, packet)
    assert screen.lines == []
    assert screen.downloads == []


def test_one_malformed_detail_does_not_hide_valid_completed_set_counts(tmp_path):
    packet = report()
    packet["sets"].insert(0, {"verified_result_attribution": {"legs": None}})
    screen = show(tmp_path, packet)
    text = "\n".join(screen.lines)
    assert "Observed set accuracy: 1/2 = 50.00%" in text
    assert "actual 3-1 / 0-0" in text


def test_workflow_preserves_and_archives_the_scheduled_diagnosis():
    from pathlib import Path
    from reconcile_capture_conflicts import REPORT_EXACT
    workflow = Path(".github/workflows/prematch-evidence.yml").read_text()
    assert workflow.index("run: python five_double_accuracy.py") < workflow.index(
        "- name: Commit permanent research history")
    # The early write-once source checkpoint has its own restricted git add.
    # Verify the final accuracy publication step, not the first git add.
    final_stage = workflow.split("- name: Commit permanent research history", 1)[1]
    final_stage = final_stage.split("- name: Archive immutable snapshots", 1)[0]
    assert "data/accuracy_diagnosis/" in final_stage.split("git add", 1)[1].split("\n", 1)[0]
    assert "data/accuracy_diagnosis/" in workflow.split("- name: Archive immutable snapshots", 1)[1]
    assert "data/accuracy_diagnosis/report.json" in REPORT_EXACT

def test_research_table_preserves_pending_sets_and_original_inputs():
    from copy import deepcopy
    from five_double_accuracy_ui import coverage_research_rows

    original = {"sets": [
        {"sydney_date": "2026-11-15", "status": "miss",
         "selected_five_model_coverage": .05923757473658,
         "full_grid_diagnosis": {"optimal_five_model_coverage": .05923757473658,
                                 "minimum_combinations_for_target_model_coverage": 156},
         "source_feature_gaps": [
             {"verified_shot_xg_available": False, "validated_lineup_adjustments_applied": False},
             {"verified_shot_xg_available": False, "validated_lineup_adjustments_applied": False}]},
        {"sydney_date": "2026-11-19", "status": "pending",
         "selected_five_model_coverage": .07105348455,
         "full_grid_diagnosis": {"optimal_five_model_coverage": .07105348455,
                                 "minimum_combinations_for_target_model_coverage": 119},
         "source_feature_gaps": [
             {"verified_shot_xg_available": True, "validated_lineup_adjustments_applied": False},
             {"verified_shot_xg_available": False, "validated_lineup_adjustments_applied": True}]},
    ]}
    preserved = deepcopy(original)
    rows = coverage_research_rows(original)
    assert len(rows) == 2
    assert rows[0]["Selected five model coverage"] == 5.92
    assert rows[1]["Best five model coverage"] == 7.11
    assert rows[1]["Result"] == "pending"
    assert rows[0]["Combos for 70% model coverage"] == 156
    assert rows[1]["Legs missing true xG"] == 1
    assert rows[1]["Legs without validated lineup adjustment"] == 1
    assert original == preserved


def test_research_table_rejects_invalid_and_out_of_range_mass():
    from five_double_accuracy_ui import coverage_research_rows

    assert coverage_research_rows(None) == []
    assert coverage_research_rows({"sets": []}) == []
    invalid = {"sets": [{"sydney_date": "2026-11-19",
               "selected_five_model_coverage": .95,
               "full_grid_diagnosis": {"optimal_five_model_coverage": .07}}]}
    assert coverage_research_rows(invalid) == []
