"""Finished-match status should not report frozen pending forecasts as losses."""
import json
from finished_result_explainer import research_archive_summary


def test_pending_127_fixtures_do_not_become_zero_accuracy(tmp_path):
    (tmp_path/"performance.json").write_text(json.dumps({
        "models":{"v8":{"frozen_fixtures":127,"verified_fixtures":0,
                        "exact_score_rate":None}},
        "five_doubles":{"frozen_pairs":6,"verified_pairs":0,
                        "awaiting_both_results":6,"hit_rate":None}
    }))
    result=research_archive_summary(tmp_path)
    assert result=={"frozen":127,"verified":0,"pending":127,
                    "doubles_frozen":6,"doubles_verified":0,
                    "doubles_pending":6,"status":"awaiting_finals"}


def test_verified_results_come_only_from_actual_saved_counts(tmp_path):
    (tmp_path/"performance.json").write_text(json.dumps({
        "models":{"v8":{"frozen_fixtures":12,"verified_fixtures":2}},
        "five_doubles":{"frozen_pairs":4,"verified_pairs":1}
    }))
    result=research_archive_summary(tmp_path)
    assert result["frozen"]==12
    assert result["verified"]==2
    assert result["pending"]==10
    assert result["doubles_pending"]==3
    assert result["status"]=="partially_settled"


def test_missing_or_corrupt_report_is_never_a_verified_win(tmp_path):
    assert research_archive_summary(tmp_path)["verified"]==0
    (tmp_path/"performance.json").write_text("{not json")
    assert research_archive_summary(tmp_path)["frozen"]==0
    (tmp_path/"settlement_status.json").write_text(
        '{"eligible_frozen_fixtures":127}')
    assert research_archive_summary(tmp_path)["frozen"]==127
    assert research_archive_summary(tmp_path)["verified"]==0
