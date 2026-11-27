from datetime import datetime, timezone
import json

import numpy as np

from professional_double_optimizer import optimal_five
import v43_replay as replay


def test_replay_scores_common_settled_pairs_and_reports_other_rows(tmp_path, monkeypatch):
    now = datetime(2026, 11, 27, tzinfo=timezone.utc)
    grid = np.full((11, 11), .1 / 120)
    grid[1, 0] = .9
    grids = [grid.tolist(), grid.tolist()]
    choices = optimal_five(*grids)
    original = {"status": "issued", "full_score_grids": grids, "five_doubles": choices,
                "estimated_coverage": sum(choice["joint_probability"] for choice in choices)}
    days = ("2026-11-21", "2026-11-22", "2026-11-23", "2026-11-24", "2026-11-25")
    paths = [tmp_path / (day + ".json") for day in days]
    for path in paths:
        path.write_text(json.dumps({"sydney_date": path.stem}), encoding="utf-8")
    paths[-1].write_text(json.dumps({"sydney_date": days[0]}), encoding="utf-8")

    def load_pair(record, **kwargs):
        day = record["sydney_date"]
        # The third record reuses the first physical pair. The fourth abstains.
        number = {days[0]: 1, days[1]: 3, days[2]: 1, days[3]: 5}[day]
        models = [{"provider": "API-Football", "fixture_id": str(i),
                   "kickoff_utc": day + "T18:00:00Z", "home_team": f"Home {i}",
                   "away_team": f"Away {i}", "score_matrix": grid.tolist()}
                  for i in (number, number + 1)]
        arm = {"status": "abstained"} if number == 5 else original.copy()
        return models, [{}, {}], {"v8_original": arm}

    monkeypatch.setattr(replay, "_final", lambda model, result, now: result)
    report = replay._cohort(paths, load_pair, names=("v8_original", "v43_replay"),
                            finals={('API-Football', '1'): (1, 0), ('API-Football', '2'): (1, 0)},
                            context={"now": now})
    assert report["archived_pairs"] == 5
    assert report["verified_pairs"] == report["pending_pairs"] == report["abstained_pairs"] == 1
    assert report["invalid_pairs"] == 2
    assert report["source_rejections"] == {
        "duplicate_physical_fixture": 1, "study_filename_day_mismatch": 1}
    assert report["first_verified_sydney_date"] == report["last_verified_sydney_date"] == days[0]
    assert report["methods"]["v8_original"]["hits"] == 1
    assert report["feasibility"]["pairs_with_grids"] == 2
    assert report["pair_details"][0]["result"] == "hit"
    assert report["pair_details"][0]["fixtures"][1]["actual_score"] == "1-0"
    assert report["pair_details"][1]["result"] == "pending"
    assert report["pair_details"][1]["fixtures"][0]["status"] == "awaiting_verified_final"
    for method in report["methods"].values():
        assert method["verified_pairs"] == 1
        assert method["score_log_loss"] > 0
        assert method["outcome_brier"] > 0


def test_empty_archive_keeps_metrics_unavailable_and_writes_nothing(tmp_path):
    report = replay.evaluate(early_root=tmp_path, late_root=tmp_path, pair_root=tmp_path,
                             source_root=tmp_path, provider_root=tmp_path, prematch_root=tmp_path)
    assert set(report["cohorts"]) == {"v42_early", "v42_late", "original_pair_archive"}
    assert report["external_requests"] == 0
    assert report["prospective_v43_evidence"] is False
    assert all(cohort["methods"]["v43_replay"]["score_log_loss"] is None
               for cohort in report["cohorts"].values())
    assert list(tmp_path.iterdir()) == []
