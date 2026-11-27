import json
from datetime import datetime, timezone
import numpy as np

import v44_authenticated_replay as replay


def test_empty_archive_never_reports_accuracy(tmp_path):
    report = replay.evaluate(pair_root=tmp_path, source_root=tmp_path,
                             prematch_root=tmp_path)
    assert report["eligible_comparison_pairs"] == 0
    assert report["comparisons"]["learned"]["hit_rate"] is None
    assert report["comparisons"]["v43_replay"]["hit_rate"] is None
    assert not report["automatic_promotion"]


def test_invalid_pair_counted_not_silently_used(tmp_path, monkeypatch):
    folder = tmp_path / "sets"
    folder.mkdir()
    (folder / "2025-01-01.json").write_text("{}")
    def invalid(*args, **kwargs):
        raise ValueError("invalid_digest")
    monkeypatch.setattr(replay, "_validated", invalid)
    report = replay.evaluate(pair_root=tmp_path, source_root=tmp_path,
                             prematch_root=tmp_path)
    assert report["source_rejections"] == {"invalid_digest": 1}
    assert report["verified_common_pairs"] == 0


def test_pending_verified_final_does_not_train(tmp_path, monkeypatch):
    folder = tmp_path / "sets"
    folder.mkdir()
    day = "2025-01-01"
    (folder / (day+".json")).write_text(json.dumps(
        {"sydney_date": day, "predicted_at_utc": day+"T01:00:00Z"}))
    models = [{"provider":"p", "fixture_id":str(i), "home_team":f"h{i}",
               "away_team":f"a{i}", "kickoff_utc":day+"T03:00:00Z"}
              for i in (1, 2)]
    monkeypatch.setattr(replay, "_validated", lambda *a, **k: (models, [], 0.0))
    report = replay.evaluate(pair_root=tmp_path, source_root=tmp_path,
                             prematch_root=tmp_path,
                             now=datetime(2025,1,2,tzinfo=timezone.utc))
    assert report["pending_pairs"] == 1
    assert report["verified_common_pairs"] == 0
    assert report["comparisons"]["v8"]["pairs"] == 0
