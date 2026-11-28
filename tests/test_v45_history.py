"""Regressions for source-qualified history recovery and archive integrity."""

from datetime import timedelta
import json
import sqlite3

import pytest

from history_pipeline import build_evidence
from test_v44_early_history import FIXTURE, NOW, evidence
import v44_early_history as early
import v45_history as history


def freeze(root, fixture, rows, when=NOW):
    packet = build_evidence(fixture, rows, when - timedelta(minutes=1))
    assert early.freeze(fixture, packet, root=root, clock=lambda: when) == "written"
    return packet


def test_recover_deduplicates_and_quarantines_conflicting_provider_history(tmp_path):
    archive = tmp_path / "v44_early_history"
    rows = evidence()["history"]
    freeze(archive, FIXTURE, rows)
    freeze(archive, {**FIXTURE, "id": 1000}, rows, NOW + timedelta(minutes=1))
    report = history.evaluate(root=tmp_path, now=NOW + timedelta(minutes=1))
    assert report["history_receipts"] == 2
    assert report["recovered_historical_fixtures"] == 1
    conflict = {**rows[0], "score": {"fullTime": {"home": 3, "away": 1}}}
    freeze(archive, {**FIXTURE, "id": 1001}, [conflict])
    report = history.evaluate(root=tmp_path, now=NOW + timedelta(minutes=1))
    assert report["recovered_historical_fixtures"] == 0
    assert report["historical_rejections"]["conflicting_historical_fixture"] == 1


def test_recover_database_authenticates_snapshot_before_filtering(tmp_path):
    freeze(tmp_path / "v44_early_history", FIXTURE, evidence()["history"])
    db = tmp_path / "history.sqlite3"
    report = history.recover_database(root=tmp_path, now=NOW, database=db)
    digest = report["snapshot_sha256"]
    assert report["database_state"] == "stored"
    before = history.recover_database(root=tmp_path, now=NOW + timedelta(minutes=1), database=db)
    assert before["database_state"] == "original_preserved"
    assert before["snapshot_sha256"] == digest
    rows = history.load_database_snapshot(digest, database=db, season=2024, competition="39", fixture_id=10)
    assert len(rows) == 1
    assert rows[0]["first_observed_at_utc"] == NOW.isoformat()
    assert rows[0]["saved_final_agrees"] is False
    assert history.load_database_snapshot(digest, database=db, provider="football-data.org") == []
    with sqlite3.connect(db) as conn:
        payload = json.loads(conn.execute(f"SELECT payload_json FROM {history.TABLE}").fetchone()[0])
        payload["rows"][0]["home_goals"] = 9
        conn.execute(f"UPDATE {history.TABLE} SET payload_json=?", (json.dumps(payload),))
    with pytest.raises(ValueError, match="integrity_check_failed"):
        history.load_database_snapshot(digest, database=db, season=2025)


def test_later_recovery_cannot_supply_an_original_cutoff_receipt(tmp_path, monkeypatch):
    rows = evidence()["history"]
    original = freeze(tmp_path / "v44_early_history", FIXTURE, rows)
    recovery_fixture = {**FIXTURE, "id": 1000}
    freeze(tmp_path / "v44_recovery_sources" / "history", recovery_fixture, rows,
           NOW + timedelta(minutes=5))
    pair_root = tmp_path / "professional_double_shadow" / "sets"
    pair_root.mkdir(parents=True)
    (pair_root / "2025-02-01.json").touch()
    model = {"provider": "API-Football", "fixture_id": "999", "home_team_id": "1", "away_team_id": "2",
             "kickoff_utc": FIXTURE["utcDate"], "input_digest": original["input_digest"],
             "evidence_as_of_utc": original["as_of_utc"], "captured_at_utc": NOW.isoformat()}
    pair = {"sydney_date": "2025-02-01", "predicted_at_utc": (NOW + timedelta(minutes=1)).isoformat()}
    monkeypatch.setattr(history, "_pair", lambda *a, **kw: (pair, [model, {**model, "fixture_id": "1000"}]))
    report = history.evaluate(root=tmp_path, now=NOW + timedelta(minutes=10))
    assert report["original_pair_legs"]["exact_receipt_available"] == 1
    assert report["original_pair_legs"]["later_recovery_only"] == 1
    assert report["original_pair_legs"]["not_due"] == 2
