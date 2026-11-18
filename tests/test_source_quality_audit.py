from copy import deepcopy
from datetime import date
import hashlib
import json

import pytest

from source_quality_audit import audit, inspect_payload, quarantine, season_window


def match(day, home="AFC Bournemouth", away="Newcastle United", goals=None):
    return {"date": day, "team1": home, "team2": away,
            "score": {"ft": [1, 4] if goals is None else goals}}


def test_wrong_year_is_flagged_without_silently_repairing_source():
    payload = {"matches": [match("2019-07-01")]}
    original = deepcopy(payload)
    report = inspect_payload(payload, season="2019-20", competition="en.1")
    assert report["quarantine_entire_source_file"]
    assert report["flagged_played_rows"] == 1
    assert report["primary_verified_date_mismatches"][0]["verified_date"] == "2020-07-01"
    assert report["flagged_rows"][0]["source_date"] == "2019-07-01"
    assert payload == original


def test_actual_late_covid_finish_is_permitted_not_assumed_corrupt():
    payload = {"matches": [match("2020-07-01"), match("2020-08-02", "A", "B")]}
    report = inspect_payload(payload, season="2019-20", competition="en.1")
    assert not report["quarantine_entire_source_file"]
    assert report["passing_this_screen_does_not_verify_all_source_dates"]
    assert season_window("2019-20")[1] == date(2020, 8, 31)


def test_missing_results_do_not_become_played_zero_zero_or_date_certification():
    report = inspect_payload({"matches": [match("2019-07-01", goals=[])]},
                             season="2019-20", competition="en.1")
    assert report["normalized_played_matches"] == 0
    assert report["flagged_played_rows"] == 0
    assert report["normalization_diagnostics"]["rejected_rows"]["missing_full_time_score"] == 1


def test_quarantine_removes_the_whole_source_file_only():
    rows = [{"season": "2019-20", "competition": "en.1", "source_date": "2019-07-01"},
            {"season": "2019-20", "competition": "en.1", "source_date": "2019-08-10"},
            {"season": "2024-25", "competition": "en.1", "source_date": "2024-08-16"},
            {"season": "2019-20", "competition": "de.1", "source_date": "2019-08-16"}]
    original = deepcopy(rows)
    result = quarantine(rows, {"quarantined_files": [{"season": "2019-20", "competition": "en.1"}]})
    assert result == rows[2:]
    assert rows == original


@pytest.mark.parametrize("season", ["2019", "2019-21", "../2019-20", None])
def test_ambiguous_season_labels_are_rejected(season):
    with pytest.raises(ValueError):
        season_window(season)


def test_hash_mismatch_stops_quality_audit(tmp_path):
    revision = "a"*40
    relative = f"{revision}/2019-20/en.1.json"
    raw = tmp_path/relative
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"matches": [match("2019-07-01")]}))
    manifest = {"source_revision": revision, "source_files": [{
        "season": "2019-20", "competition": "en.1", "cache_relative_path": relative,
        "sha256": "0"*64, "source_url": "https://example.test/pinned"}]}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source_hash"):
        audit(path, tmp_path)
    manifest["source_files"][0]["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    report = audit(path, tmp_path)
    assert report["quarantined_file_played_matches"] == 1
    assert report["date_repair_performed"] is False
    assert report["original_experiment_modified"] is False
