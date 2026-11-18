"""Source provenance, played-result filtering and date-only chronology."""
from copy import deepcopy
import hashlib
import json

import pytest

from target70_data import collect, load_normalized, normalize_file


REVISION = "a"*40


def source(*rows):
    return {"name": "Test competition", "matches": list(rows)}


def played(**changes):
    return {"date": "2024-08-01", "time": "20:00", "team1": "Home FC",
            "team2": "Away FC", "score": {"ft": [2, 1]}, **changes}


class Response:
    def __init__(self, payload):
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        pass


def test_normalization_preserves_date_without_manufacturing_utc_or_sydney():
    rows, info = normalize_file(source(played()), season="2024-25", competition="en.1")
    assert len(rows) == 1
    assert rows[0]["source_date"] == "2024-08-01"
    assert rows[0]["time_precision"] == "source_calendar_date_only"
    assert (rows[0]["home_goals"], rows[0]["away_goals"]) == (2, 1)
    assert not {"utcDate", "kickoff_utc", "kickoff", "sydney_date", "xg", "odds"} & rows[0].keys()
    assert info["normalized_played_rows"] == 1


@pytest.mark.parametrize("changes,reason", [
    ({"score": {}}, "missing_full_time_score"),
    ({"score": {"ft": [None, None]}}, "invalid_full_time_goals"),
    ({"score": {"ft": [True, 1]}}, "invalid_full_time_goals"),
    ({"score": {"ft": [-1, 1]}}, "invalid_full_time_goals"),
    ({"score": {"ft": [2.0, 1]}}, "invalid_full_time_goals"),
    ({"date": "2024-02-30"}, "invalid_source_date"),
    ({"date": "2024-08-01T20:00:00Z"}, "invalid_source_date"),
    ({"team2": "  HOME  FC  "}, "invalid_team_identity"),
    ({"team1": None}, "invalid_team_identity"),
])
def test_invalid_or_unplayed_rows_are_excluded(changes, reason):
    rows, info = normalize_file(source(played(**changes)), season="2024-25", competition="en.1")
    assert rows == []
    assert info["rejected_rows"] == {reason: 1}


def test_duplicate_result_never_adds_a_second_observation():
    first = played()
    second = played(team1="  Home   FC ")
    rows, info = normalize_file(source(first, second), season="2024-25", competition="en.1")
    assert len(rows) == 1
    assert info["rejected_rows"] == {"duplicate_match_row": 1}


def test_conflicting_result_excludes_entire_identity_in_either_order():
    first, second = played(), played(score={"ft": [3, 1]})
    for ordered in ((first, second), (second, first)):
        rows, info = normalize_file(source(*ordered), season="2024-25", competition="en.1")
        assert not rows
        assert info["conflicting_fixture_identities_excluded"] == 1


def test_pinned_collect_and_load_verify_sources_and_keep_raw_files_outside_manifest(tmp_path):
    calls = []
    def get(url, timeout):
        calls.append(url)
        return Response(source(played()))
    manifest_path = tmp_path/"manifest.json"
    cache = tmp_path/"external-cache"
    rows, manifest = collect(cache_dir=cache, manifest_path=manifest_path, revision=REVISION,
                              seasons=("2024-25",), leagues=("en.1",), get=get)
    assert len(calls) == 1
    assert f"/{REVISION}/2024-25/en.1.json" in calls[0]
    assert manifest["license"] == "CC0-1.0"
    assert manifest["collection_policy"]["maximum_concurrent_requests"] == 2
    assert manifest["collection_policy"]["paid_api_calls"] == 0
    assert manifest["utc_kickoff_or_sydney_day_inferred"] is False
    assert not manifest["contains_bookmaker_odds"]
    assert not manifest["contains_shot_xg"]
    assert manifest["normalized_played_matches"] == 1
    assert "home_team" not in manifest_path.read_text()
    assert rows == load_normalized(manifest_path, cache)
    entry = manifest["source_files"][0]
    raw = cache/entry["cache_relative_path"]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == entry["sha256"]
    def unavailable(*args, **kwargs):
        raise AssertionError("An unchanged pinned source must use its verified cache")
    repeated, _ = collect(cache_dir=cache, manifest_path=manifest_path, revision=REVISION,
                           seasons=("2024-25",), leagues=("en.1",), get=unavailable)
    assert repeated == rows
    raw.write_text(json.dumps(source(played(score={"ft": [0, 0]}))))
    with pytest.raises(ValueError, match="integrity_mismatch"):
        load_normalized(manifest_path, cache)
    with pytest.raises(ValueError, match="integrity_mismatch"):
        collect(cache_dir=cache, manifest_path=manifest_path, revision=REVISION,
                seasons=("2024-25",), leagues=("en.1",), get=unavailable)


@pytest.mark.parametrize("settings", [
    {"revision": "master"}, {"seasons": ("2030-31",)},
    {"leagues": ("unknown",)}, {"seasons": ()},
])
def test_unknown_sources_or_unpinned_revision_cannot_trigger_download(tmp_path, settings):
    def prohibited(*args, **kwargs):
        raise AssertionError("Invalid collection settings must not make HTTP requests")
    with pytest.raises(ValueError):
        collect(cache_dir=tmp_path, manifest_path=tmp_path/"m.json", get=prohibited,
                **({"revision": REVISION} | settings))


def test_same_fixture_in_two_source_files_cannot_enter_twice(tmp_path):
    def get(url, timeout):
        return Response(source(played()))
    with pytest.raises(ValueError, match="cross_file_duplicate"):
        collect(cache_dir=tmp_path/"cache", manifest_path=tmp_path/"m.json", revision=REVISION,
                seasons=("2023-24", "2024-25"), leagues=("en.1",), get=get)


def test_manifest_path_and_row_count_are_checked(tmp_path):
    manifest_path, cache = tmp_path/"m.json", tmp_path/"cache"
    _, manifest = collect(cache_dir=cache, manifest_path=manifest_path, revision=REVISION,
                           seasons=("2024-25",), leagues=("en.1",),
                           get=lambda url, timeout: Response(source(played())))
    changed = deepcopy(manifest)
    changed["source_files"][0]["cache_relative_path"] = "../outside.json"
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="source_path_mismatch"):
        load_normalized(manifest_path, cache)
    manifest["normalized_played_matches"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="count_mismatch"):
        load_normalized(manifest_path, cache)
