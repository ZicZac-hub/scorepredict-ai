"""The scheduled legacy selector may only publish new decisions before kickoff."""

from double_revisions import load_saved_pairs
from persist_prematch import write_json
from prospective_smart_double import _publish_once, evaluate, freeze


BEFORE = "2026-11-14T12:00:00+00:00"
AFTER = "2026-11-15T05:00:00+00:00"


def _forecast(fixture_id, kickoff, home_id):
    return {
        "fixture_id": fixture_id, "provider": "API-Football", "score_basis": "regulation",
        "home_team": f"Home {home_id}", "away_team": f"Away {home_id + 1}",
        "home_team_id": home_id, "away_team_id": home_id + 1,
        "kickoff_utc": kickoff, "captured_at_utc": "2026-11-14T11:00:00+00:00",
        "evidence_as_of_utc": "2026-11-14T10:59:00+00:00",
        "input_digest": "source-digest",
        "double_top10_scores": [[1, 0, .2], [1, 1, .17], [0, 1, .14], [0, 0, .1], [2, 1, .08]],
    }


def _final(forecast):
    identity = {key: forecast[key] for key in
                ("fixture_id", "provider", "kickoff_utc", "home_team_id", "away_team_id", "score_basis")}
    return {**identity, "verified_at_utc": AFTER, "status": "FT",
            "evidence_version": 2, "home_goals": 1, "away_goals": 0}


def test_freeze_requires_live_decision_and_groups_by_sydney_date(tmp_path):
    snapshots, out = tmp_path / "snapshots", tmp_path / "pairs"
    first = _forecast(100, "2026-11-14T13:00:00+00:00", 11)
    second = _forecast(200, "2026-11-15T02:00:00+00:00", 21)
    path = snapshots / "prematch_source.json"
    write_json(path, {"records": [first, second]})
    assert freeze(now=AFTER, out=out, snapshots=snapshots) == 0
    assert freeze(now="2026-11-14T12:55:00+00:00", out=out, snapshots=snapshots) == 0
    assert freeze(now=BEFORE, out=out, snapshots=snapshots) == 1
    frozen = out / "sydney-2026-11-15.json"
    original = frozen.read_bytes()
    loaded = load_saved_pairs(pair_dir=tmp_path / "no_pairs", smart_dir=out)
    assert len(loaded) == 1
    assert loaded[0]["captured_at_utc"] == BEFORE
    assert loaded[0]["first_provider"] == loaded[0]["second_provider"] == "API-Football"
    assert freeze(now="2026-11-14T12:30:00+00:00", out=out, snapshots=snapshots) == 0
    assert frozen.read_bytes() == original


def test_legacy_sets_stay_out_of_verified_denominator_and_wrong_finals_stay_pending(tmp_path):
    snapshots, out, prematch = (tmp_path / name for name in ("snapshots", "pairs", "prematch"))
    first = _forecast(100, "2026-11-14T13:00:00+00:00", 11)
    second = _forecast(200, "2026-11-15T02:00:00+00:00", 21)
    write_json(snapshots / "prematch_source.json", {"records": [first, second]})
    legacy = out / "2026-11-14.json"
    write_json(legacy, {"captured_at_utc": BEFORE, "first_fixture_id": "100", "second_fixture_id": "200"})
    assert freeze(now=BEFORE, out=out, snapshots=snapshots) == 1

    results_path = prematch / "verified_results.json"
    write_json(results_path, {"100": _final(first), "200": {**_final(second), "home_team_id": 999}})
    pending = evaluate(now=AFTER, out=out, root=tmp_path / "v10", prematch=prematch)
    assert pending["pending_days"] == 1
    assert pending["verified_days"] == 0
    assert pending["win_rate"] is None

    write_json(results_path, {"100": _final(first), "200": _final(second)})
    verified = evaluate(now=AFTER, out=out, root=tmp_path / "v10", prematch=prematch)
    assert verified["pending_days"] == 0
    assert verified["verified_days"] == verified["wins"] == 1
    assert verified["legacy_unverified_publication"]["archive_dates"] == [legacy.stem]



def test_publish_race_keeps_first_complete_packet(tmp_path):
    path = tmp_path / "sydney-2026-11-15.json"
    assert _publish_once(path, {"first": "winner"})
    original = path.read_bytes()
    assert not _publish_once(path, {"first": "later competitor"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
