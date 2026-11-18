"""Fetch and normalize CC0 OpenFootball results for retrospective research.

No bookmaker odds, xG, current provider credentials, live predictions or
invented UTC kickoff instants enter this dataset. Historical rows are used at
date precision: callers must exclude the entire candidate source date from
their training history. Raw files stay in the external workspace cache.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import requests

DEFAULT_CACHE = Path("/workspace/shared/target70-data")
SOURCE_MANIFEST = Path("data/target70_research/source_manifest.json")
SEASONS = tuple(f"{year}-{str(year+1)[-2:]}" for year in range(2019, 2025))
LEAGUES = ("en.1", "de.1", "es.1", "it.1", "fr.1")
MAX_FILES = 30
MAX_SOURCE_BYTES = 4 * 1024 * 1024
PROJECT = "openfootball/football.json"


def _team(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _identity(row):
    return (row["competition"], row["source_date"],
            row["home_team"].casefold(), row["away_team"].casefold())


def normalize_file(payload, *, season, competition):
    """Return one played fixture per identity; conflicting scores exclude it.

    An exact duplicate adds no extra observation. No full-time score means an
    unplayed/unavailable row, never an inferred 0-0 result.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise ValueError("source_matches_list_required")
    unique = {}
    conflicts = set()
    rejected = Counter()
    for raw in payload["matches"]:
        if not isinstance(raw, dict):
            rejected["invalid_match_shape"] += 1
            continue
        home, away = _team(raw.get("team1")), _team(raw.get("team2"))
        if not home or not away or home.casefold() == away.casefold():
            rejected["invalid_team_identity"] += 1
            continue
        day = raw.get("date")
        try:
            if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                raise ValueError("date_precision_required")
            date.fromisoformat(day)
        except ValueError:
            rejected["invalid_source_date"] += 1
            continue
        score = raw.get("score")
        goals = score.get("ft") if isinstance(score, dict) else None
        if not isinstance(goals, list) or len(goals) != 2:
            rejected["missing_full_time_score"] += 1
            continue
        if any(type(g) is not int or not 0 <= g <= 99 for g in goals):
            rejected["invalid_full_time_goals"] += 1
            continue
        row = {"source": "openfootball", "season": season,
               "competition": competition, "source_date": day,
               "home_team": home, "away_team": away,
               "home_goals": goals[0], "away_goals": goals[1],
               "time_precision": "source_calendar_date_only",
               "score_basis": "source_full_time_league_score"}
        key = _identity(row)
        row["source_match_id"] = hashlib.sha256(json.dumps(
            key, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        previous = unique.get(key)
        if previous is not None:
            if (previous["home_goals"], previous["away_goals"]) != tuple(goals):
                conflicts.add(key)
                rejected["conflicting_score_row"] += 1
            else:
                rejected["duplicate_match_row"] += 1
        else:
            unique[key] = row
    rows = [row for key, row in unique.items() if key not in conflicts]
    rows.sort(key=lambda row: (row["source_date"], row["competition"],
                               row["home_team"], row["away_team"]))
    return rows, {"raw_match_rows": len(payload["matches"]),
                  "normalized_played_rows": len(rows),
                  "conflicting_fixture_identities_excluded": len(conflicts),
                  "rejected_rows": dict(rejected)}


def _revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", value):
        raise ValueError("exact_source_commit_sha_required")
    return value.lower()


def _read_response(response):
    response.raise_for_status()
    content = response.content
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError("source_file_too_large")
    json.loads(content)
    return content


def resolve_revision(get=requests.get):
    response = get(f"https://api.github.com/repos/{PROJECT}/commits/master", timeout=30)
    response.raise_for_status()
    return _revision(response.json()["sha"])


def _fetch_one(job, *, cache_dir, revision, get):
    season, league = job
    relative = Path(revision)/season/(league+".json")
    path = Path(cache_dir)/relative
    metadata_path = path.with_suffix(".metadata.json")
    url = f"https://raw.githubusercontent.com/{PROJECT}/{revision}/{season}/{league}.json"
    if path.exists() and metadata_path.exists():
        content = path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("source_url") != url or
                metadata.get("sha256") != hashlib.sha256(content).hexdigest()):
            raise ValueError("cached_source_integrity_mismatch")
    else:
        content = _read_response(get(url, timeout=30))
        metadata = {"source_url": url,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n",
                                 encoding="utf-8")
    rows, diagnostics = normalize_file(json.loads(content), season=season, competition=league)
    manifest_entry = {**metadata, "season": season, "competition": league,
                      "cache_relative_path": relative.as_posix(), **diagnostics,
                      "source_date_range": ([min(r["source_date"] for r in rows),
                                             max(r["source_date"] for r in rows)]
                                            if rows else None)}
    return rows, manifest_entry


def collect(*, cache_dir=DEFAULT_CACHE, manifest_path=SOURCE_MANIFEST,
            revision=None, seasons=SEASONS, leagues=LEAGUES, get=requests.get):
    """Download at most 30 results files with at most two concurrent requests."""
    if any(season not in SEASONS for season in seasons) or any(league not in LEAGUES for league in leagues):
        raise ValueError("only_predeclared_seasons_and_leagues_allowed")
    jobs = list(dict.fromkeys((season, league) for season in seasons for league in leagues))
    if not jobs or len(jobs) > MAX_FILES:
        raise ValueError("one_to_thirty_source_files_required")
    revision = _revision(revision) if revision is not None else resolve_revision(get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        found = list(pool.map(lambda job: _fetch_one(
            job, cache_dir=cache_dir, revision=revision, get=get), jobs))
    records = [row for rows, _ in found for row in rows]
    # Catch even a cross-file duplicated/conflicting fixture. Never borrow a
    # record from another season or source when an identity collides.
    seen = {}
    for row in records:
        key = _identity(row)
        if key in seen:
            raise ValueError("cross_file_duplicate_fixture_identity")
        seen[key] = row
    records.sort(key=lambda row: (row["source_date"], row["competition"],
                                  row["home_team"], row["away_team"]))
    manifest = {"schema_version": 1, "source_project": PROJECT,
                "source_revision": revision, "license": "CC0-1.0",
                "license_url": f"https://github.com/{PROJECT}/blob/{revision}/LICENSE.md",
                "readme_url": f"https://github.com/{PROJECT}/blob/{revision}/README.md",
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "collection_policy": {"maximum_source_files": MAX_FILES,
                                      "maximum_concurrent_requests": 2,
                                      "source_files_requested": len(jobs),
                                      "paid_api_calls": 0},
                "chronology_policy": "source_date_only_exclude_entire_candidate_date_from_history",
                "utc_kickoff_or_sydney_day_inferred": False,
                "historical_source_not_prospective_prediction_evidence": True,
                "contains_bookmaker_odds": False, "contains_shot_xg": False,
                "normalized_played_matches": len(records),
                "coverage_by_season": dict(sorted(Counter(r["season"] for r in records).items())),
                "coverage_by_competition": dict(sorted(Counter(r["competition"] for r in records).items())),
                "source_files": [entry for _, entry in found]}
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return records, manifest


def load_normalized(manifest_path=SOURCE_MANIFEST, cache_dir=DEFAULT_CACHE):
    """Load only hash-matching, pinned raw sources named in the manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    revision = _revision(manifest["source_revision"])
    root = Path(cache_dir).resolve()
    rows = []
    for entry in manifest["source_files"]:
        relative = Path(revision)/entry["season"]/(entry["competition"]+".json")
        if entry.get("cache_relative_path") != relative.as_posix():
            raise ValueError("manifest_source_path_mismatch")
        path = (root/relative).resolve()
        if root not in path.parents:
            raise ValueError("source_path_outside_cache")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("cached_source_integrity_mismatch")
        records, _ = normalize_file(json.loads(content), season=entry["season"],
                                     competition=entry["competition"])
        rows.extend(records)
    if len(rows) != manifest["normalized_played_matches"]:
        raise ValueError("manifest_normalized_count_mismatch")
    return sorted(rows, key=lambda row: (row["source_date"], row["competition"],
                                        row["home_team"], row["away_team"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--revision")
    args = parser.parse_args()
    _, manifest = collect(cache_dir=args.cache_dir, manifest_path=args.manifest,
                          revision=args.revision)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
