"""Read-only date-quality audit and source-file quarantine sensitivity.

Authentic source hashes do not establish correct football dates. This module
preserves the frozen November 15 experiment and never repairs dates by guess.
The optional rerun is a separately labelled development sensitivity, not a
new independent test or an update to live forecasts.
"""
import argparse
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

from target70_data import DEFAULT_CACHE, SOURCE_MANIFEST, load_normalized, normalize_file

VERSION = "source-calendar-quarantine-sensitivity-v1"
PRIMARY_CHECKS = (
    {"season": "2019-20", "competition": "en.1",
     "home_team": "AFC Bournemouth", "away_team": "Newcastle United",
     "score": [1, 4], "verified_date": "2020-07-01",
     "primary_source": "https://www.premierleague.com/en/news/1697163/match-summary-afc-bournemouth-v-newcastle-united"},
    {"season": "2019-20", "competition": "en.1",
     "home_team": "Manchester City", "away_team": "Liverpool FC",
     "score": [4, 0], "verified_date": "2020-07-02",
     "primary_source": "https://www.premierleague.com/en/news/1701434/man-city-cruise-to-victory-over-liverpool"},
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def season_window(season):
    """A broad plausibility screen, not a source of replacement dates.

    The five evaluated European leagues start after July. The 2019-20
    coronavirus interruption permits a later August 2020 season end.
    Passing this coarse screen does not certify a date's correctness.
    """
    if not isinstance(season, str) or not re.fullmatch(r"\d{4}-\d{2}", season):
        raise ValueError("invalid_season_label")
    year = int(season[:4])
    if int(season[-2:]) != (year+1) % 100:
        raise ValueError("nonconsecutive_season_label")
    return date(year, 8, 1), date(year+1, 8, 31) if year == 2019 else date(year+1, 6, 30)


def inspect_payload(payload, *, season, competition):
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise ValueError("source_matches_list_required")
    low, high = season_window(season)
    flags = []
    checks = [r for r in PRIMARY_CHECKS if r["season"] == season and r["competition"] == competition]
    verified_mismatches = []
    for number, row in enumerate(payload["matches"]):
        if not isinstance(row, dict):
            continue
        goals = (row.get("score") or {}).get("ft")
        if not isinstance(goals, list) or len(goals) != 2:
            continue  # Unknown outcomes are never invented to complete a slate.
        observed = row.get("date")
        reasons = []
        try:
            if not isinstance(observed, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", observed):
                raise ValueError("invalid_date")
            at = date.fromisoformat(observed)
            if not low <= at <= high:
                reasons.append("outside_broad_season_date_window")
        except ValueError:
            reasons.append("invalid_source_calendar_date")
        for check in checks:
            if (row.get("team1") == check["home_team"] and row.get("team2") == check["away_team"]
                    and goals == check["score"] and observed != check["verified_date"]):
                reasons.append("date_disagrees_with_primary_match_report")
                verified_mismatches.append({**check, "source_date": observed})
        if reasons:
            flags.append({"raw_row_index": number, "source_date": observed,
                          "home_team": row.get("team1"), "away_team": row.get("team2"),
                          "score": goals, "round": row.get("round"), "reasons": reasons})
    rows, diagnostics = normalize_file(payload, season=season, competition=competition)
    return {"season": season, "competition": competition,
            "broad_plausible_date_window": [low.isoformat(), high.isoformat()],
            "normalized_played_matches": len(rows), "normalization_diagnostics": diagnostics,
            "flagged_played_rows": len(flags), "flagged_rows": flags,
            "primary_verified_date_mismatches": verified_mismatches,
            "quarantine_entire_source_file": bool(flags),
            "passing_this_screen_does_not_verify_all_source_dates": True}


def audit(manifest_path=SOURCE_MANIFEST, cache_dir=DEFAULT_CACHE):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(cache_dir).resolve()
    revision = manifest["source_revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("pinned_source_revision_required")
    inspected = []
    for entry in manifest["source_files"]:
        relative = Path(revision)/entry["season"]/(entry["competition"]+".json")
        if relative.as_posix() != entry["cache_relative_path"]:
            raise ValueError("manifest_source_path_mismatch")
        path = (root/relative).resolve()
        if root not in path.parents or file_sha(path) != entry["sha256"]:
            raise ValueError("source_hash_or_cache_path_mismatch")
        found = inspect_payload(json.loads(path.read_text()), season=entry["season"],
                                competition=entry["competition"])
        inspected.append({**found, "source_file": relative.as_posix(),
                          "sha256": entry["sha256"], "source_url": entry["source_url"]})
    quarantined = [r for r in inspected if r["quarantine_entire_source_file"]]
    return {"version": VERSION, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": file_sha(manifest_path), "source_revision": revision,
            "source_files_hash_verified": len(inspected),
            "normalized_played_matches_before_quarantine": sum(r["normalized_played_matches"] for r in inspected),
            "quarantined_files": [{k: r[k] for k in ("season", "competition", "source_file",
                                                     "sha256", "flagged_played_rows", "normalized_played_matches")}
                                  for r in quarantined],
            "quarantined_file_played_matches": sum(r["normalized_played_matches"] for r in quarantined),
            "flagged_played_rows": sum(r["flagged_played_rows"] for r in inspected),
            "verified_primary_mismatches": [x for r in inspected for x in r["primary_verified_date_mismatches"]],
            "date_repair_performed": False, "original_experiment_modified": False,
            "live_forecasts_modified": False, "target_70_percent_proven": False,
            "impact_assessment": {
                "demonstrated_future_outcome_leak_into_2023_24_or_2024_25_evaluation": False,
                "reason": "The externally checked actual events are July 2020, still earlier than the evaluated 2023-24 and 2024-25 fixtures.",
                "historical_age_weight_distortion": True,
                "earlier_2019_2020_replay_can_leak_future_results": True,
                "all_other_dates_independently_verified": False,
                "all_66_flagged_events_individually_primary_verified": False},
            "limitations": [
                "Only two flagged rows are individually checked against official match reports; the broad season rule flags the remainder.",
                "A file with any flagged date is quarantined wholesale; dates are not shifted or corrected by assumption.",
                "Raw source hashes and a CC0 license do not establish football calendar accuracy.",
                "The untouched original numerical experiment is preserved as originally run, with this new quality qualification.",
                "Any quarantine rerun is post-holdout development sensitivity, not fresh independent evidence."],
            "files": inspected}


def quarantine(rows, report):
    banned = {(r["season"], r["competition"]) for r in report["quarantined_files"]}
    return [r for r in rows if (r["season"], r.get("competition", r.get("league"))) not in banned]


def run_sensitivity(manifest_path, cache_dir, output, *, include_gradient_boost=False):
    from model_comparison_research import run
    from target70_postholdout_audit import source_audit
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("new_empty_sensitivity_output_required")
    report = audit(manifest_path, cache_dir)
    rows = load_normalized(manifest_path, cache_dir)
    kept = quarantine(rows, report)
    manifest = json.loads(Path(manifest_path).read_text())
    coverage, _, incomplete = source_audit(manifest, cache_dir, "2024-25")
    coverage["source_date_quality_quarantine"] = report["quarantined_files"]
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"version": VERSION, "quarantined_files": report["quarantined_files"],
                "input_matches": len(rows), "retained_matches": len(kept),
                "dates_repaired": False, "unseen_evaluation": False,
                "no_model_hyperparameters_changed": True,
                "original_experiment_modified": False, "include_gradient_boost": include_gradient_boost,
                "source_manifest_sha256": file_sha(manifest_path),
                "quarantine_implementation_sha256": file_sha(__file__)}
    with (output/"quarantine_protocol_before_rerun.json").open("x") as handle:
        json.dump(protocol, handle, sort_keys=True, indent=2)
        handle.write("\n")
    summary = run(kept, output, include_gradient_boost=include_gradient_boost,
                  source_manifest_sha256=file_sha(manifest_path), source_coverage=coverage,
                  incomplete_source_dates=incomplete)
    raw = output/"comparison.json"
    if raw.exists():
        (output/"comparison.json.gz").write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
    summary["source_quality_sensitivity_version"] = VERSION
    summary["dates_repaired"] = False
    summary["original_experiment_modified"] = False
    summary["full_comparison_file"] = "comparison.json.gz"
    (output/"summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path("data/source_quality_audit_v1/audit.json"))
    parser.add_argument("--sensitivity-output", type=Path)
    parser.add_argument("--include-gradient-boost", action="store_true")
    args = parser.parse_args()
    report = audit(args.manifest, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    if args.sensitivity_output:
        run_sensitivity(args.manifest, args.cache_dir, args.sensitivity_output,
                        include_gradient_boost=args.include_gradient_boost)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
