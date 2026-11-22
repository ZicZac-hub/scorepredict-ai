"""Index final-score evidence available at a report's evaluation cutoff."""
from collections import Counter
from pathlib import Path

from five_double_accuracy import _final, _load, utc


def index_verified_results(root, prematch_root, clock):
    """Reject invalid submissions before detecting disagreements between finals."""
    indexed = {}
    conflicts = set()
    rejected = Counter()
    identity_fields = ("provider", "fixture_id", "home_team_id", "away_team_id")
    paths = (Path(root) / "verified_results.json", Path(prematch_root) / "verified_results.json")
    for path in paths:
        try:
            source = _load(path, {})
        except (ValueError, OSError):
            rejected["unreadable_result_file"] += 1
            continue
        if not isinstance(source, dict):
            rejected["result_file_not_dictionary"] += 1
            continue
        for result in source.values():
            if not isinstance(result, dict):
                rejected["result_not_dictionary"] += 1
                continue
            if any(result.get(field) in (None, "") for field in identity_fields):
                rejected["missing_result_identity"] += 1
                continue
            goals = _final(result, result, clock)
            if goals is None:
                rejected["invalid_or_unavailable_verified_final"] += 1
                continue
            provider, fixture_id, home, away = (str(result[field]) for field in identity_fields)
            key = (provider, fixture_id)
            signature = (home, away, utc(result["kickoff_utc"]), goals)
            if key in indexed and indexed[key][0] != signature:
                conflicts.add(key)
            indexed.setdefault(key, (signature, result))
    return (
        {key: entry[1] for key, entry in indexed.items() if key not in conflicts},
        conflicts,
        rejected,
    )
