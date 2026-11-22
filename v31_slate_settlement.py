"""V31 original-slate settlement coverage for independent provider final-score backfill.

The latest forecast pointer is NOT a permanent enrollment roster. V28 froze the
entire candidate slate before kickoff. Verify that original source evidence
FIRST and prioritize the enrolled physical fixture IDs for settlement. Do not
reconstruct new predictions or alter any selected five-score ticket.

No provider calls here. The caller still enforces the pre-existing 12 batches
of 20 IDs per hourly run. Full-slate coverage and pre-kickoff source validity
are not five-double prediction accuracy.
"""
from collections import Counter
from datetime import timedelta
from pathlib import Path

from five_double_accuracy import utc
from v28_independent_pair_trial import (
    OUT as V28_OUT, verify_decision, _key,
)

RESULT_DELAY = timedelta(hours=3)


def enrolled_due(root, existing, now, *, v28_out=V28_OUT):
    """Return original V28 source records, not the latest-forecast replacements.

    A day is rejected in its entirety if any immutable original forecast,
    saved candidate score or selected policy fails independent reproduction.
    Results are NEVER used to choose the candidate slate.
    """
    clock = utc(now)
    due = {}
    all_sources = {}
    conflicted_ids = set()
    counters = Counter()
    invalid = Counter()
    next_due = []
    for file in sorted((Path(v28_out)/"decisions").glob("*.json")):
        try:
            packet, rows = verify_decision(file, root=root, clock=clock)
            counters["validated_frozen_slate_days"] += 1
            if packet["policy"] != "FROZEN":
                counters["original_abstention_days"] += 1
                continue
            counters["validated_active_slate_days"] += 1
            for item in rows:
                provider, fid = _key(item)
                if provider != "API-Football" or not fid.isdigit():
                    counters["excluded_non_api_football"] += 1
                    continue
                # A fixture can appear in multiple ORIGINAL slates. One
                # physical fixture receives at most one source identity.
                frozen = item["record"]
                if fid in conflicted_ids:
                    continue
                old = all_sources.get(fid)
                if old is not None:
                    fields = ("home_team_id", "away_team_id", "kickoff_utc")
                    if any(
                        (utc(old[f]) != utc(frozen[f]) if f == "kickoff_utc"
                         else str(old[f]) != str(frozen[f]))
                        for f in fields
                    ):
                        invalid["same_fixture_conflicting_archived_identity"] += 1
                        conflicted_ids.add(fid)
                        all_sources.pop(fid, None)
                        due.pop(fid, None)
                        continue
                all_sources.setdefault(fid, frozen)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            invalid[str(exc) if isinstance(exc, ValueError)
                    else type(exc).__name__] += 1
    for fid, frozen in sorted(all_sources.items()):
        if fid in existing:
            counters["already_in_saved_result_file"] += 1
            continue
        due_at = utc(frozen["kickoff_utc"]) + RESULT_DELAY
        if due_at > clock:
            counters["original_fixtures_not_yet_due"] += 1
            next_due.append(due_at.isoformat())
            continue
        due[fid] = frozen
        counters["original_fixtures_due_without_saved_result"] += 1
    info = {
        "validated_frozen_slate_days": counters["validated_frozen_slate_days"],
        "validated_active_slate_days": counters["validated_active_slate_days"],
        "original_abstention_days": counters["original_abstention_days"],
        "unique_original_fixtures_enrolled": len(all_sources),
        "already_in_saved_result_file": counters["already_in_saved_result_file"],
        "original_fixtures_not_yet_due": counters["original_fixtures_not_yet_due"],
        "original_fixtures_due_without_saved_result":
            counters["original_fixtures_due_without_saved_result"],
        "next_original_fixture_settlement_due_utc":
            min(next_due) if next_due else None,
        "excluded_non_api_football": counters["excluded_non_api_football"],
        "invalid_original_slates_excluded": dict(invalid),
        "conflicting_fixture_identities_excluded": len(conflicted_ids),
        "source_validation_precedes_any_provider_request": True,
        "new_five_score_predictions_issued": False,
        "original_forecasts_modified": False,
    }
    return due, info
