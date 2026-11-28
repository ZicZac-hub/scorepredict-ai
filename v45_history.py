"""Recover historical regulation scores from content-addressed source receipts."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3

from five_double_accuracy import _final, utc
from forecast_ledger import LedgerValidationError
from history_pipeline import RESULT_DELAY_HOURS, season_year
from v42_research_trial import _pair
import v44_early_history as early
from v44_early_history_audit import qualifies
from v44_refresh_expiry_audit import event
from verified_result_index import index_verified_results

ROOT = Path("data")
VERSION = "v45-history-recovery-v1"
TABLE = "ziczac_v45_history_snapshots"


def _identity(match):
    provider, fixture_id = match["provider"], match["id"]
    home, away = match["homeTeam"]["id"], match["awayTeam"]["id"]
    if any(type(value) not in (str, int) or not str(value).strip()
           for value in (provider, fixture_id, home, away)):
        raise ValueError("missing_historical_identity")
    return str(provider), str(fixture_id), utc(match["utcDate"]).isoformat(), str(home), str(away)


def _read_receipts(root, now):
    receipts, rejections = [], Counter()
    for name in ("v44_early_history", "v44_recovery_sources/history"):
        base = root / name
        for path in sorted((base / "receipts").glob("*.json")):
            try:
                receipt = early.load(path, root=base, now=now)
                if _identity(receipt["fixture"]) != tuple(receipt["event"]):
                    raise ValueError("invalid_receipt_identity")
                receipts.append((name, receipt))
            except (OSError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
                rejections["invalid_history_receipt"] += 1
    return receipts, rejections


def _recover_rows(receipts, finals, conflicts, now, rejections):
    candidates = {}
    conflicting_receipts = set()
    for source, receipt in receipts:
        for match in receipt["history"]:
            try:
                provider, fixture_id, kickoff, home, away = _identity(match)
                competition = match["competition"]
                score = match["score"]["fullTime"]
                observed = receipt["captured_at_utc"]
                row = {
                    "provider": provider, "fixture_id": fixture_id, "kickoff_utc": kickoff,
                    "home_team_id": home, "away_team_id": away,
                    "competition_id": competition.get("id"), "competition_code": competition.get("code"),
                    "season": season_year(match), "home_goals": score["home"], "away_goals": score["away"],
                    "first_observed_at_utc": utc(observed).isoformat(),
                    "receipt": {"archive": source, "sha256": receipt["receipt_sha256"]},
                }
                key = (provider, fixture_id)
                signature = (kickoff, home, away, score["home"], score["away"],
                             str(competition.get("id") or competition.get("code")), row["season"])
                previous = candidates.get(key)
                if previous and previous[0] != signature:
                    conflicting_receipts.add(key)
                if previous is None or row["first_observed_at_utc"] < previous[1]["first_observed_at_utc"]:
                    candidates[key] = signature, row
            except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
                rejections["historical_row_without_provider_identity"] += 1
    rows = []
    for key, (_, row) in sorted(candidates.items()):
        if key in conflicting_receipts or key in conflicts:
            rejections["conflicting_historical_fixture"] += 1
            continue
        final = finals.get(key)
        if final and _final(row, final, now) != (row["home_goals"], row["away_goals"]):
            rejections["history_disagrees_with_saved_final"] += 1
            continue
        row["saved_final_agrees"] = final is not None
        rows.append(row)
    return rows


def _pair_gaps(root, source_root, receipts, finals, conflicts, now, rejections):
    by_event = defaultdict(list)
    for source, receipt in receipts:
        by_event[tuple(receipt["event"])].append((source, receipt))
    counts, rows = Counter(), []
    for path in sorted((root / "professional_double_shadow" / "sets").glob("*.json")):
        try:
            pair, models = _pair(path, source_root=source_root, decision=now)
            cutoff = utc(pair["predicted_at_utc"])
            pair_rows = []
            for model in models:
                identity = event(model)
                saved = by_event[identity]
                if any(qualifies(receipt, model, cutoff) for _, receipt in saved):
                    history = "exact_receipt_available"
                elif any(source == "v44_recovery_sources/history"
                         and utc(receipt["captured_at_utc"]) > utc(model["captured_at_utc"])
                         for source, receipt in saved):
                    history = "later_recovery_only"
                else:
                    history = "missing_exact_receipt"
                kickoff = utc(model["kickoff_utc"])
                final = _final(model, finals.get(identity[:2]), now)
                if identity[:2] in conflicts or (identity[:2] in finals and final is None):
                    settlement = "conflicting_saved_final"
                elif final is not None:
                    settlement = "verified_final"
                elif now < kickoff + timedelta(hours=RESULT_DELAY_HOURS):
                    settlement = "not_due"
                else:
                    settlement = "missing_verified_final"
                pair_rows.append({"sydney_date": pair["sydney_date"], "provider": identity[0],
                                  "fixture_id": identity[1], "kickoff_utc": identity[2],
                                  "history_status": history, "settlement_status": settlement})
            counts["authenticated_original_pairs"] += 1
            for row in pair_rows:
                counts.update((row["history_status"], row["settlement_status"]))
            rows.extend(pair_rows)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError, LedgerValidationError):
            rejections["invalid_original_pair"] += 1
    return dict(counts), rows


def _collect(root, now):
    root = Path(root)
    checked = utc(now or datetime.now(timezone.utc))
    source_root = root / "v10_prospective"
    finals, conflicts, final_rejections = index_verified_results(source_root, root / "prematch", checked)
    receipts, rejections = _read_receipts(root, checked)
    rows = _recover_rows(receipts, finals, conflicts, checked, rejections)
    pairs, gaps = _pair_gaps(root, source_root, receipts, finals, conflicts, checked, rejections)
    seasons = Counter((row["provider"], str(row["competition_id"] or row["competition_code"]), row["season"])
                      for row in rows)
    report = {
        "report_type": VERSION, "checked_at_utc": checked.isoformat(), "external_requests": 0,
        "history_receipts": len(receipts), "recovered_historical_fixtures": len(rows),
        "earliest_history_receipt_utc": min((receipt["captured_at_utc"] for _, receipt in receipts), default=None),
        "history_matches_with_agreeing_saved_final": sum(row["saved_final_agrees"] for row in rows),
        "historical_coverage": [{"provider": provider, "competition": competition, "season": season,
                                 "regulation_scores": count}
                                for (provider, competition, season), count in sorted(seasons.items())],
        "original_pair_legs": pairs, "pair_legs": gaps,
        "historical_rejections": dict(rejections), "saved_final_rejections": dict(final_rejections),
    }
    return report, rows


def evaluate(*, root=ROOT, now=None):
    """Report what can be recovered locally and which original inputs are missing."""
    return _collect(root, now)[0]


def _snapshot(rows):
    return {"version": VERSION, "score_basis": "regulation", "rows": rows}


def recover(*, destination, root=ROOT, now=None):
    """Materialize an immutable historical snapshot ready for archive backup.

    Each row retains its actual source observation time. A recovered result
    cannot recreate a missing pre-match receipt or serve as newly verified final.
    """
    report, rows = _collect(root, now)
    snapshot = _snapshot(rows)
    digest = early.digest(snapshot)
    path = Path(destination) / (digest + ".json")
    if not early._once(path, snapshot) and early.digest(early._read(path)) != digest:
        raise ValueError("historical_snapshot_content_conflict")
    return {**report, "snapshot_path": str(path), "snapshot_sha256": digest}


@contextmanager
def _connect(*, database, database_url, read_only=False):
    if bool(database) == bool(database_url):
        raise ValueError("provide_one_history_database")
    connection = None
    try:
        if database_url:
            import psycopg
            connection = psycopg.connect(database_url, connect_timeout=7)
            placeholder = "%s"
        else:
            path = Path(database)
            if read_only:
                connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            else:
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                try:
                    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    os.close(descriptor)
                except FileExistsError:
                    pass
                connection = sqlite3.connect(path, timeout=30)
            placeholder = "?"
        with connection:
            yield connection, placeholder
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError("history_database_unavailable_" + type(exc).__name__) from None
    finally:
        if connection is not None:
            connection.close()


def _load_snapshot(connection, placeholder, digest):
    saved = connection.execute(
        f"SELECT payload_json FROM {TABLE} WHERE snapshot_sha256={placeholder}", (digest,),
    ).fetchone()
    if saved is None:
        raise ValueError("history_snapshot_not_found")
    snapshot = json.loads(saved[0])
    if (not isinstance(snapshot, dict) or snapshot.get("version") != VERSION
            or snapshot.get("score_basis") != "regulation" or not isinstance(snapshot.get("rows"), list)
            or early.digest(snapshot) != digest):
        raise ValueError("history_snapshot_integrity_check_failed")
    return snapshot


def recover_database(*, database=None, database_url=None, root=ROOT, now=None):
    """Store content-addressed historical snapshots separate from settlements."""
    report, rows = _collect(root, now)
    snapshot = _snapshot(rows)
    digest = early.digest(snapshot)
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with _connect(database=database, database_url=database_url) as (connection, placeholder):
        connection.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            snapshot_sha256 TEXT PRIMARY KEY, payload_json TEXT NOT NULL
        )""")
        inserted = connection.execute(
            f"INSERT INTO {TABLE} VALUES ({placeholder},{placeholder}) ON CONFLICT DO NOTHING", (digest, payload),
        ).rowcount
        _load_snapshot(connection, placeholder, digest)
    return {**report, "snapshot_sha256": digest, "database_backend": "postgresql" if database_url else "sqlite",
            "database_state": "stored" if inserted else "original_preserved"}


def load_database_snapshot(digest, *, database=None, database_url=None, provider=None, season=None,
                           competition=None, fixture_id=None):
    """Validate the archived snapshot before returning matching historical rows."""
    with _connect(database=database, database_url=database_url, read_only=True) as (connection, placeholder):
        snapshot = _load_snapshot(connection, placeholder, digest)
    return [row for row in snapshot["rows"] if (provider is None or row["provider"] == provider)
            and (season is None or row["season"] == season)
            and (fixture_id is None or row["fixture_id"] == str(fixture_id))
            and (competition is None or str(competition) in (str(row["competition_id"]), row["competition_code"]))]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=ROOT)
    output = cli.add_mutually_exclusive_group()
    output.add_argument("--recover-to", type=Path)
    output.add_argument("--database", type=Path)
    output.add_argument("--postgres", action="store_true", help="Use the configured DATABASE_URL")
    args = cli.parse_args()
    if args.postgres:
        url = os.environ.get("DATABASE_URL")
        if not url:
            cli.error("DATABASE_URL is required for --postgres")
        report = recover_database(root=args.root, database_url=url)
    elif args.database:
        report = recover_database(root=args.root, database=args.database)
    elif args.recover_to:
        report = recover(root=args.root, destination=args.recover_to)
    else:
        report = evaluate(root=args.root)
    print(json.dumps(report, indent=2, allow_nan=False))
