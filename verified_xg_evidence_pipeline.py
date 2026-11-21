"""Write-once historical inputs and evidence-only xG shadow evaluation.

Source packets are kept intact in data/v21_sources. Historical match inputs are
frozen at collection time, never reconstructed after an outcome. No API calls,
model promotion, or editing of the original forecast/evidence ledgers.
"""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
from pathlib import Path

from history_pipeline import build_evidence
from v10_prospective import utc
from v21_provider_evidence import valid_snapshot

ROOT = Path("data/verified_xg_replay")
PROVIDER_ROOT = Path("data/v21_sources")
RESULTS = Path("data/v10_prospective/verified_results.json")
HOLDOUT_START = "2026-12-15"
MIN_TRAIN = 30
MIN_HOLDOUT = 50
MIN_PAIRS = 20


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _identity(fixture):
    return (str(fixture["id"]), str(fixture["homeTeam"]["id"]),
            str(fixture["awayTeam"]["id"]), utc(fixture["utcDate"]))


def _source_fixture(forecast, fixture):
    fid, home, away, kickoff = _identity(fixture)
    return (str(forecast.get("provider")) == "API-Football"
            and str(forecast.get("fixture_id")) == fid
            and str(forecast.get("home_team_id")) == home
            and str(forecast.get("away_team_id")) == away
            and utc(forecast.get("kickoff_utc")) == kickoff)


def freeze_history(fixture, evidence, packet_path, *, root=ROOT,
                   provider_root=PROVIDER_ROOT):
    """Persist the exact model history when the provider packet is captured.

    An existing file is immutable. Only histories whose canonical input digest
    recomputes at the original AS-OF time are accepted.
    """
    root, provider_root, packet_path = map(Path, (root, provider_root, packet_path))
    relative = packet_path.relative_to(provider_root)
    if not relative.parts or ".." in relative.parts:
        raise ValueError("packet_outside_provider_archive")
    packet = _read(packet_path)
    source, snapshot = packet["source_forecast"], packet["provider_snapshot"]
    cutoff = utc(evidence["as_of_utc"])
    captured = utc(snapshot["captured_at_utc"])
    kickoff = utc(fixture["utcDate"])
    if (not cutoff <= captured < kickoff
            or not _source_fixture(source, fixture)
            or not valid_snapshot(snapshot, fixture, now=captured)
            or packet.get("supplemental_history_digest") != evidence.get("input_digest")
            or not isinstance(evidence.get("history"), list)):
        raise ValueError("unverified_or_mismatched_capture")
    rebuilt = build_evidence(fixture, evidence["history"], cutoff)
    if (rebuilt["input_digest"] != evidence["input_digest"]
            or len(rebuilt["history"]) != len(evidence["history"])):
        raise ValueError("historical_asof_integrity_failed")
    if not (snapshot.get("verified_historical_fixture_xg") or []):
        return None  # Still preserve the provider packet. No fabricated xG.
    record = {
        "schema": 1,
        "scope": "shadow_only_no_production_promotion",
        "fixture": rebuilt["fixture"],
        "history": rebuilt["history"],
        "as_of_utc": rebuilt["as_of_utc"],
        "history_input_digest": rebuilt["input_digest"],
        "history_sha256": _digest(rebuilt["history"]),
        "snapshot_relative_path": str(relative),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "captured_at_utc": snapshot["captured_at_utc"],
    }
    record["record_sha256"] = _digest(record)
    dest = root / "history" / (str(fixture["id"]) + "_" +
                               snapshot["snapshot_sha256"][:16] + "_" +
                               rebuilt["input_digest"][:16] + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("x", encoding="utf-8") as out:
            json.dump(record, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
    except FileExistsError:
        if _read(dest) != record:
            raise ValueError("immutable_history_conflict")
    return dest


def _authenticated_record(path, provider_root):
    record = _read(path)
    if not isinstance(record, dict):
        return None
    seal = record.get("record_sha256")
    if not isinstance(seal, str) or seal != _digest(
            {k: v for k, v in record.items() if k != "record_sha256"}):
        return None
    relative = Path(record.get("snapshot_relative_path") or "")
    if (relative.is_absolute() or not relative.parts or
            "." in relative.parts or ".." in relative.parts):
        return None
    source = _read(Path(provider_root) / relative)
    fixture, history = record.get("fixture"), record.get("history")
    if not isinstance(fixture, dict) or not isinstance(history, list):
        return None
    packet = source.get("provider_snapshot")
    if not isinstance(packet, dict) or not valid_snapshot(
            packet, fixture, now=record.get("captured_at_utc")):
        return None
    if (packet["snapshot_sha256"] != record.get("snapshot_sha256")
            or record["captured_at_utc"] != packet["captured_at_utc"]
            or not _source_fixture(source.get("source_forecast") or {}, fixture)
            or source.get("supplemental_history_digest") != record.get("history_input_digest")
            or record.get("history_sha256") != _digest(history)):
        return None
    cutoff = utc(record["as_of_utc"])
    captured = utc(record["captured_at_utc"])
    if cutoff > captured or captured >= utc(fixture["utcDate"]):
        return None
    rebuilt = build_evidence(fixture, history, cutoff)
    if (rebuilt["input_digest"] != record["history_input_digest"]
            or len(rebuilt["history"]) != len(history)):
        return None
    return {"fixture": fixture, "history": history, "snapshot": packet,
            "capture": captured}


def _verified_actual(result, fixture):
    if not isinstance(result, dict):
        return None
    try:
        fid, home, away, kickoff = _identity(fixture)
        if (result.get("provider") != "API-Football"
                or str(result.get("fixture_id")) != fid
                or str(result.get("home_team_id")) != home
                or str(result.get("away_team_id")) != away
                or utc(result.get("kickoff_utc")) != kickoff
                or result.get("status") != "FT"
                or result.get("score_basis") != "regulation"
                or utc(result.get("verified_at_utc")) <= kickoff):
            return None
        scores = (result.get("home_goals"), result.get("away_goals"))
        if any(type(n) is not int or n < 0 for n in scores):
            return None
        return list(scores)
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


def load_cases(*, root=ROOT, provider_root=PROVIDER_ROOT, results_path=RESULTS):
    """Join independently verified final results to frozen, earlier histories."""
    results = _read(results_path) if Path(results_path).exists() else {}
    candidates, rejection = {}, Counter()
    for path in sorted((Path(root) / "history").glob("*.json")):
        try:
            record = _authenticated_record(path, provider_root)
            if record is None:
                rejection["invalid_history_or_source_seal"] += 1
                continue
            fixture = record["fixture"]
            fid = str(fixture["id"])
            actual = _verified_actual(results.get(fid), fixture)
            if actual is None:
                rejection["missing_verified_final"] += 1
                continue
            # One fixture contributes once. Select earliest valid archived
            # decision without consulting the result or model probabilities.
            row = {"fixture": fixture, "history": record["history"],
                   "snapshot": record["snapshot"], "actual": actual}
            candidate = (record["capture"], str(path), row)
            if fid not in candidates or candidate[:2] < candidates[fid][:2]:
                candidates[fid] = candidate
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            rejection["unreadable_history_record"] += 1
    return [value[2] for _, value in sorted(candidates.items())], dict(rejection)


def audit_packets(*, provider_root=PROVIDER_ROOT):
    """Count source evidence, not attempts or plausible synthetic values."""
    root = Path(provider_root)
    stats = Counter()
    fixtures = set()
    for folder in (root, root / "versions"):
        for path in sorted(folder.glob("API-Football_*.json")):
            stats["packets_seen"] += 1
            try:
                record = _read(path)
                source, snapshot = record["source_forecast"], record["provider_snapshot"]
                fixture = {"provider": "API-Football", "id": source["fixture_id"],
                           "utcDate": source["kickoff_utc"],
                           "homeTeam": {"id": source["home_team_id"]},
                           "awayTeam": {"id": source["away_team_id"]}}
                if (not _source_fixture(source, fixture)
                        or not valid_snapshot(snapshot, fixture)):
                    stats["invalid_or_unsealed_packets"] += 1
                    continue
                stats["authenticated_packets"] += 1
                fixtures.add(str(fixture["id"]))
                teams = {str(source["home_team_id"]), str(source["away_team_id"])}
                historical = {team: set() for team in teams}
                for obs in snapshot.get("verified_historical_fixture_xg") or []:
                    if not isinstance(obs, dict):
                        continue
                    try:
                        if (obs.get("type") != "api_football_reported_expected_goals"
                                or utc(obs["kickoff_utc"]) + timedelta(hours=3)
                                > utc(snapshot["captured_at_utc"])):
                            continue
                        for side in ("home", "away"):
                            xg = obs[side + "_xg"]
                            if (type(xg) not in (int, float) or not 0 <= xg <= 15):
                                raise ValueError("invalid_xg")
                        for side in ("home", "away"):
                            tid = str(obs.get(side + "_team_id"))
                            if tid in historical:
                                historical[tid].add(str(obs["fixture_id"]))
                    except (ValueError, KeyError, TypeError, OverflowError):
                        continue
                if min(map(len, historical.values()), default=0) >= 3:
                    stats["packets_with_3_xg_rows_each_side"] += 1
                if min(map(len, historical.values()), default=0) >= 6:
                    stats["packets_with_6_xg_rows_each_side"] += 1
                xi = [row for row in (snapshot.get("confirmed_starting_xi") or [])
                      if isinstance(row, dict) and row.get("provider_reported") is True
                      and str(row.get("team_id")) in teams
                      and isinstance(row.get("starting_xi"), list)
                      and len(row["starting_xi"]) == 11]
                if {str(row["team_id"]) for row in xi} == teams:
                    stats["packets_with_both_provider_XIs"] += 1
            except (OSError, ValueError, KeyError, TypeError, OverflowError):
                stats["invalid_or_unsealed_packets"] += 1
    stats["unique_fixtures_with_authenticated_packets"] = len(fixtures)
    return dict(stats)


def report(*, root=ROOT, provider_root=PROVIDER_ROOT, results_path=RESULTS,
           cutoff=HOLDOUT_START):
    from verified_xg_calibration import evaluate
    cases, exclusions = load_cases(root=root, provider_root=provider_root,
                                   results_path=results_path)
    comparison = evaluate(cases, cutoff=cutoff)
    paired = comparison["five_double"]["verified_xg"]["pairs"]
    enough = (comparison["training_cases"] >= MIN_TRAIN
              and comparison["holdout_cases"] >= MIN_HOLDOUT
              and paired >= MIN_PAIRS)
    return {
        "protocol": "verified-xg-prospective-paired-v1",
        "holdout_start": cutoff,
        "authenticated_archive": audit_packets(provider_root=provider_root),
        "frozen_history_records": len(list((Path(root) / "history").glob("*.json"))),
        "independently_settled_candidate_cases": len(cases),
        "case_exclusions": exclusions,
        "minimum_evidence": {"training_cases": MIN_TRAIN,
                             "holdout_cases": MIN_HOLDOUT, "paired_days": MIN_PAIRS},
        "evidence_sufficient": enough,
        "status": ("descriptive_results_only_not_proven" if enough
                   else "insufficient_matched_authenticated_evidence"),
        "comparison": comparison,
        "live_prediction_model_changed": False,
        "automatic_promotion": False,
        "accuracy_improvement_verified": False,
        "caution": ("A passing coverage threshold is not statistical proof of superiority. "
                    "Original historical inputs must be frozen before kickoff; "
                    "old packets without archived inputs cannot be backfilled as prospective.")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="verified-xg-evidence-report.json")
    parser.add_argument("--cutoff", default=HOLDOUT_START)
    args = parser.parse_args()
    r = report(cutoff=args.cutoff)
    Path(args.output).write_text(json.dumps(r, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
    print(json.dumps({k: r[k] for k in (
        "status", "authenticated_archive", "frozen_history_records",
        "independently_settled_candidate_cases", "minimum_evidence",
        "evidence_sufficient", "automatic_promotion")}, indent=2))


if __name__ == "__main__":
    main()
