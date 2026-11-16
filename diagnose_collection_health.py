"""Read-only diagnostics of ZicZac collection, verification and evidence coverage."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("data")
def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

def main():
    prematch = load(ROOT / "prematch/accuracy_report.json") or {}
    comparison = load(ROOT / "prospective_ledger/comparison.json") or {}
    verified = load(ROOT / "prematch/verified_results.json") or {}
    verified_rows = list(verified.values()) if isinstance(verified, dict) else verified
    valid = [r for r in verified_rows if isinstance(r, dict)
             and r.get("provider") and r.get("fixture_id")
             and r.get("home_team_id") and r.get("away_team_id")
             and r.get("kickoff_utc") and r.get("score_basis") == "regulation"
             and r.get("status") in ("FT", "AET", "PEN")
             and isinstance(r.get("home_goals"), int)
             and isinstance(r.get("away_goals"), int)]
    missing = int(comparison.get("exclusions", {}).get("awaiting_verified_result", 0))
    models = {name: {"evaluated_slots": x.get("evaluated", 0),
                     "exact_hits": x.get("exact_hits", 0)}
              for name, x in comparison.get("models", {}).items()}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_unique_fixtures": prematch.get("unique_pre_kickoff_fixtures"),
        "prematch_verified_finished": prematch.get("verified_finished_matches"),
        "verified_result_rows": len(verified_rows),
        "verified_result_rows_with_identity_fields": len(valid),
        "awaiting_verified_result_slots": missing,
        "confirmed_lineups_in_earliest_snapshots": prematch.get("confirmed_lineups_in_earliest_snapshots"),
        "injury_records_in_earliest_snapshots": prematch.get("injury_records_in_earliest_snapshots"),
        "database_state_in_evaluation": comparison.get("persistence", {}).get("database", {}).get("state"),
        "models": models,
        "diagnosis": [],
    }
    if missing:
        report["diagnosis"].append("Verified result coverage is incomplete; pending does not mean lost.")
    if not valid and verified_rows:
        report["diagnosis"].append("Existing final-score rows lack strict identity or regulation-time fields.")
    if not report["confirmed_lineups_in_earliest_snapshots"]:
        report["diagnosis"].append("No confirmed lineup evidence in currently evaluated earliest snapshots.")
    if not report["injury_records_in_earliest_snapshots"]:
        report["diagnosis"].append("No injury evidence in currently evaluated earliest snapshots.")
    if report["database_state_in_evaluation"] != "connected":
        report["diagnosis"].append("Evaluation did not connect to central database; file-backed data may still exist.")
    output = ROOT / "diagnostics/collection_health.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
