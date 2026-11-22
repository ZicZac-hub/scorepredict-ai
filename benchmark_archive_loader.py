"""Read-only, identity-verified benchmark adapter for original frozen forecasts.

Both the V10 research archive and the prospective comparison ledger are read.
Final results are corroborated across the independent verification stores.
No snapshots are rewritten, no past matches are re-predicted, and no model is
promoted by this adapter.
"""
import argparse
import json
import math
from pathlib import Path

from model_comparison_audit import _time, benchmark, paired_benchmark

FORECAST_ROOTS = (
    "data/v10_prospective/forecasts",
    "data/prospective_ledger/forecasts",
)
RESULT_FILES = (
    "data/v10_prospective/verified_results.json",
    "data/prematch/verified_results.json",
)
SETTLEMENT_DIR = "data/prospective_ledger/settlements"
MAX_FILES = 3000
MAX_BYTES = 4_000_000


def _read(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("oversized archive record")
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_identity(result, fixture_id):
    """Accept finished, post-kickoff, provider-identified 90-minute scores."""
    if not isinstance(result, dict):
        return None
    if result.get("status") not in ("FT", "AET", "PEN", "FINISHED"):
        return None
    if result.get("score_basis") != "regulation":
        return None
    if not (result.get("evidence_version") == 2 or
            result.get("verified") is True or
            result.get("verification_status") == "verified"):
        return None
    provider = result.get("provider")
    if not isinstance(provider, str) or not provider:
        return None
    if str(result.get("fixture_id")) != fixture_id:
        return None
    kickoff = _time(result.get("kickoff_utc"))
    verified_at = _time(result.get("verified_at_utc"))
    if kickoff is None or verified_at is None or verified_at <= kickoff:
        return None
    home, away = result.get("home_team_id"), result.get("away_team_id")
    if any(v is None or isinstance(v, bool) or not str(v).isdigit() or
           int(v) <= 0 for v in (home, away)):
        return None
    if str(home) == str(away):
        return None
    hg, ag = result.get("home_goals"), result.get("away_goals")
    if type(hg) is not int or type(ag) is not int or min(hg, ag) < 0:
        return None
    return (provider, fixture_id, kickoff, str(int(home)), str(int(away)), hg, ag)


def _score_rows(matrix):
    if not isinstance(matrix, list) or len(matrix) < 2 or len(matrix) > 256:
        return None
    width = len(matrix[0]) if isinstance(matrix[0], list) else 0
    if width < 2 or width > 256:
        return None
    rows, values = [], []
    for h, line in enumerate(matrix):
        if not isinstance(line, list) or len(line) != width:
            return None
        for a, value in enumerate(line):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 1:
                return None
            rows.append([h, a, float(value)])
            values.append(value)
    if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-6):
        return None
    return rows


def _valid_forecast(forecast, provider, fixture_id, result):
    if not isinstance(forecast, dict) or forecast.get("mode") != "prospective":
        return None
    if forecast.get("score_basis") != "regulation":
        return None
    if forecast.get("eligible_for_comparison") is False:
        return None
    if str(forecast.get("fixture_id")) != fixture_id or forecast.get("provider") != provider:
        return None
    if any(str(forecast.get(k)) != str(result.get(k)) for k in
           ("home_team_id", "away_team_id")):
        return None
    captured = _time(forecast.get("captured_at_utc"))
    kickoff = _time(forecast.get("kickoff_utc"))
    evidence = _time(forecast.get("evidence_as_of_utc"))
    if captured is None or kickoff is None or evidence is None:
        return None
    if evidence > captured or captured >= kickoff or kickoff != _time(result.get("kickoff_utc")):
        return None
    scores = _score_rows(forecast.get("score_matrix"))
    if scores is None:
        return None
    model = forecast.get("model_version") or forecast.get("model_key")
    if not isinstance(model, str) or not model:
        return None
    probability_metrics_eligible = (
        forecast.get("probability_metrics_eligible") is not False
        and forecast.get("model_key") != "always_1_1"
        and model not in ("always_1_1", "fixed-always-1-1-v1")
    )
    return model, captured, {
        "created_at": forecast["captured_at_utc"],
        "scores": scores,
        "probability_metrics_eligible": probability_metrics_eligible,
    }


def _load_results(base, diagnostics):
    results = {}
    conflicted = set()
    paths = [base / path for path in RESULT_FILES]
    paths.extend(sorted((base / SETTLEMENT_DIR).glob("*.json")))
    for path in paths:
        if not path.is_file():
            continue
        diagnostics["result_files_read"] += 1
        try:
            packet = _read(path)
        except (OSError, ValueError, TypeError, UnicodeError):
            diagnostics["unreadable_results"] += 1
            continue
        if isinstance(packet, dict) and "fixture_id" in packet:
            items = [(str(packet["fixture_id"]), packet)]
        elif isinstance(packet, dict):
            items = list(packet.items())
        else:
            diagnostics["unreadable_results"] += 1
            continue
        for raw_id, actual in items:
            fid = str(raw_id)
            identity = _verified_identity(actual, fid)
            if identity is None:
                diagnostics["rejected_results"] += 1
                continue
            key = (identity[0], fid)
            if key in results and results[key][0] != identity:
                conflicted.add(key)
            else:
                results.setdefault(key, (identity, actual))
    for key in conflicted:
        results.pop(key, None)
    diagnostics["conflicting_result_fixtures"] = len(conflicted)
    diagnostics["verified_fixtures"] = len(results)
    return {key: result for key, (_, result) in results.items()}


def load_archive(root=".", limit=MAX_FILES):
    base = Path(root).resolve()
    diagnostics = {
        "files_checked": 0, "compatible_records": 0, "skipped_records": 0,
        "unreadable_files": 0, "truncated": False, "verified_fixtures": 0,
        "frozen_forecasts": 0, "missing_forecasts": 0,
        "result_files_read": 0, "unreadable_results": 0,
        "rejected_results": 0, "conflicting_result_fixtures": 0,
        "archived_fixture_ids": 0, "awaiting_verified_results": 0,
    }
    verified = _load_results(base, diagnostics)
    all_fixtures = set()
    for root_name in FORECAST_ROOTS:
        directory = base / root_name
        if directory.is_dir():
            for provider in directory.iterdir():
                if provider.is_dir():
                    all_fixtures.update((provider.name, p.name)
                                        for p in provider.iterdir() if p.is_dir())
    diagnostics["archived_fixture_ids"] = len(all_fixtures)
    diagnostics["awaiting_verified_results"] = len(all_fixtures - verified.keys())
    records = {}
    max_count = min(MAX_FILES, max(0, int(limit)))
    for (provider, fid), result in sorted(verified.items()):
        found = False
        for root_name in FORECAST_ROOTS:
            folder = base / root_name / provider / fid
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.json")):
                found = True
                if diagnostics["files_checked"] >= max_count:
                    diagnostics["truncated"] = True
                    break
                diagnostics["files_checked"] += 1
                try:
                    forecast = _read(path)
                    if not isinstance(forecast, dict) or (
                            forecast.get("forecast_id") and
                            forecast["forecast_id"] != path.stem):
                        diagnostics["skipped_records"] += 1
                        continue
                    candidate = _valid_forecast(forecast, provider, fid, result)
                    if candidate is None:
                        diagnostics["skipped_records"] += 1
                        continue
                    model, stamp, detail = candidate
                    row = records.setdefault((provider, fid), {
                        "fixture_id": provider + ":" + fid,
                        "kickoff": result["kickoff_utc"],
                        "actual": [result["home_goals"], result["away_goals"]],
                        "forecasts": {},
                    })
                    old = row["forecasts"].get(model)
                    if old is None or stamp < _time(old["created_at"]):
                        row["forecasts"][model] = detail
                    diagnostics["frozen_forecasts"] += 1
                except (OSError, ValueError, TypeError, UnicodeError):
                    diagnostics["unreadable_files"] += 1
            if diagnostics["truncated"]:
                break
        if not found:
            diagnostics["missing_forecasts"] += 1
        if diagnostics["truncated"]:
            break
    rows = list(records.values())
    diagnostics["compatible_records"] = len(rows)
    return rows, diagnostics


def write_summary(root=".", output="data/benchmark/summary.json"):
    """Deterministic, non-destructive report; no provider/API calls required."""
    rows, diagnostics = load_archive(root)
    report = {
        "schema_version": 1,
        "source": "immutable_pre_kickoff_and_independent_verified_finals",
        "diagnostics": diagnostics,
        "models": benchmark(rows)["models"],
        "paired": paired_benchmark(rows),
        "five_double_accuracy_not_inferred": True,
        "model_promotion": "none",
    }
    path = Path(root) / output
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != rendered:
        path.write_text(rendered, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--report", default="data/benchmark/summary.json")
    args = parser.parse_args()
    output = write_summary(args.root, args.report)
    print(json.dumps(output["diagnostics"], sort_keys=True))
