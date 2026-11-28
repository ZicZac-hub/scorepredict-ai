"""Read-only, point-in-time V45 evidence readiness for the archived match slate."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from forecast_ledger import _forecast
from v10_prospective import utc
from v42_research_trial import _components
from v44_collection_health import evaluate as collection_health, read
from v44_live_evidence_diagnostics import history_status
from v45_lineups import status_for
from v45_provider_health import archived_health

DATA = Path("data")
WEIGHTS = {"history": 35, "xg": 25, "market": 20, "lineup": 20}
QUALIFIED = {"history": "ready", "xg": "qualified", "market": "qualified", "lineup": "verified"}
LOOKAHEAD = timedelta(hours=24)
LINEUP_WINDOW = timedelta(minutes=60)


def fixture_quality(model, *, cutoff, root=DATA):
    """Award points for authenticated evidence available by the chosen cutoff."""
    root = Path(root)
    cutoff = utc(cutoff)
    kickoff = utc(model["kickoff_utc"])
    if not utc(model["captured_at_utc"]) <= cutoff < kickoff:
        raise ValueError("A cutoff from forecast capture to before kickoff is required")
    history = history_status(model, cutoff, root / "v44_early_history")
    bundles = _components(model, provider_root=root / "v21_sources", cutoff=cutoff)
    lineup = status_for(model, root=root / "v45_intelligence", cutoff=cutoff)
    lineup_state = lineup["status"]
    if lineup_state == "missing" and kickoff - cutoff > LINEUP_WINDOW:
        lineup_state = "window_not_open"
    statuses = {
        "history": history,
        "xg": "qualified" if bundles["xg"] else "missing_or_stale",
        "market": "qualified" if bundles["market"] else "missing_or_stale",
        "lineup": lineup_state,
    }
    points = {key: weight if statuses[key] == QUALIFIED[key] else 0 for key, weight in WEIGHTS.items()}
    captured = {key: item["reference"]["captured_at_utc"]
                for key in ("xg", "market") if (item := bundles[key])}
    if lineup.get("observed_at_utc"):
        captured["lineup"] = lineup["observed_at_utc"]
    return {
        "score": sum(points.values()),
        "points": points,
        "statuses": statuses,
        "evidence_captured_utc": captured,
    }


def evaluate(*, root=DATA, now=None):
    root = Path(root)
    clock = utc(now or datetime.now(timezone.utc))
    selected = {}
    conflicts = set()
    exclusions = Counter()
    for path in sorted((root / "v10_prospective" / "forecasts").glob("*/*/*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("model_key") not in {"v8", "v8_near"}:
                continue
            kickoff = utc(raw["kickoff_utc"])
            if not timedelta(0) < kickoff - clock <= LOOKAHEAD:
                continue
            model = _forecast(raw)
            key = model["provider"], model["fixture_id"]
            if (path.parent.parent.name, path.parent.name, path.stem) != (*key, raw.get("forecast_id")):
                raise ValueError("Archive identity mismatch")
            captured = utc(model["captured_at_utc"])
            if captured > clock or model.get("score_basis", "regulation") != "regulation":
                raise ValueError("Forecast unavailable by cutoff")
            previous = selected.get(key)
            if previous is not None:
                event = (model["kickoff_utc"], model["home_team_id"], model["away_team_id"])
                if event != tuple(previous[0][field] for field in ("kickoff_utc", "home_team_id", "away_team_id")):
                    conflicts.add(key)
                if captured <= previous[1]:
                    continue
            selected[key] = (model, captured)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
            exclusions["invalid_archive_entry"] += 1

    rows = []
    for key, (model, _) in sorted(selected.items(), key=lambda item: (item[1][0]["kickoff_utc"], item[0])):
        if key in conflicts:
            continue
        try:
            quality = fixture_quality(model, cutoff=clock, root=root)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
            exclusions["invalid_quality_evidence"] += 1
            continue
        rows.append({"provider": key[0], "fixture_id": key[1], "kickoff_utc": model["kickoff_utc"],
                     "home": model.get("home_team"), "away": model.get("away_team"),
                     "competition": model.get("competition"), **quality})

    return {
        "report_type": "v45_information_quality",
        "as_of_utc": clock.isoformat(),
        "lookahead_hours": 24,
        "weights": WEIGHTS,
        "fixtures": rows,
        "conflicting_fixture_identities": len(conflicts),
        "invalid_archive_entries": exclusions["invalid_archive_entry"],
        "invalid_quality_evidence": exclusions["invalid_quality_evidence"],
        "collection": collection_health(root=root, now=clock),
        "sportmonks": archived_health(root=root),
        "sportmonks_probe": read(root / "v45_intelligence" / "sportmonks_probe.json"),
        "lineup_collection": read(root / "v45_intelligence" / "lineups" / "status.json"),
        "automatic_market_capture": read(root / "v45_market_capture" / "status.json"),
        "automatic_market_trial": read(root / "v45_market_trial" / "status.json"),
        "external_requests": 0,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
