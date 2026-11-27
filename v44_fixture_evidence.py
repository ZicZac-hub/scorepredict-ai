"""Inventory individual verified pre-match fixtures for V44 research.

Never manufacture double pairs, call providers or mutate archives.
Forecast availability is not established by a final result alone.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from five_double_accuracy import _final, _result_index
from forecast_ledger import _forecast, LedgerValidationError
from v10_prospective import utc
import v42_research_trial as sources

MAX_LEAD = timedelta(hours=24)
MIN_LEAD = timedelta(minutes=5)


def _identity(row):
    return str(row["provider"]), str(row["fixture_id"])


def _event(row):
    names = tuple(" ".join(row[k].casefold().split()) for k in ("home_team", "away_team"))
    if not all(names) or names[0] == names[1]:
        raise ValueError("missing_distinct_team_names")
    return utc(row["kickoff_utc"]).isoformat(), *names


def evaluate(*, source_root=sources.SOURCE_ROOT, prematch_root=sources.PREMATCH_ROOT,
             provider_root=sources.PROVIDER_ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    finals, conflicts, result_rejections = _result_index(source_root, prematch_root)
    paths = sorted((Path(source_root) / "forecasts").glob("*/*/*.json"))
    rejected, excluded, components = Counter(), Counter(), Counter()
    candidates = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("model_key") not in ("v8", "v8_near"):
                excluded["other_model"] += 1
                continue
            if not raw.get("forecast_id") or not raw.get("input_digest"):
                raise ValueError("missing_original_identity")
            model = _forecast(raw)
            if (path.parent.parent.name, path.parent.name, path.stem) != (*_identity(model), model["forecast_id"]):
                raise ValueError("archive_path_identity_mismatch")
            if model.get("score_basis", "regulation") != "regulation":
                raise ValueError("non_regulation_forecast")
            decision, kickoff = utc(model["captured_at_utc"]), utc(model["kickoff_utc"])
            if decision > clock or not MIN_LEAD < kickoff-decision <= MAX_LEAD:
                excluded["outside_fixed_capture_window"] += 1
                continue
            event = _event(model)
            candidates.append((decision, model["forecast_id"], event, model))
        except (ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError, LedgerValidationError) as exc:
            rejected[type(exc).__name__] += 1
    # Choose earliest eligible capture, irrespective of result or component availability.
    seen_events, seen_ids, rows = set(), set(), []
    pending = missing_final = 0
    for decision, _, event, model in sorted(candidates, key=lambda r: (r[0], r[1])):
        identity = _identity(model)
        if identity in seen_ids or event in seen_events:
            excluded["duplicate_fixture_capture_or_event"] += 1
            continue
        seen_ids.add(identity)
        seen_events.add(event)
        result = finals.get(identity)
        actual = _final(model, result, clock)
        if actual is None:
            if utc(model["kickoff_utc"]) > clock:
                pending += 1
            else:
                missing_final += 1
            continue
        availability = {}
        try:
            bundle = sources._components(model, provider_root=provider_root, cutoff=decision)
            for kind in ("xg", "market", "lineup"):
                item = bundle.get(kind)
                availability[kind] = item is not None
                components[kind] += int(availability[kind])
        except (ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError):
            components["component_validation_failed"] += 1
            availability = {"xg": None, "market": None, "lineup": None}
        rows.append({"provider": identity[0], "fixture_id": identity[1],
                     "event": list(event), "forecast_id": model["forecast_id"],
                     "decision_utc": decision.isoformat(),
                     "kickoff_utc": utc(model["kickoff_utc"]).isoformat(),
                     "final_known_utc": utc(result["verified_at_utc"]).isoformat(),
                     "actual": list(actual), "components_available": availability})
    matched = {(r["provider"], r["fixture_id"]) for r in rows}
    return {"study_type": "single_fixture_evidence_inventory",
            "evaluated_at_utc": clock.isoformat(), "archive_files": len(paths),
            "unique_eligible_forecast_fixtures": len(seen_ids),
            "unique_verified_result_identities": len(finals),
            "conflicting_result_identities": len(conflicts),
            "matched_verified_fixtures": len(rows),
            "pending_kickoff": pending, "past_without_verified_final": missing_final,
            "result_identities_without_selected_match": len(set(finals)-matched),
            "component_availability_counts": dict(components),
            "forecast_rejections": dict(rejected), "exclusions": dict(excluded),
            "result_rejections": dict(result_rejections), "rows": rows,
            "external_requests": 0, "double_pairs_created": 0,
            "automatic_promotion": False,
            "limitations": ["One original V8-family capture per provider fixture and exact normalized event.",
                            "Cross-provider team aliases may still need independent reconciliation.",
                            "Individual fixtures are not verified five-double trials.",
                            "A matching lineup receipt does not establish numerical player impact."]}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
