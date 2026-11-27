"""Fixed-cutoff, read-only investigation of missing pre-match evidence.

Compare earliest archived capture with a new retrospective observation at
15 minutes before kickoff. No final score is used to select evidence.
No source relabeling, API calls, model changes, or retrospective tickets.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from forecast_ledger import _forecast, LedgerValidationError
from v10_prospective import utc
import v42_research_trial as sources
import v44_fixture_evidence as inventory

LEAD = timedelta(minutes=15)
KINDS = ("xg", "market", "lineup")


def _candidate_models(row, source_root, cutoff):
    path = Path(source_root)/"forecasts"/row["provider"]/row["fixture_id"]
    models, rejected = [], 0
    for file in sorted(path.glob("*.json")):
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
            if raw.get("model_key") not in ("v8", "v8_near"):
                continue
            if not raw.get("forecast_id") or not raw.get("input_digest"):
                raise ValueError("missing original reference")
            model = _forecast(raw)
            captured = utc(model["captured_at_utc"])
            if (file.stem != model["forecast_id"] or
                    inventory._identity(model) != (row["provider"], row["fixture_id"]) or
                    list(inventory._event(model)) != row["event"] or
                    model.get("score_basis", "regulation") != "regulation"):
                raise ValueError("source identity mismatch")
            if timedelta(0) <= cutoff-captured and utc(model["kickoff_utc"])-captured <= inventory.MAX_LEAD:
                models.append(model)
        except (ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError, LedgerValidationError):
            rejected += 1
    return models, rejected


def _select(models, provider_root, cutoff):
    """Use each packet only against its own original forecast identity."""
    found = {kind: None for kind in KINDS}
    errors = 0
    for model in models:
        try:
            bundle = sources._components(model, provider_root=provider_root, cutoff=cutoff)
            for kind in KINDS:
                item = bundle.get(kind)
                if item is None:
                    continue
                ref = item["reference"]
                order = (utc(ref["captured_at_utc"]), ref["snapshot_sha256"], model["forecast_id"])
                if order[0] > cutoff:
                    raise ValueError("post-cutoff evidence")
                if found[kind] is None or order > found[kind][0]:
                    found[kind] = (order, {**ref, "source_forecast_id": model["forecast_id"],
                                           "both_confirmed_xis": bool(item["value"].get("ready"))
                                           if kind == "lineup" else None})
        except (ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError):
            errors += 1
    return {k: v[1] if v else None for k,v in found.items()}, errors


def evaluate(*, source_root=sources.SOURCE_ROOT, prematch_root=sources.PREMATCH_ROOT,
             provider_root=sources.PROVIDER_ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    early = inventory.evaluate(source_root=source_root, prematch_root=prematch_root,
                               provider_root=provider_root, now=clock)
    counts, gaps, rows = Counter(), Counter(), []
    for row in early["rows"]:
        cutoff = utc(row["kickoff_utc"])-LEAD
        if utc(row["decision_utc"]) > cutoff:
            gaps["no_original_capture_before_fixed_cutoff"] += 1
            continue
        models, rejected = _candidate_models(row, source_root, cutoff)
        gaps["invalid_candidate_records"] += rejected
        if not models:
            gaps["no_eligible_source"] += 1
            continue
        refs, errors = _select(models, provider_root, cutoff)
        gaps["component_validation_errors"] += errors
        counts["common_fixtures"] += 1
        for kind in KINDS:
            before = row["components_available"].get(kind)
            after = refs[kind] is not None
            counts[kind+"_early"] += int(before is True)
            counts[kind+"_late"] += int(after)
            counts[kind+"_gained"] += int(before is False and after)
            counts[kind+"_lost"] += int(before is True and not after)
            counts[kind+"_early_unknown"] += int(before is None)
        counts["late_both_confirmed_xis"] += int(bool(refs["lineup"] and refs["lineup"]["both_confirmed_xis"]))
        rows.append({"fixture_id": row["fixture_id"], "provider": row["provider"],
                     "cutoff_utc": cutoff.isoformat(), "candidate_forecasts": len(models),
                     "early_availability": row["components_available"], "late_references": refs})
    return {"study_type":"fixed_15_minute_evidence_availability_replay",
            "evaluated_at_utc":clock.isoformat(), "inventory_matched":early["matched_verified_fixtures"],
            "counts":dict(counts), "gaps":dict(gaps), "rows":rows,
            "external_requests":0, "forecasts_created":0, "automatic_promotion":False,
            "limitations":["A later cutoff is not the original prediction decision.",
                           "The component validator retains existing freshness and identity requirements.",
                           "Missing receipts are not recovered from post-match observations.",
                           "This measures availability, not numerical player impact or accuracy."]}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
