"""Chronological single-fixture calibration on the verified V44 inventory.

Research only: no historical pair creation, no production weights or API calls.
Sixty previously verified individual fixtures are required for the learned arm.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np

from forecast_ledger import _forecast, LedgerValidationError
from v10_prospective import utc
import v42_research_trial as sources
import v44_fixture_evidence as inventory
from v44_authenticated_replay import _groups
from v44_family_pool_research import _grid, _weights
from v43_prediction import _combine

MIN_TRAIN_FIXTURES = 60


def _prob(p, actual):
    h, a = actual
    return float(p[h, a]) if h < p.shape[0] and a < p.shape[1] else 0.


def _metrics(p, actual):
    probability = _prob(p, actual)
    h, a = actual
    i, j = np.indices(p.shape)
    observed = [h > a, h == a, h < a]
    outcome = [p[i > j].sum(), p[i == j].sum(), p[i < j].sum()]
    return {"exact_log_loss": -math.log(max(probability, 1e-15)),
            "score_brier": float(1 + np.square(p).sum() - 2*probability),
            "outcome_brier": float(sum((v-o)**2 for v,o in zip(outcome, observed))),
            "top_score_hit": bool(tuple(np.unravel_index(p.argmax(), p.shape)) == tuple(actual))}


def _single_groups(base):
    variants = _groups([base])
    return {name: np.mean([_grid(legs[0], base.shape) for legs in methods.values()], axis=0)
            for name, methods in sorted(variants.items())}


def walk_forward(records):
    """Records are validated by evaluate; only earlier known finals train."""
    history, output, seen = [], [], set()
    for row in sorted(records, key=lambda r: (utc(r["decision_utc"]), r["forecast_id"])):
        decision = utc(row["decision_utc"])
        kickoff, known = utc(row["kickoff_utc"]), utc(row["final_known_utc"])
        identity = tuple(row["event"])
        if not decision < kickoff < known or identity in seen:
            raise ValueError("invalid or duplicate fixture")
        seen.add(identity)
        base = _grid(row["grid"])
        actual = row["actual"]
        if len(actual) != 2 or any(type(v) is not int or v < 0 for v in actual):
            raise ValueError("invalid actual")
        groups = _single_groups(base)
        grids = np.stack(list(groups.values()))
        prior = [lik for available, lik in history if available < decision]
        equal = np.full(len(groups), 1 / len(groups))
        weights = _weights(prior, len(groups)) if len(prior) >= MIN_TRAIN_FIXTURES else equal
        v43 = _combine([base], [{}])
        if v43["status"] != "issued":
            raise ValueError("V43 single control unavailable")
        output.append({"forecast_id": row["forecast_id"], "decision_utc": row["decision_utc"],
                       "training_fixtures": len(prior), "weights": dict(zip(groups, map(float, weights))),
                       "learned": _metrics(np.tensordot(weights, grids, axes=1), actual),
                       "equal": _metrics(np.tensordot(equal, grids, axes=1), actual),
                       "v8": _metrics(base, actual),
                       "v43_replay": _metrics(_grid(v43["full_score_grids"][0], base.shape), actual)})
        history.append((known, np.array([[_prob(p, actual) for p in grids]])))
    eligible = [r for r in output if r["training_fixtures"] >= MIN_TRAIN_FIXTURES]
    summary = {}
    for name in ("learned", "equal", "v8", "v43_replay"):
        metrics = [r[name] for r in eligible]
        n = len(metrics)
        summary[name] = {"fixtures": n,
                         **{k: math.fsum(m[k] for m in metrics)/n if n else None
                            for k in ("exact_log_loss", "score_brier", "outcome_brier", "top_score_hit")}}
    return {"verified_input_fixtures": len(output), "eligible_comparison_fixtures": len(eligible),
            "warmup_fixtures": MIN_TRAIN_FIXTURES, "comparisons": summary, "rows": output}


def evaluate(*, source_root=sources.SOURCE_ROOT, prematch_root=sources.PREMATCH_ROOT,
             provider_root=sources.PROVIDER_ROOT, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    report = inventory.evaluate(source_root=source_root, prematch_root=prematch_root,
                                provider_root=provider_root, now=clock)
    records, rejects = [], Counter()
    for row in report["rows"]:
        try:
            path = Path(source_root)/"forecasts"/row["provider"]/row["fixture_id"]/(row["forecast_id"]+".json")
            model = _forecast(json.loads(path.read_text(encoding="utf-8")))
            if (model["forecast_id"] != row["forecast_id"] or
                    utc(model["captured_at_utc"]) != utc(row["decision_utc"]) or
                    utc(model["kickoff_utc"]) != utc(row["kickoff_utc"]) or
                    inventory._identity(model) != (row["provider"], row["fixture_id"]) or
                    list(inventory._event(model)) != row["event"]):
                raise ValueError("inventory_source_changed")
            base = _grid(model["score_matrix"])
            _single_groups(base)
            if _combine([base], [{}])["status"] != "issued":
                raise ValueError("v43_control_unavailable")
            records.append({**row, "grid": base})
        except (ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError, LedgerValidationError) as exc:
            rejects[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    output = walk_forward(records)
    return {**output, "study_type": "retrospective_single_fixture_walk_forward",
            "evaluated_at_utc": clock.isoformat(), "inventory_matches": report["matched_verified_fixtures"],
            "source_rejections": dict(rejects), "external_requests": 0,
            "double_pairs_created": 0, "automatic_promotion": False,
            "limitations": ["Single fixture metrics are not five-double hit rates.",
                            "Correlated V8 transforms, not independently trained models.",
                            "No new xG, market or player effects; V43 control is statistical only.",
                            "Retrospective research must be prospectively validated before use."]}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
