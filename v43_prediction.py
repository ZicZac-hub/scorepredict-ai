"""One experimental v43 distribution per fixture, using qualified pre-match evidence."""

from datetime import datetime, timedelta, timezone
import math

import numpy as np

from forecast_ledger import LedgerValidationError, read_file_records
from history_pipeline import build_evidence
from professional_analyst_shadow_v25 import MAX_EARLY_SOURCE_AGE
from professional_double_optimizer import optimal_five
from research_ensemble_v8 import predict as predict_v8
from smart_double_ai import MIN_LEAD_SECONDS, pair_issue
from v10_prospective import ROOT as SOURCE_ROOT, utc
from v21_goal_model_comparison import top_scores
from v22_evidence_models import calibrate_markets, combine_goal_evidence
from verified_features_v10 import describe as history_context
from v41_unified_ensemble import WEIGHTS, core, normalized
from v42_research_engine import ensemble as v42_ensemble
from v42_research_trial import PROVIDER_ROOT, _components


VERSION = "v43-integrated-research-v1"
V42_ENSEMBLE_WEIGHT = 0.25
SOURCE_KEYS = ("provider", "fixture_id", "forecast_id", "input_digest",
               "home_team_id", "away_team_id", "kickoff_utc")
EVIDENCE_KINDS = ("xg", "market", "lineup")


class _ForecastUnavailable(ValueError):
    pass


def _abstain(reason):
    return {"version": VERSION, "status": "abstained", "reason": reason,
            "release_status": "experimental", "model_promoted": False}


def _apply_verified_evidence(grids, bundles):
    current = [core(grid) for grid in grids]
    used = {"xg": False, "market": False, "lineup": False}
    for kind in ("xg", "market"):
        evidence = [bundle.get(kind) for bundle in bundles]
        if any(item is None for item in evidence):
            continue
        if kind == "xg":
            updates = [combine_goal_evidence(grid, item["value"], weight=.25)
                       for grid, item in zip(current, evidence)]
        else:
            updates = [calibrate_markets(grid, item["value"], strength=.35, max_move=.15)
                       for grid, item in zip(current, evidence)]
        if all(item.get("ready") for item in updates):
            current = [normalized(item["matrix"], grid.shape)
                       for grid, item in zip(current, updates)]
            used[kind] = True
    return current, used


def _combine(grids, bundles):
    try:
        controls, used = _apply_verified_evidence(grids, bundles)
        matrices = [normalized((1 - V42_ENSEMBLE_WEIGHT) * grid
                               + V42_ENSEMBLE_WEIGHT * v42_ensemble(grid), grid.shape)
                    for grid in controls]
        choices = optimal_five(*matrices) if len(matrices) == 2 else []
        if len(matrices) == 2 and len(choices) != 5:
            return _abstain("Five distinct positive-probability combinations are unavailable.")
        sources = [{kind: bundle[kind]["reference"] if bundle.get(kind) else None
                    for kind in EVIDENCE_KINDS} for bundle in bundles]
    except (KeyError, ValueError, TypeError, AttributeError, ArithmeticError):
        return _abstain("The source distributions or evidence transforms failed validation.")

    return {
        "version": VERSION, "status": "issued", "release_status": "experimental",
        "model_promoted": False,
        "full_score_grids": [matrix.tolist() for matrix in matrices],
        "five_doubles": choices,
        "estimated_coverage": math.fsum(row["joint_probability"] for row in choices) if choices else None,
        "lineage": {
            "v41_weight": .90,
            "v42_hypothesis_weight": .10,
            "v42_hypothesis_weights": {name: .025 for name in
                                       ("low_goal", "high_goal", "wide_margin", "upset_sensitivity")},
            "v41_family_weights": dict(WEIGHTS),
            "evidence_used": used,
            "component_sources": sources,
            "pair_evidence_policy": (
                "Both fixtures must qualify for an xG or market update at the common cutoff."
                if len(matrices) == 2 else "The fixture must qualify for an xG or market update at the cutoff."),
            "numerical_ai_weight": 0.0,
            "numerical_lineup_weight": 0.0,
            "double_assumption": "Independent matches, mutually exclusive full-time score combinations.",
        },
    }


def combine_grids(grids, bundles):
    """Blend two V8 grids with V42-style, already qualified component bundles.

    The same V41 control underlies the four V42 perturbations, so their total
    family weight is bounded. The archive/live adapters authenticate cutoff and
    provenance before invoking this numerical function.
    """
    if len(grids) != 2 or len(bundles) != 2:
        return _abstain("The doubles calculation requires exactly two source grids and evidence bundles.")
    return _combine(grids, bundles)


def _fixture_identity(row, cutoff):
    fixture = row["fixture"]
    kickoff = utc(fixture["utcDate"])
    if row.get("kickoff") is not None and utc(row["kickoff"]) != kickoff:
        raise _ForecastUnavailable("The displayed kickoff disagrees with the provider fixture.")
    if (kickoff - cutoff <= timedelta(seconds=MIN_LEAD_SECONDS)
            or fixture.get("status") not in (None, "SCHEDULED", "TIMED")
            or fixture.get("provider_status") not in (None, "", "NS", "TBD")):
        raise _ForecastUnavailable("Choose upcoming fixtures more than five minutes from kickoff.")
    provider = str(fixture.get("provider") or "football-data.org")
    fields = [fixture.get("id"), fixture["homeTeam"].get("id"), fixture["awayTeam"].get("id")]
    if (any(value in (None, "") or type(value) is bool for value in fields)
            or str(fields[1]) == str(fields[2])):
        raise _ForecastUnavailable("The provider fixture and both teams need distinct, known identifiers.")
    return {"provider": provider, "fixture_id": str(fields[0]),
            "home_team_id": str(fields[1]), "away_team_id": str(fields[2]),
            "kickoff_utc": kickoff.isoformat()}


def _matching_archives(fixture, cutoff, source_root):
    records = read_file_records(root=source_root, provider=fixture["provider"],
                                fixture_id=fixture["fixture_id"])
    return [record for record in records if record.get("model_key") in ("v8", "v8_near")
            and record.get("score_basis", "regulation") == "regulation"
            and all(str(record[key]) == fixture[key] for key in
                    ("provider", "fixture_id", "home_team_id", "away_team_id"))
            and utc(record["kickoff_utc"]) == utc(fixture["kickoff_utc"])
            and timedelta(0) <= cutoff - utc(record["captured_at_utc"]) <= MAX_EARLY_SOURCE_AGE]


def _same_reference(record, reference):
    if not isinstance(reference, dict) or any(reference.get(key) is None for key in SOURCE_KEYS):
        return False
    return (all(str(record[key]) == str(reference[key]) for key in SOURCE_KEYS if key != "kickoff_utc")
            and utc(record["kickoff_utc"]) == utc(reference["kickoff_utc"]))


def _source_bundle(records, cutoff, provider_root):
    selected = {kind: None for kind in EVIDENCE_KINDS}
    for record in records:
        components = _components(record, provider_root=provider_root, cutoff=cutoff)
        for kind, evidence in components.items():
            if evidence is None:
                continue
            reference = evidence["reference"]
            previous = selected[kind]
            ordering = (utc(reference["captured_at_utc"]), reference["snapshot_sha256"])
            if previous is None or ordering > previous["ordering"]:
                selected[kind] = {"value": evidence["value"],
                                  "reference": {**reference, "source_forecast_id": record["forecast_id"]},
                                  "ordering": ordering}
    return selected


def _baseline(row, histories, cutoff, *, source_root, provider_root):
    identity = _fixture_identity(row, cutoff)
    fixture = row["fixture"]
    models = _matching_archives(identity, cutoff, source_root)
    if row.get("archive_frozen"):
        reference = row.get("source_forecast")
        archived = next((model for model in models if _same_reference(model, reference)), None)
        if archived is None or utc(row["frozen_at_utc"]) != utc(archived["captured_at_utc"]):
            raise _ForecastUnavailable("The selected pre-match forecast could not be verified in the original ledger.")
        baseline = archived["score_matrix"]
        context = archived.get("research_context") or {}
        metadata = {"baseline_model": archived["model_version"],
                    "baseline_capture_utc": archived["captured_at_utc"],
                    "source_forecast_id": archived["forecast_id"],
                    "input_digest": archived["input_digest"],
                    "history_matches": archived.get("history_matches"),
                    "league_history_matches": archived.get("league_history_matches"),
                    "baseline_components": archived.get("component_weights") or {}}
    else:
        packet = row.get("evidence") or {}
        prior = row.get("past") or packet.get("history") or []
        league = row.get("league_history") or packet.get("league_history") or []
        additional = (histories or {}).get(row.get("league")) or []
        evidence = build_evidence(fixture, [*prior, *league, *additional], cutoff)
        qualified = evidence["history"]
        live = predict_v8(evidence["fixture"], qualified, league_history=qualified, as_of=cutoff)
        if not live or len(live.get("component_weights", {})) < 2:
            raise _ForecastUnavailable("There is insufficient verified V8 history for the selected fixture.")
        baseline = live["matrix"]
        context = history_context(evidence["fixture"], evidence)
        metadata = {"baseline_model": live["model_version"],
                    "baseline_capture_utc": cutoff.isoformat(), "source_forecast_id": None,
                    "input_digest": evidence["input_digest"], "history_matches": len(qualified),
                    "league_history_matches": len(qualified),
                    "baseline_components": live["component_weights"]}
    bundle = _source_bundle(models, cutoff, provider_root)
    return baseline, bundle, {
        **identity, **metadata,
        "home_team": fixture["homeTeam"].get("name", ""),
        "away_team": fixture["awayTeam"].get("name", ""),
        "competition": (fixture.get("competition") or {}).get("code") or row.get("league", ""),
        "home_history_matches": (context.get("home") or {}).get("finished_matches"),
        "away_history_matches": (context.get("away") or {}).get("finished_matches"),
    }


def _fixture_model(metadata, grid, sources):
    matrix = np.asarray(grid)
    home, away = np.indices(matrix.shape)
    return {
        **metadata,
        "expected_goals": {"home": float((matrix * home).sum()), "away": float((matrix * away).sum())},
        "model_probabilities": {"home": float(matrix[home > away].sum()),
                                "draw": float(matrix[home == away].sum()),
                                "away": float(matrix[home < away].sum())},
        "top_scores": [{"score": f"{a}-{b}", "probability": p}
                       for a, b, p in top_scores(matrix) if p > 0],
        "evidence_sources": sources,
    }


def _forecast(rows, histories, as_of, source_root, provider_root, same_day):
    try:
        cutoff = utc(as_of if as_of is not None else datetime.now(timezone.utc))
        if len(rows) == 2 and (issue := pair_issue(*rows, now=cutoff, same_day=same_day)):
            return _abstain(issue)
        pieces = [_baseline(row, histories, cutoff, source_root=source_root, provider_root=provider_root)
                  for row in rows]
        grids, bundles, metadata = zip(*pieces)
        result = _combine(grids, bundles)
    except LedgerValidationError:
        return _abstain("An original fixture forecast record failed ledger verification.")
    except _ForecastUnavailable as exc:
        return _abstain(str(exc))
    except (OSError, KeyError, TypeError, ValueError, AttributeError, ArithmeticError):
        return _abstain("Pre-match fixture history or source provenance is incomplete.")
    if result["status"] != "issued":
        return result
    finished = datetime.now(timezone.utc) if as_of is None else cutoff
    if any(utc(item["kickoff_utc"]) - finished <= timedelta(seconds=MIN_LEAD_SECONDS) for item in metadata):
        return _abstain("The calculation finished outside the pre-match window.")
    result["decision_as_of_utc"] = cutoff.isoformat()
    result["fixture_models"] = [_fixture_model(item, grid, sources) for item, grid, sources in
                                zip(metadata, result["full_score_grids"], result["lineage"]["component_sources"])]
    return result


def forecast_pair(first, second, histories=None, *, as_of=None, same_day=True,
                  source_root=SOURCE_ROOT, provider_root=PROVIDER_ROOT):
    """Generate one V43 research result for the selected distinct future pair."""
    return _forecast((first, second), histories, as_of, source_root, provider_root, same_day)


def forecast_single(row, histories=None, *, as_of=None, source_root=SOURCE_ROOT,
                    provider_root=PROVIDER_ROOT):
    """Generate a single-forecast research result when only one fixture is selected."""
    return _forecast((row,), histories, as_of, source_root, provider_root, False)
