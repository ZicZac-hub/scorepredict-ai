"""V37 sparse European-score challenger, frozen research only.

For CL and EL fixtures with 2-4 authenticated same-competition matches for
at least one team, preserve a league-only Poisson control and a prespecified
five-pseudo-match shrunk team-strength challenger on the EXACT same source.
Never use historical results available after forecast time. No provider calls,
no V8 threshold changes, no live predictions or V28 ticket eligibility.
"""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np

import engine
from research_ensemble_v8 import _asof_history

VERSION = "v37-sparse-uefa-five-prior-prospective-research-v1"
ELIGIBLE = frozenset(("CL", "EL"))
MIN_TEAM_GAMES = 2
V8_TEAM_MINIMUM = 5
MIN_LEAGUE_GAMES = 35
MIN_UNRELATED_GAMES = 12
PRIOR_MATCHES = 5.0
MAX_GOALS = 10


def _clock(value):
    instant = engine._date(value)
    if instant is None or instant.tzinfo is None:
        raise ValueError("timezone_aware_source_required")
    return instant.astimezone(timezone.utc)


def _summary(matrix, label, components, control):
    x, y = np.indices(matrix.shape)
    order = np.argsort(-matrix.ravel(), kind="stable")[:10]
    return {
        "model_version": VERSION,
        "matrix": matrix,
        "scores": [(int(i//matrix.shape[1]), int(i%matrix.shape[1]),
                    float(matrix.ravel()[i])) for i in order],
        "home_xg": float(np.sum(matrix*x)),
        "away_xg": float(np.sum(matrix*y)),
        "component_weights": components,
        "v37_control_grid": control.tolist(),
        "v37_method": label,
        "v37_not_a_v8_forecast": True,
        "v37_research_only": True,
    }


def predict(fixture, history, league_history=None, *, as_of):
    """Return NONE if same-competition sparse-history evidence is inadequate."""
    if (fixture.get("provider") != "API-Football" or
            (fixture.get("competition") or {}).get("code") not in ELIGIBLE):
        return None
    cutoff = _clock(as_of)
    kickoff = _clock(fixture["utcDate"])
    if cutoff >= kickoff:
        return None
    home = (fixture.get("homeTeam") or {}).get("id")
    away = (fixture.get("awayTeam") or {}).get("id")
    if home is None or away is None or home == away:
        return None
    source = history if league_history is None else league_history
    team_prior = _asof_history(fixture, history, cutoff)
    population = _asof_history(fixture, source, cutoff)
    if len(population) < MIN_LEAGUE_GAMES:
        return None
    others = sum(
        home not in (m["homeTeam"]["id"],m["awayTeam"]["id"]) and
        away not in (m["homeTeam"]["id"],m["awayTeam"]["id"])
        for m in population
    )
    if others < MIN_UNRELATED_GAMES:
        return None
    home_games = engine._games(team_prior, home, limit=10)
    away_games = engine._games(team_prior, away, limit=10)
    home_n, away_n = len(home_games), len(away_games)
    if min(home_n,away_n) < MIN_TEAM_GAMES or min(home_n,away_n) >= V8_TEAM_MINIMUM:
        return None
    base_h, base_a = engine.league_baselines(population)
    control = engine._grid(base_h, base_a, max_goals=MAX_GOALS)
    h_stats = engine._stats(home_games, cutoff)
    a_stats = engine._stats(away_games, cutoff)

    def shrink(rate, prior, count):
        return (PRIOR_MATCHES*prior + count*rate)/(PRIOR_MATCHES+count)

    home_for = shrink(h_stats["scored"],base_h,home_n)
    away_against = shrink(a_stats["conceded"],base_h,away_n)
    away_for = shrink(a_stats["scored"],base_a,away_n)
    home_against = shrink(h_stats["conceded"],base_a,home_n)
    mu_h = max(.25, (home_for+away_against)/2)
    mu_a = max(.25, (away_for+home_against)/2)
    challenger = engine._grid(mu_h,mu_a,max_goals=MAX_GOALS)
    return _summary(challenger,"sparse_shrunk_five_pseudo_matches",
                    {"league_prior":.5,"verified_sparse_team_rates":.5},
                    control)


def frozen_arm_digest(row):
    """Bind both research arms and original evidence identity together."""
    from forecast_ledger import _utc
    source = {
        "version": VERSION,
        "provider": row["provider"], "fixture_id": row["fixture_id"],
        "kickoff_utc": _utc(row["kickoff_utc"], "kickoff_utc"),
        "evidence_as_of_utc": _utc(row["evidence_as_of_utc"],
                                   "evidence_as_of_utc"),
        "input_digest": row["input_digest"],
        "score_matrix": row["score_matrix"],
        "v37_control_grid": row["v37_control_grid"],
    }
    return hashlib.sha256(json.dumps(
        source,sort_keys=True,separators=(",",":"),allow_nan=False
    ).encode("utf-8")).hexdigest()


def freeze(fixture, evidence, captured_at, *, root=None):
    """Write one independent V37 source per future fixture, at most once.

    This shadow NEVER replaces V8 or writes a V28-eligible model key. If the
    source is insufficient, return a documented abstention and no file.
    """
    from pathlib import Path
    from forecast_ledger import append_file_record, read_file_records
    from v10_prospective import ROOT, _record, utc

    root = ROOT if root is None else Path(root)
    when = utc(captured_at)
    as_of = utc(evidence["as_of_utc"])
    kickoff = utc(fixture["utcDate"])
    if not evidence.get("input_digest") or not (as_of <= when < kickoff):
        raise ValueError("missing_authentic_pre_kickoff_evidence")
    forecast = predict(
        fixture, evidence.get("history") or [],
        evidence.get("league_history") or evidence.get("history") or [],
        as_of=as_of)
    if forecast is None:
        return {"v37_recorded": False, "v37_reason": "sparse_evidence_not_eligible"}
    provider = str(fixture["provider"])
    identifier = str(fixture["id"])
    existing = read_file_records(root, kind="forecasts",provider=provider,fixture_id=identifier)
    if any(row.get("model_key") == "v37_sparse" for row in existing):
        return {"v37_recorded": False, "v37_reason": "original_v37_already_issued"}
    completed = max(when, utc(datetime.now(timezone.utc)))
    if completed >= kickoff:
        return {"v37_recorded": False, "v37_reason": "calculation_after_kickoff"}
    row = _record(fixture,evidence,forecast,"v37_sparse",completed)
    row["v37_control_grid"] = forecast["v37_control_grid"]
    row["v37_research_only"] = True
    row["v37_no_live_model_change"] = True
    row["v37_not_eligible_for_v28"] = True
    row["v37_arm_digest_sha256"] = frozen_arm_digest(row)
    written = append_file_record(row,root=root,kind="forecasts")
    return {
        "v37_recorded": bool(written),
        "v37_reason": "new_pre_kickoff_sparse_model_issued"
            if written else "immutable_source_already_recorded",
    }
