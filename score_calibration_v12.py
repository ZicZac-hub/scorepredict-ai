"""ZicZac v12: pre-match league residual score calibration (SHADOW ONLY).

This is a conservative empirical-Bayes league score-environment adjustment,
not a trained residual model on v8 forecasts, and NOT a demonstrated increase
in accuracy. It never modifies the v8 model, saved originals, or auto doubles.

For each league, compare historical score frequencies with an independent
Poisson reference at that league's observed average home/away goals. Shrink
that ratio toward 1 with pseudocounts, limit its magnitude, and gently adjust
the existing v8 grid. The underlying matchup probabilities remain dominant.

Only provider/competition-matched final scores available BEFORE the forecast
as-of timestamp are eligible. Missing evidence means ABSTAIN, not fabricate.
"""
import math
from datetime import datetime, timezone, timedelta
import numpy as np
import engine
from research_ensemble_v8 import predict as v8_predict, _asof_history

VERSION = "ziczac-v12-league-residual-calibration-experimental"
MIN_MATCHES = 80
MAX_MATCHES = 300
LOOKBACK_DAYS = 730
HALF_LIFE_DAYS = 250.0
PRIOR_EQUIVALENT_MATCHES = 120.0
RATIO_MIN = 0.75
RATIO_MAX = 1.25
ADJUSTMENT_POWER = 0.50


def _league_rows(fixture, league_history, as_of):
    kickoff = engine._date(fixture.get("utcDate"))
    when = engine._date(as_of)
    if kickoff is None or when is None or when >= kickoff:
        return []
    boundary = when - timedelta(days=LOOKBACK_DAYS)
    result = []
    for match in _asof_history(fixture, league_history or [], when):
        played = engine._date(match.get("utcDate"))
        if played is None or played < boundary:
            continue
        if str(match.get("id")) == str(fixture.get("id")) and (
            match.get("provider") == fixture.get("provider")):
            continue
        # The shared as-of function already enforces provider, competition,
        # completed status, fixture kickoff and result availability delay.
        result.append(match)
    result.sort(key=lambda match:engine._date(match["utcDate"]))
    return result[-MAX_MATCHES:]


def _reference(mean_home, mean_away, shape):
    h = np.asarray([engine.poisson(k, mean_home) for k in range(shape[0])], dtype=float)
    a = np.asarray([engine.poisson(k, mean_away) for k in range(shape[1])], dtype=float)
    matrix = np.outer(h, a)
    total = float(matrix.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid league Poisson reference")
    return matrix / total


def calibrate_matrix(base_matrix, rows, *, as_of):
    """Return a separate calibrated score grid + auditable league diagnostics.

    None means insufficient prior, time-qualified observations. No external
    API access, no post-kickoff corrections, and no forced new modal score.
    """
    if len(rows) < MIN_MATCHES:
        return None
    base = np.asarray(base_matrix, dtype=float)
    if (base.ndim != 2 or min(base.shape) < 2 or
        not np.all(np.isfinite(base)) or np.any(base < 0) or
        not np.isclose(float(base.sum()), 1, atol=1e-7)):
        raise ValueError("Expected a normalized full exact-score matrix")
    instant = engine._date(as_of)
    if instant is None:
        raise ValueError("A dated forecast-as-of is required")
    weights = []
    scores = []
    for match in rows:
        d = engine._date(match.get("utcDate"))
        goals = (match.get("score") or {}).get("fullTime") or {}
        h, a = goals.get("home"), goals.get("away")
        if (d is None or d > instant or type(h) is not int or
            type(a) is not int or min(h, a) < 0):
            continue
        age = max(0.0, (instant-d).total_seconds()/86400.0)
        weights.append(2.0**(-age/HALF_LIFE_DAYS))
        scores.append((h, a))
    if len(scores) < MIN_MATCHES:
        return None
    w = np.asarray(weights, dtype=float)
    observed = np.asarray(scores, dtype=int)
    total_w = float(w.sum())
    if total_w <= 0:
        return None
    means = np.average(observed, weights=w, axis=0)
    reference = _reference(float(means[0]), float(means[1]), base.shape)
    counts = np.zeros_like(base)
    outside = 0
    for (h, a), weight in zip(scores, weights):
        if h >= base.shape[0] or a >= base.shape[1]:
            outside += 1
        else:
            counts[h, a] += weight
    # Do not distort the distribution if high scores fall outside the grid.
    if outside:
        return None
    empirical = (counts + PRIOR_EQUIVALENT_MATCHES*reference) / (
        total_w+PRIOR_EQUIVALENT_MATCHES)
    ratio = np.divide(empirical, reference, out=np.ones_like(base),
                      where=reference > 1e-10)
    ratio = np.clip(ratio, RATIO_MIN, RATIO_MAX)
    candidate = base * np.power(ratio, ADJUSTMENT_POWER)
    candidate /= float(candidate.sum())
    one_one = (1,1)
    return candidate, {
        "source":"completed provider/competition-matched regulation scores",
        "calibration_type":"shrunken league score residual versus league Poisson",
        "observations":len(scores),
        "weighted_observations":round(total_w,2),
        "lookback_days":LOOKBACK_DAYS,
        "prior_equivalent_games":PRIOR_EQUIVALENT_MATCHES,
        "goal_means":{"home":float(means[0]),"away":float(means[1])},
        "weighted_observed_1_1":float(counts[one_one]/total_w),
        "reference_1_1":float(reference[one_one]),
        "limited_1_1_ratio":float(ratio[one_one]),
        "total_variation":float(np.abs(candidate-base).sum()/2),
        "as_of":instant.astimezone(timezone.utc).isoformat(),
        "frozen_production_changed":False,
        "validated_accuracy_gain":False,
    }


def predict(fixture, history, league_history=None, *, as_of=None, max_goals=10):
    """Experimental pre-match v12, requiring a properly qualified league sample.

    Returns None when history is insufficient. Existing v8 records are not
    regenerated, rescored, or overwritten.
    """
    as_of = engine._date(as_of) if as_of is not None else datetime.now(timezone.utc)
    kickoff = engine._date(fixture.get("utcDate"))
    if as_of is None or kickoff is None or as_of >= kickoff:
        return None
    baseline = v8_predict(fixture, history, league_history=league_history,
                          as_of=as_of, max_goals=max_goals)
    if baseline is None:
        return None
    prior = _league_rows(fixture, history if league_history is None else league_history, as_of)
    calibrated = calibrate_matrix(baseline["matrix"], prior, as_of=as_of)
    if calibrated is None:
        return None
    matrix, diagnostic = calibrated
    flat = np.argsort(-matrix.ravel(), kind="stable")[:10]
    scores = [(int(i//matrix.shape[1]), int(i%matrix.shape[1]),
               float(matrix.ravel()[i])) for i in flat]
    home = np.arange(matrix.shape[0], dtype=float)[:,None]
    away = np.arange(matrix.shape[1], dtype=float)[None,:]
    # Shallow copy is enough; no baseline field or array is mutated.
    result = dict(baseline)
    result.update({
        "matrix":matrix,"scores":scores,
        "home_xg":float((matrix*home).sum()),
        "away_xg":float((matrix*away).sum()),
        "home_win":float(np.tril(matrix, -1).sum()),
        "draw":float(np.trace(matrix)),
        "away_win":float(np.triu(matrix, 1).sum()),
        "model_version":VERSION,
        "release_status":"experimental_shadow_only_not_promoted",
        "v12_diagnostics":diagnostic,
        "v12_original_v8_score":list(baseline["scores"][0][:2]),
        "v12_original_v8_probability":float(baseline["scores"][0][2]),
        "llm_provider_calls":0,
        "limitations":("V12 uses shrunken historical league goal-score residuals, "
                       "not a calibrated individual-v8 residual model. League "
                       "mix and matchup effects may introduce bias. Unvalidated "
                       "until independent chronological and prospective tests; "
                       "not proof of improved exact-score or double hit rates.")
    })
    return result
