"""Evidence Ensemble v8 — optional, auditable football exact-score challenger.

Preserves live v6 and fitted v7. This is an ORIGINAL statistical implementation
informed by public Dixon-Coles, Poisson and Elo literature, not proprietary AI
or external LLM outputs. Do not promote without chronological holdout evidence.
"""
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
import engine
from reference_model import fit, score_matrix

MODEL_VERSION = "ziczac-evidence-ensemble-v8-experimental"
# Predeclared weights, NOT claimed to be fitted/optimal.
MIX = (0.45, 0.35, 0.20)


def _asof_history(fixture, matches, as_of):
    """Finished, provider/league-matched evidence available before analysis."""
    instant = engine._date(as_of)
    if instant is None:
        raise ValueError("A valid forecast-as-of time is required")
    cutoff = min(instant, engine._date(fixture.get("utcDate")))
    if cutoff is None:
        return []
    out = []
    for match in engine._competition_history(fixture, matches or []):
        kickoff = engine._date(match.get("utcDate"))
        if kickoff is None or kickoff >= cutoff:
            continue
        # A finishedAt timestamp has priority if provided, otherwise use a
        # conservative three-hour delay after kickoff. Earlier same-day scores
        # must not be treated as observed merely because their kickoff passed.
        whistle = engine._date(match.get("finishedAt"))
        available = whistle or (kickoff + timedelta(hours=3))
        if available <= cutoff:
            out.append(match)
    return out


def _elo(fixture, matches):
    """Time-ordered pre-match Elo, used only when a real league population exists."""
    home_id = (fixture.get("homeTeam") or {}).get("id")
    away_id = (fixture.get("awayTeam") or {}).get("id")
    if home_id is None or away_id is None or home_id == away_id:
        return None
    population = sorted(matches, key=lambda row: engine._date(row["utcDate"]))
    if len(population) < 35:
        return None
    other = sum(home_id not in (m["homeTeam"]["id"],m["awayTeam"]["id"]) and
                away_id not in (m["homeTeam"]["id"],m["awayTeam"]["id"]) for m in population)
    if other < 12:
        return None
    rating = defaultdict(lambda: 1500.0)
    seen = defaultdict(int)
    for m in population:
        h, a = m["homeTeam"]["id"], m["awayTeam"]["id"]
        gh, ga = m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"]
        expected = 1.0/(1.0+10.0**(-((rating[h]+55.0-rating[a])/400.0)))
        observed = 1.0 if gh>ga else 0.5 if gh==ga else 0.0
        margin = min(1.8, 1.0 + 0.35*math.log1p(abs(gh-ga)))
        change = 22.0*margin*(observed-expected)
        rating[h] += change
        rating[a] -= change
        seen[h] += 1
        seen[a] += 1
    if min(seen[home_id],seen[away_id]) < 7:
        return None
    return {"home": round(rating[home_id],1), "away": round(rating[away_id],1),
            "home_matches": seen[home_id], "away_matches": seen[away_id],
            "home_advantage": 55.0, "population": len(population)}


def _result(matrix, source, baseline, components, weights, evidence, elo):
    rank = np.argsort(-matrix.ravel(), kind="stable")[:10]
    scores = [(int(i//matrix.shape[1]), int(i%matrix.shape[1]), float(matrix.ravel()[i])) for i in rank]
    hgrid = np.arange(matrix.shape[0], dtype=float)[:, None]
    agrid = np.arange(matrix.shape[1], dtype=float)[None, :]
    base_matrix = baseline["matrix"]
    same_shape = np.zeros_like(matrix)
    same_shape[:base_matrix.shape[0], :base_matrix.shape[1]] = base_matrix
    total_variation = float(np.abs(matrix-same_shape).sum()/2)
    agreeing = sum(int(np.unravel_index(np.argmax(comp["matrix"]),comp["matrix"].shape) == 
                       np.unravel_index(np.argmax(matrix),matrix.shape)) for comp in components.values())
    result = dict(baseline)
    result.update({
        "matrix": matrix, "scores": scores, "home_xg": float((matrix*hgrid).sum()),
        "away_xg": float((matrix*agrid).sum()),
        "home_win": float(np.tril(matrix,-1).sum()),
        "draw": float(np.trace(matrix)), "away_win": float(np.triu(matrix,1).sum()),
        "model_version": MODEL_VERSION, "release_status": "challenger_not_validated",
        "original_baseline_score": tuple(baseline["scores"][0][:2]),
        "original_baseline_score_probability": float(baseline["scores"][0][2]),
        "component_weights": weights, "components": {
            name: {"predicted_score":tuple(r["scores"][0][:2]),
                   "score_probability":float(r["scores"][0][2])}
            for name,r in components.items()},
        "component_top_score_agreement": agreeing,
        "component_count": len(components),
        "baseline_to_challenger_total_variation": total_variation,
        "elo": elo, "evidence": evidence,
        "source_labels": list(components),
        "llm_provider_calls": 0,
        "limitations": ("Challenger probability estimates, not verified accuracy. "
                         "No verified xG, starting-XI player strength, tactical or bookmaker "
                         "numbers are inserted. External LLM providers are not connected.")
    })
    return result


def predict(fixture, history, league_history=None, max_goals=10, as_of=None):
    """Pre-match v8 candidate; never modifies the existing baseline or ledger."""
    kickoff = engine._date(fixture.get("utcDate"))
    if kickoff is None:
        return None
    as_of = engine._date(as_of) if as_of is not None else datetime.now(timezone.utc)
    if as_of is None or as_of >= kickoff:
        # Never offer a new pre-match forecast after kickoff.
        return None
    raw_league = history if league_history is None else league_history
    prior = _asof_history(fixture, history, as_of)
    full_league = _asof_history(fixture, raw_league, as_of)
    baseline = engine.predict(fixture, prior, max_goals=max_goals, league_history=full_league)
    if baseline is None:
        return None
    components = {"preserved_v6":baseline}
    fitted = None
    target = {fixture["homeTeam"]["id"],fixture["awayTeam"]["id"]}
    other = sum(not target.intersection((m["homeTeam"]["id"],m["awayTeam"]["id"])) for m in full_league)
    if len(full_league) >= 35 and other >= 12:
        estimate = fit(fixture, full_league, regularization=2.0, half_life_days=180.0)
        if estimate:
            fitted = score_matrix(estimate,max_goals=max_goals)
            components["opponent_dixon_coles_v7"] = {
                "matrix": fitted, "scores": _top_scores(fitted)}
    elo = _elo(fixture,full_league)
    if elo:
        # Elo is a third, independently observed strength signal, not a pick.
        # Modulate relative goal rates conservatively; preserve total xG scale.
        diff = max(-400.0,min(400.0,elo["home"]+elo["home_advantage"]-elo["away"]))
        tilt = 0.40*(diff/400.0)
        mu_h = max(0.05,baseline["home_xg"]*math.exp(tilt))
        mu_a = max(0.05,baseline["away_xg"]*math.exp(-tilt))
        adjusted = engine._grid(mu_h,mu_a,max_goals=max_goals)
        components["elo_goal_contrast"] = {
            "matrix":adjusted,"scores":_top_scores(adjusted)}
    if len(components) == 3:
        weights = {"preserved_v6":MIX[0],"opponent_dixon_coles_v7":MIX[1],"elo_goal_contrast":MIX[2]}
    elif len(components) == 2 and fitted is not None:
        weights = {"preserved_v6":0.55,"opponent_dixon_coles_v7":0.45}
    elif len(components) == 2:
        weights = {"preserved_v6":0.70,"elo_goal_contrast":0.30}
    else:
        weights = {"preserved_v6":1.0}
    matrix = engine.mix_score_matrices([comp["matrix"] for comp in components.values()], list(weights.values()))
    evidence = {"forecast_as_of":as_of.isoformat(), "fixture_kickoff":kickoff.isoformat(),
                "provider":fixture.get("provider","football-data.org"),
                "prior_team_matches":len(prior), "prior_league_matches":len(full_league),
                "unrelated_league_matches":other,
                "home_last10_count":len(baseline.get("home_last10",[])),
                "away_last10_count":len(baseline.get("away_last10",[])),
                "h2h_count":len(baseline.get("h2h_last10",[]))}
    return _result(matrix,MODEL_VERSION,baseline,components,weights,evidence,elo)


def _top_scores(matrix):
    ids = np.argsort(-matrix.ravel(), kind="stable")[:10]
    return [(int(i//matrix.shape[1]),int(i%matrix.shape[1]),float(matrix.ravel()[i])) for i in ids]
