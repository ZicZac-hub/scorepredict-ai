"""Deep Research AI challenger: pre-kickoff opponent-aware ensemble.

Uses verified match results only. Team news and tactical data remain context
unless independently validated; no invented injuries, xG or lineups.
"""
import numpy as np
from engine import predict, mix_score_matrices

VERSION = "deep-research-challenger-v2-engine-v6"
VARIANTS = ("baseline", "opponent_adjusted", "team_strength",
            "venue_contrast", "strength_contrast_candidate")

def research_prediction(fixture, history):
    forecasts = {}
    for variant in VARIANTS:
        try:
            result = predict(fixture, history, variant=variant)
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            result = None
        if result is not None:
            forecasts[variant] = result
    if "baseline" not in forecasts:
        return {"available": False, "model_version": VERSION,
                "reason": "Insufficient verified pre-kickoff history"}
    # Preserve the baseline contribution; never silently promote a challenger.
    # Fixed equal weights are experimental and require holdout calibration.
    matrix = mix_score_matrices(r["matrix"] for r in forecasts.values())
    ranked = np.argsort(-matrix.ravel(), kind="stable")[:10]
    scores = [[int(i // matrix.shape[1]), int(i % matrix.shape[1]),
               float(matrix.ravel()[i])] for i in ranked]
    outcomes = {"home": float(np.tril(matrix, -1).sum()),
                "draw": float(np.trace(matrix)),
                "away": float(np.triu(matrix, 1).sum())}
    pick = max(outcomes, key=outcomes.get)
    base = forecasts["baseline"]
    return {"available": True, "model_version": VERSION,
            "variants_used": list(forecasts),
            "scores": scores, "predicted_score": scores[0][:2],
            "outcome_probabilities": outcomes,
            "outcome_pick": pick, "outcome_confidence": outcomes[pick],
            "home_xg": float(np.mean([r["home_xg"] for r in forecasts.values()])),
            "away_xg": float(np.mean([r["away_xg"] for r in forecasts.values()])),
            "evidence": {"home_last10": base["home_last10"],
                         "away_last10": base["away_last10"],
                         "home_home10": base["home_home10"],
                         "away_away10": base["away_away10"],
                         "head_to_head": base["h2h_last10"],
                         "data_quality": {"history_count": base.get("history_count", 0)}},
            "limitations": "Statistical challenger only. No validated numerical injury, lineup or tactical adjustments. No guaranteed score."}

