"""Frozen experimental ensemble for prospective, pre-kickoff comparison.

Weights are fixed from the earlier 2025 calibration, NOT refitted on
future results. Predictions are separate from immutable Baseline forecasts.
"""
import numpy as np
from engine import predict

WEIGHTS={"baseline":0.25,"team_strength":0.50,"league_opponent":0.25}
VERSION="prospective-ensemble-2025-frozen-v1"

def ensemble_prediction(fixture,history):
    forecasts={}
    for variant in WEIGHTS:
        result=predict(fixture,history,variant=variant)
        if result is None:
            return {"available":False,"model_version":VERSION,
                    "reason":"Insufficient history for one or more candidates"}
        forecasts[variant]=result
    matrix=sum(WEIGHTS[v]*forecasts[v]["matrix"] for v in WEIGHTS)
    flat=np.argsort(-matrix.ravel(),kind="stable")[:3]
    scores=[[int(np.unravel_index(int(i),matrix.shape)[0]),
             int(np.unravel_index(int(i),matrix.shape)[1]),
             float(matrix.ravel()[i])] for i in flat]
    return {"available":True,"model_version":VERSION,"weights":WEIGHTS.copy(),
            "predicted_score":scores[0][:2],
            "predicted_score_probability":scores[0][2],
            "top3_scores":[row[:2] for row in scores]}
