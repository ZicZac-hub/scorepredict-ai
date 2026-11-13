"""Five-model equal-weight consensus, experimental and auditable.

These are five statistical variants of the same underlying engine, not five
independent AI systems. No ScoreGPT proprietary model is used.
"""
import numpy as np
from engine import predict, mix_score_matrices

VARIANTS=("baseline","venue_arithmetic","team_strength","opponent_adjusted","league_opponent")
VERSION="five-statistical-model-consensus-v2-engine-v6"

def consensus_prediction(fixture,history):
    matrices=[]
    for variant in VARIANTS:
        result=predict(fixture,history,variant=variant)
        if result is None:
            return {"available":False,"model_version":VERSION,
                    "reason":"Insufficient history for one or more model variants"}
        matrices.append(result["matrix"])
    matrix=mix_score_matrices(matrices)
    ranked=np.argsort(-matrix.ravel(),kind="stable")[:10]
    scores=[[int(i//matrix.shape[1]),int(i%matrix.shape[1]),float(matrix.ravel()[i])] for i in ranked]
    return {"available":True,"model_version":VERSION,
            "variants":list(VARIANTS),"weights":{v:1/len(VARIANTS) for v in VARIANTS},
            "predicted_score":scores[0][:2],
            "predicted_score_probability":scores[0][2],
            "top3_scores":[s[:2] for s in scores[:3]],
            "top10_scores":scores,
            "home_win":float(np.tril(matrix,-1).sum()),
            "draw":float(np.trace(matrix)),
            "away_win":float(np.triu(matrix,1).sum())}

