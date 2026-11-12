import numpy as np
from unittest.mock import patch
from deep_research_ai import research_prediction, VARIANTS

def fake_predict(fixture, history, variant='baseline'):
    matrix=np.zeros((3,3))
    matrix[1,0]=.2
    matrix[2,0]=.6
    matrix[1,1]=.2
    return {'matrix':matrix,'home_xg':2.0,'away_xg':0.6,
            'home_last10':[],'away_last10':[],'home_home10':[],
            'away_away10':[],'h2h_last10':[],'data_quality':'moderate'}

def test_candidate_uses_real_model_distributions_without_forcing_low_scores():
    with patch('deep_research_ai.predict', side_effect=fake_predict):
        report=research_prediction({},[])
    assert report['available']
    assert report['predicted_score']==[2,0]
    assert len(report['variants_used'])==len(VARIANTS)
    assert abs(sum(report['outcome_probabilities'].values())-1)<1e-10

def test_no_history_reports_unavailable():
    with patch('deep_research_ai.predict', return_value=None):
        report=research_prediction({},[])
    assert not report['available']
