"""Synthetic reporting tests, not historical accuracy evidence."""
from datetime import datetime, timedelta, timezone
import numpy as np
import backtest_dynamic_dixon_coles as backtest

def test_matched_denominator_and_abstentions(monkeypatch):
    matches=[]
    for day in range(2):
        for leg in range(2):
            matches.append({'id':day*2+leg+1,
                'utcDate':(datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(days=day*3,hours=leg)).isoformat(),
                'status':'FINISHED','homeTeam':{'id':leg*2+1},
                'awayTeam':{'id':leg*2+2},
                'score':{'fullTime':{'home':0,'away':0}}})
    def fake_predict(match,past,variant):
        assert all(m['id']<match['id'] for m in past)
        if variant=='dynamic_dixon_coles' and match['id']==1:
            return None
        return {'matrix':np.full((3,3),1/9)}
    monkeypatch.setattr(backtest,'predict',fake_predict)
    report=backtest.evaluate(matches)
    assert report['offered_pairs']==2
    assert report['matched_pairs']==1
    assert report['excluded_unmatched_pairs']==1
    assert report['matched_fraction']==.5
    assert report['unavailable_pairs_by_model']=={'baseline':0,'dixon_coles':0,'dynamic_dixon_coles':1}
    assert len(report['matched_trials'])==1
    assert report['matched_trials'][0]['fixture_ids']==[3,4]
    assert all(r['pairs']==1 for r in report['models'].values())
