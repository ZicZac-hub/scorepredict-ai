"""Regression tests for calendar-aware ZicZac forecast recency.
Run: python -m unittest test_engine_recency.py
"""
import unittest
from engine import _stats, predict


class RecencyTests(unittest.TestCase):
    def test_calendar_gap_changes_weights(self):
        games=[
            {'date':'2026-10-30T12:00:00Z','gf':3,'ga':0,'result':'W'},
            {'date':'2025-10-30T12:00:00Z','gf':0,'ga':3,'result':'L'},
        ]
        rates=_stats(games,'2026-11-01T12:00:00Z')
        self.assertGreater(rates['scored'],2.0)
        self.assertLess(rates['conceded'],1.0)

    def test_same_age_equivalent(self):
        games=[
            {'date':'2026-10-01T12:00:00Z','gf':1,'ga':2,'result':'L'},
            {'date':'2026-10-01T12:00:00Z','gf':3,'ga':0,'result':'W'},
        ]
        rates=_stats(games,'2026-11-01T12:00:00Z')
        self.assertAlmostEqual(rates['scored'],2.0)
        self.assertAlmostEqual(rates['conceded'],1.0)

    def test_outlier_goal_counts_are_capped_in_rate_estimate(self):
        games=[
            {'date':'2026-10-30T12:00:00Z','gf':15,'ga':0,'result':'W'},
            {'date':'2026-10-30T12:00:00Z','gf':1,'ga':1,'result':'D'},
        ]
        rates=_stats(games,'2026-11-01T12:00:00Z')
        self.assertAlmostEqual(rates['scored'],3.0)
        self.assertEqual(rates['wins'],1)
        self.assertEqual(rates['draws'],1)

    def test_effective_sample_falls_with_unequal_recency(self):
        games=[
            {'date':'2026-10-30T12:00:00Z','gf':1,'ga':0,'result':'W'},
            {'date':'2024-01-01T12:00:00Z','gf':1,'ga':0,'result':'W'},
        ]
        rates=_stats(games,'2026-11-01T12:00:00Z')
        self.assertLess(rates['effective_played'],2.0)
        self.assertGreaterEqual(rates['effective_played'],1.0)

    def test_future_results_cannot_enter_forecast(self):
        history=[]
        for idx in range(12):
            date=f'2026-09-{idx+1:02d}T12:00:00Z'
            history.append({
                'id':idx+1,'utcDate':date,'status':'FINISHED',
                'homeTeam':{'id':1,'name':'A'} if idx%2==0 else {'id':2,'name':'B'},
                'awayTeam':{'id':2,'name':'B'} if idx%2==0 else {'id':1,'name':'A'},
                'score':{'fullTime':{'home':idx%3,'away':(idx+1)%3}}})
        fixture={'utcDate':'2026-10-01T12:00:00Z',
                 'homeTeam':{'id':1,'name':'A'},'awayTeam':{'id':2,'name':'B'}}
        baseline=predict(fixture,history)
        self.assertIsNotNone(baseline)
        future={'id':999,'utcDate':'2026-10-10T12:00:00Z','status':'FINISHED',
                'homeTeam':{'id':1,'name':'A'},'awayTeam':{'id':2,'name':'B'},
                'score':{'fullTime':{'home':10,'away':0}}}
        with_future=predict(fixture,history+[future])
        self.assertAlmostEqual(baseline['home_xg'],with_future['home_xg'])
        self.assertAlmostEqual(baseline['away_xg'],with_future['away_xg'])


if __name__=='__main__':
    unittest.main()
