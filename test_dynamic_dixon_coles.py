"""Synthetic unit tests for mechanics only; not empirical accuracy evidence."""
import unittest
from datetime import datetime,timedelta,timezone
import numpy as np
from engine import predict, _grid
from dynamic_dixon_coles import fit_forecast, five_doubles, _safe_rho, _tau

def fixture(i,h,a,gh=2,ga=1,competition=1):
    d=datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(days=i)
    return {'id':i,'utcDate':d.isoformat(),'status':'FINISHED',
            'competition':{'id':competition,'code':str(competition)},
            'homeTeam':{'id':h},'awayTeam':{'id':a},
            'score':{'fullTime':{'home':gh,'away':ga}}}

def history():
    return [fixture(i, i%6,(i+1+(i//6)%5)%6,
                    (i%4) if i%6 else 4,(i//3)%3) for i in range(150)]

class DynamicDixonColesTests(unittest.TestCase):
    def test_matrix_and_engine_integration(self):
        past=history()
        target=fixture(160,0,1)
        result=predict(target,past,variant='dynamic_dixon_coles')
        self.assertIsNotNone(result)
        self.assertEqual(result['model_version'],'dynamic-dixon-coles-v1-experimental')
        self.assertAlmostEqual(result['matrix'].sum(),1.)
        self.assertTrue(np.isfinite(result['matrix']).all())
        self.assertGreaterEqual(result['matrix'].min(),0.)
        self.assertGreaterEqual(result['dynamic_calibration']['calibration_matches'],20)
        self.assertAlmostEqual(result['home_win']+result['draw']+result['away_win'],1.)

    def test_no_future_same_time_or_other_league_leakage(self):
        target=fixture(160,0,1)
        past=history()
        a=fit_forecast(target,past)
        b=fit_forecast(target,past+[fixture(160,0,1,9,0),fixture(161,1,0,9,0),
                                    fixture(159,0,1,9,0,competition=2)])
        np.testing.assert_array_equal(a['matrix'],b['matrix'])

    def test_thin_history_abstains(self):
        self.assertIsNone(fit_forecast(fixture(160,0,1), history()[:40]))

    def test_missing_competition_abstains(self):
        target=fixture(160,0,1)
        target.pop('competition')
        self.assertIsNone(fit_forecast(target,history()))

    def test_stale_team_history_abstains_even_with_fresh_league(self):
        stale=[fixture(i,0,1) for i in range(10)]
        recent=[fixture(i,2+i%4,2+(i+1)%4) for i in range(900,1050)]
        self.assertIsNone(fit_forecast(fixture(1060,0,1),stale+recent))

    def test_rho_bounds(self):
        for h,a in ((.15,.15),(12.,12.),(12.,.15)):
            for rho in (-.2,.2):
                r=_safe_rho(h,a,rho)
                self.assertGreater(min(_tau(h,a,i,j,r) for i,j in ((0,0),(0,1),(1,0),(1,1))),0.)

    def test_five_unique_doubles_are_globally_top_ranked(self):
        a,b=_grid(1.2,.8),_grid(2.8,1.6)
        selected=five_doubles(a,b)
        brute=np.sort(np.outer(a.ravel(),b.ravel()).ravel())[::-1][:5]
        np.testing.assert_allclose([r[0] for r in selected],brute)
        self.assertEqual(len({(r[1],r[2]) for r in selected}),5)
        self.assertLessEqual(sum(r[0] for r in selected),1.)

if __name__=='__main__':
    unittest.main()
