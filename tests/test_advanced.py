import unittest
from advanced import double_chance,ranked_double_chance,five_two_leg_doubles,summarize_backtest
from history_store import connect,save_predictions,load

class TestAdvanced(unittest.TestCase):
 def test_double_chance(self):
  p={'home_win':.45,'draw':.30,'away_win':.25}
  self.assertAlmostEqual(double_chance(p)['1X'],.75)
  self.assertAlmostEqual(double_chance(p)['X2'],.55)
  self.assertAlmostEqual(double_chance(p)['12'],.70)
 def test_distinct_fixtures(self):
  rows=[]
  for i in range(6):
   rows.append({'fixture':{'id':i,'homeTeam':{'name':'A'},'awayTeam':{'name':'B'}},'result':{'home_win':.65,'draw':.20,'away_win':.15,'home_xg':1.6,'away_xg':.8},'league':'Test','kickoff':str(i)})
  picks=ranked_double_chance(rows)
  self.assertEqual(len(picks),5)
  self.assertEqual(len({p['fixture_id'] for p in picks}),5)
  doubles=five_two_leg_doubles(rows)
  self.assertEqual(len(doubles),5)
  self.assertTrue(all(p['leg1']['fixture_id']!=p['leg2']['fixture_id'] for p in doubles))
 def test_write_once(self):
  c=connect(':memory:')
  rows=[{'fixture':{'id':1,'homeTeam':{'name':'A'},'awayTeam':{'name':'B'}},'result':{'scores':[(2,1,.15)],'home_win':.5,'draw':.25,'away_win':.25},'kickoff':'2026-11-10','league':'L'}]
  self.assertEqual(save_predictions(c,rows),1)
  self.assertEqual(save_predictions(c,rows),0)
  self.assertEqual(len(load(c)),1)
if __name__=='__main__':unittest.main()
