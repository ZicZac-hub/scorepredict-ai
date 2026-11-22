"""V25 anchored future-only challenger guards and numeric invariants."""
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

from v25_anchored_challenger import (
    BASELINE_WEIGHT, _anchored_grid, _frozen_eligible,
    _portfolio, _picks_hit, _joint_mass_at_actual,
    _verify_frozen, _mixture_grid, _score_calibration, _overall_proper_scores,
)


def item(fid, home, away, *, kick=None):
    base=np.ones((5,5),dtype=float)/25
    # Distinct asymmetric 5x5 grids so shrinkage is not a no-op.
    base[2,1]+=0.075
    base/=base.sum()
    alt=np.ones((5,5),dtype=float)/25
    alt[0,0]+=0.10
    alt/=alt.sum()
    now=datetime(2026,11,23,12,tzinfo=timezone.utc)
    return {"key":("API-Football",str(fid)), "grid":base,
            "alternatives":{"baseline_v8":base,
                "poisson_marginal":alt,
                "negative_binomial_r3":alt,
                "bivariate_poisson_c012":alt},
            "teams":{("API-Football",str(home)),("API-Football",str(away))},
            "kickoff":kick or now+timedelta(hours=2)}


def original(now):
    return {
        "policy":"FROZEN",
        "sydney_day":"2026-11-24",
        "actually_issued_at_utc":now.isoformat(),
        "policy_cutoff_utc":now.isoformat(),
        "candidate_catalog_sha256":"unmodified-source-digest",
        "quality_selection":{
            "fixture_keys":[["API-Football","101"],["API-Football","102"]],
            "source_forecast_ids":["sha-a","sha-b"],
            "five_original_v25_robust_scores":[{"score1":"1-0","score2":"1-0"}]*5,
            "five_original_v8_optimal_scores":[{"score1":"2-1","score2":"2-1"}]*5,
        },
    }


class AnchoredV25Tests(unittest.TestCase):
    def test_fixed_shrinkage_normalized_and_causally_from_v8(self):
        value=item(101,1,2)
        grid=_anchored_grid(value)
        arr=np.array(grid)
        self.assertAlmostEqual(arr.sum(),1)
        self.assertTrue((arr>=0).all())
        self.assertFalse(np.allclose(arr,value["grid"]))
        self.assertEqual(BASELINE_WEIGHT,0.75)

    def test_requires_recent_original_pre_match_freeze(self):
        now=datetime(2026,11,23,12,tzinfo=timezone.utc)
        d=original(now)
        self.assertTrue(_frozen_eligible(d,now+timedelta(minutes=19)))
        self.assertFalse(_frozen_eligible(d,now+timedelta(minutes=21)))
        self.assertFalse(_frozen_eligible(d,now-timedelta(seconds=1)))
        d["policy"]="ABSTAIN"
        self.assertFalse(_frozen_eligible(d,now))

    def test_same_pair_five_unique_frozen_combinations(self):
        now=datetime(2026,11,23,12,tzinfo=timezone.utc)
        packet=_portfolio(original(now), [item(101,1,2),item(102,3,4)],now)
        self.assertEqual(len(packet["five_anchored_v25"]),5)
        self.assertEqual(len({(p["score1"],p["score2"]) for p in packet["five_anchored_v25"]}),5)
        self.assertEqual(len(packet["frozen_anchored_score_grids"]),2)
        self.assertTrue(packet["same_original_pair_for_all_three_policies"])

    def test_freeze_all_models_before_results_for_fair_comparison(self):
        now=datetime(2026,11,23,12,tzinfo=timezone.utc)
        packet=_portfolio(original(now), [item(101,1,2),item(102,3,4)],now)
        for key in ("frozen_original_v8_score_grids",
                    "frozen_v25_equal_sensitivity_score_grids",
                    "frozen_anchored_score_grids"):
            self.assertEqual(len(packet[key]),2)
            for grid in packet[key]:
                self.assertAlmostEqual(float(np.asarray(grid).sum()),1.0)
        self.assertFalse(np.allclose(packet["frozen_original_v8_score_grids"][0],
                                     packet["frozen_anchored_score_grids"][0]))
        self.assertFalse(np.allclose(packet["frozen_v25_equal_sensitivity_score_grids"][0],
                                     packet["frozen_anchored_score_grids"][0]))

    def test_same_event_calibration_reports_proper_scores(self):
        now=datetime(2026,11,23,12,tzinfo=timezone.utc)
        packet=_portfolio(original(now), [item(101,1,2),item(102,3,4)],now)
        good=_score_calibration(packet,["2-1","0-0"])
        self.assertEqual(set(good),{"original_v8","original_v25_sensitivity","anchored_v25"})
        for result in good.values():
            self.assertEqual(len(result["leg_scores"]),2)
            self.assertGreaterEqual(result["pair_mean_exact_score_log_loss"],0)
            self.assertGreaterEqual(result["pair_mean_multiclass_score_brier"],0)
        self.assertEqual(good["anchored_v25"]["leg_scores"][0]["actual_score"],"2-1")

    def test_invalid_or_pending_scores_cannot_be_fabricated(self):
        now=datetime(2026,11,23,12,tzinfo=timezone.utc)
        packet=_portfolio(original(now), [item(101,1,2),item(102,3,4)],now)
        for scores in (["0-0"], ["0-0","pending"], ["0-0","-1-0"]):
            with self.assertRaises(ValueError):
                _score_calibration(packet,scores)

    def test_physical_fixture_deduplication_keeps_calibration_comparable(self):
        now=datetime(2026,11,23,12,tzinfo=timezone.utc)
        packet=_portfolio(original(now), [item(101,1,2),item(102,3,4)],now)
        comp=_score_calibration(packet,["2-1","0-0"])
        row={"sydney_day":"2026-11-24","frozen_at_utc":now.isoformat(),
             "original_selected_fixture_keys":packet["original_selected_fixture_keys"],
             "proper_score_comparison":comp}
        totals=_overall_proper_scores([row,row])
        self.assertEqual(totals["unique_independently_verified_physical_fixtures"],2)
        self.assertEqual(totals["repeated_fixture_slots_excluded"],2)
        self.assertIsNotNone(totals["anchored_minus_v8"]["exact_score_log_loss"])
        self.assertIsNone(_overall_proper_scores([])["anchored_minus_v8"]["exact_score_log_loss"])

    def test_original_real_frozen_slate_structural_compatibility_only(self):
        # Read-only ORIGINAL pre-match fixture receipts from repository.
        # No historical V25 challenger file or falsely prospective pick issued.
        from v28_independent_pair_trial import verify_decision
        from v10_prospective import ROOT
        from five_double_accuracy import utc
        originals=sorted(Path("data/v28_independent_pair_trial/decisions").glob("*.json"))
        self.assertGreaterEqual(len(originals),2)
        for path in originals:
            decision, rows=verify_decision(path,root=ROOT,clock=datetime.now(timezone.utc))
            if decision["policy"] != "FROZEN":
                continue
            saved=_portfolio(decision,rows,utc(decision["actually_issued_at_utc"]))
            self.assertEqual(saved["original_selected_fixture_keys"],
                             decision["quality_selection"]["fixture_keys"])
            for grid in saved["frozen_anchored_score_grids"]:
                self.assertAlmostEqual(sum(sum(row) for row in grid),1.0)
        # This test is not a backdated forecast or an accuracy sample.

    def test_ineligible_close_kickoff_abstains(self):
        now=datetime(2026,11,23,12,tzinfo=timezone.utc)
        rows=[item(101,1,2,kick=now+timedelta(minutes=44)),item(102,3,4)]
        with self.assertRaisesRegex(ValueError,"not_at_least_45_minutes"):
            _portfolio(original(now),rows,now)

    def test_reject_overlapping_teams(self):
        now=datetime(2026,11,23,12,tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError,"not_four_distinct_teams"):
            _portfolio(original(now),[item(101,1,2),item(102,2,4)],now)

    def test_unverified_no_fake_win(self):
        self.assertFalse(_picks_hit([{"score1":"1-1","score2":"0-0"}],["1-0","0-0"]))
        self.assertTrue(_picks_hit([{"score1":"1-1","score2":"0-0"}],["1-1","0-0"]))

    def test_score_mass_works_at_support_boundary(self):
        g=[[[1.0]],[[1.0]]]
        self.assertEqual(_joint_mass_at_actual(g,["0-0","0-0"]),1)
        self.assertEqual(_joint_mass_at_actual(g,["4-0","0-0"]),0)

    def test_reject_malformed_sensitivity(self):
        obj=item(101,1,2)
        obj["alternatives"]["negative_binomial_r3"]=np.ones((6,6))/36
        with self.assertRaisesRegex(ValueError,"invalid_sensitivity_grid"):
            _anchored_grid(obj)


if __name__=="__main__":
    unittest.main()
