"""Provider contract and settlement regressions; no API credentials or traffic."""
import copy
from datetime import date, datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import api_football
import persist_prematch


def raw_match(status="FT"):
    return {
        "fixture":{"id":123,"date":"2020-11-01T13:00:00+01:00","status":{"short":status}},
        "teams":{"home":{"id":10,"name":"Team A"},"away":{"id":20,"name":"Team B"}},
        "league":{"id":39,"season":2020,"name":"Premier League"},
        "goals":{"home":2,"away":1},
        "score":{"fulltime":{"home":1,"away":1}},
    }


def snapshot():
    return {"fixture_id":123,"provider":"API-Football","competition":"PL",
            "home_team":"Team A","away_team":"Team B",
            "home_team_id":10,"away_team_id":20,
            "kickoff_utc":"2020-11-01T12:00:00Z",
            "captured_at_utc":"2020-11-01T10:00:00Z",
            "predicted_score":[1,1],"predicted_score_probability":.12,
            "predicted_outcome_probabilities":{"home":.6,"draw":.25,"away":.15},
            "double_top10_scores":[[1,1,.12],[2,1,.11],[1,0,.10]]}


class ProviderConversionTests(unittest.TestCase):
    def test_all_ui_date_types_work_without_network(self):
        for value in ("2026-11-13",date(2026,11,13),datetime(2026,11,13,tzinfo=timezone.utc)):
            with self.subTest(value=value), patch.object(api_football,"_request",return_value=[]) as request:
                self.assertEqual(api_football.get_fixtures("key","PL",value,14),[])
                self.assertEqual(api_football.get_history("key","PL",value),[])
                self.assertEqual(request.call_args_list[0].args[2]["from"],"2026-11-13")
                self.assertEqual(request.call_args_list[1].args[2]["season"],2026)

    def test_converter_preserves_league_identity_and_utc(self):
        match=api_football._convert(raw_match(),"WORLD")
        self.assertEqual(match["competition"],{"id":39,"code":"PL","season":2020,"name":"Premier League"})
        self.assertEqual(match["utcDate"],"2020-11-01T12:00:00Z")
        self.assertEqual(match["season"],2020)

    def test_unknown_league_has_stable_provider_code(self):
        raw=raw_match();raw["league"]["id"]=9999
        self.assertEqual(api_football._convert(raw,"WORLD")["competition"]["code"],"API-9999")

    def test_extra_time_and_penalties_use_regulation_score(self):
        for status in ("AET","PEN"):
            with self.subTest(status=status):
                match=api_football._convert(raw_match(status),"PL")
                self.assertEqual(match["score"]["fullTime"],{"home":1,"away":1})
                self.assertEqual(match["score_basis"],"regulation")

    def test_missing_extra_time_regulation_score_stays_unknown(self):
        raw=raw_match("AET");raw.pop("score")
        self.assertEqual(api_football._convert(raw,"PL")["score"]["fullTime"],{"home":None,"away":None})

    def test_ft_can_fallback_to_goals_but_boolean_is_not_score(self):
        raw=raw_match();raw.pop("score")
        self.assertEqual(api_football._convert(raw,"PL")["score"]["fullTime"],{"home":2,"away":1})
        raw["goals"]["home"]=True
        self.assertIsNone(api_football._convert(raw,"PL")["score"]["fullTime"]["home"])


class ScheduledSettlementTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.results=Path(self.directory.name)/"verified_results.json"
        self.legacy=Path(self.directory.name)/"legacy_results.json"
        for name,value in (("RESULTS",self.results),("LEGACY_RESULTS",self.legacy)):
            p=patch.object(persist_prematch,name,value);p.start();self.addCleanup(p.stop)

    def settle(self,raw,row=None):
        response=Mock();response.json.return_value={"response":[raw]}
        with patch.object(persist_prematch.requests,"get",return_value=response):
            return persist_prematch.reconcile("key",{"123":row or snapshot()})

    def test_regulation_settlement_and_highest_outcome_probability(self):
        results=self.settle(raw_match("AET"))
        self.assertEqual([results["123"]["home_goals"],results["123"]["away_goals"]],[1,1])
        report=persist_prematch.accuracy({"123":snapshot()},results)
        self.assertEqual(report["exact_score_hits"],1)
        self.assertEqual(report["winner_draw_accuracy"],0)
        self.assertEqual(report["winner_draw_modal_score_fallbacks"],0)
        self.assertEqual(report["expected_exact_hits"],.12)
        self.assertEqual(report["top3_score_accuracy"],1)

    def test_same_number_wrong_team_or_kickoff_cannot_settle(self):
        for mutation in ("team","kickoff","id"):
            raw=raw_match()
            if mutation=="team":raw["teams"]["home"]["id"]=999
            elif mutation=="kickoff":raw["fixture"]["date"]="2020-11-02T12:00:00Z"
            else:raw["fixture"]["id"]=999
            with self.subTest(mutation=mutation):
                self.assertEqual(self.settle(raw),{})

    def test_different_provider_cannot_settle(self):
        row=snapshot();row["provider"]="football-data.org"
        self.assertEqual(self.settle(raw_match(),row),{})

    def test_legacy_snapshot_requires_both_names_and_kickoff(self):
        row=snapshot();row.pop("home_team_id");row.pop("away_team_id");row.pop("provider")
        self.assertIn("123",self.settle(raw_match(),row))
        self.results.unlink()
        wrong=raw_match();wrong["teams"]["away"]["name"]="Different Team"
        self.assertEqual(self.settle(wrong,row),{})

    def test_legacy_result_preserved_and_reverified(self):
        old={"fixture_id":123,"status":"AET","home_goals":2,"away_goals":1}
        self.results.write_text(json.dumps({"123":old}))
        results=self.settle(raw_match("AET"))
        self.assertEqual(json.loads(self.legacy.read_text())["123"],old)
        self.assertEqual(results["123"]["evidence_version"],2)
        self.assertEqual(results["123"]["home_goals"],1)

    def test_unverified_legacy_result_is_excluded_when_api_unavailable(self):
        old={"fixture_id":123,"home_goals":1,"away_goals":1}
        self.results.write_text(json.dumps({"123":old}))
        with patch.object(persist_prematch.requests,"get",side_effect=api_football.requests.RequestException("offline")):
            results=persist_prematch.reconcile("key",{"123":snapshot()})
        self.assertEqual(results,{})
        self.assertEqual(persist_prematch.accuracy({"123":snapshot()},{"123":old})["verified_finished_matches"],0)
        self.assertTrue(self.legacy.exists())

    def test_verified_result_cannot_be_reused_for_changed_snapshot_identity(self):
        results=self.settle(raw_match())
        changed=snapshot();changed["home_team_id"]=999
        self.assertEqual(persist_prematch.accuracy({"123":changed},results)["verified_finished_matches"],0)


if __name__=="__main__":
    unittest.main()
