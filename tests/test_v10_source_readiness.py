"""No-network diagnostics for the real 5-team, 35-league, 12-unrelated
evidence gates. Never count an abstention as a forecast or a failed ticket.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import v10_prospective as v10
from v10_source_readiness import abstention_reason


PAIRS = ((1,2),(3,4),(1,3),(2,4),(1,4),(2,3))


def fixture(now):
    return {"id":9800,"provider":"API-Football",
            "utcDate":(now+timedelta(days=2)).isoformat(),
            "status":"SCHEDULED", "homeTeam":{"id":1,"name":"A"},
            "awayTeam":{"id":2,"name":"B"},
            "competition":{"id":44,"code":"TST"}}


def history(now,n):
    result=[]
    for i in range(n):
        h,a=PAIRS[i%len(PAIRS)]
        result.append({
            "id":i+1, "provider":"API-Football",
            "utcDate":(now-timedelta(days=n+3-i)).isoformat(),
            "status":"FINISHED", "competition":{"id":44,"code":"TST"},
            "homeTeam":{"id":h,"name":str(h)},
            "awayTeam":{"id":a,"name":str(a)},
            "score":{"fullTime":{"home":i%3,"away":(i+1)%2}},
        })
    return result


def test_v8_abstentions_are_diagnosed_from_same_pre_match_history():
    now=datetime(2035,1,10,tzinfo=timezone.utc)
    game=fixture(now)
    assert abstention_reason(game,history(now,4),history(now,4),now)==(
        "team_history_below_five")
    assert abstention_reason(game,history(now,24),history(now,24),now)==(
        "league_history_below_thirty_five")
    assert abstention_reason(game,history(now,48),history(now,48),now)==(
        "unrelated_league_history_below_twelve")
    assert abstention_reason(game,history(now,96),history(now,96),now)==(
        "independent_strength_component_unqualified")


def test_v8_abstentions_do_not_look_ahead_or_trust_bad_history():
    now=datetime(2035,1,10,tzinfo=timezone.utc)
    game=fixture(now)
    past=history(now,24)
    future=[{**row,"utcDate":(now+timedelta(hours=3)).isoformat()}
            for row in past]
    assert abstention_reason(game,future,future,now)=="team_history_below_five"
    assert abstention_reason(game,past,past,now+timedelta(days=3))==(
        "not_before_original_kickoff")
    assert abstention_reason(game,[42]*50,[],now)==(
        "unusable_authenticated_history_structure")


def test_failed_v8_retains_original_abstention_and_exposes_safe_subreason(
        monkeypatch,tmp_path):
    now=datetime(2035,1,10,tzinfo=timezone.utc)
    game=fixture(now)
    rows=history(now,24)
    monkeypatch.setattr(v10,"research_predict",lambda *a,**kw:None)
    evidence={"history":rows,"league_history":rows,
              "input_digest":"immutable-timestamped-source",
              "as_of_utc":now.isoformat()}
    result=v10.capture(game,evidence,now,root=tmp_path,
                       market_calls_remaining=0)
    assert result["v8_recorded"] is False
    assert result["reason"]=="insufficient_verified_research_evidence"
    assert result["readiness_reason"]=="league_history_below_thirty_five"
    assert not list((tmp_path/"forecasts").rglob("*.json"))


def test_collector_prefers_precise_subreason_without_changing_api_budget():
    source=Path("capture_prematch.py").read_text(encoding="utf-8")
    assert "result.get('readiness_reason') or" in source
    assert "research_backfill_limit=_positive_limit" in source
    assert "max_api_calls=_positive_limit" in source
