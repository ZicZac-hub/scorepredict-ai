import numpy as np
import pytest
from v42_research_engine import hypotheses, ensemble
from v42_research_trial import PROTOCOL, _arms

def grid():
    return np.outer([.30,.35,.20,.10,.04,.01],[.40,.30,.17,.08,.04,.01])

def test_normalized_deterministic_and_immutable():
    p=grid()
    before=p.copy()
    for q in [*hypotheses(p).values(),ensemble(p)]:
        assert np.isclose(q.sum(),1) and np.all(q>=0) and q.shape==p.shape
    assert np.array_equal(p,before)
    assert np.array_equal(ensemble(p),ensemble(p))

def test_directional_hypotheses():
    p=grid()
    h,a=np.indices(p.shape)
    arms=hypotheses(p)
    assert arms["high_goal"][h+a>=4].sum()>p[h+a>=4].sum()
    assert arms["low_goal"][h+a<=1].sum()>p[h+a<=1].sum()
    assert arms["wide_margin"][abs(h-a)>=3].sum()>p[abs(h-a)>=3].sum()

def test_invalid_and_missing_evidence():
    p=grid()
    with pytest.raises(ValueError):
        ensemble(p*2)
    arms=_arms([{"score_matrix":p.tolist()}]*2,[{},{}])
    final=arms["unified_final"]
    assert final["status"]=="issued"
    assert final["paired_xg_used"] is False and final["paired_market_used"] is False
    assert len({(x["score1"],x["score2"]) for x in final["five_doubles"]})==5
    assert PROTOCOL["automatic_model_promotion"] is False


def test_watchdog_evidence_coverage_distinguishes_missing_and_unknown():
    from v42_research_bot import evidence_coverage
    performance = {
        "valid_opportunities": 3,
        "rows": [
            {"unified_final_evidence": {"paired_xg_used": True, "paired_market_used": False}},
            {"unified_final_evidence": {"paired_xg_used": False, "paired_market_used": True}},
            {"unified_final_evidence": {"paired_xg_used": None}},
        ],
        "lineup_coverage": {"both_matches_confirmed_at_decision": 1},
    }
    summary = evidence_coverage(performance)
    assert summary["valid_pairs"] == 3
    assert summary["xg"] == {"used": 1, "not_used": 1, "unknown": 1}
    assert summary["market"] == {"used": 1, "not_used": 1, "unknown": 1}
    assert summary["confirmed_lineups_both_matches"] == 1
    with pytest.raises(ValueError, match="inconsistent_evidence_row_count"):
        evidence_coverage({**performance, "valid_opportunities": 4})


def test_watchdog_empty_archive_is_not_a_failed_or_missing_pair():
    from v42_research_bot import evidence_coverage
    summary = evidence_coverage({"valid_opportunities": 0, "rows": [],
        "lineup_coverage": {"both_matches_confirmed_at_decision": 0}})
    assert summary["xg"] == summary["market"] == {"used": 0, "not_used": 0, "unknown": 0}


@pytest.mark.parametrize("enabled,key,use_ai,reason,allowed",[
    ("true","synthetic",False,"no_ai_trigger",False),
    ("false","synthetic",True,"disabled_by_configuration",False),
    ("true","   ",True,"missing_key",False),
    (" TRUE ","synthetic",True,"eligible_for_daily_attempt",True),
])
def test_advisory_configuration_is_sanitized_and_distinguishes_skip(
        monkeypatch,enabled,key,use_ai,reason,allowed):
    import json
    from v42_research_bot import advisory_configuration
    monkeypatch.setenv("V42_AI_RESEARCH_ENABLED",enabled)
    monkeypatch.setenv("OPENAI_API_KEY",key)
    state=advisory_configuration(use_ai=use_ai)
    assert state["reason"]==reason
    assert state["attempt_allowed"] is allowed
    assert state["key_present"] is bool(key.strip())
    assert "synthetic" not in json.dumps(state)


def test_advisory_missing_or_disabled_does_not_request(monkeypatch,tmp_path):
    import v42_research_bot as bot
    def forbidden(*a,**k):
        pytest.fail("unexpected paid request")
    monkeypatch.setattr(bot.requests,"post",forbidden)
    monkeypatch.setenv("V42_AI_RESEARCH_ENABLED","true")
    monkeypatch.setenv("OPENAI_API_KEY"," ")
    assert bot.advisory({},root=tmp_path)=="missing_key"
    monkeypatch.setenv("V42_AI_RESEARCH_ENABLED","false")
    monkeypatch.setenv("OPENAI_API_KEY","synthetic")
    assert bot.advisory({},root=tmp_path)=="disabled"
    assert list(tmp_path.iterdir())==[]
