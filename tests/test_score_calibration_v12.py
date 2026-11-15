"""No-network integrity tests for the V12 experimental score calibrator."""
from datetime import datetime, timedelta, timezone
import copy
import numpy as np
from score_calibration_v12 import (
    predict, calibrate_matrix, _league_rows, MIN_MATCHES, VERSION)
from research_ensemble_v8 import predict as v8_predict


NOW=datetime(2030,6,1,12,tzinfo=timezone.utc)
PAIRS=((1,2),(2,3),(3,4),(4,5),(5,6),(6,1),
       (1,3),(2,4),(3,5),(4,6),(5,1),(6,2))


def fixture():
    return {
        "id":6001,"provider":"API-Football","status":"SCHEDULED",
        "utcDate":(NOW+timedelta(days=2)).isoformat(),
        "competition":{"id":44,"code":"TST"},
        "homeTeam":{"id":1,"name":"Home"},
        "awayTeam":{"id":2,"name":"Away"},
    }


def finished(n, *, status="FINISHED", provider="API-Football", competition=44,
             day=None):
    h,a=PAIRS[n%len(PAIRS)]
    kick=day or (NOW-timedelta(days=180-n))
    # Intentionally varied goal environment, with no requirement to lower
    # or avoid the 1-1 score; tests must not demand artificial score changes.
    goals=((n*7)%5, (n*3+1)%4)
    return {
        "id":n+100, "provider":provider,
        "utcDate":kick.isoformat(),"status":status,
        "competition":{"id":competition,"code":"TST"},
        "homeTeam":{"id":h,"name":str(h)},
        "awayTeam":{"id":a,"name":str(a)},
        "score":{"fullTime":{"home":goals[0],"away":goals[1]}},
    }


def sample(count=135):
    return [finished(n) for n in range(count)]


def test_v12_normalization_dated_league_evidence_and_unmodified_v8():
    games=sample()
    saved=copy.deepcopy(games)
    first=fixture()
    baseline=v8_predict(first,games,league_history=games,as_of=NOW)
    unchanged=baseline["matrix"].copy()
    candidate=predict(first,games,league_history=games,as_of=NOW)
    assert candidate is not None
    assert candidate["model_version"]==VERSION
    assert candidate["v12_diagnostics"]["observations"]==len(games)
    assert candidate["v12_diagnostics"]["validated_accuracy_gain"] is False
    assert candidate["release_status"]=="experimental_shadow_only_not_promoted"
    assert np.isclose(candidate["matrix"].sum(),1.0,atol=1e-10)
    assert np.all(candidate["matrix"]>=0)
    assert np.array_equal(baseline["matrix"],unchanged)
    assert games==saved
    assert candidate["llm_provider_calls"]==0
    assert len(candidate["scores"])==10
    top=np.unravel_index(int(candidate["matrix"].argmax()),candidate["matrix"].shape)
    assert top==candidate["scores"][0][:2]
    assert sum(candidate[k] for k in ("home_win","draw","away_win"))==__import__("pytest").approx(1.0)


def test_v12_future_matches_and_other_leagues_have_no_effect():
    games=sample()
    first=fixture()
    baseline=predict(first,games,league_history=games,as_of=NOW)
    assert baseline is not None
    later=games+[
        finished(145,day=NOW+timedelta(hours=5)),
        finished(146,day=NOW-timedelta(days=3),provider="other"),
        finished(147,day=NOW-timedelta(days=3),competition=12),
        finished(148,day=NOW-timedelta(hours=1)),
        finished(149,status="TIMED")
    ]
    updated=predict(first,later,league_history=later,as_of=NOW)
    assert updated is not None
    assert updated["v12_diagnostics"]["observations"]==baseline["v12_diagnostics"]["observations"]
    assert np.allclose(updated["matrix"],baseline["matrix"])


def test_v12_abstains_with_insufficient_history_and_after_kickoff():
    first=fixture()
    insufficient=sample(MIN_MATCHES-1)
    assert predict(first,insufficient,league_history=insufficient,as_of=NOW) is None
    assert predict(first,sample(),as_of=NOW+timedelta(days=3)) is None
    assert predict(first,sample(),as_of=NOW,league_history=[]) is None


def test_v12_direct_matrix_rejects_unavailable_and_incomplete_records():
    base=np.ones((11,11),dtype=float)/121.
    rows=sample(MIN_MATCHES-2)
    rows.append(finished(210,day=NOW-timedelta(minutes=30)))
    rows.append(finished(211,day=NOW-timedelta(days=1),status="TIMED"))
    assert calibrate_matrix(base,rows,as_of=NOW) is None
    # Invalid source distributions are rejected, not silently normalized.
    try:
        calibrate_matrix(base*2,sample(),as_of=NOW)
    except ValueError:
        pass
    else:
        raise AssertionError("Unnormalised probability grid was accepted")


def test_v12_source_population_is_restricted_to_correct_provider_and_league():
    games=sample()+[
        finished(211,provider="other"),
        finished(212,competition=234),
        finished(213,day=NOW+timedelta(days=1))]
    prior=_league_rows(fixture(),games,NOW)
    assert len(prior)==len(sample())
    assert all(match["provider"]=="API-Football" and match["competition"]["id"]==44
               for match in prior)
