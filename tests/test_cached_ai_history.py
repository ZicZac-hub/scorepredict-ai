"""Time-safe local AI history uses known cached facts without any HTTP calls."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest
import requests

from cached_ai_history import cached_history
from professional_match_evidence import build_dossier


NOW = datetime(2031, 11, 20, 8, tzinfo=timezone.utc)
KEY = "non-production-test-key"
ROW = {"provider": "API-Football", "fixture_id": "900", "competition": "PL",
       "kickoff_utc": (NOW+timedelta(hours=4)).isoformat(),
       "home_team_id": "11", "away_team_id": "22",
       "home_team": "Home", "away_team": "Away"}


@pytest.fixture(autouse=True)
def no_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("AI cache history must never fetch the network")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


def game(number, *, days=1, home=11, away=33, league=39, season=2031, hg=2, ag=1):
    return {"fixture": {"id": number, "date": (NOW-timedelta(days=days)).isoformat(),
                        "status": {"short": "FT"}},
            "league": {"id": league, "season": season, "name": "Verified league"},
            "teams": {"home": {"id": home, "name": "Team "+str(home)},
                      "away": {"id": away, "name": "Team "+str(away)}},
            "goals": {"home": hg, "away": ag},
            "score": {"fulltime": {"home": hg, "away": ag}}}


def write_cache(root, rows, *, stamp=NOW-timedelta(seconds=60), season=2031,
                key=KEY, payload_override=None):
    signature=json.dumps(["/fixtures", {"league": 39, "season": season},
                          hashlib.sha256(key.encode()).hexdigest()], sort_keys=True)
    path=root/(hashlib.sha256(signature.encode()).hexdigest()+".json")
    payload={"time":stamp.timestamp(), "data":rows} if payload_override is None else payload_override
    path.write_text(json.dumps(payload))
    return path


def fixture(row=ROW):
    return {"provider": row["provider"], "id": row["fixture_id"],
            "utcDate": row["kickoff_utc"], "competition": {"id":39,"code":"PL"},
            "homeTeam": {"id":row["home_team_id"], "name":row["home_team"]},
            "awayTeam": {"id":row["away_team_id"], "name":row["away_team"]}}


def test_cached_regulation_results_populate_real_5_10_20_windows(tmp_path):
    rows=[game(100+n,days=n,home=11,away=1000+n,hg=3,ag=1) for n in range(1,23)]
    rows += [game(200+n,days=n,home=2000+n,away=22,hg=0,ag=2) for n in range(1,23)]
    path=write_cache(tmp_path,rows)
    original=path.read_bytes()
    result=cached_history(ROW,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert len(result["history"])==44
    dossier=build_dossier(fixture(),result["history"],as_of=NOW)
    form=dossier["features"]["goal_form"]["value"]
    assert [form["home"]["last"+str(n)]["matches"] for n in (5,10,20)]==[5,10,20]
    assert [form["away"]["last"+str(n)]["matches"] for n in (5,10,20)]==[5,10,20]
    assert form["home"]["last20"]["goals_for_per_match"]==3
    assert form["away"]["last20"]["goals_for_per_match"]==2
    assert result["external_requests"]==0
    assert result["is_original_statistical_input"] is False
    assert path.read_bytes()==original
    assert KEY not in json.dumps(result)


def test_known_cache_fetch_time_is_not_relabelled_as_current_observation(tmp_path):
    observed=NOW-timedelta(seconds=73)
    write_cache(tmp_path,[game(1)],stamp=observed)
    result=cached_history(ROW,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert result["source_metadata"][0]["fetched_at_utc"]==observed.isoformat()
    dossier=build_dossier(fixture(),result["history"],as_of=NOW)
    assert dossier["features"]["goal_form"]["source_observed_at_utc"]==observed.isoformat()


@pytest.mark.parametrize("seconds", [-1, 901])
def test_future_or_expired_cache_is_missing_not_a_new_fetch(tmp_path,seconds):
    write_cache(tmp_path,[game(1)],stamp=NOW-timedelta(seconds=seconds))
    result=cached_history(ROW,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert result["history"]==[]
    assert result["source_metadata"]==[]
    assert result["warnings"]
    assert result["external_requests"]==0


@pytest.mark.parametrize("key", ["", "wrong-test-key"])
def test_missing_matching_cache_or_credentials_does_not_fabricate_form(tmp_path,key):
    write_cache(tmp_path,[game(1)])
    result=cached_history(ROW,as_of=NOW,api_key=key,cache_root=tmp_path)
    assert not result["history"] and result["warnings"]
    assert result["external_requests"]==0


def test_cache_results_are_provider_competition_team_and_asof_qualified(tmp_path):
    rows=[game(1), game(2,league=140), game(3,home=71,away=72),
          game(4,season=2030), game(900), game(5,days=-1), game(6,days=1/24)]
    write_cache(tmp_path,rows)
    result=cached_history(ROW,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert [row["id"] for row in result["history"]]==[1]
    changed={**ROW,"provider":"Unsupported provider"}
    assert cached_history(changed,as_of=NOW,api_key=KEY,cache_root=tmp_path)["history"]==[]


def test_conflicting_cache_results_are_excluded_instead_of_selected(tmp_path):
    original=game(1)
    conflict=deepcopy(original)
    conflict["score"]["fulltime"]["home"]=8
    write_cache(tmp_path,[original,conflict])
    result=cached_history(ROW,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert result["history"]==[]


def test_research_digest_tracks_exact_used_facts_without_changing_original_row(tmp_path):
    row={**ROW,"input_digest":"immutable-original-model-digest"}
    original=deepcopy(row)
    write_cache(tmp_path,[game(1)])
    first=cached_history(row,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    again=cached_history(row,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert first["input_digest"]==again["input_digest"]
    # Unrelated league results are outside the actual team research dossier.
    write_cache(tmp_path,[game(1),game(2,home=71,away=72)])
    unrelated=cached_history(row,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert unrelated["input_digest"]==first["input_digest"]
    assert unrelated["league_evidence_digest"]!=first["league_evidence_digest"]
    write_cache(tmp_path,[game(1,hg=3)])
    changed=cached_history(row,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert changed["input_digest"]!=first["input_digest"]
    assert row==original
    assert row["input_digest"]=="immutable-original-model-digest"


@pytest.mark.parametrize("payload", [{"time":True,"data":[]},
    {"time":NOW.timestamp(),"data":{}}, {"time":"now","data":[]}, {}])
def test_malformed_cache_remains_explicitly_unavailable(tmp_path,payload):
    write_cache(tmp_path,[],payload_override=payload)
    result=cached_history(ROW,as_of=NOW,api_key=KEY,cache_root=tmp_path)
    assert result["history"]==[]
    assert result["warnings"]
    assert result["external_requests"]==0
