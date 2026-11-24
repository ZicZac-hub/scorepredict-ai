from copy import deepcopy
from uefa_history_coverage import audit_history

def game(identifier, year, date, home, away):
    return {"provider":"API-Football","id":identifier,"utcDate":date,
            "competition":{"code":"CL","id":2,"season":year},
            "homeTeam":{"id":home},"awayTeam":{"id":away},
            "status":"FINISHED","score":{"fullTime":{"home":2,"away":1}},
            "score_basis":"regulation"}

def test_season_breakdown_preserves_source_policy_and_distinguishes_empty():
    fixture = game(99,2026,"2026-11-25T19:00:00Z",1,2)
    current = game(1,2026,"2026-11-01T19:00:00Z",1,3)
    later = game(2,2026,"2026-11-26T19:00:00Z",2,4)
    evidence = {"as_of_utc":"2026-11-24T00:00:00Z","history":[current,later],
                "exclusions":{"not_available_as_of":1}}
    cache = {("API-Football","2",2026):([current,later],{"freshness_known":True},None),
             ("API-Football","2",2025):([],{"freshness_known":False},None)}
    before = deepcopy((fixture,evidence,cache))
    report = audit_history(fixture,evidence,cache)
    assert report["qualified_team_matches"] == {"home":1,"away":0}
    assert report["seasons"][0]["returned_team_rows"] == {"home":1,"away":1}
    assert report["seasons"][1]["source_state"] == "empty_response"
    assert report["minimum_missing_to_five"] == 5
    assert report["absence_not_proof_of_nonparticipation"]
    assert (fixture,evidence,cache) == before
    cache[("API-Football","2",2025)] = ([],{}, "ProviderQuotaError")
    assert audit_history(fixture,evidence,cache)["seasons"][1]["source_state"] == "request_unavailable"
    del cache[("API-Football","2",2025)]
    assert audit_history(fixture,evidence,cache)["seasons"][1]["source_state"] == "not_observed"

def test_non_uefa_history_not_included():
    fixture = game(99,2026,"2026-11-25T19:00:00Z",1,2)
    fixture["competition"]["code"] = "PL"
    assert audit_history(fixture,{}, {}) is None
