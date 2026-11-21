"""Equivalent source timestamps settle once; actual contradictions stay pending."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json

import pytest

from ziczac_unified_audit import _collect_finals,summary

KICK=datetime(2033,2,1,12,tzinfo=timezone.utc)


def final(**changes):
    row={"provider":"API-Football","fixture_id":55,
         "kickoff_utc":KICK.isoformat(),"home_team_id":11,"away_team_id":22,
         "home_goals":4,"away_goals":0,"score_basis":"regulation",
         "evidence_version":2,"status":"FT",
         "verified_at_utc":(KICK+timedelta(hours=3)).isoformat()}
    return {**row,**changes}


def archives(tmp_path,a,b):
    v10,prematch=tmp_path/"v10",tmp_path/"prematch"
    v10.mkdir();prematch.mkdir()
    for root,row in ((prematch,a),(v10,b)):
        (root/"verified_results.json").write_text(json.dumps({"55":row}))
    return v10,prematch


def test_equivalent_offset_and_id_types_are_one_unchanged_final(tmp_path):
    first=final()
    second=final(kickoff_utc="2033-02-01T12:00:00Z",fixture_id="55",
                 home_team_id="11",away_team_id="22",
                 verified_at_utc="2033-02-01T16:00:00Z")
    v10,prematch=archives(tmp_path,first,second)
    original={path:path.read_bytes() for path in tmp_path.rglob("*.json")}
    results,conflicts=_collect_finals(v10,prematch)
    assert conflicts==set()
    assert results[("API-Football","55")]==first
    assert all(path.read_bytes()==value for path,value in original.items())


@pytest.mark.parametrize("change",[
    {"home_goals":3},{"away_team_id":33},
    {"kickoff_utc":"2033-02-01T13:00:00Z"},{"score_basis":"extra_time"}])
def test_genuinely_conflicting_score_or_identity_is_not_settled(tmp_path,change):
    v10,prematch=archives(tmp_path,final(),final(**change))
    results,conflicts=_collect_finals(v10,prematch)
    assert conflicts=={("API-Football","55")}
    assert results=={}


def test_invalid_timestamp_cannot_hide_an_otherwise_valid_final(tmp_path):
    v10,prematch=archives(tmp_path,final(kickoff_utc="unknown"),final())
    results,conflicts=_collect_finals(v10,prematch)
    assert conflicts==set()
    assert results[("API-Football","55")]["home_goals"]==4


def test_summary_cannot_grade_a_result_from_after_its_evaluation_cutoff():
    result=final()
    leg={k:result[k] for k in ("provider","fixture_id","kickoff_utc",
                              "home_team_id","away_team_id")}
    record={"id":"original","source":"control","kind":"single",
            "captured_at_utc":(KICK-timedelta(hours=2)).isoformat(),
            "legs":[leg],"choices":[["4-0"]]}
    before=summary([record],{("API-Football","55"):result},now=KICK+timedelta(hours=2))
    assert before["singles"]["verified"]==0
    assert before["singles"]["pending"]==1
    assert before["singles"]["hit_rate"] is None
    available=summary([record],{("API-Football","55"):result},now=KICK+timedelta(hours=3))
    assert available["singles"]["verified"]==1
    assert available["singles"]["hit"]==1


def test_many_model_records_for_one_match_are_not_independent_fixture_samples():
    result=final()
    leg={k:result[k] for k in ("provider","fixture_id","kickoff_utc",
                              "home_team_id","away_team_id")}
    record={"id":"first","source":"model_a","kind":"single",
            "captured_at_utc":(KICK-timedelta(hours=2)).isoformat(),
            "legs":[leg],"choices":[["4-0"]]}
    second={**record,"id":"second","source":"model_b","choices":[["1-1"]]}
    report=summary([record,second],{("API-Football","55"):result},
                   now=KICK+timedelta(hours=4))
    assert report["singles"]["verified"]==2
    assert report["unique_verified_single_fixtures"]==1
    assert report["single_record_counts_include_multiple_models"] is True
