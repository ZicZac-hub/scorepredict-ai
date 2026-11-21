"""Synthetic source identity and timetable checks; no real result claims."""
import hashlib
from copy import deepcopy
import pytest
import run_goal_distribution_2025_26_audit as audit

def samples():
    compact=[
        {"date":"2025-08-16","team1":"AFC Bournemouth",
         "team2":"Arsenal FC","score":[0,0]},
        {"date":"2025-08-17","team1":"Chelsea FC",
         "team2":"Sunderland AFC","score":{"ft":[2,1]}}
    ]
    detailed=[
        {"date":"2025-08-17","time":"12:30 UTC+1",
         "team1":"Chelsea FC","team2":"Sunderland",
         "score":{"ft":[2,1]}},
        {"date":"2025-08-16","time":"19:00 UTC+0",
         "team1":"Bournemouth","team2":"Arsenal",
         "score":{"ft":[0,0]}}
    ]
    return compact,detailed

def test_fixed_source_identity_and_explicit_full_time(monkeypatch):
    monkeypatch.setattr(audit,"EXPECTED_SEASON_EVENTS",2)
    monkeypatch.setattr(audit,"EXPECTED_COMPRESSED_SCORE_ROWS",1)
    compact,detailed=samples()
    rows,compressed=audit.join_season(compact,detailed)
    assert compressed==1
    assert [r["homeTeam"]["id"] for r in rows]==["bournemouth","chelsea"]
    assert [r["score"]["fullTime"]["home"] for r in rows]==[0,2]
    assert rows[0]["utcDate"]=="2025-08-16T19:00:00+00:00"
    assert rows[1]["utcDate"]=="2025-08-17T11:30:00+00:00"

def test_wrong_source_goal_or_missing_identity_is_rejected(monkeypatch):
    monkeypatch.setattr(audit,"EXPECTED_SEASON_EVENTS",2)
    monkeypatch.setattr(audit,"EXPECTED_COMPRESSED_SCORE_ROWS",1)
    compact,detailed=samples()
    changed=deepcopy(detailed)
    changed[1]["score"]["ft"]=[1,0]
    with pytest.raises(ValueError,match="source_full_time_disagreement"):
        audit.join_season(compact,changed)
    changed=deepcopy(detailed)
    changed[1]["team1"]="Unknown"
    with pytest.raises(ValueError,match="source_fixture_identity"):
        audit.join_season(compact,changed)

def test_pinned_git_blob_validation():
    content=b"real verified result bytes"
    blob=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\x00"+content).hexdigest()
    audit.verify_blob(content,blob)
    with pytest.raises(ValueError,match="pinned_git_blob"):
        audit.verify_blob(content+b"changed",blob)

def test_unknown_offset_never_guessed():
    with pytest.raises(ValueError,match="unsupported_source_utc_offset"):
        audit.parse_source_utc("2025-08-15","20:00")
