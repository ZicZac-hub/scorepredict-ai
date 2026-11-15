"""Protect all-source result tracking from hindsight, bad IDs and missing finals."""
from datetime import datetime,timezone,timedelta
import json
import os

import pytest

from ziczac_interactive_capture import single, pair, store, load
from ziczac_unified_audit import (
    _status, _final_matches, summary, private_records,
    archived_central_records, build)

NOW=datetime(2031,11,17,3,0,tzinfo=timezone.utc)
KICKOFF="2031-11-20T03:00:00Z"


def fixture(fid,home,away,kickoff=KICKOFF):
    return {"id":fid,"provider":"API-Football","utcDate":kickoff,
            "homeTeam":{"id":fid*2,"name":home},
            "awayTeam":{"id":fid*2+1,"name":away}}


def verified(f,score1=1,score2=1,**changes):
    result={"provider":f["provider"],"fixture_id":str(f["id"]),
            "kickoff_utc":f["utcDate"],"home_team_id":str(f["homeTeam"]["id"]),
            "away_team_id":str(f["awayTeam"]["id"]),
            "home_goals":score1,"away_goals":score2,
            "verified_at_utc":"2031-11-21T00:00:00Z",
            "score_basis":"regulation","evidence_version":2,"status":"FT"}
    result.update(changes)
    return result


def test_one_original_single_per_fixture_and_ai_provider():
    f=fixture(101,"Home","Away")
    a=single(f,[1,1],"ten_ai",ai_provider="gemini",now=NOW)
    b=single(f,[3,0],"ten_ai",ai_provider="gemini",now=NOW+timedelta(hours=1))
    c=single(f,[1,1],"ten_ai",ai_provider="openai",now=NOW)
    assert a["record_id"]==b["record_id"]
    assert a["record_id"]!=c["record_id"]
    assert a["score"]=="1-1"
    assert b["score"]=="3-0"


def test_deep_research_requires_named_method_and_original_forecast():
    f=fixture(102,"A","B")
    a=single(f,"2-1","deep_research",method="Evidence Ensemble v8",now=NOW)
    b=single(f,"1-0","deep_research",method="Market-Informed v9",now=NOW)
    assert a["record_id"]!=b["record_id"]
    with pytest.raises(ValueError):
        single(f,"2-1","deep_research",now=NOW)


def test_pre_kickoff_required_for_single_and_pair():
    a=fixture(201,"Home A","Away A")
    b=fixture(202,"Home B","Away B")
    choices=[{"score1":f"{i}-0","score2":"1-1"} for i in range(5)]
    assert len(pair(a,b,choices,now=NOW)["choices"])==5
    later=datetime(2031,11,20,3,0,tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        single(a,"1-1","ten_ai",ai_provider="gemini",now=later)
    with pytest.raises(ValueError):
        pair(a,b,choices,now=later)


def test_pair_checks_uniqueness_and_same_sydney_date():
    a=fixture(201,"Home A","Away A")
    b=fixture(202,"Home B","Away B",
              kickoff="2031-11-20T20:00:00Z") # 21 Nov Sydney
    options=[{"score1":f"{i}-0","score2":"1-1"} for i in range(5)]
    with pytest.raises(ValueError):
        pair(a,b,options,now=NOW)
    b["utcDate"]=KICKOFF
    with pytest.raises(ValueError):
        pair(a,b,[options[0]]*5,now=NOW)


def test_no_database_is_explicitly_not_saved():
    record=single(fixture(303,"A","B"),"0-0","ten_ai",
                  ai_provider="claude",now=NOW)
    assert store(record,database_url="")=={
        "durable":False,"state":"database_not_configured"}


def test_verified_result_requires_exact_provider_team_kickoff_and_regulation():
    f=fixture(444,"Home","Away")
    leg=single(f,"1-1","ten_ai",ai_provider="gemini",now=NOW)["leg"]
    final=verified(f)
    assert _final_matches(leg,final)
    for field,alter in [("provider","football-data.org"),
                        ("fixture_id","445"),
                        ("home_team_id","10000"),
                        ("kickoff_utc","2031-11-20T04:00:00Z"),
                        ("score_basis","extra_time"),
                        ("status","LIVE"),
                        ("evidence_version",1),
                        ("verified_at_utc","2031-11-19T00:00:00Z")]:
        damaged={**final,field:alter}
        assert not _final_matches(leg,damaged),field


def test_exact_hits_misses_pending_and_five_double_full_pair_match():
    f1=fixture(11,"A","B")
    f2=fixture(12,"C","D")
    a=single(f1,[1,1],"ten_ai",ai_provider="gemini",now=NOW)
    b=single(f1,[1,0],"ten_ai",ai_provider="openai",now=NOW)
    sets=[{"score1":f"{i}-0","score2":"1-1"} for i in range(5)]
    sets[0]={"score1":"1-1","score2":"2-2"}
    picks=pair(f1,f2,sets,now=NOW)
    rows=private_records([a,b,picks])
    assert len(rows)==3
    ledger={(f1["provider"],str(f1["id"])):verified(f1,1,1),
            (f2["provider"],str(f2["id"])):verified(f2,2,2)}
    report=summary(rows,ledger,now=NOW)
    assert report["singles"]["hit"]==1
    assert report["singles"]["miss"]==1
    assert report["five_doubles"]["hit"]==1
    assert report["five_doubles"]["verified"]==1
    assert report["by_source"]["ai_gemini"]["hit"]==1
    assert report["by_source"]["ai_openai"]["miss"]==1
    ledger.pop((f2["provider"],str(f2["id"])))
    pending=summary(rows,ledger,now=NOW)
    assert pending["five_doubles"]["pending"]==1
    assert pending["five_doubles"]["verified"]==0
    assert pending["five_doubles"]["hit_rate"] is None


def test_invalid_hindsight_and_missing_team_ids_are_not_grades():
    f=fixture(13,"A","B")
    post=single(f,[1,1],"ten_ai",ai_provider="gemini",now=NOW)
    post["captured_at_utc"]="2031-11-21T00:00:00Z"
    invalid=private_records([post])
    assert invalid==[]
    unverified=archived_central_records(singles=[{
        "provider":"legacy-unknown","fixture_id":31,"predicted_score":"1-1",
        "saved_at":NOW.isoformat(),"kickoff":KICKOFF}])
    assert unverified[1]==1


def test_github_public_report_cannot_contain_private_db_records(tmp_path,monkeypatch):
    import ziczac_unified_audit as audit
    monkeypatch.setattr(audit,"read_latest",lambda root:{})
    for dirname in ("v10","prematch","llm"):
        (tmp_path/dirname).mkdir()
    record=single(fixture(444,"Home","Away"),"1-1","ten_ai",
                  ai_provider="openai",now=NOW)
    public=build(v10_root=tmp_path/"v10",prematch_root=tmp_path/"prematch",
                 llm_root=tmp_path/"llm",now=NOW)
    private=build(v10_root=tmp_path/"v10",prematch_root=tmp_path/"prematch",
                  llm_root=tmp_path/"llm",now=NOW,private=[record])
    assert public["singles"]["recorded"]==0
    assert private["singles"]["recorded"]==1
    assert public["records"]==[]


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"),
                    reason="Requires GitHub Actions PostgreSQL test service")
def test_postgres_original_cannot_be_rewritten_by_later_ai_result():
    import psycopg
    address=os.environ["TEST_DATABASE_URL"]
    import uuid
    fid=int(uuid.uuid4().int%800000000)+100000000
    f=fixture(fid,"X","Y")
    first=single(f,"1-1","ten_ai",ai_provider="gemini",now=NOW)
    second=single(f,"3-0","ten_ai",ai_provider="gemini",
                  now=NOW+timedelta(hours=1))
    initial=store(first,address)
    assert initial["durable"] and initial["state"]=="saved"
    later=store(second,address)
    assert later["durable"] and later["state"]=="original_preserved"
    saved,status=load(address)
    assert status=="connected"
    row=next(r for r in saved if r["record_id"]==first["record_id"])
    assert row["score"]=="1-1"


def test_private_forecast_write_requires_owner_login(monkeypatch):
    import ziczac_interactive_capture as audit
    class FakeSite:
        secrets={"ZICZAC_ACCESS_PASSWORD":"configured-private-password",
                 "DATABASE_URL":"must-not-connect-to-db"}
        session_state={}
        def __init__(self):
            self.messages=[]
        def warning(self,text):
            self.messages.append(text)
        def caption(self,text):
            self.messages.append(text)
    site=FakeSite()
    def forbid(*args,**kwargs):
        raise AssertionError("Unauthenticated visitor must never write to private DB")
    monkeypatch.setattr(audit,"store",forbid)
    record=single(fixture(607,"Home","Away"),"2-1","ten_ai",
                  ai_provider="gemini",now=NOW)
    result=audit.store_ui(site,record)
    assert result["state"]=="private_access_required"
    assert result["durable"] is False
    assert "NOT" in site.messages[0]


def test_public_performance_board_never_queries_private_database(monkeypatch):
    import ziczac_unified_audit_ui as ui
    class FakeSite:
        secrets={}
        session_state={}
        def __init__(self):
            self.messages=[]
        def markdown(self,text): self.messages.append(text)
        def caption(self,text): self.messages.append(text)
        def info(self,text): self.messages.append(text)
        def button(self,*args,**kwargs): return True
        def spinner(self,*args,**kwargs):
            import contextlib
            return contextlib.nullcontext()
        def __getattr__(self,name):
            raise AssertionError("Unexpected UI request: "+name)
    site=FakeSite()
    def forbidden(*args,**kwargs):
        raise AssertionError("A public visitor must not inspect a private database")
    monkeypatch.setattr(ui,"build_with_private",forbidden)
    import ziczac_unified_audit as audit
    monkeypatch.setattr(audit,"build",lambda **kw:{})
    monkeypatch.setattr(ui,"load_json",lambda *args:{})
    ui.render(site,database_url="should-not-be-used")
    assert site.session_state.get("ziczac_unified_audit_status") is None


def test_five_double_methods_have_separate_write_once_ids_and_verified_results():
    first = fixture(901, "Home X", "Away X")
    second = fixture(902, "Home Y", "Away Y")
    original = [{"score1": f"{i}-0", "score2": "0-0"} for i in range(5)]
    challenger = [{"score1": f"{i}-0", "score2": "1-0"} for i in range(5)]
    challenger[0] = {"score1": "1-1", "score2": "2-2"}
    research = [{"score1": f"{i}-0", "score2": "1-1"} for i in range(5)]
    saved_original = pair(first, second, original, now=NOW)
    saved_balanced = pair(first, second, challenger, method="balanced_challenger", now=NOW)
    saved_research = pair(first, second, research, method="goal_form_research", now=NOW)
    assert len({x["record_id"] for x in
                (saved_original, saved_balanced, saved_research)}) == 3
    revised = pair(first, second, research, method="goal_form_research",
                   now=NOW + timedelta(hours=1))
    assert revised["record_id"] == saved_research["record_id"]
    finals = {(first["provider"], str(first["id"])): verified(first, 1, 1),
              (second["provider"], str(second["id"])): verified(second, 2, 2)}
    measured = summary(private_records(
        [saved_original, saved_balanced, saved_research]), finals, now=NOW)
    assert measured["by_source"]["ai_five_double"]["miss"] == 1
    assert measured["by_source"]["ai_five_double_balanced_challenger"]["hit"] == 1
    assert measured["by_source"]["ai_five_double_goal_form_research"]["miss"] == 1
    assert measured["five_doubles"]["verified"] == 3
