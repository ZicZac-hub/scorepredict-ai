from datetime import datetime,timedelta,timezone
import json
from forecast_ledger import _forecast
import v44_refresh_expiry_audit as a

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
def save(root,history,age,kind="v8_near",with_receipt=True):
    m=_forecast({"provider":"API-Football","fixture_id":"1","home_team_id":"10","away_team_id":"20",
       "kickoff_utc":(NOW+timedelta(hours=4)).isoformat(),"captured_at_utc":(NOW-timedelta(hours=age)).isoformat(),
       "evidence_as_of_utc":(NOW-timedelta(hours=age,minutes=1)).isoformat(),
       "input_digest":"digest-"+str(age),"model_key":kind,"model_version":kind+"-test",
       "horizon":"near","score_basis":"regulation","score_matrix":[[.04]*5 for _ in range(5)]})
    p=root/"forecasts"/m["provider"]/m["fixture_id"]/(m["forecast_id"]+".json")
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m))
    if with_receipt:
        key=a.early.receipt_key(list(a.event(m)),m["input_digest"])
        p=history/"receipts"/(key+".json");p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"event":list(a.event(m)),"history_input_digest":m["input_digest"],
             "as_of_utc":m["evidence_as_of_utc"],"captured_at_utc":m["evidence_as_of_utc"]}))
    return m

def test_near_capture_expires_before_window_opens(tmp_path,monkeypatch):
    source=tmp_path/"source";history=tmp_path/"history"
    m=save(source,history,23.5)
    monkeypatch.setattr(a.early,"load",lambda path,**k:json.loads(path.read_text()))
    assert a.inspect_leg(m,NOW,NOW,source,history)=="exact_history_valid_at_window_open"
    assert a.inspect_leg(m,NOW+timedelta(hours=1),NOW,source,history)=="one_shot_near_expired"

def test_missing_exact_history_distinguished_from_available_refresh(tmp_path):
    source=tmp_path/"source";history=tmp_path/"history"
    m=save(source,history,2,kind="v8",with_receipt=False)
    assert a.inspect_leg(m,NOW,NOW,source,history)=="refresh_available_missing_history"
    save(source,history,1,with_receipt=False)
    assert a.inspect_leg(m,NOW,NOW,source,history)=="one_shot_near_without_exact_history"


def test_refresh_deadline_and_fixture_identity_are_reported(tmp_path,monkeypatch):
    pairs=tmp_path/"pairs";(pairs/"sets").mkdir(parents=True)
    (pairs/"sets"/"2025-02-01.json").write_text("{}")
    first={"provider":"API-Football","fixture_id":"1","kickoff_utc":(NOW+timedelta(hours=4)).isoformat()}
    second={**first,"fixture_id":"2","kickoff_utc":(NOW+timedelta(hours=5)).isoformat()}
    monkeypatch.setattr(a.sources,"_pair",lambda *args,**kw:({"predicted_at_utc":NOW.isoformat()},[first,second]))
    monkeypatch.setattr(a,"inspect_leg",lambda m,*args:"refresh_available_missing_history" if m["fixture_id"]=="1" else "one_shot_near_without_exact_history")
    r=a.evaluate(pair_root=pairs,now=NOW)
    assert r["counts"]["existing_refresh_path_open"]==1
    row=r["rows"][0]
    assert row["priority_refresh_window_status"]=="open"
    assert row["priority_refresh_closes_utc"]==(NOW+timedelta(hours=3,minutes=15)).isoformat()
    assert row["fixtures"][0]["fixture_id"]=="1"
    assert row["fixtures"][1]["status"]=="one_shot_near_without_exact_history"
