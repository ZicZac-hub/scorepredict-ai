"""V14 automatic website display uses only frozen true prospective records."""
import json
from datetime import datetime,timezone,timedelta
from automatic_doubles_ui import read_upcoming,result_status

NOW=datetime(2031,11,19,22,tzinfo=timezone.utc)
DATE="2031-11-20"
FIRST="2031-11-20T02:00:00Z"
SECOND="2031-11-20T03:00:00Z"


def entry():
    return {
        "policy":"v14-robust-two-model-prospective-shadow-v1",
        "mode":"prospective_robust_v14","status":"frozen_pre_kickoff",
        "sydney_date":DATE,"captured_at_utc":NOW.isoformat(),
        "first":{"kickoff_utc":FIRST,"fixture_id":"100",
                 "home_team":"Team A","away_team":"Team B"},
        "second":{"kickoff_utc":SECOND,"fixture_id":"200",
                  "home_team":"Team C","away_team":"Team D"},
        "lower_envelope_score":0.055,
        "five_doubles":[{"score1":f"{i}-0","score2":"1-1",
                         "v7_joint":0.02,"v8_joint":0.021,
                         "conservative_envelope":0.02}
                        for i in range(5)]
    }


def test_robust_frozen_ui_never_substitutes_experimental_for_original(tmp_path):
    archive=tmp_path/"robust_doubles"/(DATE+".json")
    archive.parent.mkdir()
    archive.write_text(json.dumps(entry()),encoding="utf-8")
    assert read_upcoming(tmp_path,now=NOW,mode="original") is None
    packet=read_upcoming(tmp_path,now=NOW,mode="robust_v14")
    assert packet and packet["mode"]=="prospective_robust_v14"
    assert len(packet["five_doubles"])==5
    assert read_upcoming(tmp_path,now=NOW+timedelta(days=3),
                         mode="robust_v14") is None


def test_v14_verified_final_is_not_taken_from_v8_row(tmp_path):
    packet=entry()
    report={"five_doubles":{"rows":[{"sydney_date":DATE,
                                     "status":"verified","hit":True}]},
            "v14_independently_selected_five_doubles":{
                "rows":[{"sydney_date":DATE,"status":"awaiting_verified_results"}]}}
    file=tmp_path/"performance.json"
    file.write_text(json.dumps(report),encoding="utf-8")
    assert result_status(packet,file)["status"]=="awaiting_verified_results"
    report["v14_independently_selected_five_doubles"]["rows"][0]={
        "sydney_date":DATE,"status":"verified","hit":False,
        "actual_scores":["3-2","1-0"]}
    file.write_text(json.dumps(report),encoding="utf-8")
    assert result_status(packet,file)["hit"] is False


def test_v14_historical_or_after_kickoff_record_not_displayed(tmp_path):
    d=tmp_path/"robust_doubles"
    d.mkdir()
    row=entry()
    row["captured_at_utc"]=(NOW+timedelta(days=2)).isoformat()
    (d/(DATE+".json")).write_text(json.dumps(row),encoding="utf-8")
    assert read_upcoming(tmp_path,now=NOW,mode="robust_v14") is None
