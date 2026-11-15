"""V19 real-market source provenance and prospective exact-score challenger tests."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import pytest

from v19_match_intelligence import (
    VERSION, qualified_market, top_five, choose_market_pair, freeze,
    evaluate, readiness,
)

NOW=datetime(2030,1,1,4,tzinfo=timezone.utc)


def forecast(fid,home,away,model="v8",grid=None,market_minutes=-30):
    matrix=grid or [[0.40,0.10],[0.25,0.25]]
    recorded=NOW-timedelta(hours=2) if model=="v8" else NOW+timedelta(minutes=market_minutes)
    asof=NOW-timedelta(hours=3)
    f={"mode":"prospective","provider":"API-Football","model_key":model,
       "fixture_id":str(fid),"forecast_id":model+str(fid),
       "input_digest":"source"+str(fid),
       "home_team_id":str(home),"away_team_id":str(away),
       "home_team":"Team "+str(home),"away_team":"Team "+str(away),
       "kickoff_utc":(NOW+timedelta(hours=3)).isoformat(),
       "captured_at_utc":recorded.isoformat(),
       "evidence_as_of_utc":asof.isoformat(),
       "score_matrix":matrix,"market_used":model in ("v9","v10"),
       "outcome_probabilities":{"home":matrix[1][0],
                                "draw":matrix[0][0]+matrix[1][1],
                                "away":matrix[0][1]},
       "research_context":{}}
    if model in ("v9","v10"):
        f["market_snapshot"]={
            "bookmaker":"Pinnacle" if model=="v9" else "Three-book median",
            "snapshot_utc":(NOW-timedelta(hours=1)).isoformat(),
            "settlement":"regulation_90_minutes",
            "winner":[0.36,0.30,0.34],
            "bookmaker_count":3 if model=="v10" else 1,
            "bookmaker_names":["Pinnacle","Bet365","Another"] if model=="v10" else ["Pinnacle"]}
    return f


def save(root,f):
    loc=root/"forecasts"/f["provider"]/f["fixture_id"]
    loc.mkdir(parents=True,exist_ok=True)
    (loc/(f["forecast_id"]+".json")).write_text(json.dumps(f))


def pair(a,b):
    out={"mode":"prospective","status":"frozen_pre_kickoff",
         "sydney_date":(NOW+timedelta(hours=3,minutes=0)).astimezone(
             __import__("zoneinfo").ZoneInfo("Australia/Sydney")).date().isoformat(),
         "captured_at_utc":(NOW-timedelta(minutes=15)).isoformat(),
         "first":{k:a[k] for k in (
             "provider","fixture_id","forecast_id","home_team_id","away_team_id","kickoff_utc")},
         "second":{k:b[k] for k in (
             "provider","fixture_id","forecast_id","home_team_id","away_team_id","kickoff_utc")},
         "five_doubles":[{"score1":f"{i}-0","score2":"1-0",
                          "joint_probability":0.01} for i in range(5)],
         "estimated_coverage":0.05}
    return out


def actual(f,home=0,away=0,utc_delta=6):
    return {"fixture_id":int(f["fixture_id"]),"provider":f["provider"],
            "home_team_id":f["home_team_id"],
            "away_team_id":f["away_team_id"],
            "kickoff_utc":f["kickoff_utc"],"score_basis":"regulation",
            "evidence_version":2,"status":"FT",
            "home_goals":home,"away_goals":away,
            "verified_at_utc":(NOW+timedelta(hours=utc_delta)).isoformat()}


def test_timestamped_market_validates_original_source(tmp_path):
    a=forecast(42,11,12)
    m=forecast(42,11,12,model="v9")
    save(tmp_path,a);save(tmp_path,m)
    packet,reason=qualified_market(a,m,root=tmp_path,cutoff=NOW)
    assert reason is None and packet["market_model"]=="v9"
    assert packet["raw_odds_packet_independently_replayable"] is False
    m2=dict(m,input_digest="changed")
    result,reason=qualified_market(a,m2,root=tmp_path,cutoff=NOW)
    assert result is None and reason.startswith("mismatched_original_pair_source")


def test_stale_after_kickoff_and_wrong_bookmaker_rejected(tmp_path):
    base=forecast(42,11,12)
    market=forecast(42,11,12,model="v9")
    save(tmp_path,base)
    old=dict(market)
    old["market_snapshot"]=dict(market["market_snapshot"],
                                snapshot_utc=(NOW-timedelta(days=1)).isoformat())
    save(tmp_path,old)
    assert qualified_market(base,old,root=tmp_path,cutoff=NOW)[0] is None
    save(tmp_path,market)
    later=dict(market,captured_at_utc=(NOW+timedelta(hours=5)).isoformat())
    save(tmp_path,later)
    assert qualified_market(base,later,root=tmp_path,cutoff=NOW)[0] is None


def test_three_book_consensus_needs_three_independent_names(tmp_path):
    base=forecast(7,11,12);market=forecast(7,11,12,"v10")
    market["market_snapshot"]["bookmaker_names"]=["Same","same","Same"]
    save(tmp_path,base);save(tmp_path,market)
    result,reason=qualified_market(base,market,root=tmp_path,cutoff=NOW)
    assert result is None
    assert reason=="multi_book_market_not_complete"


def test_top_five_sorted_by_joint_probability():
    a=forecast(1,1,2)
    b=forecast(2,3,4)
    options,coverage=top_five(a,b)
    assert len(options)==5 and len({(x["score1"],x["score2"]) for x in options})==5
    assert coverage==pytest.approx(sum(x["joint_probability"] for x in options))
    assert all(options[i]["joint_probability"]>=options[i+1]["joint_probability"] for i in range(4))


def test_pair_freezing_and_read_only_verification(tmp_path):
    s=tmp_path/"source";d=tmp_path/"v19"
    a,b=forecast(1,101,102),forecast(2,201,202)
    ma,mb=forecast(1,101,102,"v9"),forecast(2,201,202,"v9")
    for x in (a,b,ma,mb):save(s,x)
    original=pair(a,b)
    loc=s/"doubles"/(original["sydney_date"]+".json")
    loc.parent.mkdir(parents=True,exist_ok=True)
    loc.write_text(json.dumps(original))
    selected,reason=choose_market_pair(original,[ma,mb],s,NOW)
    assert reason is None and selected["market_model"]=="v9"
    first=freeze(root=d,source_root=s,now=NOW,market_rows=[ma,mb])
    assert first["new_market_research_pairs"]==[original["sydney_date"]]
    frozen=list((d/"doubles").glob("*.json"))[0]
    evidence=frozen.read_text()
    assert not freeze(root=d,source_root=s,now=NOW,market_rows=[ma,mb])["new_market_research_pairs"]
    assert evidence==frozen.read_text()
    pending=evaluate(root=d,source_root=s,verified={str(a["fixture_id"]):actual(a)})
    assert pending["verified_sets"]==0 and pending["pending_sets"]==1
    settled=evaluate(root=d,source_root=s,
                     verified={str(a["fixture_id"]):actual(a),
                               str(b["fixture_id"]):actual(b)})
    assert settled["verified_sets"]==1
    assert settled["market_wins"]==int(any(
        x["score1"]=="0-0" and x["score2"]=="0-0"
        for x in selected["five_doubles"]))
    assert settled["real_football_probability_accuracy_improvement_not_yet_proven"]


def test_missing_market_on_second_leg_abstains(tmp_path):
    a,b=forecast(1,101,102),forecast(2,201,202)
    ma=forecast(1,101,102,"v9")
    for x in (a,b,ma):save(tmp_path,x)
    selected,reason=choose_market_pair(pair(a,b),[ma],tmp_path,NOW)
    assert selected is None
    assert reason["reason"]=="two_leg_archived_market_evidence_incomplete"


def test_source_mismatch_and_result_identity_not_awarded(tmp_path):
    s=tmp_path/"src";d=tmp_path/"v19"
    a,b=forecast(1,101,102),forecast(2,201,202)
    ma,mb=forecast(1,101,102,"v9"),forecast(2,201,202,"v9")
    for x in (a,b,ma,mb):save(s,x)
    original=pair(a,b)
    p=s/"doubles"/(original["sydney_date"]+".json")
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(original))
    freeze(root=d,source_root=s,now=NOW,market_rows=[ma,mb])
    wrong=actual(b);wrong["away_team_id"]="not-original"
    report=evaluate(root=d,source_root=s,verified={"1":actual(a),"2":wrong})
    assert report["verified_sets"]==0
    wrongscore=dict(ma);wrongscore["score_matrix"]=[[.25,.25],[.25,.25]]
    save(s,wrongscore)
    report=evaluate(root=d,source_root=s,verified={"1":actual(a),"2":actual(b)})
    assert report["verified_sets"]==0
    assert report["excluded_integrity_failures"]


def test_no_hindsight_and_readiness_truthful(tmp_path):
    s=tmp_path/"source";d=tmp_path/"v19"
    a,b=forecast(1,101,102),forecast(2,201,202)
    ma,mb=forecast(1,101,102,"v9"),forecast(2,201,202,"v9")
    for x in (a,b,ma,mb):save(s,x)
    selection,reason=choose_market_pair(pair(a,b),[ma,mb],s,NOW+timedelta(hours=4))
    assert selection is None
    original=pair(a,b)
    p=s/"doubles"/(original["sydney_date"]+".json")
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(original))
    assert freeze(root=d,source_root=s,now=NOW+timedelta(hours=4),market_rows=[ma,mb])["new_market_research_pairs"]==[]
    status=readiness(source_root=s,root=d)
    assert status["true_shot_quality_xg_feed_integrated"] is False
    assert status["timestamped_lineup_adjustment_trained"] is False
    assert status["market_quotes_raw_http_response_audited"] is False
