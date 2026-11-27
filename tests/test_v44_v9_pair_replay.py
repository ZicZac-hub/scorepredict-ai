from datetime import datetime,timedelta,timezone
import v44_v9_pair_replay as replay

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)

def model(i="1",key="v8_control",captured=NOW):
    return {"provider":"API-Football","fixture_id":i,"home_team_id":i+"1","away_team_id":i+"2",
            "home_team":"Home "+i,"away_team":"Away "+i,
            "kickoff_utc":(NOW+timedelta(hours=2)).isoformat(),
            "captured_at_utc":captured.isoformat(),"evidence_as_of_utc":(captured-timedelta(minutes=1)).isoformat(),
            "input_digest":"same","model_key":key,"market_used":key=="v9",
            "market_snapshot":{"snapshot_utc":captured.isoformat(),"settlement":"regulation_90_minutes",
                               "winner":[.5,.25,.25],"totals":[.5,.5]}}

def test_source_matching_never_accepts_future_or_wrong_team():
    a,b=model(),model(key="v9")
    assert replay.match_leg(a,(a,b),NOW)==(a,b)
    assert replay.match_leg(a,(a,b),NOW-timedelta(seconds=1)) is None
    assert replay.match_leg({**a,"home_team_id":"wrong"},(a,b),NOW) is None
    assert replay.match_leg(a,(a,{**b,"input_digest":"different"}),NOW) is None

def test_training_excludes_both_legs_aliases_and_late_finals():
    a,b=model("1"),model("2")
    def row(i,known,event=None):
        m=model(i)
        return {"identity":replay.inventory._identity(m),
                "event":event or replay.inventory._event(m),"known":known}
    old=NOW-timedelta(days=1)
    rows=[row("1",old),row("2",old),row("3",old,replay.inventory._event(b)),
          row("4",NOW),row("5",NOW+timedelta(seconds=1)),row("6",old)]
    assert replay.earlier_training(rows,[a,b],NOW)==[rows[-1]]

def test_no_accuracy_claim_without_matched_results():
    for report in replay.summary([],replay.FIXED).values():
        assert report["pairs"]==0
        assert report["hit_rate"] is None
        assert report["score_log_loss"] is None
