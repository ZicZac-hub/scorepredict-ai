from datetime import datetime,timedelta,timezone
import json
import pytest
from forecast_ledger import _forecast
import v44_market_ablation as audit

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)

def model(key="v9",**changes):
    row={"mode":"prospective","provider":"API-Football","fixture_id":"11",
         "model_key":key,"model_version":key+"-test","horizon":"market_control" if key=="v8_control" else "research",
         "home_team_id":"1","away_team_id":"2","home_team":"A","away_team":"B",
         "input_digest":"original-digest","evidence_as_of_utc":(NOW-timedelta(minutes=5)).isoformat(),
         "captured_at_utc":NOW.isoformat(),"kickoff_utc":(NOW+timedelta(hours=2)).isoformat(),
         "score_basis":"regulation","score_matrix":[[.04]*5 for _ in range(5)],
         "market_used":key=="v9","market_snapshot":{"snapshot_utc":NOW.isoformat(),
         "settlement":"regulation_90_minutes","winner":[.5,.25,.25],"totals":[.5,.5]} if key=="v9" else None}
    return _forecast({**row,**changes})

def save(root,row):
    p=root/"forecasts"/row["provider"]/row["fixture_id"]/(row["forecast_id"]+".json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(row))

def test_exact_capture_and_evidence_match(tmp_path):
    save(tmp_path,model("v8_control"))
    save(tmp_path,model())
    rows,counts,rejects=audit.match_archive(tmp_path,NOW)
    assert len(rows)==1 and not rejects
    assert counts["matched_fixtures"]==1

def test_different_evidence_or_time_not_matched(tmp_path):
    save(tmp_path,model("v8_control"))
    save(tmp_path,model(input_digest="different"))
    rows,counts,_=audit.match_archive(tmp_path,NOW)
    assert not rows and counts["unmatched_or_ambiguous_groups"]==2
    assert audit.match_key(model())!=audit.match_key(model(captured_at_utc=(NOW+timedelta(seconds=1)).isoformat()))

def test_future_and_invalid_market_rejected():
    row=model()
    row["market_snapshot"]["snapshot_utc"]=(NOW+timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError,match="outside_capture"):
        audit.qualify_market(row)
    row=model()
    row["market_snapshot"]["winner"]=[.5,.25,.4]
    with pytest.raises(ValueError,match="invalid_winner"):
        audit.qualify_market(row)

def test_empty_report_does_not_claim_accuracy():
    summary,paired=audit._summary([])
    assert summary["v9"]["hit_rate"] is None
    assert paired["mean_market_minus_control_log_loss"] is None
