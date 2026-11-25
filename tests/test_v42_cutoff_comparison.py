from datetime import datetime, timezone
import json
import v42_cutoff_comparison as audit

def row(day, hit=True, status="verified", xg=False):
    return {"sydney_date":day,"status":"pending" if status=="pending" else "verified",
        "actual_scores":[[1,0],[2,1]] if status!="pending" else [None,None],
        "methods":{"unified_final":{"status":status,"metrics":{"hit":hit,
            "score_log_loss":2.0 if hit else 3.0,"outcome_brier":0.3 if hit else 0.5} if status=="verified" else None}},
        "unified_final_evidence":{"paired_xg_used":xg,"paired_market_used":False},
        "lineup_context":[{"both_confirmed_xis":False},{"both_confirmed_xis":False}]}

def receipt(root,day,minute,identity="same"):
    p=root/"trials"
    p.mkdir(parents=True,exist_ok=True)
    (p/(day+".json")).write_text(json.dumps({"baseline_pair":{"record_sha256":identity},
        "source_forecast_refs":["a","b"],"decision_as_of_utc":f"2026-11-25T10:{minute:02d}:00+00:00"}))

def test_empty_is_not_zero_accuracy(tmp_path):
    x=audit.compare_reports({"rows":[]},{"rows":[]},early_root=tmp_path,late_root=tmp_path)
    assert x["matched_verified_issued_pairs"]==0
    assert x["matched_late"]["hit_rate"] is None
    assert x["score_log_loss_advantage"]["mean"] is None

def test_common_identity_cutoff_and_abstentions(tmp_path):
    a,b=tmp_path/"early",tmp_path/"late"
    days=["2026-11-25","2026-11-26","2026-11-27","2026-11-28"]
    for i,d in enumerate(days[:3]):
        receipt(a,d,10)
        receipt(b,d,20 if i!=2 else 10)
    x=audit.compare_reports({"rows":[row(days[0],False),row(days[1],status="abstained"),row(days[2]),row(days[3])]},
        {"rows":[row(days[0],True,xg=True),row(days[1]),row(days[2])]},early_root=a,late_root=b)
    assert x["matched_valid_pairs"]==2
    assert x["matched_verified_issued_pairs"]==1
    assert x["late_only_wins"]==1
    assert x["score_log_loss_advantage"]["mean"]==1
    assert x["matched_late_evidence"]["paired_xg_used"]==1
    assert x["early_only_days"]==1
    assert x["pair_rejections"]=={"late_cutoff_not_strictly_later":1}

def test_conflicting_identity_or_result_rejected(tmp_path):
    a,b=tmp_path/"early",tmp_path/"late"
    d="2026-11-25"
    receipt(a,d,10)
    receipt(b,d,20,"different")
    x=audit.compare_reports({"rows":[row(d)]},{"rows":[row(d)]},early_root=a,late_root=b)
    assert x["pair_rejections"]=={"different_pair_identity":1}
    receipt(b,d,20)
    y=row(d)
    y["actual_scores"]=[[0,0],[2,1]]
    x=audit.compare_reports({"rows":[row(d)]},{"rows":[y]},early_root=a,late_root=b)
    assert x["pair_rejections"]=={"settlement_disagreement":1}
