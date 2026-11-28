from datetime import datetime,timedelta,timezone
import copy
import pytest
import v44_paired_uncertainty as u

NOW=datetime(2025,1,1,12,tzinfo=timezone.utc)

def rows(days=10,per_day=3):
    out=[]
    for d in range(days):
        for j in range(per_day):
            a={"exact_log_loss":3.0,"score_brier":1.0,"outcome_brier":.6,"top_score_hit":False}
            b={"exact_log_loss":2.5,"score_brier":.9,"outcome_brier":.5,"top_score_hit":True}
            out.append({"identity":["API-Football",str(d*per_day+j)],
                "kickoff_utc":(NOW+timedelta(days=d)).isoformat(),
                "metrics":{"v8_control":a,"v9":b}})
    return out

def test_constant_differences_are_reproduced():
    r=u.summarize(rows())
    assert r["fixtures"]==30 and r["blocks"]==10
    assert r["status"]=="descriptive_resampling_available"
    assert r["metrics"]["exact_log_loss"]["mean_v9_minus_v8"]==-.5
    assert r["metrics"]["exact_log_loss"]["descriptive_95_percentile_interval"]==[-.5,-.5]
    assert r["metrics"]["top_score_hit"]["mean_v9_minus_v8"]==1.0

def test_few_blocks_and_empty_samples_do_not_invent_intervals():
    r=u.summarize(rows(3,20))
    assert r["fixtures"]==60 and r["blocks"]==3
    assert r["metrics"]["exact_log_loss"]["descriptive_95_percentile_interval"] is None
    assert u.summarize(rows(),period="week")["status"]=="insufficient_distinct_blocks_or_fixtures"
    r=u.summarize([])
    assert r["metrics"]["exact_log_loss"]["mean_v9_minus_v8"] is None

def test_paired_block_resampling_is_reproducible_with_variable_block_sizes():
    data=rows()
    extra=copy.deepcopy(data[0])
    extra["identity"][1]="extra"
    extra["metrics"]["v9"]["exact_log_loss"]=18.0
    data.append(extra)
    a=u.summarize(data);b=u.summarize(data)
    assert a==b
    assert a["metrics"]["exact_log_loss"]["mean_v9_minus_v8"]==0.0
    low,high=a["metrics"]["exact_log_loss"]["descriptive_95_percentile_interval"]
    assert low<0<high

def test_invalid_or_duplicate_rows_rejected():
    data=rows()
    with pytest.raises(ValueError,match="duplicate_fixture"):
        u.summarize(data+[data[0]])
    data[0]["metrics"]["v9"]["exact_log_loss"]=float("nan")
    with pytest.raises(ValueError,match="invalid_loss"):
        u.summarize(data)

def test_empty_archive_has_no_results_or_promotion(tmp_path):
    r=u.evaluate(source_root=tmp_path/"original",prematch_root=tmp_path/"prematch",now=NOW)
    assert r["daily_blocks"]["fixtures"]==0
    assert r["external_requests"]==0 and not r["automatic_promotion"]
    assert not list(tmp_path.iterdir())
