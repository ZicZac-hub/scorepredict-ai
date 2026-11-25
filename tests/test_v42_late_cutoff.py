"""The late cutoff is a separate protocol; no calls or original receipt edits."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import v42_research_trial as early
import v42_late_research_trial as late

NOW = datetime(2031, 10, 1, 10, tzinfo=timezone.utc)

def test_late_cutoff_boundaries_and_distinct_archive():
    assert late.ROOT != early.ROOT
    assert late.VERSION != early.VERSION
    for minutes, expected in [(181,False),(180,False),(66,False),(65,True),(6,True),(5,False),(0,False),(-1,False)]:
        models=[{"kickoff_utc":(NOW+timedelta(minutes=minutes)).isoformat()},
                {"kickoff_utc":(NOW+timedelta(minutes=minutes+30)).isoformat()}]
        assert late._window(models,NOW) is expected
    assert early.PROTOCOL["maximum_lead_seconds_inclusive"] == 10800
    assert late.PROTOCOL["maximum_lead_seconds_inclusive"] == 3900
    assert late.PROTOCOL["maximum_component_age_seconds"] == early.PROTOCOL["maximum_component_age_seconds"]
    assert late.PROTOCOL["automatic_model_promotion"] is False

def test_late_protocol_immutably_registers_separately(tmp_path):
    a = early.register_protocol(root=tmp_path/"early", now=NOW)
    p = tmp_path/"early"/"protocol.json"
    before = p.read_bytes()
    b = late.register_protocol(root=tmp_path/"late", now=NOW)
    assert a["protocol_sha256"] != b["protocol_sha256"]
    assert p.read_bytes() == before
    assert "v42_late_research_trial.py" in b["implementation_sha256"]
    assert late.register_protocol(root=tmp_path/"late", now=NOW+timedelta(minutes=1)) == b
