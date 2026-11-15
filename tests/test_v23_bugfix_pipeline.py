"""Regressions for actual V23 accuracy pipeline availability bugs.

No fabricated high-score forecasts, no direct provider API calls.
"""
from datetime import datetime,timedelta,timezone
import hashlib
import json

import numpy as np

from smart_double_ai import rank_pairs
from ai_five_double_engine import merged_five, pair_check
from ai_five_double_ui import archive_rows,repository_provider_packet
from v21_capture_sources import source_packet_path


NOW=datetime(2032,8,11,4,tzinfo=timezone.utc)
KICK=(NOW+timedelta(hours=5)).isoformat()
UPDATE=(NOW-timedelta(hours=2)).isoformat()
SCORES=[[1,1,.15],[2,1,.11],[1,0,.10],[0,1,.08],[2,2,.06]]
GRID=np.zeros((11,11))
GRID[1,1]=.40
GRID[1,0]=.35
GRID[2,1]=.25


def match(fid,home,away):
    # Legitimate source rows can have utcDate without a duplicate top-level
    # 'kickoff' key. The five-double engine must accept them.
    return {"fixture":{
        "id":fid,"provider":"API-Football","utcDate":KICK,
        "homeTeam":{"id":home,"name":"Team-"+str(home)},
        "awayTeam":{"id":away,"name":"Team-"+str(away)}},
        "result":{"scores":SCORES,"matrix":GRID.tolist(),
                  "model_version":"v8"}}


def test_fixture_date_only_does_not_break_manual_or_auto_five_doubles():
    first,second=match(700,11,12),match(701,21,22)
    assert pair_check(first,second,now=NOW) is None
    result=merged_five(first,second,now=NOW)
    assert result["status"]=="ready"
    assert len(result["doubles"])==5
    assert len(rank_pairs([first,second]))==1
    assert result["sydney_date"]==(NOW+timedelta(hours=5)).astimezone(
        __import__("zoneinfo").ZoneInfo("Australia/Sydney")).date().isoformat()


def test_archived_v8_keeps_entire_verified_probability_grid(monkeypatch):
    import v10_prospective
    saved={
        "provider":"API-Football","fixture_id":"700","model_key":"v8",
        "forecast_id":"original700","input_digest":"historysha700",
        "model_version":"v8-research","home_team":"Team-11",
        "away_team":"Team-12","home_team_id":"11","away_team_id":"12",
        "kickoff_utc":KICK,"captured_at_utc":UPDATE,
        "top10_scores":SCORES,"score_matrix":GRID.tolist(),
        "competition":"PL","history_matches":42,
        "league_history_matches":120,
        "expected_goals":{"home":1.4,"away":1.1}}
    monkeypatch.setattr(v10_prospective,"read_latest",
                        lambda root:{("API-Football","700","v8"):saved})
    archive=archive_rows(now=NOW)
    assert len(archive)==1
    row=archive[0]
    assert row["result"]["matrix"]==GRID.tolist()
    assert row["league_history_matches"]==120
    assert row["source_forecast"]["input_digest"]=="historysha700"


def test_repository_saved_odds_packet_does_not_require_database(tmp_path):
    fixture=match(700,11,12)["fixture"]
    source={
        "model_key":"v8","provider":"API-Football","fixture_id":"700",
        "forecast_id":"original700","input_digest":"historysha700",
        "home_team_id":"11","away_team_id":"12",
        "kickoff_utc":KICK,"captured_at_utc":UPDATE}
    snapshot={
        "schema":1,"status":"captured_pre_kickoff","provider":"API-Football",
        "fixture_id":"700","home_team_id":"11","away_team_id":"12",
        "kickoff_utc":KICK,"captured_at_utc":NOW.isoformat(),
        "multi_market":{
            "1x2":{"last_source_update_utc":UPDATE},
            "over_2_5":{"last_source_update_utc":UPDATE}},
        "verified_historical_fixture_xg":[]}
    encoded=json.dumps(snapshot,sort_keys=True,separators=(",",":"),allow_nan=False)
    snapshot["snapshot_sha256"]=hashlib.sha256(encoded.encode()).hexdigest()
    record={"source_forecast":{k:source[k] for k in (
            "provider","fixture_id","forecast_id","input_digest",
            "home_team_id","away_team_id","kickoff_utc")},
            "provider_snapshot":snapshot}
    path=source_packet_path(tmp_path,source)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record))
    chosen=repository_provider_packet(fixture,now=NOW,
                   originals=[source],packet_root=tmp_path,priority="market")
    assert chosen is not None
    assert chosen["snapshot_sha256"]==snapshot["snapshot_sha256"]
    wrong=json.loads(json.dumps(fixture))
    wrong["awayTeam"]["id"]=777
    assert repository_provider_packet(wrong,now=NOW,
                   originals=[source],packet_root=tmp_path) is None
    assert repository_provider_packet(fixture,now=NOW+timedelta(hours=6),
                   originals=[source],packet_root=tmp_path) is None


def test_top_ten_only_current_row_does_not_hide_complete_frozen_v8_grid():
    from ai_five_double_ui import options
    live=match(700,11,12)
    archived=match(700,11,12)
    live["kickoff"]=KICK
    archived["kickoff"]=KICK
    live["result"].pop("matrix")
    archived["archive_frozen"]=True
    chosen=options([live],now=NOW,archived=[archived])
    assert len(chosen)==1
    assert chosen[0]["archive_frozen"] is True
    assert chosen[0]["result"]["matrix"]==GRID.tolist()
    live["result"]["matrix"]=GRID.tolist()
    chosen=options([live],now=NOW,archived=[archived])
    assert len(chosen)==1
    assert not chosen[0].get("archive_frozen",False)
