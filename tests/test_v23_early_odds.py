"""V23 early odds: one checked request, immutable packet, no repeat billing."""
from datetime import datetime,timedelta,timezone
import json

from v23_early_odds import capture
from v21_capture_sources import best_packet

NOW=datetime(2031,8,9,5,tzinfo=timezone.utc)
LEG={"provider":"API-Football","fixture_id":"3001",
     "forecast_id":"forecast3001","input_digest":"evidence3001",
     "home_team_id":"10","away_team_id":"20",
     "kickoff_utc":(NOW+timedelta(hours=8)).isoformat()}


def odds():
    return [{"update":(NOW-timedelta(minutes=10)).isoformat(),
             "bookmakers":[{"name":"A Genuine Book","bets":[
                 {"name":"Match Winner","values":[
                     {"value":"Home","odd":"2.10"},
                     {"value":"Draw","odd":"3.70"},
                     {"value":"Away","odd":"3.50"}]},
                 {"name":"Goals Over/Under","values":[
                     {"value":"Over 2.5","odd":"1.80"},
                     {"value":"Under 2.5","odd":"2.10"}]}]}]}]


def test_one_real_source_market_snapshot_and_no_recapture(monkeypatch,tmp_path):
    import v23_early_odds as collector
    monkeypatch.setattr(collector,"eligible_legs",lambda root,now:[LEG])
    requests=[]
    def req(key,endpoint,params):
        requests.append((endpoint,params))
        return odds()
    first=capture(now=NOW,enabled=True,api_key="stub-no-network",
                  source_root=tmp_path,root=tmp_path,request=req)
    assert first["queries"]==1
    assert first["new_packets"]==["3001"]
    assert len(requests)==1
    saved=best_packet(tmp_path,LEG,now=NOW,priority="market")
    assert saved and saved["provider_snapshot"]["multi_market"]["over_2_5"]
    assert saved["provider_snapshot"]["verified_historical_fixture_xg"]==[]
    assert saved["source_forecast"]["input_digest"]=="evidence3001"
    second=capture(now=NOW,enabled=True,api_key="stub-no-network",
                   source_root=tmp_path,root=tmp_path,request=req)
    assert second["queries"]==0 and len(requests)==1
    assert len(list((tmp_path/"versions").glob("*.json")))==1


def test_optout_makes_no_api_calls(monkeypatch,tmp_path):
    import v23_early_odds as collector
    monkeypatch.setattr(collector,"eligible_legs",lambda root,now:[LEG])
    def fail(*a,**kw):
        raise AssertionError("unapproved API call")
    status=capture(now=NOW,enabled=False,api_key="stub",
                   source_root=tmp_path,root=tmp_path,request=fail)
    assert not status["enabled"] and status["queries"]==0
