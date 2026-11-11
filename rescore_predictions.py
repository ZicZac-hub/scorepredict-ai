"""Re-score previously captured upcoming fixtures using the current live engine.

Original evidence is immutable. This script only writes separate revision
files for matches still in the future at the time of recalculation.
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from api_football import LEAGUES, _convert
from capture_prematch import request_fixtures
from engine import predict, LIVE_MODEL_VERSION
from persist_prematch import earliest_snapshots, parse_time, write_json

OUTPUT = Path("data/prematch/rescores")
MIN_RECALC_HOURS = 0

def main():
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY secret required")
    now = datetime.now(timezone.utc)
    originals = earliest_snapshots()
    pending = defaultdict(list)
    for fid, row in originals.items():
        try:
            kickoff = parse_time(row["kickoff_utc"])
            code = row["competition"]
            if kickoff > now + timedelta(hours=MIN_RECALC_HOURS) and code in LEAGUES:
                pending[code].append((fid,row))
        except (ValueError, KeyError, TypeError):
            continue
    output = {"rescored_at_utc":now.isoformat(),"model_version":LIVE_MODEL_VERSION,
              "revisions":[],"skipped":[],"note":"Only pre-kickoff rescoring. Original snapshots never modified. Historical results are never used as prediction features."}
    for code, items in sorted(pending.items()):
        try:
            season = now.year if code == "BSA" or now.month >= 7 else now.year-1
            raw_history = request_fixtures(key, {"league":LEAGUES[code],
                "season":season,"from":(now-timedelta(days=180)).date().isoformat(),
                "to":now.date().isoformat(),"status":"FT"})
            history = [m for raw in raw_history if (m:=_convert(raw,code)).get("status")=="FINISHED"
                       and parse_time(m["utcDate"]) < now]
        except Exception as exc:
            output["skipped"].append({"competition":code,"reason":"history unavailable: "+str(exc)[:150]})
            continue
        for fid, original in items:
            try:
                raw_fixture = request_fixtures(key, {"id":str(fid)})
                if not raw_fixture:
                    raise ValueError("fixture unavailable")
                fixture = _convert(raw_fixture[0],code)
                kickoff = parse_time(fixture["utcDate"])
                if kickoff <= now or fixture.get("status") not in ("SCHEDULED","TIMED"):
                    output["skipped"].append({"fixture_id":fid,"reason":"not an upcoming fixture"})
                    continue
                # Never use results on or after the current run or kickoff.
                past = [m for m in history if parse_time(m["utcDate"]) < now and
                        parse_time(m["utcDate"]) < kickoff]
                forecast = predict(fixture,past,variant="baseline")
                if not forecast:
                    raise ValueError("insufficient current pre-match history")
                revised = [int(forecast["scores"][0][0]),int(forecast["scores"][0][1])]
                record = {"fixture_id":int(fid),"competition":code,
                    "home_team":original.get("home_team"),"away_team":original.get("away_team"),
                    "kickoff_utc":fixture["utcDate"],"rescored_at_utc":now.isoformat(),
                    "original_captured_at_utc":original["captured_at_utc"],
                    "original_score":original["predicted_score"],"updated_score":revised,
                    "changed":revised!=original["predicted_score"],
                    "original_model":original.get("forecast_model"),
                    "updated_model":forecast.get("model_version",LIVE_MODEL_VERSION),
                    "updated_score_probability":float(forecast["scores"][0][2]),
                    "updated_expected_goals":{"home":float(forecast["home_xg"]),"away":float(forecast["away_xg"])},
                    "updated_outcome_probabilities":{"home":float(forecast["home_win"]),
                        "draw":float(forecast["draw"]),"away":float(forecast["away_win"])}}
                # One latest revision per fixture; immutable originals in snapshots.
                write_json(OUTPUT/(str(fid)+".json"),record)
                output["revisions"].append(record)
            except Exception as exc:
                output["skipped"].append({"fixture_id":fid,"reason":str(exc)[:150]})
    write_json(OUTPUT/"latest_report.json",output)
    print(json.dumps({"rescored":len(output["revisions"]),
        "changed":sum(x["changed"] for x in output["revisions"]),
        "skipped":output["skipped"]},indent=2))

if __name__=="__main__":
    main()
