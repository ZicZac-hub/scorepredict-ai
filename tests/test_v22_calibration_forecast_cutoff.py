"""A result before kickoff can still be unavailable at forecast creation."""
from datetime import datetime,timedelta,timezone
import numpy as np

from v22_calibration import chronological_evaluation


def test_prequential_fit_uses_forecast_timestamp_instead_of_later_kickoff():
    day=datetime(2034,1,1,tzinfo=timezone.utc)
    rows=[]
    finals={}
    for number,hour in ((1,9),(2,18)):
        kickoff=day+timedelta(hours=hour)
        row={"provider":"API-Football","fixture_id":str(number),
             "home_team_id":str(number*2),"away_team_id":str(number*2+1),
             "home_team":f"Home{number}","away_team":f"Away{number}",
             "model_key":"v8","forecast_id":f"original{number}",
             "kickoff_utc":kickoff.isoformat(),
             "captured_at_utc":(kickoff-timedelta(hours=4)).isoformat(),
             "competition":"PL","score_basis":"regulation",
             "score_matrix":np.full((5,5),1/25).tolist()}
        rows.append(row)
        finals[str(number)]={**row,"evidence_version":2,"status":"FT",
            "home_goals":1,"away_goals":0,
            "verified_at_utc":(kickoff+timedelta(hours=6)).isoformat()}
    # First result at15:00 is before target18:00 kickoff, after14:00 forecast.
    report=chronological_evaluation(rows,finals,min_samples=1,as_of=day+timedelta(days=1,hours=3))
    assert report["leagues"]["PL"]["verified_evaluated"]==0
    finals["1"]["verified_at_utc"]=(day+timedelta(hours=13)).isoformat()
    valid=chronological_evaluation(rows,finals,min_samples=1,as_of=day+timedelta(days=1,hours=3))
    assert valid["leagues"]["PL"]["verified_evaluated"]==1
