"""Independent Sportmonks correct-score probability research.

This never overwrites an API-Football forecast or changes a five-double pick.
Fixture IDs are provider-specific: match only unambiguous home/away names AND
kickoff before requesting licensed Sportmonks prediction probabilities.
"""
import json
import math
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from persist_prematch import load_json, verified_result, write_json
from v10_prospective import ROOT as V10_ROOT, read_latest, utc

ROOT = Path("data/independent_football")
API_ROOT = "https://api.sportmonks.com/v3/football"
WINDOW = timedelta(hours=36)


def _name(value):
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return re.sub(r"[^a-z0-9]+", " ", "".join(
        char for char in value if not unicodedata.combining(char)
    )).strip()


def _kickoff(item):
    try:
        if item.get("starting_at_timestamp") is not None:
            return datetime.fromtimestamp(int(item["starting_at_timestamp"]),timezone.utc)
        date = item.get("starting_at")
        if date:
            return utc(date)
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _participants(item):
    teams = {}
    for part in item.get("participants") or []:
        where = (part.get("meta") or {}).get("location")
        if where in ("home","away"):
            teams[where] = part.get("name")
    return teams


def match_fixture(forecast, items):
    """No fuzzy/ambiguous identity mapping between API-Football and Sportmonks."""
    candidates = []
    target = utc(forecast["kickoff_utc"])
    for item in items:
        when, teams = _kickoff(item), _participants(item)
        if (when is None or abs((when-target).total_seconds())>300
                or _name(teams.get("home")) != _name(forecast.get("home_team"))
                or _name(teams.get("away")) != _name(forecast.get("away_team"))
                or not teams.get("home") or not teams.get("away")):
            continue
        try:
            candidates.append(int(item["id"]))
        except (KeyError,TypeError,ValueError):
            continue
    return candidates[0] if len(set(candidates)) == 1 else None


def _numeric_percent(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError("Probability must be numerical")
    if not math.isfinite(float(value)) or not 0<=float(value)<=100:
        raise ValueError("Probability percentage outside valid bounds")
    return float(value)/100.0


def parse_predictions(payload, fixture_id):
    """Only named Sportmonks markets; percentages never confused with fractions."""
    rows = payload.get("data") or []
    if not isinstance(rows,list):
        raise ValueError("Prediction response is not a list")
    found = {}
    for entry in rows:
        if str(entry.get("fixture_id")) != str(fixture_id):
            continue
        market = ((entry.get("type") or {}).get("developer_name") or "").upper()
        predictions = entry.get("predictions") or {}
        if not isinstance(predictions,dict):
            continue
        if market == "FULLTIME_RESULT_PROBABILITY":
            if not all(k in predictions for k in ("home","draw","away")):
                continue
            outcome = {k:_numeric_percent(predictions[k]) for k in ("home","draw","away")}
            if abs(sum(outcome.values())-1)>0.035:
                continue
            found["outcome_probabilities"]=outcome
        if market == "CORRECT_SCORE_PROBABILITY":
            scores = {}
            for raw,val in predictions.items():
                key = re.fullmatch(r"([0-9])-([0-9])",str(raw))
                if key:
                    value = _numeric_percent(val)
                    if value:
                        scores[str(raw)] = value
            if scores and sum(scores.values()) <= 1.04:
                found["score_probabilities"] = scores
                found["top_scores"] = sorted(scores, key=lambda x:(-scores[x],x))[:5]
    return found


def _get(api_token, url, *, params=None, request=None):
    get = request or requests.get
    response = get(url,params=params,headers={"Authorization":api_token,
                   "Accept":"application/json"},timeout=20,allow_redirects=False)
    response.raise_for_status()
    if response.status_code!=200:
        raise RuntimeError("Prediction API did not return HTTP 200")
    body=response.json()
    if not isinstance(body,dict) or body.get("errors"):
        raise ValueError("Provider returned invalid or error response")
    return body


def _discover(api_token, clock, *, request=None, pages=2):
    begin=(clock-timedelta(days=1)).date().isoformat()
    end=(clock+WINDOW+timedelta(days=1)).date().isoformat()
    url=f"{API_ROOT}/fixtures/between/{begin}/{end}"
    items=[]
    for page in range(1,pages+1):
        body=_get(api_token,url,params={"include":"participants","per_page":50,
                                        "page":page},request=request)
        items.extend(body.get("data") or [])
        if not (body.get("pagination") or {}).get("has_more"):
            break
    return items


def _choices(rows, clock):
    by_fixture={}
    for row in rows.values():
        try:
            if (row.get("provider")!="API-Football"
                    or row.get("model_key") not in ("v8","v8_near")
                    or not row.get("forecast_id") or not row.get("input_digest")):
                continue
            kickoff, recorded=utc(row["kickoff_utc"]),utc(row["captured_at_utc"])
            if not recorded<=clock<kickoff or recorded>=kickoff:
                continue
            if not timedelta(minutes=45)<kickoff-clock<=WINDOW:
                continue
            key=str(row["fixture_id"])
            previous=by_fixture.get(key)
            if previous is None or (row["model_key"]=="v8_near" and previous["model_key"]!="v8_near"):
                by_fixture[key]=row
        except (ValueError,TypeError,KeyError):
            continue
    return sorted(by_fixture.values(),key=lambda r:(utc(r["kickoff_utc"]),str(r["fixture_id"])))


def capture(*, now=None, environ=None, root=ROOT, source=V10_ROOT, request=None,
            max_matches=2):
    clock=utc(now or datetime.now(timezone.utc))
    env=os.environ if environ is None else environ
    enabled=env.get("SPORTMONKS_PREDICTIONS_ENABLED")=="true"
    token=env.get("THIRD_FOOTBALL_API_KEY") or env.get("SPORTMONKS_API_TOKEN")
    status={"checked_at_utc":clock.isoformat(),"enabled":enabled,
            "token_present":bool(token),"provider":"Sportmonks","attempted":0,
            "captured":0,"unmatched":0,"no_correct_scores":0,
            "error_type":None,"reason":""}
    path=Path(root)
    if not enabled or not token:
        status["reason"]="disabled" if not enabled else "token_missing"
        write_json(path/"status.json",status)
        return status
    candidates=_choices(read_latest(source),clock)
    try:
        fixtures=_discover(token,clock,request=request)
        for row in candidates:
            if status["attempted"]>=max(0,min(5,max_matches)):
                break
            dest=path/"forecasts"/("API-Football_"+str(row["fixture_id"])+".json")
            if dest.exists():
                continue
            sportmonks_id=match_fixture(row,fixtures)
            if sportmonks_id is None:
                status["unmatched"]+=1
                continue
            status["attempted"]+=1
            response=_get(token,f"{API_ROOT}/predictions/probabilities/fixtures/{sportmonks_id}",
                          params={"include":"type"},request=request)
            probabilities=parse_predictions(response,sportmonks_id)
            if not probabilities.get("score_probabilities"):
                status["no_correct_scores"]+=1
                continue
            frozen=datetime.now(timezone.utc)
            if frozen>=utc(row["kickoff_utc"]):
                continue
            record={"mode":"prospective","source":"Sportmonks Predictions add-on",
                    "provider":"API-Football","fixture_id":str(row["fixture_id"]),
                    "sportmonks_fixture_id":sportmonks_id,
                    "home_team_id":str(row["home_team_id"]),
                    "away_team_id":str(row["away_team_id"]),
                    "home_team":row["home_team"],"away_team":row["away_team"],
                    "kickoff_utc":row["kickoff_utc"],"captured_at_utc":frozen.isoformat(),
                    "source_forecast_id":row["forecast_id"],
                    "input_digest":row["input_digest"],"score_basis":"regulation",
                    "score_probabilities":probabilities["score_probabilities"],
                    "top_scores":probabilities["top_scores"],
                    "outcome_probabilities":probabilities.get("outcome_probabilities"),
                    "original_statistical_score":row["predicted_score"],
                    "research_only":True,"no_auto_promotion":True}
            dest.parent.mkdir(parents=True,exist_ok=True)
            try:
                with dest.open("x",encoding="utf-8") as output:
                    output.write(json.dumps(record,sort_keys=True,indent=2)+"\n")
                status["captured"]+=1
            except FileExistsError:
                pass
        status["reason"]="completed_bounded_check"
    except (requests.RequestException,ValueError,TypeError,RuntimeError,KeyError) as exc:
        status["error_type"]=type(exc).__name__
        status["reason"]="provider_access_or_subscription_error"
    write_json(path/"status.json",status)
    return status


def evaluate(root=ROOT,results_path=V10_ROOT/"verified_results.json"):
    results=load_json(Path(results_path),{})
    actual_n=score_hits=original_hits=0
    for path in (Path(root)/"forecasts").glob("*.json"):
        row=load_json(path,{})
        fixture_id=str(row.get("fixture_id"))
        actual=results.get(fixture_id)
        if not verified_result(row,actual,fixture_id):
            continue
        try:
            if utc(row["captured_at_utc"])>=utc(row["kickoff_utc"]):
                continue
            best=(row.get("top_scores") or [None])[0]
            if best is None:
                continue
            actual_n+=1
            exact=f"{actual['home_goals']}-{actual['away_goals']}"
            score_hits+=int(best==exact)
            original_hits+=int(row.get("original_statistical_score")==
                               [actual["home_goals"],actual["away_goals"]])
        except (ValueError,KeyError,TypeError):
            continue
    report={"verified_matched_fixtures":actual_n,
            "sportmonks_top_score_hits":score_hits,
            "original_statistical_hits":original_hits,
            "sportmonks_exact_rate":score_hits/actual_n if actual_n else None,
            "original_exact_rate":original_hits/actual_n if actual_n else None,
            "note":"Same fixture with pre-kickoff source correspondence and independently "
                   "verified 90-minute goals; no retroactive scores or model promotion."}
    write_json(Path(root)/"evaluation.json",report)
    return report


def main():
    print(json.dumps({"capture":capture(),"evaluation":evaluate()},indent=2))


if __name__=="__main__":
    main()
