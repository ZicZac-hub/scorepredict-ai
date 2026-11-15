"""V24 broad, quota-bounded market research for API-Football fixtures.

One provider /odds?date request can return multiple fixtures (10/page).
Store timestamped odds for ALL returned fixtures, even BEFORE the V8 engine
models them. Later bind to real V8 forecasts by exact fixture identity.
Never backdate a quote, create a fake team identity, or alter original scores.

Opt-in is the EXISTING V23_EARLY_ODDS_ENABLED=true setting. The default
global budget for this separate broad collector is 12 API requests per UTC
day, at most two per run. Other GitHub workflows may consume API quota too.
"""
import hashlib
import json
import os
from datetime import datetime,timedelta,timezone
from pathlib import Path

from api_football import _request
from persist_prematch import load_json,write_json
from v10_prospective import ROOT as ORIGINAL_ROOT,read_latest,utc
from v16_pair_research import original_model
from v21_capture_sources import source_packet_path,best_packet
from v21_provider_evidence import valid_snapshot
from v22_evidence_models import parse_markets

ROOT=Path("data/v24_market_catalog")
SOURCE_ROOT=Path("data/v21_sources")
MAX_PER_RUN=2
DEFAULT_DAILY_REQUESTS=12
MAX_DAILY_REQUESTS=60
MAX_PAGE=3
DATE_HORIZON=14
STALE_LIMIT_HOURS=12


def _safe_positive_int(value):
    try:
        n=int(value)
        if n>0 and str(n)==str(value):
            return n
    except (TypeError,ValueError):
        pass
    return None


def _hash(record):
    return hashlib.sha256(json.dumps(
        record,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _day_priority(clock):
    """Rotate discovery dates so distant fixtures are not permanently starved."""
    today=clock.date()
    shift=(clock.toordinal()*8+clock.hour//3)%DATE_HORIZON
    offsets=[(shift+i)%DATE_HORIZON for i in range(DATE_HORIZON)]
    return [(today+timedelta(days=i)).isoformat() for i in offsets]


def _slot(clock):
    return clock.strftime("%Y%m%d")+"_"+str(clock.hour//3)


def _attempt(root,clock,date,page):
    return Path(root)/"requests"/clock.date().isoformat()/(
        date+"_p"+str(page)+"_slot"+_slot(clock)+".json")


def _day_count(root,clock):
    return len(list((Path(root)/"requests"/clock.date().isoformat()).glob("*.json")))


def _bookmakers_market(item,observed,fetch_time):
    return parse_markets([item],observed_at=observed,
                         verified_provider_fetched_at=fetch_time)


def _save_fixture_market(item,*,clock,fetched,root,query_date):
    if not isinstance(item,dict):
        return None
    fixture=item.get("fixture") or {}
    fid=_safe_positive_int(fixture.get("id"))
    if fid is None:
        return None
    quoted=_bookmakers_market(item,clock,fetched)
    if not quoted:
        return None
    # Source-provided kickoff is optional. When absent, never invent it:
    # actual 90-minute fixture/time validation occurs during V8 binding.
    actual_start=fixture.get("date")
    try:
        actual_start=utc(actual_start).isoformat() if actual_start else None
    except (TypeError,ValueError):
        actual_start=None
    if actual_start is not None and utc(actual_start)<=clock:
        return None
    league=item.get("league") or {}
    row={"schema":1,"provider":"API-Football",
         "fixture_id":str(fid),"source_fixture_kickoff_utc":actual_start,
         "source_league_id":league.get("id"),
         "query_date":query_date,
         "captured_at_utc":clock.isoformat(),
         "provider_fetched_at_utc":fetched,
         "market_types":quoted,
         "has_verified_both_teams":False,
         "cannot_predict_without_original_model":True}
    sha=_hash(row)
    row["snapshot_sha256"]=sha
    path=Path(root)/"fixtures"/str(fid)/(sha+".json")
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open("x",encoding="utf-8") as out:
            out.write(json.dumps(row,sort_keys=True,indent=2,allow_nan=False)+"\n")
        return str(fid)
    except FileExistsError:
        return None


def _valid_catalog(row,source,now):
    try:
        if not isinstance(row,dict) or row.get("provider")!="API-Football":
            return False
        if str(row["fixture_id"])!=str(source["fixture_id"]):
            return False
        digest=row.get("snapshot_sha256")
        if digest!=_hash({k:v for k,v in row.items() if k!="snapshot_sha256"}):
            return False
        captured=utc(row["captured_at_utc"])
        fetched=utc(row["provider_fetched_at_utc"])
        kickoff=utc(source["kickoff_utc"])
        clock=utc(now)
        if not (fetched<=captured<=clock<kickoff):
            return False
        if row.get("source_fixture_kickoff_utc") and (
                abs((utc(row["source_fixture_kickoff_utc"])-kickoff).total_seconds())>300):
            return False
        market=row.get("market_types") or {}
        if "1x2" not in market or not ({"over_2_5","over_3_5"} & set(market)):
            return False
        for kind in ("1x2","over_2_5","over_3_5"):
            if kind not in market:
                continue
            updated=utc(market[kind]["last_source_update_utc"])
            if not (updated<=clock and updated<=kickoff and
                    clock-updated<=timedelta(hours=STALE_LIMIT_HOURS)):
                return False
        return True
    except (TypeError,KeyError,ValueError):
        return False


def _bind_packet(row,source,now):
    """Create a true source identity packet from an actual archived V8 source."""
    fixture={"id":source["fixture_id"],"provider":source["provider"],
             "utcDate":source["kickoff_utc"],
             "homeTeam":{"id":source["home_team_id"]},
             "awayTeam":{"id":source["away_team_id"]}}
    clock=utc(now)
    if not _valid_catalog(row,source,clock):
        return None
    packet={"schema":1,"status":"captured_pre_kickoff",
            "provider":"API-Football",
            "fixture_id":str(source["fixture_id"]),
            "home_team_id":str(source["home_team_id"]),
            "away_team_id":str(source["away_team_id"]),
            "kickoff_utc":source["kickoff_utc"],
            "captured_at_utc":row["captured_at_utc"],
            "market":None,"multi_market":row["market_types"],
            "confirmed_starting_xi":[],"reported_injuries":[],
            "verified_historical_fixture_xg":[],
            "available_sources":{"timestamped_multi_market":True,
                                 "historical_xg":False,"lineups":False},
            "source_limits":{"bulk_market_catalog_sha256":row["snapshot_sha256"],
                             "provider_fetched_at_utc":row["provider_fetched_at_utc"],
                             "source_fixture_kickoff_utc":row.get("source_fixture_kickoff_utc"),
                             "does_not_claim_verified_shot_xg":True}}
    packet["snapshot_sha256"]=_hash(packet)
    if not valid_snapshot(packet,fixture,now=clock):
        return None
    record={"schema":1,
            "source_forecast":{k:source[k] for k in (
                "provider","fixture_id","forecast_id","input_digest",
                "home_team_id","away_team_id","kickoff_utc")},
            "provider_snapshot":packet,
            "not_baseline_forecast_digest":True,
            "source_scope":"audited_date_paged_API_Football_odds_source_not_an_original_forecast"}
    return record


def bind_catalog(*,now,root=ROOT,original_root=ORIGINAL_ROOT,
                 source_root=SOURCE_ROOT,originals=None):
    """Bind ALL supported saved V8 fixtures to authentic pre-match market data.

    Catalog collection is independent of whether the football model has
    already analysed a fixture; binding waits for its immutable V8 identity.
    """
    clock=utc(now)
    forecast_rows=(originals if originals is not None else read_latest(original_root))
    forecasts=(forecast_rows.values() if isinstance(forecast_rows,dict)
               else forecast_rows)
    found={}
    summary={"v8_checked":0,"v8_with_catalog":0,"new_bindings":[],
             "missing_catalog":0,"out_of_window":0}
    for row in forecasts or []:
        try:
            if row.get("model_key") not in ("v8","v8_near") or (
                    row.get("provider")!="API-Football"):
                continue
            kickoff=utc(row["kickoff_utc"])
            if not (clock+timedelta(minutes=5)<kickoff<=clock+timedelta(days=14)):
                summary["out_of_window"]+=1
                continue
            source=original_model(original_root,row)
            if (not source or str(source.get("input_digest"))!=
                    str(row.get("input_digest"))):
                continue
            key=(str(source["provider"]),str(source["fixture_id"]),
                 str(source["forecast_id"]))
            found.setdefault(key,source)
        except (KeyError,TypeError,ValueError):
            continue
    for source in found.values():
        summary["v8_checked"]+=1
        fid=str(source["fixture_id"])
        folder=Path(root)/"fixtures"/fid
        accepted=[]
        for path in sorted(folder.glob("*.json")):
            row=load_json(path,{})
            if _valid_catalog(row,source,clock):
                accepted.append(row)
        if not accepted:
            summary["missing_catalog"]+=1
            continue
        summary["v8_with_catalog"]+=1
        freshest=max(accepted,key=lambda item:(
            item["market_types"]["1x2"]["last_source_update_utc"],
            item["captured_at_utc"]))
        record=_bind_packet(freshest,source,clock)
        if record is None:
            continue
        base=source_packet_path(source_root,source)
        packet=record["provider_snapshot"]
        path=Path(source_root)/"versions"/(
            base.stem+"_"+packet["snapshot_sha256"][:16]+".json")
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            with path.open("x",encoding="utf-8") as out:
                out.write(json.dumps(record,sort_keys=True,indent=2,
                                     allow_nan=False)+"\n")
            summary["new_bindings"].append(fid)
        except FileExistsError:
            pass
    return summary


def _request_dates(root,clock,max_requests,daily_budget,request,key):
    run={"queries":0,"new_catalog_fixtures":[],"empty_pages":0,
         "skipped_stale":0,"provider_errors":[],"attempts":[]}
    days=_day_priority(clock)
    # Marker ledger bounds cumulative UTC-day spending across all runs.
    remaining=max(0,min(max_requests,daily_budget-_day_count(root,clock)))
    for date in days:
        if run["queries"]>=remaining:
            break
        for page in range(1,MAX_PAGE+1):
            if run["queries"]>=remaining:
                break
            marker=_attempt(root,clock,date,page)
            if marker.exists():
                existing=load_json(marker,{})
                if existing.get("result_count",0)<10:
                    break
                continue
            if page>1:
                previous=load_json(_attempt(root,clock,date,page-1),{})
                if not previous or previous.get("result_count",0)<10:
                    break
            marker.parent.mkdir(parents=True,exist_ok=True)
            try:
                with marker.open("x",encoding="utf-8") as file:
                    file.write(json.dumps({"status":"attempt_reserved",
                                           "date":date,"page":page,
                                           "reserved_at_utc":clock.isoformat()})+"\n")
            except FileExistsError:
                continue
            run["queries"]+=1
            run["attempts"].append({"date":date,"page":page})
            try:
                if request is _request:
                    envelope=request(key,"/odds",{"date":date,"page":page},
                                     with_metadata=True)
                    meta=envelope.get("metadata") or {}
                    if meta.get("stale") or not meta.get("fetched_at_utc"):
                        run["skipped_stale"]+=1
                        results=[]
                        fetched=None
                    else:
                        fetched=meta["fetched_at_utc"]
                        results=envelope.get("data") or []
                else:
                    # Test callbacks must supply source bookmaker update
                    # timestamps. They cannot invent an API fetch timestamp.
                    results=request(key,"/odds",{"date":date,"page":page})
                    fetched=None
                if not isinstance(results,list):
                    results=[]
                for item in results:
                    saved=_save_fixture_market(item,clock=clock,fetched=fetched,
                                               root=root,query_date=date)
                    if saved is not None:
                        run["new_catalog_fixtures"].append(saved)
                write_json(marker,{"date":date,"page":page,
                                   "reserved_at_utc":clock.isoformat(),
                                   "result_count":len(results),
                                   "source_response_available":bool(results),
                                   "saved_market_fixture_count":len(run["new_catalog_fixtures"])})
                if len(results)<10:
                    run["empty_pages"]+=int(not results)
                    break
            except Exception as exc:
                # A failed call still consumes one request reservation; later
                # runs can safely move to other date windows instead.
                run["provider_errors"].append({"date":date,"page":page,
                                               "kind":type(exc).__name__})
                write_json(marker,{"date":date,"page":page,
                                   "result_count":0,"provider_error_type":type(exc).__name__,
                                   "reserved_at_utc":clock.isoformat()})
                break
    return run


def capture(*,now=None,enabled=None,api_key=None,root=ROOT,
            original_root=ORIGINAL_ROOT,source_root=SOURCE_ROOT,
            request=None,max_per_run=MAX_PER_RUN,daily_limit=None,
            originals=None):
    clock=utc(now or datetime.now(timezone.utc))
    active=(os.environ.get("V23_EARLY_ODDS_ENABLED")=="true"
            if enabled is None else bool(enabled))
    key=api_key if api_key is not None else os.environ.get("API_FOOTBALL_KEY","")
    status={"at_utc":clock.isoformat(),"enabled":active,
            "request_budget_per_run":max_per_run,
            "requests_used_today":_day_count(root,clock),
            "new_catalog_fixtures":[],"queries":0,
            "new_source_bindings":[]}
    if not active or not key:
        status["reason"]="not_enabled_or_api_key_missing"
        return status
    if daily_limit is None:
        try:
            daily_limit=int(os.environ.get("V24_DAILY_ODDS_REQUESTS",
                                           DEFAULT_DAILY_REQUESTS))
        except (ValueError,TypeError):
            daily_limit=DEFAULT_DAILY_REQUESTS
    daily_limit=max(0,min(MAX_DAILY_REQUESTS,int(daily_limit)))
    status["daily_limit"]=daily_limit
    cb=request or _request
    run=_request_dates(root,clock,min(MAX_PER_RUN,max(0,int(max_per_run))),
                       daily_limit,cb,key)
    status.update(run)
    status["requests_used_today"]=_day_count(root,clock)
    bound=bind_catalog(now=clock,root=root,original_root=original_root,
                       source_root=source_root,originals=originals)
    status["binding"]=bound
    status["new_source_bindings"]=bound["new_bindings"]
    status["all_supported_odds_not_guaranteed"]=True
    status["note"]="Date pages capped by quota, market availability, original fixture identity and kickoff"
    write_json(Path(root)/"status.json",status)
    return status


if __name__=="__main__":
    print(json.dumps(capture(),indent=2))
