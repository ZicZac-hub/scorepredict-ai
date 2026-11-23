"""Opt-in, quota-bounded live discovery for AI Five-Double Today/Tomorrow.

Never manufacture fixtures or statistical scores. A user click performs a
single date-scoped API-Football fixture search and a small bounded batch of
model evaluations; results are session-only and not counted as frozen wins.
"""
from datetime import datetime, timezone

from ai_five_double_engine import eligible
from global_fixtures import global_fixtures
from worldwide_model import model_worldwide
from on_demand_ai import utc


MAX_NEW_MODELS_PER_CLICK = 12
DEFAULT_NEW_MODELS_PER_CLICK = 4


def _quota_limited(message):
    text=str(message or "").casefold()
    return any(part in text for part in ("quota", "rate limit", "too many requests",
                                         "http 429", "providerquotaerror"))


def research_day(api_key, local_day, *, known_ids=None, limit=DEFAULT_NEW_MODELS_PER_CLICK,
                 discover=global_fixtures, model=model_worldwide, now=None):
    clock = utc(now or datetime.now(timezone.utc))
    report = {"day": local_day.isoformat(), "discovered": 0, "checked": 0,
              "modeled": [], "unmodeled": 0, "errors": [], "remaining": 0,
              "quota_limited": False, "scope": "API-Football listed pre-match fixtures"}
    if not api_key:
        report["errors"].append("API_FOOTBALL_KEY missing from Streamlit Secrets")
        return report
    events, source_errors = discover(api_key, local_day, 1)
    report["errors"].extend(str(e)[:150] for e in source_errors[:3])
    report["quota_limited"] = any(_quota_limited(e) for e in source_errors)
    future=[]
    for event in events:
        if event.get("source")!="API-Football":
            continue
        fixture=event.get("fixture") or {}
        kickoff=utc(fixture.get("utcDate"))
        if (fixture.get("provider")=="API-Football" and kickoff is not None
                and (kickoff-clock).total_seconds()>300
                and fixture.get("id") is not None
                and (fixture.get("homeTeam") or {}).get("id") is not None
                and (fixture.get("awayTeam") or {}).get("id") is not None):
            future.append(event)
    report["discovered"]=len(future)
    # Real provider-sourced fixtures, including those that cannot yet be
    # modelled. Never convert a fixture listing into an invented scoreline.
    # This gives V25 a truthful Today/Tomorrow match list before the model
    # receives sufficient historical observations.
    report["fixtures"]=[
        {
            "provider":"API-Football",
            "fixture_id":str(event["fixture"]["id"]),
            "kickoff_utc":utc(event["fixture"]["utcDate"]).isoformat(),
            "home_team":str(event["fixture"]["homeTeam"].get("name") or ""),
            "away_team":str(event["fixture"]["awayTeam"].get("name") or ""),
            "competition":str(event.get("league") or ""),
            "country":str(event.get("country") or ""),
            "source_status":"provider-listed; model not guaranteed"
        }
        for event in future
    ]
    previously=set(known_ids or [])
    queued=[event for event in future
            if str((event.get("fixture") or {}).get("id")) not in previously]
    # A user-selected larger batch does not alter the default four-call
    # policy; the hard cap protects the provider allowance.
    max_attempts=max(0,min(int(limit),MAX_NEW_MODELS_PER_CLICK))
    for event in queued[:max_attempts]:
        report["checked"]+=1
        try:
            row,err=model(api_key,event,model_variant="research_v8")
            if isinstance(row,dict) and eligible(row,now=clock):
                report["modeled"].append(row)
            else:
                report["unmodeled"]+=1
                if err:
                    report["errors"].append(str(err)[:160])
                    if _quota_limited(err):
                        report["quota_limited"]=True
        except Exception as exc:
            report["unmodeled"]+=1
            report["errors"].append("Model research failed: "+type(exc).__name__)
            if _quota_limited(type(exc).__name__):
                report["quota_limited"]=True
        # A quota failure affects the whole account. Never spin through the
        # rest of the day's candidate fixtures after detecting it.
        if report["quota_limited"]:
            break
    report["remaining"]=max(0,len(queued)-report["checked"])
    report["checked_ids"]=[str((event["fixture"] or {})["id"])
                           for event in queued[:report["checked"]]]
    return report
