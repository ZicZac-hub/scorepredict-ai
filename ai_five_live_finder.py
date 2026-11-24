"""Opt-in, quota-bounded live discovery for AI Five-Double Today/Tomorrow.

Never manufacture fixtures or statistical scores. A user click performs a
single date-scoped API-Football fixture search and a small bounded batch of
model evaluations; results are session-only and not counted as frozen wins.
"""
from datetime import datetime, timezone, timedelta

from ai_five_double_engine import eligible
from global_fixtures import global_fixtures
from worldwide_model import model_worldwide
from on_demand_ai import utc


MAX_NEW_MODELS_PER_CLICK = 12
DEFAULT_NEW_MODELS_PER_CLICK = 4
# The listed competitions typically have better supported league histories.
# This is a collection priority, NEVER a claim that their scores are safer.
HISTORY_PRIORITY_LEAGUES = frozenset((2,3,39,40,61,78,88,94,135,140))


def _history_priority(event):
    """Fetch order only: known established leagues before sparse youth leagues.

    Never invent missing team histories, goals, probabilities or bookmaker data.
    Every other genuine provider fixture stays eligible for a later batch.
    """
    fixture = event.get("fixture") or {}
    league = event.get("league_id")
    if league is None:
        league = (fixture.get("competition") or {}).get("id")
    try:
        established = int(league) in HISTORY_PRIORITY_LEAGUES
    except (TypeError,ValueError):
        established = False
    start = utc(fixture.get("utcDate"))
    return (not established, start or datetime.max.replace(tzinfo=timezone.utc),
            str(fixture.get("id") or ""))


def _quota_backoff(report, clock):
    report["quota_limited"] = True
    report["quota_retry_after_utc"] = (clock + timedelta(minutes=30)).isoformat()
    report["quota_retry_note"] = (
        "Retry is manual after the cooldown; this does not refill the "
        "provider's actual daily/monthly allowance."
    )



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
    if any(_quota_limited(e) for e in source_errors):
        _quota_backoff(report, clock)
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
    previously={str(x) for x in (known_ids or [])}
    queued = []
    already = set(previously)
    for event in sorted(future, key=_history_priority):
        fid = str((event.get("fixture") or {}).get("id"))
        if fid not in already:
            queued.append(event)
            already.add(fid)
    report["selection_policy"] = "established_league_history_first_then_kickoff"
    report["prioritization_not_a_win_probability"] = True
    # If discovery already failed on quota, do not spend another call on
    # historical modelling. Preserve the previously retrieved fixture list.
    if report["quota_limited"]:
        report["remaining"] = len(queued)
        report["checked_ids"] = []
        return report
    max_attempts = max(0,min(int(limit),MAX_NEW_MODELS_PER_CLICK))
    completed_ids = []
    for event in queued[:max_attempts]:
        fid = str(event["fixture"]["id"])
        report["checked"] += 1
        limited = False
        try:
            row, err = model(api_key,event,model_variant="research_v8")
            if isinstance(row,dict) and eligible(row,now=clock):
                report["modeled"].append(row)
            else:
                report["unmodeled"] += 1
                if err:
                    report["errors"].append(str(err)[:160])
                    limited = _quota_limited(err)
        except Exception as exc:
            report["unmodeled"] += 1
            report["errors"].append("Model research failed: "+type(exc).__name__)
            limited = _quota_limited(type(exc).__name__) or _quota_limited(str(exc))
        if limited:
            # A rate-limit failure is not a completed historical-model check:
            # keep this match retryable after genuine provider access returns.
            _quota_backoff(report, clock)
            report["blocked_fixture_id"] = fid
            break
        completed_ids.append(fid)
    report["checked_ids"] = completed_ids
    report["remaining"] = len(queued) - len(completed_ids)
    return report
