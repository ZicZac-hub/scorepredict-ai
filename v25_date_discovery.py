"""Today-forward date and real fixture browsing for V25 (Sydney timezone).

This is a fixture-discovery view, not a forecast generator: provider-sourced
games can appear before a valid full score-distribution model exists.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SYDNEY=ZoneInfo("Australia/Sydney")
DEFAULT_HORIZON_DAYS=14


def visible_days(now, modelled_by_day=None, *, horizon=DEFAULT_HORIZON_DAYS):
    """Include TODAY and TOMORROW even with 0 modelled fixtures."""
    if type(horizon) is not int or horizon<2 or horizon>31:
        raise ValueError("Horizon must cover 2-31 days")
    start=now.astimezone(SYDNEY).date()
    floor={ (start+timedelta(days=i)).isoformat() for i in range(horizon) }
    extras={
        day for day in (modelled_by_day or {})
        if isinstance(day,str) and day>=start.isoformat()
        and len(day)==10 and day[4]=="-" and day[7]=="-"
    }
    return sorted(floor|extras)


def date_label(day, now, *, eligible_count=0, discovered_count=0):
    current=now.astimezone(SYDNEY).date()
    chosen=datetime.strptime(day,"%Y-%m-%d").date()
    offset=(chosen-current).days
    prefix=("Today · " if offset==0 else
            "Tomorrow · " if offset==1 else "")
    label=prefix+chosen.strftime("%a %d %b")
    if eligible_count:
        return label+f" · {eligible_count} ready"
    if discovered_count:
        return label+f" · {discovered_count} fixtures, no score model"
    return label+" · search fixtures"


def merge_modelled(current, new):
    """Deduplicate session-only modelled rows by provider, ID and kickoff."""
    from ai_five_double_engine import identity
    unique={}
    for row in list(current or [])+list(new or []):
        try:
            key=identity(row)
            if key:
                unique.setdefault(key,row)
        except (ValueError,TypeError,AttributeError,KeyError):
            continue
    return list(unique.values())


def fixture_table(day, scored, discovered):
    """Display provider-listed games separately from independently modelled ones."""
    items={}
    for row in discovered or []:
        if not isinstance(row,dict):
            continue
        try:
            captured=datetime.fromisoformat(str(row["kickoff_utc"]).replace("Z","+00:00"))
            if captured.tzinfo is None or captured.astimezone(SYDNEY).date().isoformat()!=day:
                continue
            key=(str(row["provider"]),str(row["fixture_id"]))
            items[key]={
                "Kickoff (Sydney)":captured.astimezone(SYDNEY).strftime("%I:%M %p").lstrip("0"),
                "Match":str(row.get("home_team") or "Home")+" vs "+str(row.get("away_team") or "Away"),
                "Score model":"Fixture found · not yet qualified",
            }
        except (KeyError,ValueError,TypeError):
            continue
    for entry in scored or []:
        try:
            match=entry["row"]["fixture"]
            key=(str(match.get("provider") or "football-data.org"),
                 str(match["id"]))
            time=entry["kickoff"].astimezone(SYDNEY)
            if time.date().isoformat()!=day:
                continue
            items[key]={
                "Kickoff (Sydney)":time.strftime("%I:%M %p").lstrip("0"),
                "Match":entry["title"],
                "Score model":"V25 ready · full pre-match grid",
            }
        except (KeyError,TypeError,ValueError):
            continue
    return sorted(items.values(),key=lambda row:(datetime.strptime(
        row["Kickoff (Sydney)"],"%I:%M %p").time(),row["Match"]))
