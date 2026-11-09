"""Worldwide date-first soccer fixtures from TheSportsDB's limited free v1 API.

Free eventsday responses can be truncated (documented limit: three events).
Never present the result as a complete global schedule.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
import streamlit as st

SYDNEY=ZoneInfo("Australia/Sydney")
ENDPOINT="https://www.thesportsdb.com/api/v1/json/123/eventsday.php"

@st.cache_data(ttl=3600,show_spinner=False)
def day_events(utc_day):
    response=requests.get(ENDPOINT,params={"d":utc_day,"s":"Soccer"},timeout=15)
    response.raise_for_status()
    data=response.json()
    return data.get("events") or []

def parse_event(event):
    raw=(event.get("strTimestamp") or "").strip()
    if not raw:
        day=(event.get("dateEvent") or "").strip()
        hour=(event.get("strTime") or "00:00:00").strip()
        raw=f"{day}T{hour}Z"
    try:
        kickoff=datetime.fromisoformat(raw.replace("Z","+00:00"))
        if kickoff.tzinfo is None:kickoff=kickoff.replace(tzinfo=timezone.utc)
        kickoff=kickoff.astimezone(timezone.utc)
    except (TypeError,ValueError):
        return None
    if event.get("intHomeScore") not in (None,"") or event.get("intAwayScore") not in (None,""):
        return None
    return {"id":event.get("idEvent"),"kickoff":kickoff,
            "local":kickoff.astimezone(SYDNEY),
            "country":event.get("strCountry") or "Unknown",
            "league":event.get("strLeague") or "Unknown",
            "home":event.get("strHomeTeam") or "Unknown home",
            "away":event.get("strAwayTeam") or "Unknown away",
            "source":"TheSportsDB free"}

def worldwide(start_day,number_days):
    # Include neighboring UTC dates so Sydney calendar dates are never lost.
    start=datetime.combine(start_day,datetime.min.time(),tzinfo=SYDNEY)
    stop=start+timedelta(days=number_days)
    dates=set()
    d=(start.astimezone(timezone.utc)-timedelta(days=1)).date()
    last=(stop.astimezone(timezone.utc)+timedelta(days=1)).date()
    while d<=last:
        dates.add(d.isoformat());d+=timedelta(days=1)
    found={};warnings=[]
    for day in sorted(dates):
        try:
            items=day_events(day)
            for e in items:
                m=parse_event(e)
                if m and start<=m["local"]<stop and m["kickoff"]>datetime.now(timezone.utc):
                    found[m["id"] or (m["kickoff"],m["home"],m["away"])]=m
        except (requests.RequestException,ValueError) as exc:
            warnings.append(f"{day}: {exc}")
    return sorted(found.values(),key=lambda m:m["kickoff"]),warnings
