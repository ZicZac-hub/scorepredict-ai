"""Football-data.org adapter with bounded requests and last-good disk snapshots.

Snapshots survive Streamlit reruns on the same instance, but not necessarily
free-host restarts. Never present old data as verified live results.
"""
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qs
import json
import os
import threading
import time
from pathlib import Path
from uuid import uuid4
import requests
import streamlit as st

ROOT = 'https://api.football-data.org/v4'
CACHE_DIR = Path('/tmp/scorepredict_api_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCK = threading.RLock()
MIN_GAP = 7.0  # conservative spacing for free API accounts
TTL = 3600
STALE_LIMIT = 7 * 86400
_last_request = 0.0
_blocked_until = 0.0

def _file(path, key):
    digest = hashlib.sha256((path + ':' + hashlib.sha256(key.encode()).hexdigest()).encode()).hexdigest()
    return CACHE_DIR / (digest + '.json')

def _snapshot(path, key):
    try:
        payload = json.loads(_file(path, key).read_text())
        if time.time() - payload['saved_at'] <= STALE_LIMIT:
            return payload
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None

def _write(path, key, value):
    target = _file(path, key)
    temp = target.with_name(target.name+'.'+uuid4().hex+'.tmp')
    fetched_at=time.time()
    try:
        temp.write_text(json.dumps({'saved_at': fetched_at, 'data': value}))
        os.replace(temp, target)
    except OSError:
        try: temp.unlink(missing_ok=True)
        except OSError: pass
    return fetched_at

def cache_ttl(path):
    parsed=urlsplit(path)
    query=parse_qs(parsed.query)
    if parsed.path == '/competitions': return 86400
    if parsed.path.startswith('/matches/'): return 60
    if '/matches' in parsed.path and query.get('season'):
        try:
            return 86400 if int(query['season'][0]) < datetime.now(timezone.utc).year-1 else 900
        except (ValueError,TypeError): return 900
    if '/matches' in parsed.path: return 300
    return TTL


def _expose(data, fetched_at, ttl, cache_status, with_metadata):
    if not with_metadata:return data
    now=time.time()
    return {'data':data,'metadata':{
        'provider':'football-data.org',
        'fetched_at_utc':datetime.fromtimestamp(fetched_at,timezone.utc).isoformat(),
        'retrieved_at_utc':datetime.fromtimestamp(now,timezone.utc).isoformat(),
        'age_seconds':max(0.0,now-fetched_at),'cache_status':cache_status,
        'stale':cache_status=='stale_fallback','ttl_seconds':ttl,'freshness_known':True}}


def get_json(path, api_key, with_metadata=False):
    global _last_request, _blocked_until
    ttl=cache_ttl(path)
    with LOCK:
        cached = _snapshot(path, api_key)
        age=time.time()-cached['saved_at'] if cached else None
        if cached and age < ttl:
            return _expose(cached['data'],cached['saved_at'],ttl,'fresh_cache',with_metadata)
        stale_limit=min(STALE_LIMIT,max(300,ttl*4))
        stale=cached if cached and age < stale_limit else None
        if time.time() < _blocked_until:
            if stale:
                st.warning('API cooldown active: using a previously downloaded snapshot (may be outdated).')
                return _expose(stale['data'],stale['saved_at'],ttl,'stale_fallback',with_metadata)
            raise RuntimeError('API cooldown active; no recent saved snapshot for this competition.')
        gap = MIN_GAP - (time.monotonic() - _last_request)
        if gap > 0:
            time.sleep(gap)
        _last_request = time.monotonic()
        try:
            response = requests.get(ROOT + path, headers={'X-Auth-Token': api_key}, timeout=25)
            if response.status_code == 429:
                retry = response.headers.get('Retry-After', '90')
                try: delay = min(max(float(retry), 60), 900)
                except ValueError: delay = 90
                _blocked_until = time.time() + delay
                if stale:
                    st.warning('Football API rate limit reached: showing labelled cached data, which may be outdated.')
                    return _expose(stale['data'],stale['saved_at'],ttl,'stale_fallback',with_metadata)
                raise RuntimeError(f'Football API rate limit reached. Wait about {int(delay)} seconds; no recent data available.')
            if response.status_code in (401, 403):
                raise RuntimeError('API access denied. Check key and competition permissions.')
            response.raise_for_status()
            data = response.json()
            fetched_at=_write(path, api_key, data)
            return _expose(data,fetched_at,ttl,'network',with_metadata)
        except requests.RequestException as exc:
            if stale:
                st.warning('Football API temporarily unavailable: showing a labelled saved snapshot, which may be outdated.')
                return _expose(stale['data'],stale['saved_at'],ttl,'stale_fallback',with_metadata)
            raise RuntimeError('Football API request failed ('+type(exc).__name__+')') from exc

def competitions(key):
    data = get_json('/competitions', key)
    return sorted([c for c in data.get('competitions', []) if c.get('code')],
                  key=lambda c: (c.get('area', {}).get('name', ''), c.get('name', '')))

def fixtures(key, code, date_from, date_to):
    # Stable broad date window avoids new API calls for each slider adjustment.
    data = get_json(f'/competitions/{code}/matches?dateFrom={date_from}&dateTo={date_to}', key)
    return [m for m in data.get('matches', []) if m.get('status') in ('SCHEDULED', 'TIMED')]

def history(key, code, now=None, with_evidence=False):
    from history_pipeline import load_evidence
    now=now or datetime.now(timezone.utc)
    fixture={'utcDate':now.isoformat() if hasattr(now,'isoformat') else str(now),
             'provider':'football-data.org','competition':{'code':code}}
    def source(path,with_metadata=False):
        return get_json(path,key,with_metadata=with_metadata)
    evidence=load_evidence(fixture,as_of=now,primary_get=source)
    return evidence if with_evidence else evidence['history']
