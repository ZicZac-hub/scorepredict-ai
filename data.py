"""Football-data.org adapter with bounded requests and last-good disk snapshots.

Snapshots survive Streamlit reruns on the same instance, but not necessarily
free-host restarts. Never present old data as verified live results.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
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
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps({'saved_at': time.time(), 'data': value}))
    os.replace(temp, target)

def get_json(path, api_key):
    global _last_request, _blocked_until
    with LOCK:
        cached = _snapshot(path, api_key)
        if cached and time.time() - cached['saved_at'] < TTL:
            return cached['data']
        if time.time() < _blocked_until:
            if cached:
                st.warning('API cooldown active: using a previously downloaded snapshot (may be outdated).')
                return cached['data']
            raise RuntimeError('API cooldown active; no saved snapshot for this competition yet.')
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
                if cached:
                    st.warning('Football API rate limit reached: showing cached data, which may be outdated.')
                    return cached['data']
                raise RuntimeError(f'Football API rate limit reached. Wait about {int(delay)} seconds; no cached data available.')
            if response.status_code in (401, 403):
                raise RuntimeError('API access denied. Check key and competition permissions.')
            response.raise_for_status()
            data = response.json()
            _write(path, api_key, data)
            return data
        except requests.RequestException as exc:
            if cached:
                st.warning('Football API temporarily unavailable: showing a saved snapshot, which may be outdated.')
                return cached['data']
            raise RuntimeError(f'Football API request failed: {exc}') from exc

def competitions(key):
    data = get_json('/competitions', key)
    return sorted([c for c in data.get('competitions', []) if c.get('code')],
                  key=lambda c: (c.get('area', {}).get('name', ''), c.get('name', '')))

def fixtures(key, code, date_from, date_to):
    # Stable broad date window avoids new API calls for each slider adjustment.
    data = get_json(f'/competitions/{code}/matches?dateFrom={date_from}&dateTo={date_to}', key)
    return [m for m in data.get('matches', []) if m.get('status') in ('SCHEDULED', 'TIMED')]

def history(key, code):
    data = get_json(f'/competitions/{code}/matches?status=FINISHED', key)
    return data.get('matches', [])
