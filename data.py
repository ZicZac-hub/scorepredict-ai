"""Football-data.org v4 API adapter. No scraped/fictional live fixtures."""
import requests
import streamlit as st

ROOT = 'https://api.football-data.org/v4'

@st.cache_data(ttl=3600, show_spinner=False)
def get_json(path, api_key):
    response = requests.get(ROOT + path, headers={'X-Auth-Token': api_key}, timeout=25)
    if response.status_code in (401, 403):
        raise RuntimeError('API access denied. Check your key or free-plan competition permissions.')
    if response.status_code == 429:
        raise RuntimeError('Free API rate limit reached. Please try again later.')
    response.raise_for_status()
    return response.json()


def competitions(key):
    data = get_json('/competitions', key)
    return sorted([c for c in data.get('competitions', []) if c.get('code')],
                  key=lambda c: (c.get('area', {}).get('name', ''), c.get('name', '')))


def fixtures(key, code, date_from, date_to):
    data = get_json(f'/competitions/{code}/matches?dateFrom={date_from}&dateTo={date_to}', key)
    return [m for m in data.get('matches', []) if m.get('status') in ('SCHEDULED', 'TIMED')]


def history(key, code):
    # API returns matches for current season by default. Past-season availability varies by plan.
    data = get_json(f'/competitions/{code}/matches?status=FINISHED', key)
    return data.get('matches', [])
