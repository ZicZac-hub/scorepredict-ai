"""Fetch regulation-time results from the provider that supplied a forecast."""
from copy import deepcopy

from api_football import _convert
from history_store import identity_matches


def verification_batch(rows, cursor=0, limit=10):
    """Rotate bounded checks so unavailable recent matches cannot block older ones."""
    if not rows or limit <= 0:
        return [], 0
    start = cursor % len(rows)
    count = min(limit, len(rows))
    return [rows[(start+i) % len(rows)] for i in range(count)], (start+count) % len(rows)


def primary_result(payload):
    """Normalize a football-data.org result without confusing extra time with 90 minutes."""
    match = deepcopy(payload)
    match['provider'] = 'football-data.org'
    score = match.get('score') or {}
    if score.get('duration') in ('EXTRA_TIME', 'PENALTY_SHOOTOUT'):
        score['fullTime'] = score.get('regularTime') or {'home': None, 'away': None}
        score['duration'] = 'REGULAR'
    match['score'] = score
    return match


def fetch_saved_result(saved, primary_get=None, secondary_get=None):
    """Return a matching result only; callbacks carry credentials outside this module.

    Old archives lack provider identity. Both providers may be checked, but only
    an exact team and kickoff match may establish their source.
    """
    provider = saved.get('provider', 'legacy-unknown')
    providers = ['API-Football', 'football-data.org'] if provider == 'legacy-unknown' else [provider]
    errors = []
    for source in providers:
        try:
            if source == 'API-Football' and secondary_get:
                candidates = [_convert(raw, saved.get('league', 'WORLD'))
                              for raw in secondary_get('/fixtures', {'id': saved['fixture_id']})]
            elif source == 'football-data.org' and primary_get:
                candidates = [primary_result(primary_get(f"/matches/{saved['fixture_id']}"))]
            else:
                continue
            for match in candidates:
                full = (match.get('score') or {}).get('fullTime') or {}
                valid_score = all(type(full.get(side)) is int and full[side] >= 0
                                  for side in ('home', 'away'))
                if match.get('status') == 'FINISHED' and valid_score and identity_matches(saved, match):
                    return match, errors
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            errors.append(f'{source}: {str(exc)[:120]}')
    return None, errors
