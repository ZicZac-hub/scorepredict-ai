"""Highest-coverage five exact-score pairs, with explicit independence estimates.

Only real score distributions are used. Alternatives for one fixture pair are
mutually exclusive; their summed mass is coverage, not a measured win rate.
"""
import math
from datetime import datetime, timezone
from itertools import combinations
from zoneinfo import ZoneInfo

SYDNEY = ZoneInfo('Australia/Sydney')
MIN_LEAD_SECONDS = 300


def fixture_identity(row):
    fixture = row.get('fixture') or {}
    provider = fixture.get('provider') or 'football-data.org'
    return str(provider), str(fixture.get('id'))


def same_fixture(first, second):
    if fixture_identity(first) == fixture_identity(second):
        return True
    a, b = first.get('fixture') or {}, second.get('fixture') or {}
    names_a = tuple(' '.join(str((a.get(side+'Team') or {}).get('name') or '').casefold().split()) for side in ('home','away'))
    names_b = tuple(' '.join(str((b.get(side+'Team') or {}).get('name') or '').casefold().split()) for side in ('home','away'))
    # A duplicate event from two providers is still one match, not two legs.
    kickoff = _instant(first.get('kickoff') or a.get('utcDate'))
    return bool(all(names_a) and names_a == names_b and kickoff is not None and
                kickoff == _instant(second.get('kickoff') or b.get('utcDate')))


def _scores(row):
    """Validate and deduplicate cells rather than count the same outcome twice."""
    cells = {}
    for candidate in (row.get('result') or {}).get('scores', []):
        try:
            h, a, probability = candidate
            if type(h) is not int or type(a) is not int or min(h, a) < 0:
                continue
            probability = float(probability)
            if not math.isfinite(probability) or not 0 < probability <= 1:
                continue
            key = (h, a)
            cells[key] = max(cells.get(key, 0.0), probability)
        except (ValueError, TypeError):
            continue
    if sum(cells.values()) > 1.0 + 1e-8:
        return []
    return sorted(((h, a, p) for (h, a), p in cells.items()),
                  key=lambda x: (-x[2], x[0], x[1]))


def five_doubles(first, second, limit=5):
    """Highest-probability distinct pairs maximize covered independent mass.

    At most ``limit`` leading cells in either leg can occur in the best
    ``limit`` Cartesian products, so keeping that many cells is sufficient.
    """
    if limit <= 0 or same_fixture(first, second):
        return []
    left, right = _scores(first)[:limit], _scores(second)[:limit]
    options = [{'Score 1': f'{h}-{a}', 'Score 2': f'{h2}-{a2}',
                'P1': p, 'P2': p2, 'Joint': p*p2}
               for h, a, p in left for h2, a2, p2 in right]
    return sorted(options, key=lambda x: (-x['Joint'], x['Score 1'], x['Score 2']))[:limit]


def _instant(value):
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace('Z', '+00:00'))
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _sydney_day(value):
    """Use the user's Sydney matchday, including daylight saving changes."""
    result = _instant(value)
    return result.astimezone(SYDNEY).date() if result is not None else None


def pair_issue(first, second, *, now=None, same_day=True):
    """Validate a two-fixture opportunity before ranking live picks."""
    fixtures = [row.get('fixture') or {} for row in (first, second)]
    if any(fixture.get('id') in (None, '') for fixture in fixtures):
        return 'Both matches need a fixture ID.'
    if same_fixture(first, second):
        return 'Choose two different fixtures.'
    names = [' '.join(str((fixture.get(side + 'Team') or {}).get('name') or '')
                      .casefold().split())
             for fixture in fixtures for side in ('home', 'away')]
    if not all(names) or len(set(names)) != 4:
        return 'Choose a pair involving four different teams.'
    team_ids = [(fixture.get('provider') or 'football-data.org', str(team['id']))
                for fixture in fixtures for side in ('home', 'away')
                if (team := fixture.get(side + 'Team') or {}).get('id') not in (None, '')]
    if len(set(team_ids)) != len(team_ids):
        return 'Choose a pair involving four different teams.'
    times = [_instant(row.get('kickoff') or fixture.get('utcDate'))
             for row, fixture in zip((first, second), fixtures)]
    if any(kickoff is None for kickoff in times):
        return 'Both matches need a valid kickoff time.'
    if now is not None:
        clock = _instant(now)
        if clock is None or any((kickoff - clock).total_seconds() <= MIN_LEAD_SECONDS
                                for kickoff in times):
            return 'Both matches must be more than five minutes from kickoff.'
    if same_day and len({kickoff.astimezone(SYDNEY).date() for kickoff in times}) != 1:
        return 'Choose two matches on the same Sydney calendar date.'
    return None


def rank_pairs(rows, limit=5, *, now=None):
    """Compare distinct-team same-Sydney-day pairs by modeled coverage.

    Live callers supply ``now`` so cached fixtures that have started cannot
    become new predictions. Omitting it supports dated historical replays.
    """
    eligible = [r for r in rows if _scores(r)]
    candidates = []
    for first, second in combinations(eligible, 2):
        if pair_issue(first, second, now=now):
            continue
        picks = five_doubles(first, second)
        if len(picks) < 5:
            continue
        coverage = sum(x['Joint'] for x in picks)
        candidates.append({'first': first, 'second': second, 'doubles': picks,
                           'coverage': coverage, 'probability_assumption': 'independent matches'})
    return sorted(candidates, key=lambda x: (-x['coverage'],
                  fixture_identity(x['first']), fixture_identity(x['second'])))[:limit]
