# V24 — Capture more real football data, without inventing results

## What changed

Previous V23 odds collection only looked at the two matches already frozen
into a five-double set. With no qualifying pair it made **zero requests**,
despite the owner's `V23_EARLY_ODDS_ENABLED=true` setting.

V24 now has two complementary collection paths:

1. `v23_early_odds.py` scans **every individually frozen API-Football V8
   forecast**, not just two-match selections. It may request fixture-specific
   pre-match odds in its two limited windows, before kickoff.
2. `v24_bulk_market_capture.py` independently discovers **all fixture
   markets returned by selected API-Football /odds?date requests**, even
   when no statistical V8 forecast exists yet. This supports dates from
   today through 13 days ahead. Provider results are paginated at 10
   fixture entries per page and collected in timestamped, write-once
   source records under `data/v24_market_catalog/fixtures/`.

Only when a saved fixture can later be matched to an independently frozen
V8 forecast by provider fixture ID, team identities, exact kickoff and
source forecast ID/digest does the collector bind the market data for V23
scoring. There is **no retrospective remodelling, hidden match result,
fabricated player injury, assumed bookmaker update timestamp, or fake xG**.

Pre-match odds include independently observed 1X2, O/U 2.5, O/U 3.5 and
BTTS where the source actually provides the required complete prices.
Other bookmaker results that do not meet this minimal source standard
are **not** described as verified usable model inputs.

## Quotas and opt-in

The existing GitHub Actions variable
`V23_EARLY_ODDS_ENABLED=true` enables this additional broad collector.
It also requires a working repository secret `API_FOOTBALL_KEY`.
No further variable is required to begin date-wide market discovery.

**Default V24 budget: 12 date-page calls per UTC day, maximum 2 per run.**
This is an **additional** limit for V24 only; the other provider
pipelines also use API-Football. You can explicitly set the optional
Actions variable `V24_DAILY_ODDS_REQUESTS` to a whole number, at
most 60; leave it unset to conserve quota. No run can exceed
2 date-page attempts. Provider access/quota errors halt the collector
instead of triggering more requests.

Source dates are rotated across the 14-day horizon to avoid permanently
starving farther matches. Each 3-hour refresh slot deduplicates
date/page attempts, including failed API requests. All captured
market snapshots are saved under a SHA-256 fingerprint, with the
original fetched-at UTC instant and bookmaker-update versus API
retrieval provenance kept separate. If a source has no verified kickoff
or team identities, those fields remain explicitly unknown until a
matching saved forecast becomes available.

The core /odds API is date-filterable and paginated at 10 fixture
entries per page; odds are usually available 1–14 days before kickoff
and are refreshed roughly every 3 hours, but actual coverage varies
by competition and subscription.

## What this does NOT claim

- **All football matches worldwide:** API-Football only returns the
  records available within your plan, request budget, source dates
  and pagination limits. A source record may be incomplete.
- **Every website or every AI:** GitHub Actions runs the named
  API-Football collector only. Arbitrary external website browsing,
  proprietary independent xG providers, and paid AI research are NOT
  activated by this market-data option.
- **All true xG, lineups and injuries:** The separate
  `V21_EVIDENCE_ENABLED=true` variable authorizes bounded pre-match
  research for original V8 fixtures. Without that separate opt-in,
  the broad collector does NOT make player/lineup API calls.
  That V21 step now also scans all individual frozen V8 fixtures.
- **Guaranteed better exact scores:** V8 remains untouched and V23
  remains a research-only challenger. The timestamped V23 double
  hit/miss records are graded only after verified final scores, on
  the same pair as the original.

## Where to see evidence

- `data/v24_market_catalog/status.json`:
  actual requests, request budget, fixture markets found, and
  original forecasts successfully bound. 0 is reported as 0.
- `data/v24_market_catalog/fixtures/`: immutable bookmaker market
  observations, including those without corresponding V8 forecasts.
- `data/v21_sources/versions/`: source-matched append-only
  market packets usable by V23 after a valid original V8 forecast.
- `data/v23_prospective/status.json` and `performance.json`:
  actual V23 pairs selected, pending and verified results.
- The **AI Five-Double Studio** research dashboard shows live-source
  availability as reported by the last GitHub workflow, not by
  treating test success as prediction accuracy.

## Running it

The existing `Capture verified pre-match evidence` GitHub workflow runs
at minutes 2, 17, 32 and 47 of each hour, and can also be started
manually via GitHub Actions **Run workflow**.

The broad collector runs on a schedule or manual workflow dispatch,
not on source-code pushes; no new paid calls are made by code tests.
The next successful scheduled run creates the source report. An enabled
step does **not** prove any market was returned: inspect the counts.

Test: `tests/test_v24_bulk_market_capture.py`, together with the
previous `tests/test_v23_early_odds.py` and other same-pair tests.
