# Shared evidence and prospective forecast ledger

All live, worldwide, scheduled and revised forecasts now use the same
`two-season-asof-v1` evidence policy: the target competition's current and
previous seasons, at most 730 days, provider-qualified IDs, regulation scores,
and results available by the frozen cutoff. `finishedAt` is preferred; absent
that field, availability is conservatively estimated as kickoff plus three
hours. Conflicting duplicate results are excluded. A deterministic input digest
and separate retrieval/cache provenance make the inputs auditable.

This changes the evidence protocol, not the numerical model. Earlier v6/v7
163/157 validation counts belong to the old protocol. A fixed T−60m replay under
the new policy produced 163 v6 and 159 fitted-v7 hits on 1,385 paired matches;
league-average Poisson and always-1–1 each hit 158. This is not an accuracy gain
caused by model changes, and no model is automatically promoted. Fresh forecasts
are collected to establish a genuine comparison.

## Central database

The app reads `DATABASE_URL` from Streamlit Secrets or its environment and uses
PostgreSQL through psycopg 3. It creates only these namespaced tables/indexes:

- `forecast_ledger_forecasts`: immutable complete score grids and evidence.
- `forecast_ledger_settlements`: identity-checked regulation-time results.
- `forecast_archive_snapshots`: versioned backups of existing singles/doubles.

No existing application tables are dropped or replaced. Originals are restored
before new write-once predictions can be saved after a restart. Failed central
restoration pauses those new original saves; it cannot silently substitute a
later forecast. New doubles are backed up immediately. Versioned legacy records
remain separate from genuine prospective forecasts and do not manufacture a
new pre-match accuracy record.

The app's Archive page reports connection and synchronization status. PostgreSQL
is marked durable; a local SQLite/file outbox is explicitly local-only. A failed
central connection preserves local records for retry and shows partial coverage.
Connection strings and provider credentials are excluded from stored records and
error displays.

For scheduled collection to write directly to PostgreSQL, configure the same
`DATABASE_URL` as a GitHub Actions secret. When that secret is absent, Actions
commits an immutable file ledger under `data/prospective_ledger/`; the app
imports those files into its configured database. The private server archive
stays private: its originals are backed up to PostgreSQL, not automatically
published into GitHub.

## Genuine comparisons

`prospective_ledger.capture_pair` runs preserved v6, fitted v7, a training-only
league-Poisson baseline and the fixed 1–1 point benchmark on identical evidence.
It stores full grids, support and tail policy; numerical model/release versions;
actual capture and evidence times; the input digest; and freshness metadata.
Historical reconstructions, targets with known scores, late computations and
stale/backdated collection clocks cannot be entered as prospective records.
Fallback v7 outputs are labelled and excluded from paired comparisons.

Capture slots are research, T−24h, T−60m and T−30m, with documented time windows.
A slot is complete only when all four eligible contributors share one capture,
evidence digest and identity. Interrupted packages remain retryable. The first
complete common package is graded; an old v5 original does not block a new v6/v7
comparison. Original and retrospective accuracy pages remain unchanged.

Metrics include exact and top-three hits, full-grid log loss, 1X2 Brier and
reliability bins, paired wins/losses, and league/horizon breakdowns. Always 1–1
is a point benchmark; its artificial one-hot grid is excluded from probabilistic
metrics. Repeated horizons are separate strata, not independent matches. Pooled
exact McNemar inference is suppressed when fixtures repeat; horizon-specific
tests and clustered uncertainty are reported instead.

## Collection and recovery

The repository collector runs every 15 minutes with discovery caching, a shared
per-run history cache, explicit API/time/fixture budgets, and full coverage/skip
reports. There is no silent first-30-fixtures restriction. Near-kickoff fixture
status and available team news are refreshed before freezing a pair. Endpoint
cache lifetimes differ by data type, and stale fallbacks retain their actual
retrieval time. Team news is stored as evidence, not an unvalidated goal boost.

The heavier legacy verification/rescoring pass remains on a six-hour cadence.
All immutable file records and partial coverage reports are preserved even when
database synchronization fails. Retry imports the outbox before deciding which
slots are already complete.

Run the comparison independently with:

```sh
python evaluate_prospective_ledger.py
python reference_provider_benchmark.py data/backtest_finished_matches.json --forecast-lead-minutes 60
python -m pytest -q
```

Regression CI includes a dedicated PostgreSQL 16 service through
`TEST_DATABASE_URL`; those tests never use the production `DATABASE_URL`.
