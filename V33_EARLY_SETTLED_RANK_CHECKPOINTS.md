# V33 — Early-settled, originally frozen ranking checkpoints

V32 requires **all** original fixtures on a Sydney-day candidate slate to have
independently verified finals. V33 reports earlier rank-lift observations
without waiting for unrelated games on that slate.

For each V28 original frozen day, reverify the full unmodified decision,
source receipts, original ranking metrics and candidate digest. Rank all
original candidates without looking at results. Predefine disjoint top and
bottom 1/5/10-pair cohorts for each quality and robustness ranking. A
checkpoint becomes scoreable only once **every candidate pair in both arms**
has verified regulation-time finals through the existing V30 verification
path. Other original candidates can still be pending. The comparisons
always have 5 exact-score tickets per candidate pair per arm.

Unverified or partially completed checkpoints remain **pending**, never
losses. A full original day, rather than each overlapping candidate pair,
remains one statistical unit. Within-day correlated results and different
fixture coverage prevent an iid pair hit-rate interpretation. Aggregates
remain exploratory until 30 verified day-checkpoint observations; that is
not evidence of a 70% live five-double hit rate.

Output: `data/v33_early_rank_checkpoints/status.json`.
Run: `python -m unittest -q test_v33_early_rank_checkpoints.py` then
`python v33_early_rank_checkpoints.py`.

This is strictly research-only. No external API requests, retroactive picks,
ticket edits, rank changes, or live selection policy changes.
