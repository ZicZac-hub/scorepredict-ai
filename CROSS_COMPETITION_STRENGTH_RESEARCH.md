# Cross-competition strength research v1

Purpose: test whether domestic league-relative attack and defence information can improve a sparse UEFA score distribution without lowering V8's five-match requirement or treating unavailable European history as fabricated data.

## Fixed research hypothesis

- API-Football team identifiers only; current-season participation in exactly one supported domestic league is required.
- Existing history cache only, no extra requests. Source observation must be known, not stale, and at or before the decision cutoff. Existing history policy excludes unavailable results and conflicting duplicates.
- At least 50 domestic league results, eight team results, and three relevant venue results per team. Both teams must qualify.
- Six regularized opponent-adjustment iterations, 180-day half-life and prior of eight. A bounded 0.85–1.15 goal tilt modifies the existing European control grid without changing its support.
- Domestic relative strength is not absolute strength across countries. The method does not estimate league quality or UEFA coefficients, nor does it claim an accuracy gain.

## Next validation gate

The proposed scheduled integration uses the existing pre-match cache, with no extra provider calls. It preserves content-addressed source packets and implementation-bound write-once controls/challengers before kickoff. The read-only grader reproduces both score grids and checks independently verified regulation results; it reports matched exact-score hits and log loss. Outcome Brier, paired uncertainty and prospective five-double trials remain follow-up work. Actual source coverage and live scheduled execution have not yet been verified. Report missing/ambiguous contexts and coverage. Do not use earlier completed fixtures as prospective results, change existing V37/V41/V42 archives, or automatically promote. A five-double improvement requires a separate prospective matched-pair trial.

Run: `python -m pytest -q tests/test_cross_competition_strength.py`. No real provider calls or paid AI requests are needed for these synthetic tests.

## Domestic history availability follow-up

The first production capture evaluated 19 European candidates and reported missing or ambiguous domestic context for all 19. The collector's in-memory league history cache is normally populated only by fixtures due for the regular capture. This can omit domestic histories during a UEFA-only collection run.

The proposed end-of-run research pass revisits still-future European fixtures within 48 hours, after ordinary forecast and market collection. It reuses existing cache entries, and on the existing heavy cadence only may use up to 20 spare adapter calls (two UEFA leagues and eight supported domestic leagues, two seasons each). It cannot exceed the existing configured main collector call ceiling, and light runs are cache-only. It does not authorize new subscriptions or raise that ceiling; actual call consumption can increase within it. The sparse model eligibility and source freshness/identity checks remain unchanged. The research pass reports requested, eligible and captured counts separately; no claim of improved coverage or accuracy until a real run demonstrates it.
