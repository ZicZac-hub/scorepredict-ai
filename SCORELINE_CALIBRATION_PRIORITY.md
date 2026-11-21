# Exact-score calibration priority — 22 November 2026 (Sydney)

## Verified case and causal limitation

The archived identity-matched results include Como 2–0 Cagliari (fixture 1550200)
and Lazio 3–0 Lecce (fixture 1550202), both 21 November in Italy.

The original v8_near Lazio forecast captured 19 Nov 2026 at 20:27:45 UTC
(forecast ID 6b3f81042ebecce9e6df174ccdb319823ed397dcfbc43277c59ae8c8862dc65e)
gave Lazio 3–0 a model probability of 6.955151%, its fifth-highest
individual scoreline. A fifth-ranked individual leg does not necessarily
appear among the highest five two-match joint combinations.

A 3–0 final has three total goals, not four or more. The historical replay
(data/accuracy_diagnosis/historical_replay.json) reports the four-plus-goal
observed share as 33.0789% and predicted share as 33.1692% on 393
retrospective development fixtures. This does not support blanket
inflation of high-goal scores.

A direct read-only reproduction from the two archived 19 Nov near-kickoff
score grids gives 7.271889995% model mass for the five highest joint cells.
The actual Como 2–0 / Lazio 3–0 cell has 0.860453985% model mass and ranks
21st among all score-pair cells. This is why it was absent from the highest
five joint combinations despite Lazio 3–0 ranking fifth by itself.
These are uncalibrated model masses, not historical hit probabilities.

The screenshots depict a manually selected Como–Lazio pair; this is
different from the V10 automatically frozen daily pair and the V25 archived
automatic shadow pair. Do not insert screenshot selections into either
immutable prospective ledger after kickoff.

## New audit

Run: python scoreline_calibration_audit.py

The read-only script selects the latest pre-kickoff snapshot per fixture
and V8 family, with at least 30 minutes of lead time. It rejects wrong
identity, conflicting result sources, invalid distributions and late
verification; it never calls an external provider.

It measures predicted versus actual rates and Brier scores for exact 3–0,
exact 0–3, four-plus total goals, and margins of three or more. It reports
the model probability and rank of each verified final score with forecast
ID provenance. V8 and V8-near remain separate groups, not independent
matching observations.

The verified-results backfill workflow also regenerates the analysis in
data/accuracy_diagnosis/scoreline_calibration.json, without rewriting
an unchanged report.

## Model promotion criteria

Compare predefined models on equal, time-matched historical and
independent prospective samples. Evaluate score-distribution log loss,
event Brier, 3–0 and 0–3 calibration, high-margin tails, and verified
five-double exact-score hits. Do not fit an arbitrary 3–0 multiplier to
this one result. Keep the model unchanged without demonstrated
out-of-sample improvement.

No 70% accuracy, ROI, bookmaker value, new source xG, or model promotion
is asserted by this research-only update.
