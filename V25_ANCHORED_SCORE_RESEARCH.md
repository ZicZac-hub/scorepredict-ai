# V25 probability-anchored score selection experiment

V25 has **1/4 verified five-double robust wins** vs **0/4 original V8
five-double wins** on the same four archived pairs. Those sets lack independently
frozen V25 probability grids: legacy results cannot validate a new V25
calibration method. This change is not a demonstrated accuracy improvement.

## Mathematical challenger

V25's original five exact-score doubles maximize the **lower envelope**
of several goal-model sensitivities; that envelope is not calibrated or a
probability lower bound. The V25 mixture average excludes V8. A prespecified
alternate model uses **75% original V8 score matrix + 25% equal mix of the
three V25 sensitivity transformations per match**. The five highest
products of the resulting two match grids give the challenger five doubles.

These are the SAME original V28 frozen quality-selected two fixtures. The V25
robust portfolio and the original V8 top-five are saved as explicit controls.
The algorithm is not refitted to the four historical outcomes; weights are
fixed in code. A new cohort is captured only within 20 minutes of the original
V28 decision's actual issuance, with at least 45 minutes before each kickoff.

## Provenance, comparison, and gates

Every candidate source is reverified against its write-once original V28
candidate catalog, original forecast receipts, same fixed source IDs and all
ranking values. The anchored two full probability grids and five picks are
written to a separate **write-once** day file and never appended to earlier
V25 or V28 forecasts. A later verifier replays the original frozen inputs
and compares the entire saved challenger object before result grading.

Results are sourced only through the independent V30 original-fixture
regulation-time index. Pending games are never counted as losses. A matched
same-pair, same-day comparison records original V8, original V25 robust, and
anchored V25 wins; this is a genuine *score-selection* comparison, unlike
different-fixture ranking experiments. An initial review target is 100 verified
paired days, with paired-discordance review required for promotion.
No inferred or AI-generated finals are accepted.

Command: `python v25_anchored_challenger.py` for scheduled capture and
readout. `python v25_anchored_challenger.py --verify-only` in CI/backfill
never issues picks. Artifacts are in
`data/v25_anchored_challenger/{sets,status.json}`.

The new policy stays research-only, with no new paid API calls and no live
model promotion. Neither this prospective experiment nor passed unit tests
prove 70% five-double accuracy.
