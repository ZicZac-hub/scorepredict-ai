# Reference-informed v7 model and release decision

V7 adds an independently implemented statistical model, a preserved-v6
comparison, an evidence dossier and explicit research selection. It does not
call ScoreGPT, ChatGPT/OpenAI, Claude, DeepSeek or Gemini. Their public material
is research context, not model votes or proprietary model access.

The fitted contributor estimates time-decayed team attack/defence and home/away
league intercepts from regulation scores. Ridge regularisation stabilises thin
samples. A Dixon-Coles low-score parameter is fitted from prior results, with
positive probability constraints. Complete provider-qualified competition
history is required; selected-team-only data and sparse populations disclose a
v6 fallback. Public modelling references include
[Dixon and Coles (1997)](https://doi.org/10.1111/1467-9876.00065) and
[Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437).

## Parameter selection and independent evaluation

Public [football-data.co.uk](https://www.football-data.co.uk/data.php) Premier
League/Bundesliga CSVs for 2022/23 initialise histories. All six combinations of
ridge penalty (2, 6, 12) and half-life (180, 365 days), and fitted-model weights
(0, .25, .5, .75, 1), are compared on 2023/24 only. Each fit excludes the fixture
and every same-day/future match. Exact-score hits are the primary selection
criterion, then top-three coverage, then full-grid log loss. The selected
parameters are penalty **2**, half-life **180 days**, fitted weight **1**.

On the 661 paired 2023/24 validation matches, the selected model recorded 88
exact hits versus v6's 83. This is model-selection data, not an accuracy claim.
An earlier log-loss-first development comparison had already inspected 2024/25
(76 v6 exact hits versus 62 for that earlier candidate). Therefore 2024/25 is
explicitly treated as development evidence for the revised exact-first policy;
it is not described as an untouched final test.

After the exact-first parameters were frozen, an independent replay used the
existing API-Football historical artifact from
[the repository backtest](https://github.com/ZicZac-hub/scorepredict-ai/actions/runs/43352996768).
Its 2,255 input matches cover five leagues in 2025/2026; 1,385 eligible paired
2026 forecasts use strictly earlier same-provider/league results. These are
historical reconstructions, separate from the user's 236 original forecasts.
No provider-2026 results selected or adjusted parameters.

| Independent 2026 metric | Preserved v6 | Fitted v7 candidate |
| --- | ---: | ---: |
| Paired forecasts | 1,385 | 1,385 |
| Exact-score hits | 163 (11.77%) | 157 (11.34%) |
| Top-three coverage | 30.90% | 30.90% |
| Full-grid log loss (lower better) | 2.97593 | 2.96773 |
| Outcome Brier (lower better) | 0.61372 | 0.59777 |
| Share of 1-1 top scores | 78.41% | 48.45% |

V7 improves these probability scores but does not improve the primary exact-hit
metric. **V6 stays the default active forecast.** The fitted v7 model is
available as an explicitly selected research model and for model-versioned
revisions/retrospective comparisons. Fewer 1-1 forecasts alone is not a reason
to promote it. No future exact-score improvement is established. Counts are
finite-sample historical observations; shared teams and histories limit simple
independence assumptions.

## API and reproducibility

`forecast_model.predict(fixture, history, league_history=complete_history)`
returns the preserved v6 distribution with model version
`reference-informed-v7-v6-retained`, `active_model_version` pointing to v6,
`candidate_summary`, `candidate_disagreement`, `promotion_evidence`, and
`selection_policy`. The `methods` field remains a dictionary of numeric tuples.
An explicit `variant='reference_v7'` selects the fitted candidate; insufficient
evidence returns `reference-informed-v7-fallback-v6` with `fallback_reason`.
Other experimental variants continue to delegate to the unchanged v6 engine.

Both selections derive the top exact score and outcome probabilities from the
same normalised score grid. The candidate additionally exposes a clearly
labelled score within the favoured result category. This alternative never
replaces its unconditional top score. Expected goal counts are estimated from
results and are not observed shot-quality xG.

Use existing cached CSVs or download them with the earlier
`model_repair_benchmark.py --download` command. Reproduce public selection with:

```sh
python reference_model_benchmark.py --data-dir work/model_benchmark_data
python reference_provider_benchmark.py data/backtest_finished_matches.json
python -m pytest tests/test_reference_model.py -q
```

Benchmark files default to `work/`. Provider histories require the existing
authorised API-Football data workflow; no secrets or provider data are embedded
in model code. Thirteen regressions cover provenance, future-result exclusion,
actual-score exclusion, fitted positivity, high-scoring evidence, sparse-data
fallback, comparison labelling, and retention of the better exact-hit baseline.
