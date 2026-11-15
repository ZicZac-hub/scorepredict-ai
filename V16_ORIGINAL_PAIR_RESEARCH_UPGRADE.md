# V16 — True frozen-pair AI research and credible accuracy evaluation

**15 November 2026.** Experimental integration. The original production V8 five-double selector, archived exact scores and historical records are unchanged. Neither the new audit nor the tests calls paid AI APIs.

## Why the previous six-AI research was incomplete

Six configured credentials are not six successful model votes. The scheduled collector could select a generic earliest same-day pair different from the two fixtures selected by the original frozen five-double selector. This left the V15 AI panel without two comparable legs. A V8 refresh may also have a new source forecast ID or digest for the same fixture, so matching fixtures by ID alone is not a sound controlled comparison.

## V16 implementation

1. The new v16_pair_research.py reads the exact original two-match V8 pair and original write-once forecast source files from data/v10_prospective. It verifies teams, IDs, dates, digest, same Sydney calendar day, four distinct teams, original pre-match timestamps and five frozen score options. Eligible original pairs are ranked using their already-frozen estimated five-double coverage; final results never enter selection.
2. In an owner-opted-in two-match AI batch, llm_research_panel.py first attempts to research the eligible original frozen two-leg pair before trying a generic pair. Existing published AI research is never charged for again just to force completeness.
3. The V15 research joiner matches BOTH legs to original provider, fixture, team IDs, forecast IDs, source digests and timestamps. A mismatched vote is an abstention, not a synthetic consensus. An original archived model is used rather than an unrelated later fixture refresh.
4. The read-only source audit validates the same real provider schema and reports source match failures per leg. Duplicate records do not count as new independent sources.
5. The V15 evaluation and website panel show actual same-pair V8 versus AI wins, wins unique to either model and 95% Wilson intervals. Pending final results do not count as losses.

## Cost safety and opt-in

Paid research remains LIMITED to one fixture per batch by default. The owner must deliberately set GitHub Actions variable LLM_RESEARCH_MAX_MATCHES=2 to permit a paired batch. Two fixtures can approximately double calls/costs for the enabled model providers. The collector never raises this limit automatically. GitHub Actions and Streamlit secrets are separate. No passwords or API tokens are included in source code or reports.

## What this does and does not prove

This is an evidence-quality and coverage upgrade, not verified better predictive accuracy. It cannot establish seven wins out of ten exact-score five-double sets. Six language models reading the same statistical briefing are correlated; their chosen scores are not six independent calibrated probabilities. Keep the live V8 baseline and historical records unchanged unless future truly frozen, independently verified comparisons prove a meaningful improvement.

## Validation

New regression tests cover exact original pair priority, no future evidence, source identity and digest matching, duplicate sources, malformed votes, missing panels, uncertainty ranges, opt-in billing and immutable existing records. CI runs the full suite on a PR. Do not claim pass counts for V16 until the GitHub checks have completed successfully.
