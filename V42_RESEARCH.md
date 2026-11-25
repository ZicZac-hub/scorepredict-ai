# V42 isolated research programme
V41 source and records remain unchanged because the registered protocol hashes its implementation.
V42 compares a same-cutoff V41 control with four fixed score-shape hypotheses and one fixed ensemble. The changes are experimental, not established improvements. A separate preregistered trial reuses authenticated original pairs and regulation results; no historical issuance is allowed. The decision window is 5–180 minutes before earliest kickoff, so this comparison is not identical to the existing V41 operational schedule. No AI or lineup numerical weights are inferred without validation. Missing paired xG/market evidence is not fabricated.
 
Run tests: `python -m pytest -q tests/test_v42_research.py`.
One refresh: `python v42_research_bot.py --no-ai`.
Minute watchdog: `python v42_research_bot.py --watch` on a persistent supervised host with an independently refreshed provider archive. It does not fetch new provider data by itself. Optional `V42_AI_RESEARCH_ENABLED=true` and existing OpenAI key enable one advisory attempt per UTC day. No automatic generated-code execution, deployment, or live promotion. GitHub cron has a five-minute minimum and can be delayed; it is not a guaranteed minute service. Keep only one writer per research archive. Do not use a shared mutable checkout concurrently with the existing collector.
