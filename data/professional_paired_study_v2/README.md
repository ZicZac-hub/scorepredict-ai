# Versioned professional-pair ablation research (v2)

The existing `data/professional_paired_study/` directory is an immutable **v1** research archive. Its registered code hashes no longer match the current implementation, so rewriting its protocol or silently accepting a different algorithm would invalidate evidence.

This directory is a separate prospective **v2** study. The first scheduled run registers `protocol.json` with the current source-code fingerprints and records new trial decisions before kickoff. Previously finished games and old v1 trials are **not** imported or reissued as v2 forecasts.

The baseline and source-evidence budgets are unchanged. Neither the v1 nor v2 study can auto-promote models. If the implementation later changes, a new study version is required; do not overwrite the v2 protocol.
