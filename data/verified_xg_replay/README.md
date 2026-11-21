# Verified historical xG shadow evidence

This directory contains **write-once**, pre-kickoff historical match inputs captured alongside API-Football xG source packets. Each record references an immutable, hash-checked source packet under data/v21_sources/ and stores its original as-of input digest. Finished scores are joined only later from data/v10_prospective/verified_results.json.

Records are created prospectively by the existing scheduled v21_capture_sources.py collection. An old provider packet with no frozen historical input must **not** be reconstructed after the result and called prospective evidence.

The read-only research runner (python verified_xg_evidence_pipeline.py) publishes readiness and matched comparisons, using a predeclared 2026-12-15 holdout cutoff. A coverage threshold is not statistical proof of improvement. No automatic production model promotion is permitted.

This tracked README also makes the directory safe to stage when there are no new observations; an empty directory by itself would cause the scheduled GitHub git add to fail.
