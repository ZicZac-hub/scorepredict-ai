"""Independently grade immutable V37 sparse-UEFA forecasts and league control.

Research only: never issue forecasts or recompute predictions after results.
Only original write-once source grids with matching digest and independently
verified regulation finals count. Pending is NOT a loss. One physical match
counts once regardless of how often a provider report is seen.
"""
import json
from datetime import datetime, timezone
from math import fsum
from pathlib import Path

from v10_prospective import ROOT, utc
from forecast_ledger import read_file_records
from v10_evaluate import _qualified
from verified_result_index import index_verified_results
from v25_score_accuracy_research import describe_score, summarize_scores
from v37_sparse_history_research import VERSION, frozen_arm_digest

OUT = Path("data/v37_sparse_prospective/status.json")
PREMATCH = Path("data/prematch")
MIN_REVIEW_FIXTURES = 100
METHODS = ("league_only", "sparse_shrunk")


def evaluate(*, root=ROOT, prematch=PREMATCH, now=None, rows=None,
             finals=None):
    clock = utc(now or datetime.now(timezone.utc))
    # read_latest() intentionally excludes experimental V37 from all V8/V28
    # source selection. The V37-only grader must inspect original immutable
    # ledger records directly; NEVER change read_latest's V8 selection gates.
    sources = (read_file_records(root,kind="forecasts")
               if rows is None else
               (list(rows.values()) if isinstance(rows,dict) else list(rows)))
    if finals is None:
        finals, conflicts, rejected = index_verified_results(root,prematch,clock)
    else:
        conflicts, rejected = set(), {}
    grouped = {method: [] for method in METHODS}
    pending, invalid, frozen = 0, 0, 0
    # A physical fixture cannot be counted twice even if two different
    # immutable V37 forecast IDs exist. Reject the whole duplicate group,
    # never select one ex post to flatter score accuracy.
    fixtures = {}
    for row in sources:
        if not isinstance(row,dict) or row.get("model_key") != "v37_sparse":
            continue
        fixture_key = (str(row.get("provider")),str(row.get("fixture_id")))
        fixtures.setdefault(fixture_key,[]).append(row)
    frozen = len(fixtures)
    for fixture_key, group in sorted(fixtures.items()):
        if len(group) != 1:
            invalid += 1
            continue
        row = group[0]
        try:
            if (row.get("v37_research_only") is not True or
                    row.get("v37_no_live_model_change") is not True or
                    row.get("v37_not_eligible_for_v28") is not True or
                    row["v37_arm_digest_sha256"] != frozen_arm_digest(row) or
                    not utc(row["evidence_as_of_utc"]) <=
                        utc(row["captured_at_utc"]) < utc(row["kickoff_utc"])):
                raise ValueError("invalid_source_integrity")
            final = finals.get(fixture_key)
            if final is None or not _qualified(row,final):
                pending += 1
                continue
            actual = (int(final["home_goals"]),int(final["away_goals"]))
            matched = (
                describe_score(row["v37_control_grid"],actual),
                describe_score(row["score_matrix"],actual),
            )
            for method, scored in zip(METHODS,matched):
                grouped[method].append(scored)
        except (ValueError, KeyError, TypeError, OverflowError):
            invalid += 1

    method_reports = {
        method:summarize_scores(grouped[method])
        for method in METHODS
    }
    n = len(grouped[METHODS[0]])
    if n != len(grouped[METHODS[1]]):
        raise ValueError("methods_must_use_the_exact_same_verified_fixtures")
    if frozen != n + pending + invalid:
        raise ValueError("missing_or_double_counted_original_v37_fixtures")
    return {
        "version": VERSION,
        "evaluated_at_utc":clock.isoformat(),
        "frozen_unique_physical_fixtures":frozen,
        "verified_matched_physical_fixtures":n,
        "pending":pending,
        "invalid_or_tampered_excluded":invalid,
        "independent_final_conflicts_excluded":len(conflicts),
        "invalid_verified_result_rows_excluded":dict(rejected),
        "methods":method_reports,
        "minimum_review_verified_fixtures":MIN_REVIEW_FIXTURES,
        "ready_for_model_promotion":False,
        "not_a_five_double_trial":True,
        "no_backdated_scores":True,
        "v8_unchanged":True,
        "no_live_model_change":True,
        "no_provider_calls":True,
    }


def run(out=OUT, **kwargs):
    result = evaluate(**kwargs)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if not out.exists() or out.read_text(encoding="utf-8") != payload:
        out.write_text(payload,encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "frozen":report["frozen_unique_physical_fixtures"],
        "verified":report["verified_matched_physical_fixtures"],
        "pending":report["pending"],
        "invalid_excluded":report["invalid_or_tampered_excluded"],
        "methods":list(report["methods"]),
        "promoted":False,
    },sort_keys=True))
