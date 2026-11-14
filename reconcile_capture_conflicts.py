"""Safely reconcile generated-report conflicts in the capture workflow.

This never overwrites immutable forecast records. Concurrent jobs may
freeze different predictions for the same Sydney date; after validating
both candidates as pre-kickoff, the FIRST already published date wins.
On rebase, stage 2 is newer upstream and stage 3 the capture job.
Generated reports prefer upstream; verified final scores are unioned
only after matching exact source identities.
"""
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import subprocess
import sys


REPORT_PREFIXES = (
    "data/prematch/rescores/",
    "data/prematch/double_rescores/",
    "data/prospective_ledger/discovery/",
)
REPORT_EXACT = frozenset((
    "data/prematch/accuracy_report.json",
    "data/prematch/prospective_comparison.json",
    "data/prematch/prospective_double_comparison.json",
    "data/prematch/five_model_comparison.json",
    "data/prematch/smart_double_comparison.json",
    "data/prematch/legacy_results.json",
    "data/prospective_ledger/comparison.json",
    "data/prospective_ledger/capture_coverage.json",
    "data/prospective_ledger/cadence.json",
    "data/v10_prospective/performance.json",
    "data/v10_prospective/settlement_status.json",
    "data/llm_research/status.json",
    "data/llm_research/evaluation.json",
    "data/independent_football/status.json",
    "data/independent_football/evaluation.json",
))
VERIFIED = "data/prematch/verified_results.json"
V10_VERIFIED = "data/v10_prospective/verified_results.json"
VERIFIED_FILES = frozenset((VERIFIED,V10_VERIFIED))
FREEZE_PREFIXES = ("data/daily_picks/","data/v10_prospective/doubles/")
RESULT_FIELDS = ("provider","fixture_id","kickoff_utc","home_team_id",
                 "away_team_id","score_basis","home_goals","away_goals",
                 "evidence_version")


def merge_verified(upstream, local):
    """Union genuinely verified results; never silently change final scores."""
    if not isinstance(upstream,dict) or not isinstance(local,dict):
        raise ValueError("Verified-result inputs must be fixture dictionaries")
    merged=dict(upstream)
    for fixture_id,row in local.items():
        if fixture_id not in merged:
            if not isinstance(row,dict) or row.get("evidence_version") != 2:
                raise ValueError("Cannot merge an unverified-result candidate")
            merged[fixture_id]=row
        elif any(merged[fixture_id].get(field)!=row.get(field)
                 for field in RESULT_FIELDS):
            raise ValueError("Conflicting final-score identity for "+str(fixture_id))
        # Prefer the already published final-score proof and original
        # verification timestamp over rewriting it on each run.
    return merged


def _timestamp(value):
    date=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if date.tzinfo is None:
        raise ValueError("Original freeze must be timezone-aware")
    return date.astimezone(timezone.utc)


def valid_freeze(path,row):
    """Validate published and pending picks before keeping first publication."""
    if not isinstance(row,dict) or not isinstance(path,str):
        return False
    try:
        if path.startswith("data/daily_picks/"):
            day=Path(path).stem
            original=_timestamp(row["captured_at_utc"])
            kickoff=_timestamp(row["kickoff_utc"])
            return bool(row.get("prediction_date")==day
                        and row.get("status")=="frozen_in_capture_workflow"
                        and original<kickoff
                        and kickoff.astimezone(ZoneInfo("Australia/Sydney")).date().isoformat()==day
                        and isinstance(row.get("exact_score"),list)
                        and len(row["exact_score"])==2
                        and all(type(n) is int and n>=0 for n in row["exact_score"]))
        if path.startswith("data/v10_prospective/doubles/"):
            day=Path(path).stem
            original=_timestamp(row["captured_at_utc"])
            first=_timestamp(row["first"]["kickoff_utc"])
            second=_timestamp(row["second"]["kickoff_utc"])
            combinations=row.get("five_doubles")
            if not isinstance(combinations,list) or len(combinations)!=5:
                return False
            pairs={(x["score1"],x["score2"]) for x in combinations}
            return bool(row.get("sydney_date")==day
                        and row.get("status")=="frozen_pre_kickoff"
                        and original<min(first,second)
                        and first.astimezone(ZoneInfo("Australia/Sydney")).date().isoformat()==day
                        and second.astimezone(ZoneInfo("Australia/Sydney")).date().isoformat()==day
                        and len(pairs)==5
                        and str(row["first"]["fixture_id"])!=str(row["second"]["fixture_id"]))
    except (ValueError,TypeError,KeyError,AttributeError):
        return False
    return False


def _git(*args):
    result=subprocess.run(["git",*args],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError("Git reconciliation failed for "+str(args[0])+
                           ": "+result.stderr[:500])
    return result.stdout


def reconcile():
    raw=_git("diff","--name-only","--diff-filter=U")
    paths=[x.strip() for x in raw.splitlines() if x.strip()]
    if not paths:
        raise RuntimeError("No eligible generated-file conflicts to reconcile")
    # Inspect all paths before changing ANY of them. Original files and
    # score snapshots are NEVER automatically discarded or overwritten.
    disallowed=[x for x in paths
                if x not in VERIFIED_FILES and x not in REPORT_EXACT
                and not x.startswith(REPORT_PREFIXES)
                and not x.startswith(FREEZE_PREFIXES)]
    if disallowed:
        raise RuntimeError("Immutable forecast conflicts need manual review: "+
                           ", ".join(disallowed[:8]))
    # Validate every conflicting freeze before mutating any merge contents.
    for path in paths:
        if path.startswith(FREEZE_PREFIXES):
            upstream=json.loads(_git("show",":2:"+path))
            local=json.loads(_git("show",":3:"+path))
            if not valid_freeze(path,upstream) or not valid_freeze(path,local):
                raise RuntimeError("Unverified pre-match frozen pick conflict: "+path)
    for path in paths:
        if path in VERIFIED_FILES:
            upstream=json.loads(_git("show",":2:"+path))
            local=json.loads(_git("show",":3:"+path))
            merged=merge_verified(upstream,local)
            Path(path).parent.mkdir(parents=True,exist_ok=True)
            Path(path).write_text(json.dumps(merged,indent=2,sort_keys=True)+"\n",
                                  encoding="utf-8")
        else:
            # REBASE semantics: --ours means first already published on main.
            # This preserves the ORIGINAL frozen score on date collisions.
            _git("checkout","--ours","--",path)
        _git("add","--",path)
    print("Reconciled derived reports without touching immutable originals:",len(paths))


if __name__=="__main__":
    try:
        reconcile()
    except (RuntimeError,ValueError,TypeError,KeyError) as exc:
        print("SAFETY STOP:",str(exc),file=sys.stderr)
        sys.exit(1)
