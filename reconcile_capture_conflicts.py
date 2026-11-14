"""Safely reconcile generated-report conflicts in the capture workflow.

This intentionally does NOT auto-resolve conflicts involving immutable
original forecasts, saved five-double selections or daily research picks.
On rebase, git stage 2 is the latest upstream state and stage 3 is the
capture job's generated state. An existing newer upstream report wins over
a stale generated report; verified final-score dictionaries are merged only
when both observations agree on the exact fixture and result identity.
"""
import json
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
))
VERIFIED = "data/prematch/verified_results.json"
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
                if x!=VERIFIED and x not in REPORT_EXACT
                and not x.startswith(REPORT_PREFIXES)]
    if disallowed:
        raise RuntimeError("Immutable forecast conflicts need manual review: "+
                           ", ".join(disallowed[:8]))
    for path in paths:
        if path==VERIFIED:
            upstream=json.loads(_git("show",":2:"+path))
            local=json.loads(_git("show",":3:"+path))
            merged=merge_verified(upstream,local)
            Path(path).parent.mkdir(parents=True,exist_ok=True)
            Path(path).write_text(json.dumps(merged,indent=2,sort_keys=True)+"\n",
                                  encoding="utf-8")
        else:
            # REBASE semantics: --ours means the newer upstream main state.
            _git("checkout","--ours","--",path)
        _git("add","--",path)
    print("Reconciled derived reports without touching immutable originals:",len(paths))


if __name__=="__main__":
    try:
        reconcile()
    except (RuntimeError,ValueError,TypeError,KeyError) as exc:
        print("SAFETY STOP:",str(exc),file=sys.stderr)
        sys.exit(1)
