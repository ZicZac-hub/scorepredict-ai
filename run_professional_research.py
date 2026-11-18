"""Reproduce date-safe research from pinned CC0 data; never deploy a model."""
import argparse
import hashlib
import json
from pathlib import Path

from target70_data import collect, load_normalized
from target70_benchmark import run_validation, run_holdout
from source_quality_audit import audit, run_sensitivity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=Path("data/target70_research/source_manifest.json"))
    parser.add_argument("--cache-dir",type=Path,default=Path("work/research_cache"))
    parser.add_argument("--output",type=Path,default=Path("work/professional_research"))
    parser.add_argument("--include-gradient-boost",action="store_true")
    parser.add_argument("--reproduce-original",action="store_true",
                        help="Also reproduce the preserved, source-date-affected original control.")
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Use a new empty output directory to preserve previous experiments.")
    frozen=json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True,exist_ok=True)
    collect(cache_dir=args.cache_dir,revision=frozen["source_revision"],
            manifest_path=args.output/"download_manifest.json")
    # The original manifest defines the accepted files and hashes, while the
    # new download manifest records this reproduction's actual fetch times.
    rows=load_normalized(args.manifest,args.cache_dir)
    manifest_sha=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    quality=audit(args.manifest,args.cache_dir)
    (args.output/"source_quality_audit.json").write_text(json.dumps(quality,indent=2,sort_keys=True)+"\n")
    original=None
    if args.reproduce_original:
        experiment=args.output/"frozen_protocol_reproduction"
        run_validation(rows,experiment,source_manifest_sha256=manifest_sha)
        original=run_holdout(rows,experiment,source_manifest_sha256=manifest_sha)
    development=run_sensitivity(args.manifest,args.cache_dir,
                        args.output/"source_quality_screened_development",
                        include_gradient_boost=args.include_gradient_boost)
    summary={"reproduction_not_new_unseen_evidence":True,
             "original_selected_model":original["selected_model"] if original else None,
             "original_policy_results":original["chosen_model_threshold_results"] if original else None,
             "development_model_results":development["development_five_double_thresholds"],
             "source_revision":frozen["source_revision"],"source_matches":len(rows),
             "quarantined_source_files":quality["quarantined_files"],
             "live_model_promoted":False,"external_paid_provider_requests":0,
             "target_70_percent_proven":False}
    (args.output/"reproduction_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
