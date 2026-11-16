"""Read only compatible settled forecast benchmark records from local archive.

Never infer an outcome from a prediction or rewrite original archives.
Unrecognized historical formats are counted as skipped, not silently converted.
"""
import json
from pathlib import Path

ROOTS=("data/v10_prospective","data/prematch")
MAX_FILES=500
MAX_BYTES=4_000_000

def load_archive(root=".", limit=MAX_FILES):
    base=Path(root).resolve()
    rows=[]
    diagnostics={"files_checked":0,"compatible_records":0,"skipped_records":0,
                 "unreadable_files":0,"truncated":False}
    paths=[]
    for folder in ROOTS:
        location=(base/folder)
        if location.is_dir():
            paths.extend(location.rglob("*.json"))
            paths.extend(location.rglob("*.jsonl"))
    for path in sorted(set(paths)):
        if diagnostics["files_checked"]>=min(max(0,int(limit)),MAX_FILES):
            diagnostics["truncated"]=True
            break
        diagnostics["files_checked"]+=1
        try:
            if path.stat().st_size>MAX_BYTES:
                diagnostics["unreadable_files"]+=1
                continue
            if path.suffix==".jsonl":
                with path.open(encoding="utf-8") as handle:
                    payload=[json.loads(line) for line in handle if line.strip()]
            else:
                payload=json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload,dict):
                payload=payload.get("records",payload.get("fixtures",[payload]))
            if not isinstance(payload,list):
                diagnostics["skipped_records"]+=1
                continue
            for item in payload:
                if (isinstance(item,dict) and
                    all(k in item for k in ("fixture_id","kickoff","actual","forecasts"))):
                    rows.append(item)
                    diagnostics["compatible_records"]+=1
                else:
                    diagnostics["skipped_records"]+=1
        except (OSError,ValueError,TypeError,UnicodeError):
            diagnostics["unreadable_files"]+=1
    return rows,diagnostics
