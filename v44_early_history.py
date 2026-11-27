"""Content-addressed early league-history receipts, independent of xG capture.

Freeze only actual in-memory, as-of-qualified histories during the existing
pre-match capture. Never reconstruct a missing historic receipt after kickoff.
No API calls or change to forecast selection. One receipt per fixture event.
"""
from datetime import datetime,timezone
import hashlib,json,os,tempfile
from pathlib import Path

from history_pipeline import build_evidence
from v10_prospective import utc

ROOT=Path("data/v44_early_history")
VERSION="early-history-v1"


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _once(path,obj):
    """Atomic write-once publication; never replace a conflicting receipt."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=".pending-",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as out:
            json.dump(obj,out,sort_keys=True,separators=(",",":"),allow_nan=False)
            out.write("\n")
        try:
            os.link(temp,path)
            return True
        except FileExistsError:
            return False
    finally:
        os.unlink(temp)


def _fixture(f):
    return {k:f[k] for k in ("provider","id","utcDate","homeTeam","awayTeam","competition")}


def _event(f):
    return [str(f["provider"]),str(f["id"]),utc(f["utcDate"]).isoformat(),
            str(f["homeTeam"]["id"]),str(f["awayTeam"]["id"])]


def _minimal(row):
    # Preserve all fields used by history_pipeline's canonical input digest.
    compact={k:row[k] for k in ("provider","id","utcDate","status","homeTeam","awayTeam","competition","score")}
    if row.get("finishedAt") is not None:
        compact["finishedAt"]=row["finishedAt"]
    if row.get("season") is not None:
        compact["season"]=row["season"]
    if row.get("score_basis") is not None:
        compact["score_basis"]=row["score_basis"]
    compact["homeTeam"]={"id":row["homeTeam"]["id"]}
    compact["awayTeam"]={"id":row["awayTeam"]["id"]}
    compact["score"]={"fullTime":row["score"]["fullTime"],
                      "duration":row["score"].get("duration")}
    return compact


def freeze(fixture,evidence,*,root=ROOT,clock=None):
    clock=clock or (lambda:datetime.now(timezone.utc))
    root=Path(root)
    target=_fixture(fixture)
    start=utc(clock())
    cutoff=utc(evidence["as_of_utc"])
    kickoff=utc(target["utcDate"])
    if not cutoff<=start<kickoff:
        return "outside_pre_match_window"
    path=root/"receipts"/(digest(_event(target))+".json")
    if path.exists():
        load(path,root=root,now=start)
        return "original_preserved"
    original=evidence["history"]
    if not isinstance(original,list) or not original:
        return "no_history"
    compact=[_minimal(row) for row in original]
    rebuilt=build_evidence(target,compact,cutoff)
    if (rebuilt["input_digest"]!=evidence["input_digest"]
            or len(rebuilt["history"])!=len(original)):
        raise ValueError("history_input_digest_mismatch")
    # Blob hash is independent of target and can be reused across fixtures.
    blob_sha=digest(compact)
    blob=root/"blobs"/(blob_sha+".json")
    if not _once(blob,compact) and digest(_read(blob))!=blob_sha:
        raise ValueError("immutable_history_blob_conflict")
    observed=utc(clock())
    if not start<=observed<kickoff:
        return "capture_finished_outside_window"
    receipt={"version":VERSION,"fixture":target,"event":_event(target),
             "as_of_utc":cutoff.isoformat(),"captured_at_utc":observed.isoformat(),
             "history_sha256":blob_sha,"history_input_digest":evidence["input_digest"],
             "history_count":len(compact)}
    receipt["receipt_sha256"]=digest(receipt)
    if _once(path,receipt):
        return "written"
    load(path,root=root,now=observed)
    return "original_preserved"


def load(path,*,root=ROOT,now=None):
    root=Path(root)
    path=Path(path)
    record=_read(path)
    if (record["version"]!=VERSION or record["receipt_sha256"]!=
            digest({k:v for k,v in record.items() if k!="receipt_sha256"})
            or record["event"]!=_event(record["fixture"])
            or path.stem!=digest(record["event"])):
        raise ValueError("invalid_receipt")
    cutoff,captured,kickoff=map(utc,(record["as_of_utc"],record["captured_at_utc"],record["fixture"]["utcDate"]))
    if not cutoff<=captured<kickoff or captured>utc(now or datetime.now(timezone.utc)):
        raise ValueError("invalid_capture_time")
    sha=record["history_sha256"]
    if not isinstance(sha,str) or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid_history_hash")
    history=_read(root/"blobs"/(sha+".json"))
    if digest(history)!=sha or len(history)!=record["history_count"]:
        raise ValueError("invalid_history_blob")
    rebuilt=build_evidence(record["fixture"],history,cutoff)
    if (rebuilt["input_digest"]!=record["history_input_digest"]
            or len(rebuilt["history"])!=len(history)):
        raise ValueError("invalid_history_digest")
    return {**record,"history":history}
