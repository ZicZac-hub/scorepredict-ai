"""V42 watchdog: refresh research reports, not production code or past forecasts.
Run --watch on a continuously running host with refreshed provider archives.
One optional bounded AI advisory attempt per UTC day; no code execution.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
import requests
from v42_research_trial import main, ROOT, PAIR_ROOT, SOURCE_ROOT, _pair, utc, PROTOCOL
from persist_prematch import write_json

def advisory(report, root=ROOT):
    if os.getenv("V42_AI_RESEARCH_ENABLED","").lower() not in ("1","true","yes"):
        return "disabled"
    key=os.getenv("OPENAI_API_KEY","")
    if not key:
        return "missing_key"
    day=datetime.now(timezone.utc).date().isoformat()
    receipt=Path(root)/("ai_attempt_"+day+".json")
    receipt.parent.mkdir(parents=True,exist_ok=True)
    try:
        with receipt.open("x",encoding="utf-8") as f:
            json.dump({"status":"reserved","day":day},f)
    except FileExistsError:
        return "daily_attempt_already_reserved"
    try:
        evaluation=report["evaluation"]
        snapshot={k:evaluation[k] for k in ("methods","same_cutoff_v41_comparison","lineup_coverage","source_rejections","result_rejections")}
        response=requests.post("https://api.openai.com/v1/responses",
            headers={"Authorization":"Bearer "+key},timeout=25,
            json={"model":"gpt-4o-mini","store":False,"max_output_tokens":350,
                  "input":[{"role":"system","content":"Review this football probability research audit. Identify one data-quality problem and one falsifiable future-only experiment. No executable code, betting advice, secrets, or accuracy guarantees."},
                           {"role":"user","content":json.dumps(snapshot)}]})
        response.raise_for_status()
        memo=" ".join(c.get("text","") for item in response.json().get("output",[]) for c in item.get("content",[]) if c.get("type")=="output_text").strip()
        if not memo:
            raise ValueError("empty_memo")
        write_json(Path(root)/("ai_memo_"+day+".json"),{"memo":memo[:1800],"advisory_only":True})
        return "saved"
    except requests.HTTPError as exc:
        code = getattr(exc.response, "status_code", None)
        return "failed_http_" + str(code) if type(code) is int and 400 <= code <= 599 else "failed_HTTPError"
    except (requests.RequestException,ValueError,TypeError,KeyError) as exc:
        return "failed_"+type(exc).__name__

def evidence_coverage(performance):
    """Summarize validated trial rows without changing the frozen protocol."""
    rows = performance["rows"]
    if len(rows) != performance["valid_opportunities"]:
        raise ValueError("inconsistent_evidence_row_count")
    result = {"valid_pairs": len(rows)}
    for label, key in (("xg", "paired_xg_used"), ("market", "paired_market_used")):
        values = [row.get("unified_final_evidence", {}).get(key) for row in rows]
        result[label] = {"used": sum(v is True for v in values),
                         "not_used": sum(v is False for v in values),
                         "unknown": sum(type(v) is not bool for v in values)}
    result["confirmed_lineups_both_matches"] = performance["lineup_coverage"]["both_matches_confirmed_at_decision"]
    return result

def window_diagnostics(*, now, root=ROOT, pair_root=PAIR_ROOT, source_root=SOURCE_ROOT):
    """Read-only schedule audit; does not change registered forecast rules."""
    now = utc(now)
    counts = {"already_frozen": 0, "too_early": 0, "in_window": 0,
              "too_late": 0, "invalid": 0}
    next_window = None
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            record, models = _pair(path, source_root=source_root, decision=now)
            if (Path(root)/"trials"/(record["sydney_date"]+".json")).exists():
                counts["already_frozen"] += 1
                continue
            kickoff = min(utc(m["kickoff_utc"]) for m in models)
            lead = (kickoff-now).total_seconds()
            if lead > PROTOCOL["maximum_lead_seconds_inclusive"]:
                counts["too_early"] += 1
                opens = kickoff-timedelta(seconds=PROTOCOL["maximum_lead_seconds_inclusive"])
                if next_window is None or opens < next_window:
                    next_window = opens
            elif lead <= PROTOCOL["minimum_lead_seconds_exclusive"]:
                counts["too_late"] += 1
            else:
                counts["in_window"] += 1
        except (ValueError, KeyError, TypeError, OSError, AttributeError, OverflowError):
            counts["invalid"] += 1
    return {**counts, "as_of_utc": now.isoformat(),
            "next_window_opens_utc": next_window.isoformat() if next_window else None}

def tick(*, use_ai=True):
    clock=datetime.now(timezone.utc)
    try:
        report=main(now=None)
        e=report["evaluation"]
        performance=json.loads((ROOT/"performance.json").read_text(encoding="utf-8"))
        if performance["evaluated_at_utc"] != e["evaluated_at_utc"]:
            raise ValueError("performance_snapshot_mismatch")
        coverage=evidence_coverage(performance)
        status={"checked_at_utc":clock.isoformat(),"status":"ok",
                "verified_pairs":e["verified_opportunities"],"pending_pairs":e["pending_opportunities"],
                "capture_rejections":report["capture"]["rejections"],
                "outside_decision_window":report["capture"]["outside_decision_window"],
                "paired_data":e["lineup_coverage"],"evidence_coverage":coverage,
                "decision_windows":window_diagnostics(now=clock),
                "ai_mode":"configured_advisory" if use_ai else "no_ai_smoke",
                "ai":advisory(report) if use_ai else "disabled",
                "auto_promotes":False,"auto_edits_code":False}
    except Exception as exc:
        status={"checked_at_utc":clock.isoformat(),"status":"failed","error_class":type(exc).__name__,
                "auto_promotes":False,"auto_edits_code":False}
    write_json(ROOT/"watchdog.json",status)
    print(json.dumps(status),flush=True)
    return status

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--no-ai",action="store_true")
    args=parser.parse_args()
    while True:
        start=time.monotonic()
        result=tick(use_ai=not args.no_ai)
        if not args.watch:
            raise SystemExit(0 if result["status"]=="ok" else 1)
        time.sleep(max(0,60-(time.monotonic()-start)))
