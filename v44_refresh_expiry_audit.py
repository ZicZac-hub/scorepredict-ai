"""Audit expiration of one-shot near-kickoff captures before research windows.

Read-only; no provider calls, predictions, changes to frozen records, or
new pair selection. Identifies structural collection gaps rather than accuracy.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
from v10_prospective import utc
from forecast_ledger import _forecast,LedgerValidationError
import v44_early_history as early
from v44_early_history_audit import qualifies
import v42_research_trial as sources

MAX_AGE=timedelta(hours=24)

def event(m):
    return (str(m["provider"]),str(m["fixture_id"]),utc(m["kickoff_utc"]).isoformat(),
            str(m["home_team_id"]),str(m["away_team_id"]))

def inspect_leg(original,opening,now,source_root,history_root):
    e=event(original)
    near=False
    exact=[];usable=[]
    for path in sorted((Path(source_root)/"forecasts"/e[0]/e[1]).glob("*.json")):
        try:
            raw=json.loads(path.read_text())
            if raw.get("model_key") not in ("v8","v8_near") or not raw.get("forecast_id"):
                continue
            m=_forecast(raw)
            at=utc(m["captured_at_utc"])
            if m["forecast_id"]!=path.stem or event(m)!=e or at>now or m.get("score_basis")!="regulation":
                continue
            near|=m["model_key"]=="v8_near"
            for key in (early.receipt_key(list(e),m["input_digest"]),early.digest(list(e))):
                p=Path(history_root)/"receipts"/(key+".json")
                if not p.is_file():
                    continue
                try:
                    r=early.load(p,root=history_root,now=now)
                    if qualifies(r,m,now):
                        exact.append(at)
                        if timedelta(0)<=opening-at<=MAX_AGE:
                            usable.append(at)
                except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
                    continue
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError,LedgerValidationError):
            continue
    if usable:
        return "exact_history_valid_at_window_open"
    if near:
        return "one_shot_near_expired" if exact else "one_shot_near_without_exact_history"
    return "refresh_available_but_history_expired" if exact else "refresh_available_missing_history"

def evaluate(*,source_root=sources.SOURCE_ROOT,history_root=early.ROOT,pair_root=sources.PAIR_ROOT,now=None):
    now=utc(now or datetime.now(timezone.utc))
    counts=Counter();legs=Counter();rows=[]
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            pair,models=sources._pair(path,source_root=source_root,decision=now)
            first=min(utc(m["kickoff_utc"]) for m in models)
            opens=max(now,first-timedelta(hours=3),utc(pair["predicted_at_utc"]))
            if first-timedelta(minutes=5)<=opens:
                counts["closed_or_ineligible"]+=1;continue
            statuses=[inspect_leg(m,opens,now,source_root,history_root) for m in models]
            legs.update(statuses)
            counts["upcoming_pairs"]+=1
            counts["pairs_with_"+str(statuses.count("exact_history_valid_at_window_open"))+"_usable_legs"]+=1
            refresh_opens=first-timedelta(hours=24)
            refresh_closes=first-timedelta(minutes=45)
            phase=("not_yet_open" if now<refresh_opens else
                   "closed" if now>=refresh_closes else "open")
            for m,status in zip(models,statuses):
                if status.startswith("refresh_available"):
                    counts["existing_refresh_path_"+phase]+=1
            rows.append({"day":path.stem,"audit_window_utc":opens.isoformat(),
                         "priority_refresh_opens_utc":refresh_opens.isoformat(),
                         "priority_refresh_closes_utc":refresh_closes.isoformat(),
                         "priority_refresh_window_status":phase,"leg_statuses":statuses,
                         "fixtures":[{"provider":m["provider"],"fixture_id":str(m["fixture_id"]),
                                      "kickoff_utc":utc(m["kickoff_utc"]).isoformat(),
                                      "status":status} for m,status in zip(models,statuses)]})
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            counts["invalid_pair"]+=1
    return {"study_type":"one_shot_refresh_expiry_audit","counts":dict(counts),
            "leg_statuses":dict(legs),"rows":rows,"external_requests":0,
            "automatic_promotion":False,
            "limitations":["Readiness is conditional on records currently available; new captures may change it.",
                           "A history receipt does not establish model readiness or predictive accuracy."]}

if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
