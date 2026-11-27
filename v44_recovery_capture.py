"""Isolated recovery capture for already-frozen near forecasts missing history.

Consumes an existing collector's in-memory evidence. No API calls, new pair
selection, changes to V8/V8-near, or changes to a registered study. An entirely
separate archive is intentionally invisible to the live prediction readers.
"""
from datetime import datetime,timedelta,timezone
from pathlib import Path

from forecast_ledger import append_file_record,_forecast
from source_capture_clock import capture_clock
from v10_prospective import utc,read_latest,research_predict,_record
import v44_early_history as early
from v44_early_history_audit import qualifies
from v44_refresh_priority import fixture_key,model_key,targets
import v42_research_trial as sources

ROOT=Path("data/v44_recovery_sources")
MODEL_KEY="v44_recovery_control"
VERSION="v44-recovery-control-v1"


def capture(fixture,evidence,*,root=ROOT,source_root=sources.SOURCE_ROOT,
            pair_root=sources.PAIR_ROOT,history_root=early.ROOT,now=None,clock=None):
    started,observed=capture_clock(now,clock)
    root=Path(root)
    key=fixture_key(fixture)
    cutoff=utc(evidence["as_of_utc"])
    # This is only for genuinely later collection within the existing
    # original-pair refresh window. An old receipt is never reconstructed.
    if not timedelta(minutes=45)<key[2]-started<=timedelta(hours=48):
        return {"status":"outside_refresh_window"}
    required=targets(now=started,pair_root=pair_root,source_root=source_root)
    if key not in required:
        return {"status":"not_authenticated_original_pair_leg"}
    earliest=required[key]
    if not timedelta(minutes=45)<earliest-started<=timedelta(hours=24):
        return {"status":"outside_common_refresh_window"}
    original=read_latest(source_root).get((*key[:2],"v8_near"))
    if original is None or model_key(original)!=key:
        return {"status":"existing_near_capture_not_found"}
    if not utc(original["captured_at_utc"])<cutoff<=started:
        return {"status":"history_not_genuinely_newer"}
    # Require a receipt for the existing near forecast only if it is
    # available and remains usable through the first research decision.
    original_event=[str(original["provider"]),str(original["fixture_id"]),
                    utc(original["kickoff_utc"]).isoformat(),
                    str(original["home_team_id"]),str(original["away_team_id"])]
    opens=max(started,earliest-timedelta(hours=3))
    for rid in (early.receipt_key(original_event,original["input_digest"]),early.digest(original_event)):
        path=Path(history_root)/"receipts"/(rid+".json")
        if not path.is_file():
            continue
        try:
            receipt=early.load(path,root=history_root,now=started)
            if (qualifies(receipt,original,started)
                    and timedelta(0)<=opens-utc(original["captured_at_utc"])<=timedelta(hours=24)):
                return {"status":"existing_exact_history_usable"}
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            continue
    # At most one recovery forecast per fixture event, to prevent repeated
    # collection or changes to earlier records. The original live archive
    # is never written by this module.
    destination=root/"archive"
    existing=read_latest(destination).get((*key[:2],MODEL_KEY))
    if existing is not None:
        if model_key(existing)!=key:
            return {"status":"existing_recovery_identity_mismatch"}
        return {"status":"recovery_preserved"}
    history=evidence.get("history") or []
    league=evidence.get("league_history") or history
    if not history or not evidence.get("input_digest"):
        return {"status":"missing_verified_history"}
    # Persist the exact supplied evidence before calculating a forecast.
    outcome=early.freeze(fixture,evidence,root=root/"history",clock=observed)
    if outcome not in ("written","original_preserved"):
        return {"status":"receipt_"+outcome}
    forecast=research_predict(fixture,history,league_history=league,as_of=cutoff)
    if forecast is None or len(forecast.get("component_weights") or {})<2:
        return {"status":"insufficient_verified_model_history"}
    completed=observed()
    if not started<=completed<earliest-timedelta(minutes=5):
        return {"status":"calculation_finished_outside_common_window"}
    row=_record(fixture,evidence,{**forecast,"model_version":VERSION},MODEL_KEY,completed)
    row.update({"horizon":"isolated_recovery","numerical_model_version":forecast["model_version"],
                "previous_near_forecast_id":original["forecast_id"],
                "release_status":"shadow_validation_only"})
    model=_forecast(row)
    rid=early.receipt_key(original_event,evidence["input_digest"])
    receipt=early.load(root/"history"/"receipts"/(rid+".json"),root=root/"history",now=completed)
    if not qualifies(receipt,model,completed):
        raise ValueError("recovery_receipt_mismatch")
    row["recovery_receipt_sha256"]=receipt["receipt_sha256"]
    written=append_file_record(row,root=destination,kind="forecasts")
    return {"status":"recovery_written" if written else "recovery_preserved",
            "forecast_id":model["forecast_id"],"receipt_sha256":receipt["receipt_sha256"],
            "external_requests":0,"automatic_promotion":False}
