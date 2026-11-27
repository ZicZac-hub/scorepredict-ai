"""Research-only historical xG age challenger.

Past-match xG and live team information have different lifetimes. Compare
3h versus 24h packet age for historical xG only, at a fixed pre-match cutoff.
Keep exact source hashes, minimum sample/venue rules and completed-match gate.
No provider calls, prospective tickets, archive edits or model promotion.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import numpy as np

from v10_prospective import utc
import v42_research_trial as sources
import v44_fixture_evidence as inventory
from v44_late_evidence_audit import _candidate_models, LEAD
from v22_evidence_models import xg_profile, combine_goal_evidence
from v41_unified_ensemble import core
from v44_single_calibration import _metrics

STRICT_AGE = timedelta(hours=3)
HISTORICAL_AGE = timedelta(hours=24)


def _profile(packet, model, cutoff):
    captured = utc(packet["captured_at_utc"])
    rows = packet.get("verified_historical_fixture_xg") or []
    if not rows:
        return {"ready":False,"reason":"no_genuine_xg_rows"}
    seen = set()
    for row in rows:
        identity = str(row["fixture_id"])
        if (identity in seen or identity == str(model["fixture_id"])
                or row.get("type") != "api_football_reported_expected_goals"
                or utc(row["kickoff_utc"])+timedelta(hours=3)>captured
                or str(row["home_team_id"]) == str(row["away_team_id"])
                or any(type(row[k]) not in (int,float) or not math.isfinite(row[k])
                       or not 0<=row[k]<=8 for k in ("home_xg","away_xg"))):
            return {"ready":False,"reason":"unqualified_historical_xg_rows"}
        seen.add(identity)
    return xg_profile(sources._fixture(model), packet, as_of=cutoff, minimum=6)


def _find(models, provider_root, cutoff):
    root = Path(provider_root)
    counts, seen, strict, extended = Counter(), set(), [], []
    for model in models:
        base = sources.source_packet_path(root, model)
        for path in [base, *sorted((root/"versions").glob(base.stem+"_*.json"))]:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            counts["packet_files"] += 1
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not sources.qualified_packet(record,model,now=cutoff):
                    counts["identity_hash_or_cutoff_rejected"] += 1
                    continue
                packet = record["provider_snapshot"]
                captured = utc(packet["captured_at_utc"])
                age = cutoff-captured
                if not timedelta(0)<=age<=HISTORICAL_AGE:
                    counts["outside_historical_age_window"] += 1
                    continue
                limits = packet.get("source_limits") or {}
                cap = limits.get("max_historical_fixtures_per_team")
                if type(cap) is int and cap<6:
                    counts["capture_cap_below_six"] += 1
                result = _profile(packet,model,cutoff)
                if not result.get("ready"):
                    counts[result.get("reason","xg_not_ready")] += 1
                    continue
                ref = {"path":path.relative_to(root).as_posix(),
                       "snapshot_sha256":packet["snapshot_sha256"],
                       "captured_at_utc":packet["captured_at_utc"],
                       "source_forecast_id":model["forecast_id"]}
                item = ((captured,ref["snapshot_sha256"],ref["source_forecast_id"]),ref,result)
                extended.append(item)
                if age<=STRICT_AGE:
                    strict.append(item)
            except (ValueError,TypeError,KeyError,OSError,AttributeError,ArithmeticError):
                counts["packet_validation_error"] += 1
    def latest(items):
        return max(items,key=lambda i:i[0]) if items else None
    return latest(strict),latest(extended),counts


def evaluate(*, source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,
             provider_root=sources.PROVIDER_ROOT,now=None):
    clock = utc(now or datetime.now(timezone.utc))
    audit = inventory.evaluate(source_root=source_root,prematch_root=prematch_root,
                               provider_root=provider_root,now=clock)
    counts, diagnostics, rows = Counter(),Counter(),[]
    for row in audit["rows"]:
        cutoff = utc(row["kickoff_utc"])-LEAD
        models, rejected = _candidate_models(row,source_root,cutoff)
        diagnostics["invalid_forecast_candidates"] += rejected
        if not models:
            counts["no_cutoff_source"] += 1
            continue
        strict, extended, reasons = _find(models,provider_root,cutoff)
        diagnostics.update(reasons)
        counts["eligible_fixtures"] += 1
        counts["strict_ready"] += int(strict is not None)
        counts["extended_ready"] += int(extended is not None)
        counts["extended_only"] += int(extended is not None and strict is None)
        if not extended:
            continue
        # Fixed latest eligible baseline independent of xG selection and result.
        model = max(models,key=lambda m:(utc(m["captured_at_utc"]),m["forecast_id"]))
        base = core(model["score_matrix"])
        changed = combine_goal_evidence(base,extended[2],weight=.25)
        if not changed.get("ready"):
            counts["transform_abstained"] += 1
            continue
        actual = row["actual"]
        rows.append({"fixture_id":row["fixture_id"],"provider":row["provider"],
                     "cutoff_utc":cutoff.isoformat(),"baseline_forecast_id":model["forecast_id"],
                     "strict_reference":strict[1] if strict else None,
                     "historical_reference":extended[1],
                     "baseline":_metrics(base,actual),
                     "historical_xg":_metrics(np.asarray(changed["matrix"]),actual)})
    n=len(rows)
    summary={k:{"fixtures":n,"mean_exact_log_loss":
                 math.fsum(r[k]["exact_log_loss"] for r in rows)/n if n else None,
                 "exact_score_hit_rate":math.fsum(r[k]["top_score_hit"] for r in rows)/n if n else None}
             for k in ("baseline","historical_xg")}
    return {"study_type":"historical_xg_age_challenger_replay",
            "evaluated_at_utc":clock.isoformat(),"inventory_matches":audit["matched_verified_fixtures"],
            "counts":dict(counts),"packet_diagnostics":dict(diagnostics),
            "paired_metrics":summary,"rows":rows,"external_requests":0,
            "automatic_promotion":False,"forecasts_created":0,
            "limitations":["24h applies only to historical xG, not odds or lineups.",
                           "No missing history is invented or combined across packets.",
                           "Conditional matched subset is not all-fixture accuracy.",
                           "Retrospective comparison cannot establish prospective improvement."]}


if __name__ == "__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
