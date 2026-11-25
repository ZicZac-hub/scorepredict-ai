"""Write-once cross-competition research receipts; no provider calls.

The caller supplies the already observed league history cache from the real
pre-match collection. Only source-qualified V37-sparse European controls
are eligible. Raw history is content-addressed and stored separately so
a future grader can reproduce the candidate without fetching newer data.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from cross_competition_strength import research, DOMESTIC, VERSION
from history_pipeline import build_evidence, _date, season_year
from v37_sparse_history_research import predict

ROOT = Path("data/cross_competition_research")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))+"\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        return True
    except FileExistsError:
        return False


def freeze(fixture, evidence, cache, *, root=ROOT, clock=None):
    now = clock or (lambda: datetime.now(timezone.utc))
    decision = _date(now())
    kickoff = _date(fixture["utcDate"])
    if fixture.get("provider") != "API-Football" or fixture.get("competition", {}).get("code") not in ("CL","EL"):
        return "not_supported"
    if decision >= kickoff:
        return "not_pre_kickoff"
    fid = str(fixture["id"])
    if not fid.isdecimal():
        raise ValueError("invalid_fixture_id")
    root = Path(root)
    destination = root / "trials" / (fid+".json")
    if destination.exists():
        return "original_preserved"
    # Require the original shared history digest and reconstruct it before use.
    original_cutoff = _date(evidence["as_of_utc"])
    if original_cutoff > decision:
        return "future_european_evidence"
    rebuilt = build_evidence(fixture, evidence["history"], original_cutoff)
    if rebuilt["input_digest"] != evidence.get("input_digest"):
        return "european_digest_mismatch"
    # Authenticate the European source observation as well as the results.
    league = str(fixture["competition"]["id"])
    season = season_year(fixture)
    european = []
    for year in (season, season-1):
        entry = cache.get(("API-Football", league, year))
        if entry is None:
            return "european_source_missing"
        rows, meta, error = entry
        if error or not meta.get("freshness_known") or meta.get("stale") is not False:
            return "european_source_unqualified"
        if not meta.get("fetched_at_utc") or _date(meta["fetched_at_utc"]) > decision:
            return "european_observation_unqualified"
        european.extend(rows)
    if build_evidence(fixture, european, original_cutoff)["input_digest"] != rebuilt["input_digest"]:
        return "european_cache_digest_mismatch"
    control = predict(fixture, rebuilt["history"], as_of=original_cutoff)
    if control is None:
        return "sparse_control_unavailable"
    result = research(fixture, control["matrix"], cache, as_of=decision)
    if result["status"] != "research_candidate":
        return result["reason"]
    # Save every observed supported domestic source, not only the chosen league.
    # This preserves the evidence used to rule out ambiguous team membership.
    refs = []
    eligible_leagues = {str(v) for v in DOMESTIC.values()} | {league}
    for key, entry in sorted(cache.items()):
        if key[0] != "API-Football" or key[1] not in eligible_leagues or key[2] not in (season,season-1):
            continue
        rows, metadata, error = entry
        if error or not metadata.get("freshness_known") or metadata.get("stale") is not False:
            continue
        if not metadata.get("fetched_at_utc") or _date(metadata["fetched_at_utc"]) > decision:
            continue
        # Source metadata whitelist; never retain provider credentials.
        packet = {"key": list(key), "rows": rows,
            "metadata": {k: metadata[k] for k in ("freshness_known","stale","fetched_at_utc")}}
        sha = digest(packet)
        path = root / "sources" / (sha+".json")
        _write_once(path, packet)
        if digest(json.loads(path.read_text())) != sha:
            raise ValueError("source_receipt_digest_mismatch")
        refs.append(sha)
    completed = _date(now())
    if not decision <= completed < kickoff:
        return "calculation_after_kickoff"
    here = Path(__file__).resolve().parent
    receipt = {"version": VERSION, "fixture": fixture,
        "decision_as_of_utc": decision.isoformat(), "predicted_at_utc": completed.isoformat(),
        "european_as_of_utc": original_cutoff.isoformat(), "european_input_digest": rebuilt["input_digest"],
        "source_sha256": refs, "model": result,
        "implementation_sha256": {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
            for name in ("cross_competition_capture.py","cross_competition_strength.py","history_pipeline.py",
                         "v37_sparse_history_research.py","research_ensemble_v8.py","engine.py")},
        "research_only": True, "external_requests": 0}
    receipt["record_sha256"] = digest(receipt)
    return "frozen" if _write_once(destination, receipt) else "original_preserved"
