"""Read-only summary of independently frozen v8 and five-double evidence.

The Streamlit live forecast database and the GitHub prospective snapshot ledger
are different stores. Never report pending records as model losses.
"""
import json
from pathlib import Path


ROOT = Path("data/v10_prospective")


def _read(path):
    try:
        obj=json.loads(Path(path).read_text(encoding="utf-8"))
        return obj if isinstance(obj,dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _count(value):
    if type(value) is int and value >= 0:
        return value
    return 0


def research_archive_summary(root=ROOT):
    performance=_read(Path(root)/"performance.json")
    settlement=_read(Path(root)/"settlement_status.json")
    model=(performance.get("models") or {}).get("v8") or {}
    doubles=performance.get("five_doubles") or {}
    frozen=_count(model.get("frozen_fixtures")) or _count(
        settlement.get("eligible_frozen_fixtures"))
    verified=_count(model.get("verified_fixtures"))
    sets=_count(doubles.get("frozen_pairs"))
    verified_sets=_count(doubles.get("verified_pairs"))
    return {
        "frozen": frozen,
        "verified": min(frozen,verified),
        "pending": max(0,frozen-verified),
        "doubles_frozen": sets,
        "doubles_verified": min(sets,verified_sets),
        "doubles_pending": max(0,sets-verified_sets),
        "status": "awaiting_finals" if frozen and not verified else
                  "partially_settled" if frozen and verified<frozen else
                  "no_saved_research" if not frozen else "fully_settled"
    }
