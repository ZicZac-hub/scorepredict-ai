"""V23 end-to-end: frozen V8 and V23 on the SAME two fixtures, verified settlement.

Synthetic data only. This proves write-once integrity and hit/miss logic,
NOT real football prediction accuracy.
"""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
from v21_goal_model_comparison import top_scores
import v23_prospective as v23
import v22_prospective as v22

NOW = datetime(2032, 6, 11, 9, tzinfo=timezone.utc)
KICK = (NOW + timedelta(hours=3)).isoformat()

def _candidate(fid, home, away):
    base = np.zeros((11, 11))
    base[1, 1] = .70
    base[1, 0] = .15
    base[2, 1] = .10
    base[0, 0] = .05
    challenger = np.zeros((11, 11))
    challenger[3, 2] = .70
    challenger[2, 2] = .20
    challenger[0, 0] = .10
    source = {
        "provider": "API-Football", "fixture_id": str(fid),
        "forecast_id": "immutable" + str(fid),
        "input_digest": "originaldigest" + str(fid),
        "home_team_id": str(home), "away_team_id": str(away),
        "model_key": "v8", "kickoff_utc": KICK
    }
    return {
        "source": source,
        "snapshot_hash": "a" * 64,
        "v8": {"scores": top_scores(base, 10)},
        "v22": {"scores": top_scores(challenger, 10)}
    }

def _actual(candidate, home=1, away=1, valid=True):
    leg = candidate["source"]
    return {
        "evidence_version": 2, "score_basis": "regulation",
        "provider": "API-Football" if valid else "another-provider",
        "fixture_id": leg["fixture_id"],
        "home_team_id": leg["home_team_id"],
        "away_team_id": leg["away_team_id"],
        "kickoff_utc": leg["kickoff_utc"],
        "status": "FT",
        "verified_at_utc": (NOW + timedelta(hours=6)).isoformat(),
        "home_goals": home, "away_goals": away
    }

def test_v23_immutable_same_pair_pending_hit_miss_and_bad_identity(monkeypatch, tmp_path):
    first, second = _candidate(3001, 11, 12), _candidate(3002, 21, 22)
    monkeypatch.setattr(v23, "frozen_candidates", lambda **kwargs: [first, second])

    first_freeze = v23.freeze(now=NOW, root=tmp_path)
    assert len(first_freeze["newly_frozen_days"]) == 1
    assert not v23.freeze(now=NOW, root=tmp_path)["newly_frozen_days"]

    files = list((tmp_path / "pairs").glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert len(record["doubles"]["baseline_v8"]) == 5
    assert len(record["doubles"]["v23_market_form"]) == 5
    assert record["captured_at_utc"] < KICK
    assert record["no_production_promotion"]
    assert json.loads(files[0].read_text()) == record

    # Preserve pre-match source identity in the settlement code, rather
    # than replacing GitHub's immutable source inputs with a fake result.
    known = {x["source"]["fixture_id"]: x["source"] for x in (first, second)}
    monkeypatch.setattr(v22, "original_model",
                        lambda root, leg: known.get(str(leg["fixture_id"])))

    pending = v23.evaluate(root=tmp_path, source_root=tmp_path,
                           verified={"3001": _actual(first)})
    assert pending["methods"]["baseline_v8"]["pending"] == 1
    assert pending["methods"]["v23_market_form"]["pending"] == 1
    assert pending["methods"]["v23_market_form"]["hit_rate"] is None

    # Both matches must have provider-matched independently verified finals.
    completed = v23.evaluate(root=tmp_path, source_root=tmp_path, verified={
        "3001": _actual(first), "3002": _actual(second)
    })
    assert completed["methods"]["baseline_v8"]["verified"] == 1
    assert completed["methods"]["baseline_v8"]["hits"] == 1
    assert completed["methods"]["v23_market_form"]["verified"] == 1
    assert completed["methods"]["v23_market_form"]["hits"] == 0

    # Tampered provider identity does NOT become a false loss or a true hit.
    tampered = v23.evaluate(root=tmp_path, source_root=tmp_path, verified={
        "3001": _actual(first), "3002": _actual(second, valid=False)
    })
    assert tampered["methods"]["v23_market_form"]["pending"] == 1
    assert tampered["methods"]["v23_market_form"]["verified"] == 0
    assert json.loads(files[0].read_text()) == record
