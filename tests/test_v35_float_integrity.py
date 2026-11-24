"""V35 archived scores are exact; only microscopic numeric drift is valid."""
from copy import deepcopy

from v35_continuous_double_trials import _same_portfolio_with_float_tolerance


def frozen():
    # Synthetic, normalized complete score grids and five unique joint picks.
    grid = [[.04]*5 for _ in range(5)]
    five = [{"score1":f"0-{i}","score2":"1-1",
             "joint_probability":.0016} for i in range(5)]
    return {
        "version":"v35-synthetic","sydney_day":"2035-01-01",
        "original_catalog_sha256":"original-immutable-hash",
        "original_selected_fixture_keys":[["API-Football","101"],
                                          ["API-Football","202"]],
        "policies":{"v8_original":{
            "score_grids":[grid,deepcopy(grid)],
            "five_joint_scores":five}},
        "not_hindsight_selected":True,
    }


def test_only_roundoff_permitted_with_identical_original_tickets():
    base = frozen()
    near = deepcopy(base)
    near["policies"]["v8_original"]["score_grids"][0][0][0] += 1e-15
    near["policies"]["v8_original"]["score_grids"][0][0][1] -= 1e-15
    near["policies"]["v8_original"]["five_joint_scores"][0][
        "joint_probability"] += 1e-15
    assert _same_portfolio_with_float_tolerance(base, near)
    assert base == frozen()  # no frozen source rewritten


def test_real_score_grid_change_still_fails_integrity_gate():
    base = frozen()
    tampered=deepcopy(base)
    grid=tampered["policies"]["v8_original"]["score_grids"][0]
    grid[0][0] += .00001
    grid[0][1] -= .00001  # still sums to one
    assert not _same_portfolio_with_float_tolerance(base,tampered)


def test_changed_ticket_and_source_identity_cannot_pass():
    base=frozen()
    wrong=deepcopy(base)
    wrong["policies"]["v8_original"]["five_joint_scores"][0]["score1"]="4-4"
    assert not _same_portfolio_with_float_tolerance(base,wrong)
    wrong=deepcopy(base)
    wrong["original_catalog_sha256"]="changed-source"
    assert not _same_portfolio_with_float_tolerance(base,wrong)


def test_non_numeric_grid_or_missing_ticket_is_rejected():
    base=frozen()
    wrong=deepcopy(base)
    wrong["policies"]["v8_original"]["score_grids"][0][0][0]="0.04"
    assert not _same_portfolio_with_float_tolerance(base,wrong)
    wrong=deepcopy(base)
    wrong["policies"]["v8_original"]["five_joint_scores"].pop()
    assert not _same_portfolio_with_float_tolerance(base,wrong)
    wrong=deepcopy(base)
    wrong["policies"]["v8_original"]["five_joint_scores"][0]["score1"]="0-1"
    assert not _same_portfolio_with_float_tolerance(base,wrong)
