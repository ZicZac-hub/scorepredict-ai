"""The final panel publishes only after its prospective evidence gate opens."""

import pytest

import final_five_doubles
from final_five_doubles_ui import render


class Screen:
    def __init__(self):
        self.details = []
        self.tables = []

    def __getattr__(self, name):
        if name in ("subheader", "info", "caption", "metric", "write"):
            return lambda *args: self.details.extend(args)
        raise AttributeError(name)

    def dataframe(self, rows, **kwargs):
        self.tables.append(rows)


@pytest.mark.parametrize("state, ready, show_pick", [
    ("awaiting_validation", False, False),
    ("incomplete_slate", True, False),
    ("expired", True, False),
    ("available", False, False),
    ("available", True, True),
])
def test_final_panel_withholds_match_names_and_scores(monkeypatch, state, ready, show_pick):
    decisions = {
        "sydney_day": "2030-01-01", "eligible_fixtures": 3, "eligible_pairs": 3,
        "compared_pairs": 3, "estimate": 0.15, "methods": ["baseline"],
        "evidence_counts": {"xg": 0, "market": 0}, "names": ["Alpha vs Beta", "Gamma vs Delta"],
        "five_doubles": [{"score1": f"{index}-0", "score2": "2-0", "joint_probability": 0.03}
                         for index in range(5)],
    }
    validation = {"wins": 80, "verified": 100, "pending": 1, "abstained": 2,
                  "wilson_95": [0.71, 0.87], "target": 0.70, "minimum_verified": 100,
                  "minimum_issued_fraction": 0.80, "invalid": 0, "ready": ready,
                  "first_cohort_verified": 100, "first_cohort_wilson_95": [0.71, 0.87],
                  "publication_wilson_95": [0.70, 0.87]}
    monkeypatch.setattr(final_five_doubles, "presentation", lambda: {
        "state": state, "decision": decisions, "validation": validation,
    })
    screen = Screen()

    render(screen)

    assert "Final Five Doubles" in screen.details
    assert any("3/3 eligible fixture pairs compared" in text for text in screen.details)
    assert any("Alpha vs Beta" in text for text in screen.details) is show_pick
    assert bool(screen.tables) is show_pick
    if show_pick:
        assert len(screen.tables[0]) == 5
