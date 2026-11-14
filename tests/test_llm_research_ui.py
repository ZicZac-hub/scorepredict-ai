"""Mobile-friendly read-only AI display; no provider calls or changed forecasts."""
from llm_research_ui import _outcome_of, _pct, _score_explanation, _score_text


def test_saved_ai_score_is_visible_even_without_aligned_consensus():
    votes = {"gemini": {"score": [1, 1],
                        "home": 0.43, "draw": 0.27, "away": 0.30}}
    panel = {"outcome": "home", "suggested_score": None}
    explanation = _score_explanation(panel, votes)
    assert "No consensus exact score" in explanation
    assert "Individual AI picks" in explanation
    assert _score_text(votes["gemini"]["score"]) == "1–1"
    assert _score_text(panel["suggested_score"]) == "Not available"


def test_probabilities_are_labelled_without_ascribing_model_accuracy():
    assert _pct(0.43) == "43.0%"
    assert _pct(1) == "100.0%"
    assert _pct(None) == "Not available"
    assert _pct(True) == "Not available"
    assert _pct(1.4) == "Not available"


def test_consensus_alignment_explanation():
    panel = {"outcome": "home", "suggested_score": [2,1]}
    votes = {"gemini": {"score": [2,1]}}
    assert "2–1" in _score_explanation(panel,votes)
    assert _outcome_of([1,1]) == "draw"
    assert _outcome_of([2,1]) == "home"
    assert _outcome_of([0,1]) == "away"


def test_never_print_malformed_score_as_real():
    assert _score_text(["1","1"]) == "Not available"
    assert _score_text([1]) == "Not available"
    assert _score_text([True,0]) == "Not available"
