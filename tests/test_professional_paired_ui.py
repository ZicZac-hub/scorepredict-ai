"""Future comparison display never turns stale flags or missing arms into wins."""
from copy import deepcopy
import json

from streamlit.testing.v1 import AppTest

import professional_paired_ui as ui


def stats(*, issued=0, abstained=0, pending=0, verified=0, hits=0):
    return {"issued": issued, "abstained": abstained, "pending": pending,
            "verified": verified, "hits": hits,
            "hit_rate": hits/verified if verified else None,
            "wilson_95": [0.1, 0.9] if verified else None,
            "single_score_log_loss": 2.9 if verified else None,
            "outcome_brier": 0.6 if verified else None}


def app_for(tmp_path):
    return AppTest.from_string(
        "import streamlit as st\nfrom pathlib import Path\n"
        "from professional_paired_ui import render\n"
        f"render(st, root=Path({str(tmp_path)!r}))\n")


def protocol(tmp_path):
    path=tmp_path/"professional_paired_study"
    path.mkdir()
    (path/"protocol.json").write_text(json.dumps({"fixed": True}))
    return path


def test_missing_protocol_does_not_create_a_study_or_accuracy(tmp_path, monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("No protocol means no eligible study to evaluate")
    monkeypatch.setattr(ui, "_evaluate", unexpected)
    app=app_for(tmp_path).run(timeout=15)
    assert not app.exception
    assert any("waiting for its fixed protocol" in text.value for text in app.info)
    assert len(app.dataframe)==0
    assert not (tmp_path/"professional_paired_study").exists()


def test_no_settled_results_ignore_fabricated_cached_performance(tmp_path, monkeypatch):
    path=protocol(tmp_path)
    (path/"performance.json").write_text(json.dumps({"methods": {
        "baseline_v8": stats(issued=100, verified=100, hits=100)}}))
    seen=[]
    def evaluate(**kwargs):
        seen.append(kwargs)
        return {"methods": {
            "baseline_v8": stats(issued=2, pending=2),
            "market_only": stats(abstained=2),
            "xg_only": stats(abstained=2),
            "xg_market": stats(abstained=2)}, "comparisons": {}}
    monkeypatch.setattr(ui, "_evaluate", evaluate)
    app=app_for(tmp_path).run(timeout=15)
    assert not app.exception
    table=app.dataframe[0].value
    assert set(table["Observed five-double hit rate"])=={"Unavailable"}
    assert list(table["Verified sets"])==[0]*4
    assert list(table["Missing evidence / abstained"])==[0,2,2,2]
    assert list(table["Pending"])==[2,0,0,0]
    assert any("No settled matched results yet" in text.value for text in app.info)
    assert any("70% verified hit rate is not established" in text.value for text in app.caption)
    assert seen[0]["source_root"]==tmp_path/"v10_prospective"
    assert seen[0]["provider_root"]==tmp_path/"v21_sources"
    assert seen[0]["pair_root"]==tmp_path/"professional_double_shadow"


def test_available_arm_rates_keep_their_actual_sample_and_abstentions(tmp_path, monkeypatch):
    protocol(tmp_path)
    report={"methods": {
        "baseline_v8": stats(issued=3, pending=1, verified=2, hits=1),
        "market_only": stats(issued=1, abstained=2, verified=1, hits=0)},
        "comparisons": {"market_only": {"matched_verified_sets": 1}}}
    original=deepcopy(report)
    monkeypatch.setattr(ui, "_evaluate", lambda **kwargs: report)
    app=app_for(tmp_path).run(timeout=15)
    assert not app.exception
    table=app.dataframe[0].value
    assert list(table["Observed five-double hit rate"])==["50.00%", "0.00%"]
    assert list(table["Missing evidence / abstained"])==[0,2]
    assert report==original
    assert len(app.get("download_button"))==2


def test_authentication_failure_never_falls_back_to_cached_accuracy(tmp_path, monkeypatch):
    path=protocol(tmp_path)
    (path/"performance.json").write_text(json.dumps({"hit_rate": 1.0}))
    def invalid(**kwargs):
        raise ValueError("fixture digest mismatch")
    monkeypatch.setattr(ui, "_evaluate", invalid)
    app=app_for(tmp_path).run(timeout=15)
    assert not app.exception
    assert len(app.dataframe)==0
    assert any("could not authenticate" in text.value for text in app.warning)


def test_inconsistent_verified_counts_are_not_displayed(tmp_path, monkeypatch):
    protocol(tmp_path)
    monkeypatch.setattr(ui, "_evaluate", lambda **kwargs: {"methods": {
        "baseline_v8": {**stats(), "hits": 5}}})
    app=app_for(tmp_path).run(timeout=15)
    assert not app.exception
    assert len(app.dataframe)==0
    assert any("No accuracy result is displayed" in text.value for text in app.warning)


def test_partial_final_result_stays_pending_and_clock_uses_sydney(tmp_path, monkeypatch):
    protocol(tmp_path)
    monkeypatch.setattr(ui, "_evaluate", lambda **kwargs: {
        "evaluated_at_utc": "2031-11-20T23:00:00+00:00",
        "methods": {"baseline_v8": stats(issued=1, pending=1)},
        "rows": [{"sydney_date": "2031-11-21", "status": "pending",
                  "actual_scores": [[0, 1], None]}]})
    app=app_for(tmp_path).run(timeout=15)
    assert not app.exception
    date_table=app.dataframe[1].value
    assert date_table.iloc[0]["Sydney match date"]=="2031-11-21"
    assert date_table.iloc[0]["Set status"]=="pending"
    assert date_table.iloc[0]["First verified final score"]=="0–1"
    assert date_table.iloc[0]["Second verified final score"]=="Pending verification"
    assert any("21 Nov 2031, 10:00 AEDT" in text.value for text in app.caption)


def test_real_empty_study_runtime_is_read_only_and_makes_no_provider_calls(tmp_path, monkeypatch):
    import api_football
    import professional_paired_study as study
    import requests
    path=tmp_path/"professional_paired_study"
    study.register_protocol(root=path)
    original=(path/"protocol.json").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("The evidence display must never call a provider")
    monkeypatch.setattr(api_football, "_request", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    app=app_for(tmp_path).run(timeout=15)
    assert not app.exception
    table=app.dataframe[0].value
    assert list(table["Verified sets"])==[0]*4
    assert set(table["Observed five-double hit rate"])=={"Unavailable"}
    assert (path/"protocol.json").read_bytes()==original
    assert not (path/"performance.json").exists()
