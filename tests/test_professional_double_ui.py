"""Ready Premium selections must render/save; abstention cannot imply success."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import professional_double_ui as ui
from professional_double_optimizer import AUTO_SELECTION, POLICY, manual

NOW=datetime(2035,11,20,tzinfo=timezone.utc)


class Clock:
    @classmethod
    def now(cls, timezone_value=None):
        return NOW


def row(number):
    p=np.array([.20,.30,.25,.15,.06,.025,.01,.005])
    matrix=np.outer(p,p)
    scores=sorted([(h,a,float(v)) for h,line in enumerate(matrix)
                   for a,v in enumerate(line)],key=lambda x:-x[2])[:10]
    kickoff=(NOW+timedelta(hours=4+number)).isoformat()
    return {"fixture":{"provider":"API-Football","id":number,"utcDate":kickoff,
                       "homeTeam":{"id":number*2,"name":f"Home {number}"},
                       "awayTeam":{"id":number*2+1,"name":f"Away {number}"}},
            "kickoff":kickoff,"archive_frozen":True,
            "frozen_at_utc":(NOW-timedelta(hours=1)).isoformat(),
            "result":{"scores":scores,"matrix":matrix.tolist(),"model_version":"v1"}}


class Screen:
    secrets={}
    def __init__(self,mode="Premium AI Auto Select",pressed=()):
        self.mode=mode
        self.pressed=set(pressed)
        self.lines=[]
        self.metrics={}
        self.tables=[]
    def selectbox(self,title,values,**kwargs):
        return list(values)[0]
    def radio(self,title,values,**kwargs):
        return self.mode
    def button(self,title,**kwargs):
        return kwargs.get("key") in self.pressed
    def markdown(self,text): self.lines.append(text)
    def caption(self,text): self.lines.append(text)
    def warning(self,text): self.lines.append(text)
    def info(self,text): self.lines.append(text)
    def write(self,value): self.lines.append(str(value))
    def metric(self,title,value): self.metrics[title]=value
    def dataframe(self,values,**kwargs): self.tables.append(values)
    def expander(self,*args,**kwargs): return nullcontext()


@pytest.fixture
def isolated(monkeypatch):
    import ai_five_double_ui as old_ui
    import on_demand_ai
    import professional_double_shadow
    import ziczac_interactive_capture as capture
    monkeypatch.setattr(ui,"datetime",Clock)
    monkeypatch.setattr(old_ui,"options",lambda rows,**kwargs:rows)
    monkeypatch.setattr(old_ui,"repository_provider_packet",lambda *args,**kwargs:None)
    monkeypatch.setattr(on_demand_ai,"private_gate_ready",lambda st:False)
    monkeypatch.setattr(professional_double_shadow,"load_verified_trials",lambda:[])
    saved=[]
    monkeypatch.setattr(capture,"store_ui",lambda st,record:saved.append(record))
    return saved


def ready_report(first,second,*,scope="premium_max_calibrated_coverage_v1"):
    report=manual(first,second,now=NOW)
    report["selection_scope"]=scope
    report["coverage_is_calibrated"]=True
    report["calibration"]={"calibrated_coverage_estimate":.12}
    return report


def test_premium_ready_displays_actual_selected_pair_and_saves_scope(monkeypatch,isolated):
    rows=[row(i) for i in (1,2,3,4)]
    report=ready_report(rows[2],rows[3])
    monkeypatch.setattr(ui,"premium",lambda *args,**kwargs:{"status":"ready","selection":report})
    screen=Screen()
    ui.render(screen,rows,{})
    assert "Calibrated five-combination coverage estimate" in screen.metrics
    assert len(screen.tables[0])==5
    assert "Home 3" in "\n".join(screen.lines) and "Home 4" in "\n".join(screen.lines)
    assert len(isolated)==1
    assert [leg["fixture_id"] for leg in isolated[0]["legs"]]==["3","4"]
    assert isolated[0]["method"]==POLICY
    assert isolated[0]["policy_metadata"]["selection_scope"]=="premium_max_calibrated_coverage_v1"
    assert any("70% verified hit rate is not established" in line for line in screen.lines)


def test_abstention_never_renders_or_saves_a_premium_prediction(monkeypatch,isolated):
    monkeypatch.setattr(ui,"premium",lambda *args,**kwargs:{"status":"abstained",
        "selection":None,"eligible_pairs_compared":1,"rejections":{"insufficient_calibration":1}})
    screen=Screen()
    ui.render(screen,[row(1),row(2)],{})
    assert any("Premium Auto Select abstained" in line for line in screen.lines)
    assert screen.tables==[] and screen.metrics=={} and isolated==[]
    assert "Premium automatic selection" not in "\n".join(screen.lines)


def test_public_verified_trials_use_real_loader_and_fresh_decision_clock(monkeypatch,isolated):
    import professional_double_shadow
    observed=[]
    public=[{"source":"genuinely_verified_public_trial"}]
    monkeypatch.setattr(professional_double_shadow,"load_verified_trials",lambda:public)
    def premium(rows,*,now,verified_trials):
        observed.append((now,verified_trials))
        return {"status":"abstained","selection":None,
                "eligible_pairs_compared":1,"rejections":{}}
    monkeypatch.setattr(ui,"premium",premium)
    ui.render(Screen(),[row(1),row(2)],{})
    assert observed==[(NOW,public)]


def test_research_bootstrap_is_explicit_and_saved_as_automatic_research(monkeypatch,isolated):
    monkeypatch.setattr(ui,"premium",lambda *args,**kwargs:{"status":"abstained",
        "selection":None,"eligible_pairs_compared":1,"rejections":{}})
    screen=Screen(pressed={"professional_auto_research_generate"})
    ui.render(screen,[row(1),row(2)],{})
    assert len(screen.tables[0])==5
    assert len(isolated)==1
    assert isolated[0]["policy_metadata"]["selection_scope"]==AUTO_SELECTION
    assert any("not a qualified Premium pick" in line for line in screen.lines)
    assert "Calibrated five-combination coverage estimate" not in screen.metrics


def test_manual_ready_and_failed_durable_save_keep_display_truthful(monkeypatch,isolated):
    import ziczac_interactive_capture as capture
    def reject(*args,**kwargs): raise ValueError("invalid provenance")
    monkeypatch.setattr(capture,"pair",reject)
    screen=Screen(mode="All upcoming · manual",pressed={"professional_pair_generate"})
    ui.render(screen,[row(1),row(2)],{})
    assert len(screen.tables[0])==5 and isolated==[]
    assert any("was not saved" in line and "No durable save is claimed" in line for line in screen.lines)


def test_changed_selected_source_cannot_render_or_save(monkeypatch,isolated):
    rows=[row(1),row(2)]
    report=ready_report(*rows)
    report["first_fixture"]=deepcopy(report["first_fixture"])
    report["first_fixture"]["id"]=99999
    monkeypatch.setattr(ui,"premium",lambda *args,**kwargs:{"status":"ready","selection":report})
    screen=Screen()
    ui.render(screen,rows,{})
    assert screen.tables==[] and isolated==[]
    assert any("could not be matched" in line for line in screen.lines)


def test_streamlit_premium_ready_runtime_renders_and_saves(monkeypatch,isolated):
    from streamlit.testing.v1 import AppTest
    import ziczac_interactive_capture as capture
    rows=[row(1),row(2)]
    report=ready_report(*rows)
    monkeypatch.setattr(ui,"premium",lambda *args,**kwargs:{"status":"ready","selection":report})
    def store(st,record):
        st.session_state["saved_scope"]=record["policy_metadata"]["selection_scope"]
        return {"durable":False,"state":"test_capture_only"}
    monkeypatch.setattr(capture,"store_ui",store)
    app=AppTest.from_string("import streamlit as st\nfrom professional_double_ui import render\n"
                            f"render(st,{rows!r},{{}})\n").run(timeout=15)
    app.radio[0].set_value("Premium AI Auto Select").run(timeout=15)
    assert not app.exception
    assert app.session_state["saved_scope"]=="premium_max_calibrated_coverage_v1"
    assert any(metric.label=="Calibrated five-combination coverage estimate" for metric in app.metric)
    assert len(app.dataframe[0].value)==5


@pytest.mark.parametrize("wrapped",[False,True])
def test_real_correct_score_quote_packets_accept_list_and_legacy_envelope(monkeypatch,isolated,wrapped):
    from correct_score_market import parse_correct_score_quotes
    import ai_five_double_ui as old_ui
    rows=[row(1),row(2)]
    report=ready_report(*rows)
    packets={}
    for i,source in enumerate(rows):
        fixture=source["fixture"]
        quote_response=[{"fixture":{"id":fixture["id"],"date":fixture["utcDate"]},
            "bookmakers":[{"name":"Source bookmaker","bets":[{"name":"Correct Score",
                "values":[{"value":report["choices"][0][f"score{i+1}"],"odd":"8.0"}]}]}]}]
        parsed=parse_correct_score_quotes(quote_response,fixture,observed_at=NOW,
                                          verified_provider_fetched_at=NOW)
        assert parsed["ready"]
        packets[fixture["id"]]={"correct_score_market_quotes":parsed if wrapped else parsed["quotes"]}
    monkeypatch.setattr(old_ui,"repository_provider_packet",lambda fixture,**kwargs:packets[fixture["id"]])
    monkeypatch.setattr(ui,"premium",lambda *args,**kwargs:{"status":"ready","selection":report})
    screen=Screen()
    ui.render(screen,rows,{})
    prices=screen.tables[1]
    assert prices[0]["leg_price_product"]==64
    assert prices[0]["executable_double_quote_verified"] is False
    assert any("uncalibrated per-score probabilities" in line for line in screen.lines)
    assert any("actual ROI is unavailable" in line for line in screen.lines)
    assert len(isolated)==1


def test_analysis_finishing_after_kickoff_does_not_render_a_fresh_prediction(monkeypatch,isolated):
    rows=[row(1),row(2)]
    report=ready_report(*rows)
    current=[NOW]
    class MovingClock:
        @classmethod
        def now(cls,timezone_value=None): return current[0]
    monkeypatch.setattr(ui,"datetime",MovingClock)
    def delayed_premium(*args,**kwargs):
        current[0]=NOW+timedelta(hours=7)
        return {"status":"ready","selection":report}
    monkeypatch.setattr(ui,"premium",delayed_premium)
    screen=Screen()
    ui.render(screen,rows,{})
    assert screen.metrics=={} and screen.tables==[] and isolated==[]
    assert any("match has started; no new prediction was displayed or saved" in line for line in screen.lines)
