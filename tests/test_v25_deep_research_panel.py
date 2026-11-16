"""V25 Deep Research visibility, mobile-friendly status and frozen records."""
from contextlib import nullcontext
import json
from pathlib import Path

from v25_deep_research_panel import load_status, latest_frozen_set, render


class Screen:
    def __init__(self):
        self.messages=[]
        self.links=[]

    def expander(self,label,expanded=False):
        self.messages.append(("expander",label,expanded))
        return nullcontext()

    def caption(self,value):
        self.messages.append(("caption",value))

    def write(self,value):
        self.messages.append(("write",value))

    def info(self,value):
        self.messages.append(("info",value))

    def link_button(self,title,url):
        self.links.append((title,url))


def data():
    return {
        "model_version":"pro-analyst-v25-robust-sensitivity-shadow",
        "total_frozen_sets":3, "verified_sets":1,"pending_sets":2,
        "robust_wins":1,"baseline_wins_on_same_pairs":0
    }


def set_data():
    return {
        "decision":"FROZEN_RESEARCH_ONLY", "sydney_date":"2035-02-11",
        "first":{"home_team":"Home 1","away_team":"Away 1","forecast_id":"v8-1"},
        "second":{"home_team":"Home 2","away_team":"Away 2","forecast_id":"v8-2"},
        "five_doubles":[{"score1":f"{n}-0","score2":"1-0"} for n in range(5)]
    }


def test_no_report_still_visible_with_no_fake_zero_hit_rate(tmp_path):
    screen=Screen()
    render(screen,report_path=tmp_path/"missing.json",sets_dir=tmp_path/"sets")
    all_text=" ".join(str(x) for x in screen.messages)
    assert screen.messages[0]==(
        "expander","🧮 V25 Professional Analyst · Five Doubles",True)
    assert "no saved research run" in all_text
    assert "not counted as a loss" in all_text
    assert "Real five-double accuracy" not in all_text
    assert screen.links and screen.links[0][1].endswith("prematch-evidence.yml")


def test_report_counts_and_frozen_five_visible_in_order(tmp_path):
    status=tmp_path/"status.json"
    status.write_text(json.dumps(data()))
    directory=tmp_path/"sets"
    directory.mkdir()
    file=directory/"2035-02-11.json"
    file.write_text(json.dumps(set_data()))
    screen=Screen()
    render(screen,report_path=status,sets_dir=directory)
    text=" ".join(str(x) for x in screen.messages)
    assert "V25 five-double hits: **1/1**" in text
    assert "Original V8 on identical pairs: **0/1**" in text
    assert "Awaiting results: **2**" in text
    assert "Home 1 vs Away 1" in text
    assert "Home 2 vs Away 2" in text
    assert "**4-0**" in text
    assert latest_frozen_set(directory)["five_doubles"] == set_data()["five_doubles"]


def test_invalid_and_conflicting_counts_are_not_displayed_as_verified(tmp_path):
    target=tmp_path/"status.json"
    row=data()
    row["verified_sets"]=4
    target.write_text(json.dumps(row))
    assert load_status(target) is None
    row=data()
    row["robust_wins"]=2
    target.write_text(json.dumps(row))
    assert load_status(target) is None
    row=data()
    row["verified_sets"]="1"
    target.write_text(json.dumps(row))
    assert load_status(target) is None
    target.write_text("{broken")
    assert load_status(target) is None


def test_five_doubles_only_displays_immutable_frozen_research_sets(tmp_path):
    directory=tmp_path/"sets"
    directory.mkdir()
    original=set_data()
    (directory/"2035-02-11.json").write_text(json.dumps(original))
    invalid=dict(original,decision="retrospective")
    (directory/"2035-02-12.json").write_text(json.dumps(invalid))
    assert latest_frozen_set(directory)["sydney_date"]=="2035-02-11"
    original["five_doubles"]=[{"score1":"1-1","score2":"0-0"}]
    (directory/"2035-02-11.json").write_text(json.dumps(original))
    assert latest_frozen_set(directory) is None


def test_deep_research_page_wires_v25_before_match_filter_or_buttons():
    source=Path("research_ui.py").read_text(encoding="utf-8")
    assert source.index("    render_v25(st)") < source.index("    choices = {}")
    assert source.index("    render_v25(st)") < source.index(
        '    method = st.radio("Prediction method",')
    assert "v25_deep_research_panel" in source
