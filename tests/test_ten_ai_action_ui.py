"""Ten-AI panel must offer a real run button, without accidental paid calls."""
from datetime import datetime, timedelta, timezone
from contextlib import nullcontext

from ten_ai_action_ui import available, render


def example_fixture(minutes_ahead=90):
    kickoff=(datetime.now(timezone.utc)+timedelta(minutes=minutes_ahead)).isoformat()
    return {
        "fixture":{
            "provider":"API-Football", "id":1942,
            "utcDate":kickoff,
            "homeTeam":{"id":11,"name":"Home"},
            "awayTeam":{"id":22,"name":"Away"},
        },
        "league":"PL",
        "result":{"scores":[[1,1,.13],[2,1,.11]]},
        "past":[]
    }


def test_ten_ai_match_menu_uses_only_upcoming_valid_fixtures():
    clock=datetime(2030,11,14,14,tzinfo=timezone.utc)
    good=example_fixture()
    good["fixture"]["utcDate"]="2030-11-15T14:00:00Z"
    after=example_fixture()
    after["fixture"]["utcDate"]="2030-11-14T13:00:00Z"
    missing=example_fixture()
    missing["fixture"]["id"]=None
    five=example_fixture()
    five["fixture"]["utcDate"]="2030-11-14T14:04:00Z"
    matches=available([good,after,missing,five],now=clock)
    assert len(matches)==1
    assert next(iter(matches.values())) is good


class StreamlitFake:
    def __init__(self, *, login=False, credentials=False):
        self.session_state={"ziczac_access_granted":login}
        self.secrets={}
        if login:
            self.secrets["ZICZAC_ACCESS_PASSWORD"]="test-password"
        if credentials:
            self.secrets.update({"GEMINI_API_KEY":"dummy-no-real-api",
                                 "GEMINI_MODEL":"test-model"})
        self.messages=[]
        self.clicks=0
    def markdown(self,value,*args,**kwargs):self.messages.append(("markdown",value))
    def caption(self,value,*args,**kwargs):self.messages.append(("caption",value))
    def warning(self,value,*args,**kwargs):self.messages.append(("warning",value))
    def info(self,value,*args,**kwargs):self.messages.append(("info",value))
    def selectbox(self,label,options,**kwargs):
        self.messages.append(("selectbox",label))
        return options[0]
    def checkbox(self,label,**kwargs):
        self.messages.append(("checkbox",label))
        return True
    def button(self,label,**kwargs):
        self.messages.append(("button",label))
        self.clicks+=1
        return True
    def spinner(self,*args,**kwargs):return nullcontext()
    def expander(self,*args,**kwargs):return nullcontext()


def test_run_button_exists_even_without_streamlit_api_keys_and_never_bills(monkeypatch):
    import ten_ai_action_ui as mod
    monkeypatch.setattr(mod,"frozen_for_fixture",lambda *args,**kwargs:None)
    def reject(*args,**kwargs):
        raise AssertionError("No paid API calls allowed on a public site")
    monkeypatch.setattr(mod,"run_selected",reject)
    st=StreamlitFake()
    render(st,rows=[example_fixture()],histories={})
    assert any("Run Ten-AI Research & Predict" in text
               for kind,text in st.messages if kind=="button")
    assert any("Streamlit" in text for kind,text in st.messages if kind=="warning")
    assert "ziczac_ten_ai_results" in st.session_state
    assert not st.session_state["ziczac_ten_ai_results"]


def test_authenticated_run_uses_its_server_credentials_once_per_session(monkeypatch):
    import ten_ai_action_ui as mod
    monkeypatch.setattr(mod,"frozen_for_fixture",lambda *args,**kwargs:None)
    called=[]
    def pretend(*args,**kwargs):
        called.append(args)
        assert list(args[4])==["gemini"]
        return {"status":"completed", "votes":{}, "errors":{},
                "attempted":["gemini"],"consensus":None}
    monkeypatch.setattr(mod,"run_selected",pretend)
    st=StreamlitFake(login=True,credentials=True)
    record=example_fixture()
    render(st,rows=[record],histories={})
    render(st,rows=[record],histories={})
    assert len(called)==1
    assert any("previous AI answers" in text for kind,text in st.messages
               if kind=="info")


def test_pending_and_started_fixture_cannot_be_selected():
    item=example_fixture(minutes_ahead=-5)
    assert not available([item])
