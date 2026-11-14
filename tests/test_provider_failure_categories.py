"""Regression checks for provider access and safe high-frequency collection."""
from datetime import datetime,timezone

import pytest
import api_football
from capture_prematch import collect


class Reply:
    def __init__(self,status_code,payload=None):
        self.status_code=status_code
        self.payload=payload or {}
    def raise_for_status(self):
        if self.status_code>=400:
            raise RuntimeError("Bad status")
    def json(self):
        return self.payload


@pytest.mark.parametrize("status,category",[
    (429, api_football.ProviderQuotaError),
    (401, api_football.ProviderAccessError),
    (403, api_football.ProviderAccessError),
])
def test_provider_access_failures_are_categorized_safely(tmp_path,monkeypatch,status,category):
    monkeypatch.setattr(api_football,"CACHE",tmp_path)
    monkeypatch.setattr(api_football.requests,"get",
                        lambda *args,**kwargs:Reply(status))
    with pytest.raises(category):
        api_football._request("never-print-this-key","/fixtures",
                              {"league":39,"season":2026})


def test_provider_quota_in_success_http_body_is_not_hidden(tmp_path,monkeypatch):
    monkeypatch.setattr(api_football,"CACHE",tmp_path)
    monkeypatch.setattr(api_football.requests,"get",
        lambda *args,**kwargs:Reply(200,{"errors":{
            "requests":"You have reached the limit of calls for this day"}}))
    with pytest.raises(api_football.ProviderQuotaError):
        api_football._request("not-a-live-key","/fixtures",
                              {"league":39,"season":2026})


@pytest.mark.parametrize("error,reason",[
    (api_football.ProviderQuotaError,"provider_quota"),
    (api_football.ProviderAccessError,"provider_credentials_or_plan")
])
def test_collector_stops_retrying_all_leagues_with_unusable_provider_key(
        tmp_path,error,reason):
    calls=[]
    def fail(key,endpoint,params,**kwargs):
        calls.append(params["league"])
        raise error("A safe simulated API block")
    report=collect("test",now=datetime(2026,11,15,2,tzinfo=timezone.utc),
        codes=("PL","PD","BL1","SA"),ledger_root=tmp_path,
        api_request=fail,database_url=None)
    diagnostics=report["diagnostics"]
    assert len(calls)==1
    assert diagnostics["collector_blocker"]["reason"]==reason
    assert not diagnostics["coverage_complete"]
    assert report["records"]==[]
