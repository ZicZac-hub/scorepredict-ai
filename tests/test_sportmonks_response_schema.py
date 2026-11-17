"""Sportmonks schema validation: cursor/page pagination, safe redacted diagnostics."""
import json
from datetime import timedelta

from sportmonks_xg_research import (
    _decode_page, _query_pages, _schema_diagnostic,
    RequestBudgetExhausted,
)
from test_sportmonks_xg_research import NOW, original, next_match


def test_verified_cursor_pagination_does_not_mix_page_and_per_page(tmp_path):
    calls = []
    def request(url,params):
        calls.append(dict(params))
        if len(calls)==1:
            assert params["per_page"]==50 and params["page"]==1
            return {"data":[],"pagination":{
                "has_more":True,"next_cursor":"opaque-only-cursor"}}
        assert "per_page" not in params and "page" not in params
        assert params["cursor"]=="opaque-only-cursor"
        return {"data":[next_match()],
                "pagination":{"has_more":False,"next_cursor":None}}
    got,reason,details=_query_pages(original(),NOW,tmp_path,request)
    assert reason=="source_identity_matched"
    assert got["fixture_id"]==54321
    assert details["pages_read"]==2
    assert details["response_schema"]["data_kind"]=="list"
    assert details["response_schema"]["has_more_kind"]=="bool"
    assert len(calls)==2
    def must_not_request(*args,**kwargs):
        raise AssertionError("Repeated page requests should be cached")
    replay,why,cached=_query_pages(original(),NOW+timedelta(minutes=10),
                                    tmp_path,must_not_request)
    assert replay==got and why=="source_identity_matched"
    assert cached["cached_pages_used"]==2


def test_next_cursor_without_has_more_is_explicit_forward_pointer():
    rows,state,error,shape=_decode_page(
        {"data":[],"pagination":{"next_cursor":"opaque-next"}},1)
    assert error is None
    assert state["mode"]=="cursor" and state["more"] is True
    assert shape["next_cursor_present"]


def test_legacy_next_page_url_parses_page_without_persisting_auth():
    rows,state,error,diag=_decode_page({
        "data":[next_match()],
        "pagination":{"has_more":True,"next_page":
            "https://api.sportmonks.com/v3/football/fixtures/date/2032-08-17?page=2&api_token=TOP_SECRET"}
    },1)
    assert error is None and state=={"mode":"page","more":True,"page":2}
    assert "TOP_SECRET" not in json.dumps(diag)
    assert "TOP_SECRET" not in json.dumps(state)


def test_next_page_other_host_rejected_without_assuming_nonempty_response_finished():
    rows,state,error,diag=_decode_page({
        "data":[next_match()],
        "pagination":{"next_page":"https://bad.example/fixtures?page=2"}
    },1)
    assert rows is None and error=="invalid_discovery_response"


def test_empty_no_results_message_is_terminal_not_a_fixture(tmp_path):
    seen=[]
    def request(url,params):
        seen.append(dict(params))
        return {"data":[],"message":"No results found."}
    match,reason,meta=_query_pages(original(),NOW,tmp_path,request)
    assert match is None
    assert reason=="no_unique_cross_provider_fixture_match"
    assert len(seen)==2  # verified empty filtered search and date fallback
    assert meta["response_schema"]["pagination_kind"]=="NoneType"
    assert meta["response_schema"]["row_count"]==0
    assert meta["response_schema"]["message_present"]


def test_absent_pagination_with_nonempty_data_is_invalid_not_assumed_complete(tmp_path):
    match,reason,meta=_query_pages(
        original(),NOW,tmp_path,lambda url,params:
        {"data":[next_match()],"message":"no paging metadata"})
    assert match is None and reason=="invalid_discovery_response"
    assert meta["pages_read"]==0
    assert meta["response_schema"]["pagination_kind"]=="NoneType"
    assert meta["response_schema"]["row_count"]==1


def test_missing_has_more_and_no_next_pointer_abstains():
    rows,state,error,shape=_decode_page({
        "data":[next_match()],"pagination":{"count":1,"current_page":1}
    },1)
    assert error=="invalid_discovery_response"


def test_unsafe_has_more_strings_rejected():
    rows,state,error,shape=_decode_page({
        "data":[],"pagination":{"has_more":"false"}
    },1)
    assert error=="invalid_discovery_response"
    assert shape["has_more_kind"]=="str"


def test_diagnostics_allowlist_cannot_expose_response_secret_strings(tmp_path):
    secret="MY-PRIVATE-TOKEN-DO-NOT-LOG"
    packet={"data":[],"pagination":{"has_more":"invalid",
                     "next_cursor":secret,"api_token":secret},
            "message":"Authentication error "+secret,
            "private":secret}
    match,reason,summary=_query_pages(
        original(),NOW,tmp_path,lambda url,params:packet)
    assert match is None and reason=="invalid_discovery_response"
    assert secret not in json.dumps(summary)
    assert secret not in json.dumps(_schema_diagnostic(packet))
    assert not list(tmp_path.rglob("*.json"))


def test_budget_exhaustion_stays_distinct_and_no_unverified_match(tmp_path):
    def no_calls(url,params):
        raise RequestBudgetExhausted
    match,reason,summary=_query_pages(original(),NOW,tmp_path,no_calls)
    assert match is None and reason=="discovery_request_budget_exhausted"
    assert summary["pages_read"]==0


def test_empty_no_data_is_validated_and_falls_back_to_unfiltered_date(tmp_path):
    calls=[]
    def request(url, params):
        calls.append(dict(params))
        if "filters" in params:
            return {"data":[],"message":"No data found"}
        return {"data":[next_match()],"pagination":{"has_more":False}}
    found,reason,meta=_query_pages(original(),NOW,tmp_path,request)
    assert found["fixture_id"]==54321
    assert reason=="source_identity_matched"
    assert len(calls)==2
    assert meta["response_schema"]["message_class"]=="absent"


def test_provider_subscription_message_is_not_treated_as_empty_or_saved(tmp_path):
    secret="SECRET-DO-NOT-LOG"
    body={"data":[],"message":"No data: subscription required "+secret}
    rows,state,reason,shape=_decode_page(body,1)
    assert rows is None and state is None
    assert reason=="provider_access_or_subscription"
    assert shape["message_class"]=="provider_access_or_subscription"
    assert secret not in json.dumps(shape)
    result,why,meta=_query_pages(original(),NOW,tmp_path,
                                  lambda url,params:body)
    assert result is None and why==reason
    assert secret not in json.dumps(meta)
    assert not list(tmp_path.rglob("*.json"))


def test_empty_unclassified_message_fails_closed_without_credential_leak(tmp_path):
    secret="TOP-PRIVATE-TOKEN"
    body={"data":[],"message":"Unexpected provider reply "+secret}
    rows,state,reason,shape=_decode_page(body,1)
    assert rows is None and state is None
    assert reason=="unclassified_provider_message"
    assert secret not in json.dumps(shape)
    match,why,meta=_query_pages(original(),NOW,tmp_path,
                                lambda url,params:body)
    assert match is None and why==reason
    assert meta["response_schema"]["row_count"]==0
    assert secret not in json.dumps(meta)
    assert not list(tmp_path.rglob("*.json"))


def test_rate_limit_and_invalid_request_are_distinct_safe_categories():
    examples=(
        ("Too many requests: token=HIDDEN", "provider_quota_or_rate_limit"),
        ("Invalid filter value: token=HIDDEN", "provider_invalid_request"),
        ("Unauthorized token HIDDEN", "provider_access_or_subscription"),
    )
    for message,category in examples:
        rows,state,reason,shape=_decode_page({"data":[],"message":message},1)
        assert rows is None and state is None
        assert reason==category
        assert shape["message_class"]==category
        assert "HIDDEN" not in json.dumps(shape)
