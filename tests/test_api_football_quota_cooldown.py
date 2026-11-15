"""Do not repeatedly hit a football API after a 429 or waste date-query quota."""
import pytest

import api_football as api


def test_date_scoped_fixture_cache_is_longer_than_five_minutes():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today=datetime.now(ZoneInfo("Australia/Sydney")).date()
    future=today.fromordinal(today.toordinal()+1)
    assert api.cache_ttl("/fixtures",{"date":today.isoformat(),
                                     "timezone":"Australia/Sydney"}) == 900
    assert api.cache_ttl("/fixtures",{"date":future.isoformat(),
                                     "timezone":"Australia/Sydney"}) == 3600


def test_quota_429_pauses_further_network_requests_for_same_key(tmp_path,monkeypatch):
    monkeypatch.setattr(api,"CACHE",tmp_path)
    monkeypatch.setattr(api,"_QUOTA_BLOCK_UNTIL",{})
    calls=[]
    class Response:
        status_code=429
    def reject(url,**kwargs):
        calls.append(url)
        return Response()
    monkeypatch.setattr(api.requests,"get",reject)
    with pytest.raises(api.ProviderQuotaError):
        api._request("test-key-nonsecret","/fixtures",
                     {"date":"2030-11-15","timezone":"Australia/Sydney"})
    assert len(calls)==1
    with pytest.raises(api.ProviderQuotaError):
        api._request("test-key-nonsecret","/fixtures",
                     {"date":"2030-11-16","timezone":"Australia/Sydney"})
    # The in-process account circuit breaker stops repeated rejected calls.
    assert len(calls)==1
    assert len(api._QUOTA_BLOCK_UNTIL)==1
    assert "test-key-nonsecret" not in str(api._QUOTA_BLOCK_UNTIL)


def test_different_api_keys_not_blocked_by_anothers_quota(tmp_path,monkeypatch):
    monkeypatch.setattr(api,"CACHE",tmp_path)
    monkeypatch.setattr(api,"_QUOTA_BLOCK_UNTIL",{})
    calls=[]
    class Response:
        status_code=429
    def reject(url,**kwargs):
        calls.append(url)
        return Response()
    monkeypatch.setattr(api.requests,"get",reject)
    for key in ("account-one","account-two"):
        with pytest.raises(api.ProviderQuotaError):
            api._request(key,"/fixtures",{"date":"2030-11-15"})
    assert len(calls)==2


def test_recent_cached_data_still_works_during_quota_cooldown(tmp_path,monkeypatch):
    import time, hashlib, json
    monkeypatch.setattr(api,"CACHE",tmp_path)
    monkeypatch.setattr(api,"_QUOTA_BLOCK_UNTIL",{})
    key="test-cached-key"
    params={"date":"2030-11-15"}
    signature=json.dumps(["/fixtures",params,hashlib.sha256(key.encode()).hexdigest()],
                         sort_keys=True)
    path=tmp_path/(hashlib.sha256(signature.encode()).hexdigest()+".json")
    path.write_text(json.dumps({"time":time.time(),"data":[{"fixture":{"id":77}}]}))
    def forbid(*args,**kwargs):
        raise AssertionError("The cached fixture query must not hit the network")
    monkeypatch.setattr(api.requests,"get",forbid)
    api._QUOTA_BLOCK_UNTIL[hashlib.sha256(key.encode()).hexdigest()]=time.time()+900
    assert api._request(key,"/fixtures",params)==[{"fixture":{"id":77}}]
