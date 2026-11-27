from datetime import datetime,timedelta,timezone
import copy
import v44_recovery_source_freshness as f
import v44_early_history as early

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
FIXTURE={"provider":"API-Football","id":1,"utcDate":(NOW+timedelta(hours=2)).isoformat(),
         "homeTeam":{"id":10},"awayTeam":{"id":20},
         "competition":{"id":39,"season":2024,"code":"PL"}}

def packet():
    rows=[{"provider":"API-Football","season":s,
           "fetched_at_utc":(NOW-timedelta(minutes=30)).isoformat(),
           "retrieved_at_utc":(NOW-timedelta(minutes=20)).isoformat(),
           "age_seconds":600,"ttl_seconds":3600,"freshness_known":True,
           "stale":False,"cache_status":"hit","source_season_mismatches":0}
          for s in (2024,2023)]
    return {"input_digest":"exact-input","provenance":{"sources":rows}}

def test_two_seasons_and_hash_are_bound_to_actual_input():
    p=packet();r=f.inspect(FIXTURE,p,NOW)
    assert r["status"]=="adapter_fresh"
    assert r["seasons"][0]["age_at_observation_seconds"]==1800
    assert r["sha256"]==early.digest({k:v for k,v in r.items() if k!="sha256"})
    assert r["provider_update_independently_verified"] is False
    assert r["input_digest"]=="exact-input"

def test_stale_expired_and_missing_metadata_are_distinct():
    p=packet();p["provenance"]["sources"][0]["stale"]=True
    assert f.inspect(FIXTURE,p,NOW)["seasons"][0]["status"]=="stale_fallback"
    p=packet();p["provenance"]["sources"][0]["ttl_seconds"]=900
    assert f.inspect(FIXTURE,p,NOW)["seasons"][0]["status"]=="expired_adapter_ttl"
    p=packet();p["provenance"]["sources"].pop()
    assert f.inspect(FIXTURE,p,NOW)["seasons"][1]["status"]=="missing_metadata"
    assert f.inspect(FIXTURE,{"input_digest":"a"},NOW)["status"]=="not_fully_verified_fresh"

def test_future_and_inconsistent_timestamps_fail_closed():
    p=packet();p["provenance"]["sources"][0]["retrieved_at_utc"]=(NOW+timedelta(seconds=1)).isoformat()
    assert f.inspect(FIXTURE,p,NOW)["seasons"][0]["status"]=="inconsistent_metadata"
    p=packet();p["provenance"]["sources"][0]["age_seconds"]=0
    assert f.inspect(FIXTURE,p,NOW)["seasons"][0]["status"]=="inconsistent_metadata"

def test_no_arbitrary_metadata_or_secret_fields_persisted():
    p=packet();p["provenance"]["sources"][0].update({"api_key":"private","endpoint":"private","raw_response":"private"})
    assert "private" not in str(f.inspect(FIXTURE,p,NOW))
    p["provenance"]["sources"].append(copy.deepcopy(p["provenance"]["sources"][0]))
    assert f.inspect(FIXTURE,p,NOW)["seasons"][0]["status"]=="duplicate_metadata"

def test_unknown_freshness_and_wrong_provider():
    p=packet();p["provenance"]["sources"][0]["freshness_known"]=False
    assert f.inspect(FIXTURE,p,NOW)["seasons"][0]["status"]=="unknown_freshness"
    p=packet();p["provenance"]["sources"][0]["provider"]="other"
    r=f.inspect(FIXTURE,p,NOW)
    assert r["unexpected_source_count"]==1 and r["status"]=="not_fully_verified_fresh"
