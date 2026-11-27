"""Record source freshness without changing any registered forecast study.

The adapter's timestamps are provenance assertions, not independent proof
that a provider's underlying data were updated. Unknown, stale, incomplete
and inconsistent metadata are reported separately; no silent fresh label.
Never persist raw responses, provider credentials, or arbitrary metadata.
"""
from datetime import timedelta
import math
from history_pipeline import season_year
from v10_prospective import utc
import v44_early_history as early

VERSION="recovery-source-freshness-v1"
# Restrict the age of even a previously cached response when assessing
# current-season history. Prior-season historical results can be much older,
# so report their age against the provider-supplied TTL rather than inventing
# a freshness guarantee.
CURRENT_MAX_AGE=timedelta(hours=24)

def inspect(fixture,evidence,observed):
    observed=utc(observed)
    current=season_year(fixture)
    provider=str(fixture["provider"])
    expected=(current,current-1)
    sources=(evidence.get("provenance") or {}).get("sources") or []
    rows=[]
    if not isinstance(sources,list):
        sources=[]
    for season in expected:
        matches=[s for s in sources if isinstance(s,dict)
                 and type(s.get("season")) is int and s["season"]==season
                 and s.get("provider")==provider]
        row={"season":season,"status":"missing_metadata"}
        if len(matches)>1:
            row["status"]="duplicate_metadata"
        elif matches:
            s=matches[0]
            try:
                if s.get("freshness_known") is not True:
                    row["status"]="unknown_freshness"
                else:
                    fetched,retrieved=utc(s["fetched_at_utc"]),utc(s["retrieved_at_utc"])
                    ttl=s.get("ttl_seconds")
                    age=(observed-fetched).total_seconds()
                    reported=s.get("age_seconds")
                    if (not fetched<=retrieved<=observed or type(ttl) not in (int,float)
                            or not math.isfinite(ttl) or ttl<=0
                            or type(reported) not in (int,float) or not math.isfinite(reported)
                            or reported<0
                            or abs(reported-(retrieved-fetched).total_seconds())>2):
                        raise ValueError("invalid_metadata")
                    stale=s.get("stale")
                    if type(stale) is not bool:
                        raise ValueError("invalid_stale_flag")
                    row.update({"fetched_at_utc":fetched.isoformat(),
                                "retrieved_at_utc":retrieved.isoformat(),
                                "age_at_observation_seconds":round(age,3),
                                "ttl_seconds":ttl})
                    if stale or s.get("cache_status")=="stale_fallback":
                        row["status"]="stale_fallback"
                    elif age>ttl:
                        row["status"]="expired_adapter_ttl"
                    elif season==current and observed-fetched>CURRENT_MAX_AGE:
                        row["status"]="current_season_older_than_24h"
                    elif s.get("source_season_mismatches",0)!=0:
                        row["status"]="source_season_mismatch"
                    else:
                        row["status"]="adapter_fresh"
            except (ValueError,TypeError,KeyError,AttributeError,ArithmeticError):
                row={"season":season,"status":"inconsistent_metadata"}
        rows.append(row)
    unexpected=sum(not isinstance(s,dict) or s.get("provider")!=provider
                   or type(s.get("season")) is not int or s["season"] not in expected
                   for s in sources)
    status="adapter_fresh" if not unexpected and all(r["status"]=="adapter_fresh" for r in rows) else "not_fully_verified_fresh"
    result={"version":VERSION,"event":early._event(fixture),"input_digest":evidence["input_digest"],
            "observed_at_utc":observed.isoformat(),"status":status,"seasons":rows,
            "unexpected_source_count":unexpected,
            "provider_update_independently_verified":False}
    result["sha256"]=early.digest(result)
    return result
