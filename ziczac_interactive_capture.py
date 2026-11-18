"""Write-once central audit for interactive ZicZac forecasts.

Only generated, user-requested pre-kickoff predictions are saved. A configured
PostgreSQL DATABASE_URL is required for durable website captures; the local
Streamlit session is NOT a durable forecast record. Never store API keys,
prompt text, user identity or billed provider responses.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

SOURCES = {"deep_research","ten_ai","ai_five_double"}
PROVIDERS = {"openai","claude","gemini","grok","deepseek","perplexity",
             "mistral","qwen","groqcloud","cohere"}
_SCORE = re.compile(r"^((?:0|[1-9][0-9]?))-((?:0|[1-9][0-9]?))$")
_TABLE = """CREATE TABLE IF NOT EXISTS ziczac_interactive_forecasts (
 record_id TEXT PRIMARY KEY,
 source TEXT NOT NULL,
 provider TEXT NOT NULL,
 fixture_id TEXT NOT NULL,
 captured_at_utc TEXT NOT NULL,
 kickoff_utc TEXT NOT NULL,
 payload_json TEXT NOT NULL
)"""


def utc(value):
    if isinstance(value,datetime):
        instant=value
    else:
        instant=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if instant.tzinfo is None:
        raise ValueError("A timezone-aware timestamp is required")
    return instant.astimezone(timezone.utc)


def score(value):
    if isinstance(value,(tuple,list)) and len(value)==2 and all(
            type(v) is int and 0<=v<=99 for v in value):
        return f"{value[0]}-{value[1]}"
    if isinstance(value,str) and _SCORE.fullmatch(value):
        return value
    raise ValueError("Score must contain two nonnegative integer goal counts from 0 to 99")


def leg(fixture,now):
    if not isinstance(fixture,dict) or fixture.get("id") in (None,""):
        raise ValueError("A real provider-qualified fixture is required")
    provider=str(fixture.get("provider") or "football-data.org")
    if provider not in ("API-Football","football-data.org"):
        raise ValueError("Unsupported provider")
    kickoff=utc(fixture.get("utcDate"))
    if not now<kickoff:
        raise ValueError("A prediction must be captured before kickoff")
    home,away=fixture.get("homeTeam") or {},fixture.get("awayTeam") or {}
    ids=[home.get("id"),away.get("id")]
    if any(v in (None,"") for v in ids) or str(ids[0])==str(ids[1]):
        raise ValueError("Both distinct provider team IDs are required")
    return {"provider":provider,"fixture_id":str(fixture["id"]),
            "kickoff_utc":kickoff.isoformat(),
            "home_team_id":str(ids[0]),"away_team_id":str(ids[1]),
            "home_team":str(home.get("name") or "")[:120],
            "away_team":str(away.get("name") or "")[:120]}


def _freeze(source,record,now):
    if source not in SOURCES:
        raise ValueError("Unknown audit source")
    if source in ("deep_research","ten_ai"):
        legs=[record["leg"]]
        scores=[record["score"]]
    else:
        legs=record["legs"]
        scores=[x["score1"] for x in record["choices"]]
    if any(utc(item["kickoff_utc"])<=now for item in legs):
        raise ValueError("Prediction generated at or after kickoff")
    fingerprint=[source,[(v["provider"],v["fixture_id"]) for v in legs]]
    if source=="deep_research":
        fingerprint.append(record.get("method"))
    if source=="ten_ai":
        fingerprint.append(record.get("ai_provider"))
    if source=="ai_five_double" and record.get("method"):
        # Preserve existing baseline IDs. Distinct predeclared research
        # methods must not overwrite each other on the same fixture pair.
        fingerprint.append(record["method"])
    # First original per source / fixture (or per declared method) wins.
    key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    return {"record_id":key,"source":source,
            "provider":legs[0]["provider"],"fixture_id":legs[0]["fixture_id"],
            "captured_at_utc":now.isoformat(),**record}


def single(fixture,predicted,source,*,method="",ai_provider="",now=None,evidence_digest=""):
    instant=utc(now or datetime.now(timezone.utc))
    if source not in ("deep_research","ten_ai"):
        raise ValueError("Use pair() for five-double records")
    if source=="ten_ai" and ai_provider not in PROVIDERS:
        raise ValueError("AI provider was not among the ten configured adapters")
    if source=="deep_research" and not method:
        raise ValueError("Research method is required")
    record={"leg":leg(fixture,instant),"score":score(predicted)}
    if evidence_digest:
        if not isinstance(evidence_digest,str) or not re.fullmatch(r"[0-9a-f]{64}",evidence_digest):
            raise ValueError("Real SHA-256 evidence digest required, or omit")
        record["source_evidence_sha256"]=evidence_digest
    if source=="deep_research":
        record["method"]=str(method)[:120]
    else:
        record["ai_provider"]=ai_provider
    return _freeze(source,record,instant)


def pair(first,second,choices,*,now=None,method="",policy_metadata=None):
    instant=utc(now or datetime.now(timezone.utc))
    legs=[leg(first,instant),leg(second,instant)]
    if (legs[0]["provider"],legs[0]["fixture_id"]) == (
            legs[1]["provider"],legs[1]["fixture_id"]):
        raise ValueError("Cannot form a double from one fixture")
    if utc(legs[0]["kickoff_utc"]).astimezone(
            __import__("zoneinfo").ZoneInfo("Australia/Sydney")).date()!=(
            utc(legs[1]["kickoff_utc"]).astimezone(
            __import__("zoneinfo").ZoneInfo("Australia/Sydney")).date()):
        raise ValueError("Doubles must have the same Sydney match date")
    if not isinstance(choices,list) or len(choices)!=5:
        raise ValueError("Exactly five score combinations are required")
    if method not in ("", "balanced_challenger", "goal_form_research", "v22_xg_market", "v23_market_form", "v25_professional_analyst", "professional-probability-five-v1"):
        raise ValueError("Unrecognized five-double research method")
    frozen=[{"score1":score(x["score1"]),"score2":score(x["score2"])}
            for x in choices]
    if len({(x["score1"],x["score2"]) for x in frozen})!=5:
        raise ValueError("Five unique score combinations are required")
    record = {"legs": legs, "choices": frozen}
    if policy_metadata is not None:
        import math
        if method != "professional-probability-five-v1" or not isinstance(policy_metadata,dict):
            raise ValueError("Policy metadata requires the professional optimizer version")
        probability=policy_metadata.get("estimated_coverage")
        versions=policy_metadata.get("model_versions")
        times=policy_metadata.get("source_times_utc")
        scope=policy_metadata.get("selection_scope","manual_same_day_v1")
        if (type(probability) not in (int,float) or not math.isfinite(probability)
                or not 0<probability<=1 or not isinstance(versions,list) or len(versions)!=2
                or any(not isinstance(v,str) or not v or len(v)>200 for v in versions)
                or not isinstance(times,list) or len(times)!=2
                or any(utc(t)>instant for t in times)
                or scope not in ("manual_same_day_v1","auto_max_raw_coverage_same_day_v1",
                                 "premium_max_calibrated_coverage_v1")):
            raise ValueError("Valid pre-match model probability and provenance required")
        record["policy_metadata"]={"estimated_coverage":float(probability),
                                   "model_versions":versions,
                                   "selection_scope":scope,
                                   "source_times_utc":[utc(t).isoformat() for t in times]}
        references=policy_metadata.get("source_forecast_refs")
        if references is not None:
            allowed=("provider","fixture_id","forecast_id","input_digest", "home_team_id",
                     "away_team_id","kickoff_utc")
            if not isinstance(references,dict) or set(references)!={"first","second"}:
                raise ValueError("Two source forecast reference mappings required")
            record["policy_metadata"]["source_forecast_refs"]={
                side:{key:value for key,value in ref.items() if key in allowed}
                for side,ref in references.items() if isinstance(ref,dict)}
    if method:
        record["method"] = method
    return _freeze("ai_five_double",record,instant)


def store(record,database_url=None):
    """Return durable status; never report success without an acknowledged DB write."""
    address=database_url if database_url is not None else os.environ.get("DATABASE_URL","")
    if not address:
        return {"durable":False,"state":"database_not_configured"}
    import psycopg
    try:
        evidence_legs=[record["leg"]] if record.get("leg") else record["legs"]
        deadline=min(utc(item["kickoff_utc"]) for item in evidence_legs)
        with psycopg.connect(address,connect_timeout=7) as conn:
            with conn.cursor() as cursor:
                cursor.execute(_TABLE)
                cursor.execute(
                    """INSERT INTO ziczac_interactive_forecasts
                       (record_id,source,provider,fixture_id,captured_at_utc,
                        kickoff_utc,payload_json)
                       SELECT %s,%s,%s,%s,%s,%s,%s
                       WHERE clock_timestamp() < %s::timestamptz
                       ON CONFLICT DO NOTHING""",
                    (record["record_id"],record["source"],record["provider"],
                     record["fixture_id"],record["captured_at_utc"],
                     (record.get("leg") or record["legs"][0])["kickoff_utc"],
                     json.dumps(record,sort_keys=True,separators=(",",":")),deadline))
                inserted=cursor.rowcount>0
                if not inserted:
                    cursor.execute("""SELECT payload_json FROM ziczac_interactive_forecasts
                                      WHERE record_id=%s""",(record["record_id"],))
                    original=cursor.fetchone()
                    if not original:
                        return {"durable":False,"state":"not_persisted_before_kickoff"}
                return {"durable":True,"state":"saved" if inserted else "original_preserved"}
    except Exception as exc:
        return {"durable":False,"state":"database_unavailable",
                "error_type":type(exc).__name__}


def load(database_url):
    """Read only confirmed central records. An unavailable database is explicit."""
    if not database_url:
        return [],"database_not_configured"
    import psycopg
    try:
        with psycopg.connect(database_url,connect_timeout=7) as conn:
            with conn.cursor() as cursor:
                cursor.execute(_TABLE)
                cursor.execute("""SELECT payload_json FROM ziczac_interactive_forecasts
                                  ORDER BY captured_at_utc,record_id""")
                return [json.loads(row[0]) for row in cursor.fetchall()], "connected"
    except Exception as exc:
        return [],"database_unavailable:"+type(exc).__name__


def store_ui(st,record):
    """Visible, owner-authenticated, truthful persistence status."""
    from on_demand_ai import private_gate_ready
    if not private_gate_ready(st):
        st.warning("Your prediction is visible in this session but NOT "
                   "permanently saved. Enable Private Access with "
                   "ZICZAC_ACCESS_PASSWORD and sign in to allow secure "
                   "central audit writes.")
        return {"durable":False,"state":"private_access_required"}
    try:
        address=st.secrets.get("DATABASE_URL","") or os.environ.get("DATABASE_URL","")
    except (OSError,KeyError,RuntimeError,AttributeError):
        address=os.environ.get("DATABASE_URL","")
    status=store(record,database_url=address)
    if status["durable"]:
        st.caption("🔒 Original pre-match forecast saved permanently to the "
                   "central database." if status["state"]=="saved"
                   else "🔒 The original pre-match forecast is already preserved.")
    else:
        st.warning("This on-demand prediction is NOT centrally saved ("+
                   status["state"]+"). It remains in this session only. "
                   "Configure a working DATABASE_URL in Streamlit Secrets "
                   "before relying on tracked results.")
    return status
