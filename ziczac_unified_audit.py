"""Read-only, source-labelled ZicZac HIT/MISS/PENDING audit.

First frozen pre-kickoff scores stay immutable. Full-time scores must be
independently verified, identity-matched and regulation-time. A GitHub report
contains only the already-public scheduled forecasts; private interactive
captures are merged exclusively inside the Streamlit Performance screen.
"""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from persist_prematch import verified_result
from v10_prospective import read_latest, utc

ROOT = Path("data/tracking")


def _json(path,default):
    try:
        value=json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value,type(default)) else default
    except (OSError,TypeError,ValueError):
        return default


def _goals(value):
    if isinstance(value,(tuple,list)) and len(value)==2 and all(
            type(x) is int and 0<=x<=99 for x in value):
        return f"{value[0]}-{value[1]}"
    if isinstance(value,str):
        parts=value.split("-")
        if len(parts)==2 and all(p.isdigit() and p==str(int(p)) for p in parts):
            return value
    return None


def _leg(row):
    return {"provider":str(row.get("provider") or ""),
            "fixture_id":str(row.get("fixture_id") or ""),
            "kickoff_utc":row.get("kickoff_utc"),
            "home_team_id":str(row.get("home_team_id") or ""),
            "away_team_id":str(row.get("away_team_id") or ""),
            "home_team":row.get("home_team") or "",
            "away_team":row.get("away_team") or ""}


def _prematch(row):
    try:
        return utc(row["captured_at_utc"])<utc(row["kickoff_utc"])
    except (KeyError,ValueError,TypeError):
        return False


def _final_matches(leg,final):
    if not isinstance(final,dict) or final.get("score_basis")!="regulation":
        return False
    if final.get("evidence_version")!=2 and final.get("verification_status")!="verified" and final.get("verified") is not True:
        return False
    if final.get("status") not in ("FT","AET","PEN","FINISHED"):
        return False
    try:
        if not all(str(leg[k])==str(final[k]) for k in (
                "provider","fixture_id","home_team_id","away_team_id")):
            return False
        if utc(leg["kickoff_utc"])!=utc(final["kickoff_utc"]):
            return False
        if utc(final["verified_at_utc"])<=utc(final["kickoff_utc"]):
            return False
        if any(type(final[k]) is not int or final[k]<0 for k in ("home_goals","away_goals")):
            return False
        if leg["home_team_id"]==leg["away_team_id"]:
            return False
    except (KeyError,ValueError,TypeError):
        return False
    return True


def _status(legs,choices,results):
    if not choices or not legs:
        return "invalid",[],[]
    actual=[]
    for leg in legs:
        key=(str(leg["provider"]),str(leg["fixture_id"]))
        final=results.get(key)
        if not _final_matches(leg,final):
            return "pending",[],[]
        actual.append(f"{final['home_goals']}-{final['away_goals']}")
    choices=[tuple(row) for row in choices]
    picked=[all(value==observed for value,observed in zip(choice,actual))
            for choice in choices]
    return "hit" if any(picked) else "miss",actual,picked


def _collect_finals(v10_root,prematch_root,extra=None):
    finals={}
    conflicted=set()
    sources=[_json(Path(prematch_root)/"verified_results.json",{}),
             _json(Path(v10_root)/"verified_results.json",{})]
    if extra:
        sources.append({str(i):row for i,row in enumerate(extra)})
    for source in sources:
        for row in source.values():
            if not isinstance(row,dict):
                continue
            key=(str(row.get("provider") or ""),str(row.get("fixture_id") or ""))
            if not all(key):
                continue
            existing=finals.get(key)
            signature=tuple(row.get(k) for k in (
                "kickoff_utc","home_team_id","away_team_id","home_goals","away_goals"))
            if existing is not None:
                old=tuple(existing.get(k) for k in (
                    "kickoff_utc","home_team_id","away_team_id","home_goals","away_goals"))
                if signature!=old:
                    conflicted.add(key)
            else:
                finals[key]=row
    for key in conflicted:
        finals.pop(key,None)
    return finals,conflicted


def _item(source,key,legs,choices,captured,*,model="",detail="",is_private=False):
    if not legs or not choices or len(choices[0])!=len(legs):
        return None
    if any(_goals(value) is None for pick in choices for value in pick):
        return None
    try:
        observed=utc(captured)
        if any(observed>=utc(leg["kickoff_utc"]) for leg in legs):
            return None
    except (KeyError,TypeError,ValueError):
        return None
    if any(not all(leg.get(k) for k in ("provider","fixture_id",
                                       "home_team_id","away_team_id")) for leg in legs):
        return None
    return {"id":str(key),"source":source,"kind":"double" if len(legs)==2 else "single",
            "model":str(model or "")[:100],"detail":str(detail or "")[:120],
            "captured_at_utc":observed.isoformat(),
            "legs":legs,"choices":choices,"private":is_private}


def scheduled_records(v10_root=Path("data/v10_prospective"),
                      prematch_root=Path("data/prematch"),
                      llm_root=Path("data/llm_research")):
    records=[]
    try:
        frozen=read_latest(v10_root)
    except (OSError,ValueError,TypeError):
        frozen={}
    for (provider,fid,model),row in frozen.items():
        pick=_goals(row.get("predicted_score"))
        if pick is None:
            continue
        record=_item("frozen_"+str(model),(provider,fid,model),
                     [_leg(row)],[[pick]],row.get("captured_at_utc"),model=model)
        if record:
            records.append(record)
    # Earliest saved automatic website prediction per fixture, never a later
    # revised screenshot or post-result score.
    first={}
    for path in Path(prematch_root).glob("snapshots/*/*.json"):
        saved=_json(path,{})
        if not _prematch(saved):
            continue
        key=str(saved.get("fixture_id") or "")
        old=first.get(key)
        if key and (old is None or utc(saved["captured_at_utc"])<utc(old["captured_at_utc"])):
            first[key]=saved
    for fid,row in first.items():
        pick=_goals(row.get("predicted_score"))
        if pick:
            record=_item("website_original",("website_original",fid),
                         [_leg({**row,"provider":row.get("provider") or "API-Football"})],
                         [[pick]],row.get("captured_at_utc"),
                         model=row.get("forecast_model"))
            if record:
                records.append(record)
    # First immutable six-/ten-provider research panel for each fixture.
    ai_panels=[]
    for path in Path(llm_root).glob("forecasts/*.json"):
        saved=_json(path,{})
        if saved.get("research_only") is True and _prematch(saved):
            ai_panels.append(saved)
    ai_panels.sort(key=lambda x:utc(x["captured_at_utc"]))
    seen=set()
    for panel in ai_panels:
        leg=_leg(panel)
        for ai,vote in sorted((panel.get("votes") or {}).items()):
            ident=(leg["provider"],leg["fixture_id"],str(ai))
            if ident in seen:
                continue
            if not isinstance(vote,dict):
                continue
            pick=_goals(vote.get("score"))
            if not pick:
                continue
            record=_item("ai_"+str(ai),("ai",*ident),[leg],[[pick]],
                         panel["captured_at_utc"],model=(panel.get("models") or {}).get(ai),
                         detail="scheduled_verified_evidence")
            if record:
                records.append(record)
                seen.add(ident)
    for path in Path(v10_root).glob("doubles/*.json"):
        saved=_json(path,{})
        if saved.get("mode")!="prospective" or saved.get("status")!="frozen_pre_kickoff":
            continue
        legs=[_leg(saved.get("first") or {}),_leg(saved.get("second") or {})]
        choices=[[row.get("score1"),row.get("score2")]
                 for row in (saved.get("five_doubles") or [])]
        if len(choices)!=5:
            continue
        record=_item("automatic_five_double",("automatic_double",path.stem),
                     legs,choices,saved.get("captured_at_utc"),model="v8")
        if record:
            records.append(record)
    return records


def private_records(rows):
    output=[]
    for record in rows:
        if not isinstance(record,dict):
            continue
        source=record.get("source")
        if source in ("deep_research","ten_ai"):
            leg=record.get("leg") or {}
            pick=_goals(record.get("score"))
            if not pick:
                continue
            model=(record.get("ai_provider") if source=="ten_ai"
                   else record.get("method"))
            kind=("ai_"+str(model)) if source=="ten_ai" else "deep_research"
            entry=_item(kind,record.get("record_id"),[leg],[[pick]],
                        record.get("captured_at_utc"),model=model,
                        detail="website_on_demand",is_private=True)
        elif source=="ai_five_double":
            legs=record.get("legs") or []
            choices=[[row.get("score1"),row.get("score2")]
                     for row in (record.get("choices") or [])]
            if len(choices)!=5:
                continue
            method = record.get("method") or "probability"
            if method not in ("probability", "balanced_challenger", "goal_form_research"):
                continue
            source_name = ("ai_five_double" if method == "probability" else
                           "ai_five_double_" + method)
            entry=_item(source_name,record.get("record_id"),legs,choices,
                        record.get("captured_at_utc"),model=method,
                        detail="website_on_demand",is_private=True)
        else:
            continue
        if entry:
            output.append(entry)
    return output


def archived_central_records(forecasts=(),singles=(),doubles=()):
    """Normalize originals from the existing durable forecast ledger.

    Pre-migration manual pairs without real provider and team identities are
    retained in the source archive, but not counted as verified losses.
    """
    rows=[]
    missing_identity=0
    for row in forecasts:
        pick=_goals(row.get("predicted_score"))
        if not pick:
            continue
        record=_item("website_model_ledger",row.get("forecast_id") or
                     ("ledger",row.get("provider"),row.get("fixture_id"),
                      row.get("model_version"),row.get("captured_at_utc")),
                     [_leg(row)],[[pick]],row.get("captured_at_utc"),
                     model=row.get("model_version"),is_private=True)
        if record:
            rows.append(record)
    for saved in singles:
        leg=_leg({
            "provider":saved.get("provider"),"fixture_id":saved.get("fixture_id"),
            "home_team_id":saved.get("home_id"),"away_team_id":saved.get("away_id"),
            "home_team":saved.get("home"),"away_team":saved.get("away"),
            "kickoff_utc":saved.get("kickoff")})
        pick=_goals(saved.get("predicted_score"))
        if not pick:
            continue
        record=_item("website_saved_original",
                     ("website",leg["provider"],leg["fixture_id"],saved.get("saved_at")),
                     [leg],[[pick]],saved.get("saved_at"),
                     model=saved.get("model_version"),is_private=True)
        if record:
            rows.append(record)
        else:
            missing_identity+=1
    for saved in doubles:
        legs=[]
        for name in ("first","second"):
            legs.append(_leg({
                "provider":saved.get(name+"_provider"),
                "fixture_id":saved.get(name+"_id"),
                "kickoff_utc":saved.get(name+"_kickoff"),
                "home_team_id":saved.get(name+"_home_id"),
                "away_team_id":saved.get(name+"_away_id"),
                "home_team":saved.get(name+"_name",""),
                "away_team":""
            }))
        try:
            saved_choices=json.loads(saved["predictions_json"])
        except (KeyError,ValueError,TypeError):
            continue
        picks=[[entry.get("score1"),entry.get("score2")]
               for entry in saved_choices]
        if len(picks)!=5:
            continue
        record=_item("website_manual_five_double",saved.get("snapshot_key"),
                     legs,picks,saved.get("saved_at"),is_private=True)
        if record:
            rows.append(record)
        else:
            missing_identity+=1
    return rows,missing_identity


def central_sources(database_url):
    """Do not copy private website picks into GitHub's public data directory."""
    from ziczac_interactive_capture import load as load_interactive
    private,status=load_interactive(database_url)
    if not database_url:
        return [],[],"database_not_configured",0,[]
    try:
        from forecast_ledger import open_ledger
        with open_ledger(database_url) as store:
            records,legacy=archived_central_records(
                forecasts=store.iter_forecasts(),
                singles=store.iter_archive_records("single"),
                doubles=store.iter_archive_records("double"))
            finals=store.get_settlements()
        return private,records,status,legacy,finals
    except Exception as exc:
        return private,[],"ledger_unavailable:"+type(exc).__name__,0,[]


def summary(records,finals,*,now=None,private_connection="not_loaded",conflicts=()):
    clock=utc(now or datetime.now(timezone.utc))
    graded=[]
    for record in records:
        status,actual,per_option=_status(record["legs"],record["choices"],finals)
        graded.append({**record,"status":status,"actual_scores":actual,
                       "option_hits":per_option,
                       "scheduled_kickoff_utc":min(leg["kickoff_utc"]
                                                   for leg in record["legs"])})
    groups={}
    for name in sorted({r["source"] for r in graded}):
        cohort=[r for r in graded if r["source"]==name]
        totals=Counter(r["status"] for r in cohort)
        checked=totals["hit"]+totals["miss"]
        groups[name]={"recorded":len(cohort),"verified":checked,
                      "hit":totals["hit"],"miss":totals["miss"],
                      "pending":totals["pending"],
                      "hit_rate":totals["hit"]/checked if checked else None}
    singles=[r for r in graded if r["kind"]=="single"]
    doubles=[r for r in graded if r["kind"]=="double"]
    def totals(items):
        count=Counter(r["status"] for r in items)
        n=count["hit"]+count["miss"]
        return {"recorded":len(items),"verified":n,"hit":count["hit"],
                "miss":count["miss"],"pending":count["pending"],
                "hit_rate":count["hit"]/n if n else None}
    # The public GitHub report only publishes scheduled records; private
    # Postgres details are rendered on the password-protected website.
    ordered=sorted(graded,key=lambda r:r["scheduled_kickoff_utc"],reverse=True)
    return {"schema_version":1,"checked_at_utc":clock.isoformat(),
            "policy":"first_saved_pre_kickoff_only_verified_regulation_time",
            "singles":totals(singles),"five_doubles":totals(doubles),
            "by_source":groups,"records":ordered,
            "private_database_status":private_connection,
            "identity_conflicts":len(conflicts),
            "note":"Pending results are neither hits nor misses. AI providers are "
                   "tracked individually; no guarantee of exact-score accuracy."}


def build(*,v10_root=Path("data/v10_prospective"),
          prematch_root=Path("data/prematch"),llm_root=Path("data/llm_research"),
          private=(),central=(),extra_finals=(),now=None,private_connection="not_loaded",
          legacy_unverifiable=0):
    finals,conflicts=_collect_finals(v10_root,prematch_root,extra_finals)
    records=scheduled_records(v10_root,prematch_root,llm_root)
    records.extend(private_records(private))
    records.extend(central)
    result=summary(records,finals,now=now,
                   private_connection=private_connection,conflicts=conflicts)
    result["legacy_records_without_verifiable_identity"]=legacy_unverifiable
    return result


def build_with_private(database_url,**kwargs):
    private,central,status,legacy,finals=central_sources(database_url)
    return build(private=private,central=central,extra_finals=finals,
                 private_connection=status,
                 legacy_unverifiable=legacy,**kwargs)


def publish(root=ROOT):
    report=build()
    path=Path(root)/"status.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                    encoding="utf-8")
    return report


if __name__=="__main__":
    report=publish()
    print(json.dumps({"checked_at_utc":report["checked_at_utc"],
                      "singles":report["singles"],
                      "five_doubles":report["five_doubles"],
                      "sources":list(report["by_source"])},indent=2))
