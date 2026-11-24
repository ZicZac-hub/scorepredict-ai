"""Quota-bounded scheduled V21 research capture for ORIGINAL frozen forecasts.

The owner-enabled GitHub workflow runs every scheduled/dispatch pass, whether
or not Actions variables were previously configured. No calls on code pushes.
Capture source evidence strictly before kickoff, preserve originals, perform at
most one near-kickoff refresh, and keep a persisted daily attempt budget.
Missing API evidence is recorded as missing rather than invented.
"""
import json
import math
import os
from datetime import datetime,timezone,timedelta
from pathlib import Path

from v10_prospective import ROOT as ORIGINAL_ROOT, utc
from v16_pair_research import original_model
from history_pipeline import load_evidence
from v21_provider_evidence import capture,valid_snapshot
from v21_prospective import rank_pairs,_models
from persist_prematch import load_json,write_json
from source_capture_clock import capture_clock

ROOT=Path("data/v21_sources")
NEAR_REFRESH_MAX_MINUTES=65
SOURCE_FRESHNESS_HOURS=3
PAIR_ROOT=Path("data/professional_double_shadow")


def targeted_pair_models(*, source_root, pair_root=PAIR_ROOT, now):
    """Prioritize one authenticated fixed pair before its common cutoff.

    The later leg may not yet be in its own near-kickoff window. Observing
    BOTH legs before the earlier match starts is necessary for a prospective
    same-pair comparison. This changes priority, never the attempt budgets.
    """
    from professional_double_shadow import _validated
    clock=utc(now)
    choices=[]
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            record=load_json(path,{})
            models,_,_=_validated(record,source_root=source_root,now=clock)
            if record["sydney_date"]!=path.stem:
                continue
            kickoffs=[utc(model["kickoff_utc"]) for model in models]
            if (all(model["provider"]=="API-Football" for model in models)
                    and clock+timedelta(minutes=8)<min(kickoffs)<=
                        clock+timedelta(minutes=NEAR_REFRESH_MAX_MINUTES)
                    and max(kickoffs)<=clock+timedelta(hours=SOURCE_FRESHNESS_HOURS)):
                choices.append((min(kickoffs),path.stem,models))
        except (KeyError,TypeError,ValueError,OSError,AttributeError,OverflowError):
            continue
    return min(choices,key=lambda item:(item[0],item[1]))[2] if choices else []


def source_availability(leg, snapshot, *, now):
    """Describe usable source coverage without relying on list lengths.

    Called only for an identity/hash-qualified packet. A dozen rows for one
    team, duplicate XIs or stale markets do not establish complete evidence.
    Missing injury reports are deliberately not treated as a completeness test.
    """
    clock=utc(now)
    captured=None
    try:
        captured=utc(snapshot["captured_at_utc"])
    except (KeyError,TypeError,ValueError):
        pass
    ids={str(leg["home_team_id"]),str(leg["away_team_id"])}
    xg={team:set() for team in ids}
    for row in snapshot.get("verified_historical_fixture_xg") or []:
        try:
            if not isinstance(row,dict) or captured is None:
                continue
            stamp=utc(row["kickoff_utc"])
            fixture_id=str(row["fixture_id"])
            home,away=str(row["home_team_id"]),str(row["away_team_id"])
            values=(row["home_xg"],row["away_xg"])
            if (not fixture_id or home==away or stamp+timedelta(hours=3)>captured
                    or stamp+timedelta(hours=3)>clock
                    or any(type(v) not in (int,float) or not math.isfinite(v)
                           or not 0<=v<=8 for v in values)):
                continue
            for team in ids.intersection((home,away)):
                xg[team].add(fixture_id)
        except (KeyError,TypeError,ValueError):
            continue
    xi={}
    xi_conflicts=set()
    for row in snapshot.get("confirmed_starting_xi") or []:
        if not isinstance(row,dict):
            continue
        team=str(row.get("team_id"))
        names=row.get("starting_xi")
        if (team in ids and isinstance(names,list) and len(names)==11
                and all(isinstance(name,str) and name.strip() for name in names)
                and len({" ".join(name.casefold().split()) for name in names})==11):
            item=(tuple(names),row.get("formation"))
            if team in xi and xi[team]!=item:
                xi_conflicts.add(team)
            else:
                xi[team]=item
    xi_teams=set(xi)-xi_conflicts
    markets=snapshot.get("multi_market") or {}
    fresh=[]
    for kind in ("1x2","over_2_5","over_3_5"):
        try:
            market=markets[kind]
            stamp=utc(market["last_source_update_utc"])
            keys=("home","draw","away") if kind=="1x2" else ("yes","no")
            probabilities=market["probabilities"]
            values=[probabilities[key] for key in keys]
            normalized=(all(type(value) in (int,float) and math.isfinite(value)
                            and 0<=value<=1 for value in values)
                        and math.isclose(sum(values),1.0,rel_tol=0,abs_tol=1e-5))
            if (normalized and captured is not None and stamp<=captured<=clock
                    and clock-stamp<=timedelta(hours=12)):
                fresh.append(kind)
        except (KeyError,TypeError,ValueError,AttributeError):
            continue
    recent=bool(captured is not None and timedelta(0)<=clock-captured<=
                timedelta(hours=SOURCE_FRESHNESS_HOURS))
    counts={team:len(rows) for team,rows in sorted(xg.items())}
    market_ready="1x2" in fresh and bool({"over_2_5","over_3_5"}.intersection(fresh))
    reasons=[]
    if not recent:
        reasons.append("source_not_observed_within_3_hours")
    if min(counts.values(),default=0)<6:
        reasons.append("fewer_than_6_genuine_xg_fixtures_for_either_team")
    if xi_teams!=ids:
        reasons.append("both_identity_matched_starting_XIs_unavailable")
    if not market_ready:
        reasons.append("fresh_1x2_and_goal_total_markets_unavailable")
    return {"source_captured_at_utc":captured.isoformat() if captured else None,
            "source_observed_within_3_hours":recent,"xg_fixtures_by_team":counts,
            "confirmed_xi_team_ids":sorted(xi_teams),
            "conflicting_xi_team_ids":sorted(xi_conflicts),"fresh_market_types":fresh,
            "refresh_reasons":reasons,
            "lineups_have_zero_numerical_weight":True}


def source_packet_path(root,leg):
    def safe(v):
        value=str(v)
        if not value or len(value)>120 or any(c not in
             "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-." for c in value) or value in (".",".."):
            raise ValueError("Unsafe source identifier")
        return value
    return Path(root)/(safe(leg["provider"])+"_"+safe(leg["fixture_id"])+"_"+
                      safe(leg["forecast_id"])+".json")


def qualified_packet(record,leg,now=None):
    if not isinstance(record,dict):
        return False
    source=record.get("source_forecast") or {}
    keys=("provider","fixture_id","forecast_id","input_digest","home_team_id",
          "away_team_id","kickoff_utc")
    if any(str(source.get(k))!=str(leg.get(k)) for k in keys):
        return False
    fixture={"provider":source["provider"],"id":source["fixture_id"],
             "utcDate":source["kickoff_utc"],
             "homeTeam":{"id":source["home_team_id"]},
             "awayTeam":{"id":source["away_team_id"]}}
    return valid_snapshot(record.get("provider_snapshot"),fixture,now=now)



def confirmed_provider_xi_ids(leg, snapshot, *, now):
    """Require genuine provider-reported, unique, nonconflicting starting XIs."""
    filtered = dict(snapshot)
    filtered["confirmed_starting_xi"] = [
        row for row in snapshot.get("confirmed_starting_xi") or []
        if isinstance(row, dict) and row.get("provider_reported") is True
    ]
    return source_availability(leg, filtered, now=now)["confirmed_xi_team_ids"]


def best_packet(root, leg, *, now=None, priority="research"):
    """Best currently valid write-once packet for a NEW forecast.

    First frozen forecasts retain their own source hash; selecting stronger
    pre-match evidence for a new forecast never rewrites those outcomes.
    Priority uses only source availability and timestamps, NEVER match results.
    """
    root=Path(root)
    base=source_packet_path(root,leg)
    paths=[base]
    paths.extend(sorted((root/"versions").glob(base.stem+"_*.json")))
    valid=[]
    for path in paths:
        record=load_json(path,{})
        if not qualified_packet(record,leg,now=now):
            continue
        snap=record["provider_snapshot"]
        xg=snap.get("verified_historical_fixture_xg") or []
        ids=(str(leg["home_team_id"]),str(leg["away_team_id"]))
        counts=[len({str(r.get("fixture_id")) for r in xg
                     if team_id in (str(r.get("home_team_id")),
                                    str(r.get("away_team_id")))})
                for team_id in ids]
        markets=snap.get("multi_market") or {}
        nmarkets=len(markets)
        valid_market=False
        latest_market=""
        try:
            clock=utc(now or datetime.now(timezone.utc))
            captured=utc(snap["captured_at_utc"])
            valid_times={}
            for kind in ("1x2","over_2_5","over_3_5"):
                try:
                    stamp=utc(markets[kind]["last_source_update_utc"])
                    if stamp<=captured<=clock and clock-stamp<=timedelta(hours=12):
                        valid_times[kind]=stamp
                except (KeyError,TypeError,ValueError,AttributeError):
                    continue
            valid_market=("1x2" in valid_times and
                          bool({"over_2_5","over_3_5"} & set(valid_times)))
            if valid_market:
                latest_market=max(t.isoformat() for t in valid_times.values())
        except (KeyError,TypeError,ValueError):
            valid_market=False
        if priority=="lineup":
            clock=utc(now or datetime.now(timezone.utc))
            xi=confirmed_provider_xi_ids(leg,snap,now=clock)
            quality=(len(xi),utc(snap["captured_at_utc"]),int(valid_market))
        elif priority=="market":
            quality=(int(valid_market),latest_market,nmarkets,
                     min(counts),str(snap.get("captured_at_utc") or ""))
        else:
            quality=(int(min(counts)>=6 and valid_market),
                     min(counts),int(valid_market),nmarkets,
                     str(snap.get("captured_at_utc") or ""))
        valid.append((quality,record))
    return max(valid,key=lambda x:x[0])[1] if valid else None


def current_pair(root=ORIGINAL_ROOT,now=None):
    """Identify frozen original legs 8 minutes to 48 hours pre-kickoff.

    A pair's two matches need not kick off simultaneously. A later match
    remains eligible after the earlier one has started, but it must itself
    still be pre-match and match the ORIGINAL frozen source.
    """
    clock=utc(now or datetime.now(timezone.utc))
    options=[]
    seen=set()
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        pair=load_json(path,{})
        try:
            if pair.get("status")!="frozen_pre_kickoff" or utc(pair["captured_at_utc"])>clock:
                continue
            for leg in (pair["first"],pair["second"]):
                kickoff=utc(leg["kickoff_utc"])
                if not clock+timedelta(minutes=8)<kickoff<=clock+timedelta(hours=48):
                    continue
                key=(str(leg["provider"]),str(leg["fixture_id"]))
                if key in seen:
                    continue
                model=original_model(root,leg)
                if model is not None and model.get("input_digest")==leg.get("input_digest"):
                    seen.add(key)
                    options.append((kickoff,model))
        except (KeyError,TypeError,ValueError):
            continue
    # Include individual frozen V8 fixtures even when no double was frozen.
    from v10_prospective import read_latest
    rows=read_latest(root)
    for leg in rows.values() if isinstance(rows,dict) else rows:
        try:
            if (leg.get("model_key") not in ("v8","v8_near") or
                    leg.get("provider")!="API-Football"):
                continue
            kickoff=utc(leg["kickoff_utc"])
            if not clock+timedelta(minutes=8)<kickoff<=clock+timedelta(hours=48):
                continue
            key=(str(leg["provider"]),str(leg["fixture_id"]))
            if key in seen:
                continue
            model=original_model(root,leg)
            if model is not None and model.get("input_digest")==leg.get("input_digest"):
                seen.add(key)
                options.append((kickoff,model))
        except (KeyError,TypeError,ValueError):
            continue
    # Research tomorrow's games earlier, while prioritising imminent lineups.
    options.sort(key=lambda x:(
        0 if x[0]<=clock+timedelta(minutes=90) else 1,
        x[0],str(x[1]["fixture_id"])))
    return [x[1] for x in options]


def capture_pair(*,root=ROOT,source_root=ORIGINAL_ROOT,now=None,
                 api_key=None,enabled=None,api_request=None,clock=None,
                 pair_root=PAIR_ROOT,archive_root=Path("data/verified_xg_replay")):
    clock,observation_time=capture_clock(now,clock)
    active=(os.environ.get("V21_EVIDENCE_ENABLED")=="true" if enabled is None else bool(enabled))
    result={"at_utc":clock.isoformat(),"enabled":active,
            "new_packets":[],"calls":0,"skipped":[],
            "capture_window_minutes":[8,2880],"attempted_legs":0,
            "near_refresh_window_minutes":[8,NEAR_REFRESH_MAX_MINUTES],
            "maximum_near_refresh_attempts_per_original":1,
            "source_availability":{},"targeted_pair_fixture_ids":[]}
    root=Path(root)
    key=api_key if api_key is not None else os.environ.get("API_FOOTBALL_KEY","")
    # A daily cap on *attempts*, not just successfully saved packets.
    # load_evidence may also call the provider; this is not an HTTP request cap.
    try:
        daily_limit=int(os.environ.get("V21_DAILY_CAPTURE_ATTEMPTS") or "6")
    except (TypeError,ValueError):
        daily_limit=6
    daily_limit=max(0,min(24,daily_limit))
    usage_path=root/"usage"/(clock.strftime("%Y-%m-%d")+".json")
    used=max(0,int(load_json(usage_path,{}).get("attempts",0)))
    result["daily_attempts_limit"]=daily_limit
    result["daily_attempts_used"]=used

    def finish(reason):
        result["reason"]=reason
        result["completed_at_utc"]=observation_time().isoformat()
        result["daily_attempts_used"]=used
        write_json(root/"status.json",result)
        return result

    if not active:
        return finish("collection_disabled")
    if not key:
        return finish("API_FOOTBALL_KEY_missing")
    from api_football import _request,LEAGUES
    cb=api_request or _request
    models=current_pair(source_root,clock)
    targeted=targeted_pair_models(source_root=source_root,pair_root=pair_root,now=clock)
    targets={(str(row["provider"]),str(row["fixture_id"]),str(row["forecast_id"]))
             for row in targeted}
    if targeted:
        identities={(str(row["provider"]),str(row["fixture_id"])) for row in targeted}
        models=list(targeted)+[row for row in models if
            (str(row["provider"]),str(row["fixture_id"])) not in identities]
        result["targeted_pair_fixture_ids"]=[str(row["fixture_id"]) for row in targeted]
    result["eligible_legs"]=len(models)
    if not models:
        return finish("no_original_frozen_leg_in_8_min_to_48_hour_window")
    cache={}
    for leg in models:
        try:
            clock=observation_time()
            path=source_packet_path(root,leg)
            kickoff=utc(leg["kickoff_utc"])
            if not clock+timedelta(minutes=8)<kickoff<=clock+timedelta(hours=48):
                result["skipped"].append(str(leg["fixture_id"])+":outside_capture_window_at_attempt")
                continue
            previous=best_packet(root,leg,now=clock)
            existing=(previous or {}).get("provider_snapshot") or {}
            availability=source_availability(leg,existing,now=clock)
            result["source_availability"][str(leg["fixture_id"])]=availability
            # Original packets are write-once. When collected early, allow one
            # additional near-kickoff pass to capture announced starting XIs.
            # Do not poll the same match 15 minutes all day.
            late_window=(clock+timedelta(minutes=8)<kickoff<=
                         clock+timedelta(minutes=NEAR_REFRESH_MAX_MINUTES))
            target=(str(leg["provider"]),str(leg["fixture_id"]),str(leg["forecast_id"])) in targets
            refresh_needed=bool(path.exists() and (late_window or target) and
                                availability["refresh_reasons"])
            # An early fixed-pair refresh must not consume the later XI refresh.
            # Preserve legacy markers for the near-kickoff phase conservatively.
            phase = "refresh_attempts" if late_window else "early_target_refresh_attempts"
            marker=root/phase/path.name
            if path.exists() and not refresh_needed:
                result["skipped"].append(str(leg["fixture_id"])+":original_preserved")
                continue
            if refresh_needed and marker.exists():
                result["skipped"].append(str(leg["fixture_id"])+":refresh_already_attempted")
                continue
            comp=str(leg.get("competition") or "")
            league=LEAGUES.get(comp)
            if league is None:
                result["skipped"].append(str(leg["fixture_id"])+":league_not_supported")
                continue
            if used>=daily_limit:
                result["skipped"].append(str(leg["fixture_id"])+":daily_attempt_budget_exhausted")
                continue
            # Reserve up to two daily source attempts for near-kickoff lineups.
            # Otherwise early scouting of future fixtures can spend the whole
            # daily allowance before confirmed lineups become available.
            early=kickoff>clock+timedelta(minutes=90) and not target
            reserved=min(2,max(0,daily_limit-1))
            if early and used>=daily_limit-reserved:
                result["skipped"].append(str(leg["fixture_id"])+
                    ":near_kickoff_budget_reserved")
                continue
            if result["attempted_legs"]>=2:
                result["skipped"].append(str(leg["fixture_id"])+":per_run_attempt_budget_exhausted")
                continue
            # Reserve quota BEFORE HTTP requests, including source failures.
            # The workflow permanently saves this counter with other evidence.
            used+=1
            write_json(usage_path,{"date_utc":clock.strftime("%Y-%m-%d"),
                                   "attempts":used,"limit":daily_limit,
                                   "last_attempt_utc":clock.isoformat()})
            result["attempted_legs"]+=1
            if refresh_needed:
                marker.parent.mkdir(parents=True,exist_ok=True)
                with marker.open("x",encoding="utf-8") as file:
                    file.write(json.dumps({"attempted_at_utc":clock.isoformat(),
                        "fixture_id":str(leg["fixture_id"]),
                        "reason":"one_bounded_near_kickoff_source_refresh",
                        "eligibility_window_minutes":[8,NEAR_REFRESH_MAX_MINUTES],
                        "fixed_pair_common_cutoff_priority":target,
                        "source_refresh_reasons":availability["refresh_reasons"]},
                        sort_keys=True)+"\n")
            fixture={"provider":leg["provider"],"id":int(leg["fixture_id"]),
                     "utcDate":leg["kickoff_utc"],
                     "homeTeam":{"id":int(leg["home_team_id"]),"name":leg.get("home_team")},
                     "awayTeam":{"id":int(leg["away_team_id"]),"name":leg.get("away_team")},
                     "competition":{"code":comp,"id":league}}
            evidence=load_evidence(fixture,as_of=clock,api_key=key,
                                   api_request=cb,cache=cache)
            clock=observation_time()
            if not clock+timedelta(minutes=8)<kickoff:
                result["skipped"].append(str(leg["fixture_id"])+":history_completed_outside_capture_window")
                continue
            provider=capture(fixture,evidence["history"],key,
                             now=observation_time(),request=cb,clock=observation_time,
                             max_history_per_team=(6 if
                                 os.environ.get("V22_DEEP_XG_ENABLED")=="true"
                                 else 1),
                             include_provider_prediction=True)
            clock=observation_time()
            result["calls"]+=(provider.get("diagnostics") or {}).get("queries",0)
            if not valid_snapshot(provider,fixture,now=clock):
                result["skipped"].append(str(leg["fixture_id"])+":no_qualified_packet")
                continue
            result["source_availability"][str(leg["fixture_id"])]=source_availability(
                leg,provider,now=clock)
            if not any((provider.get("available_sources") or {}).values()):
                result["skipped"].append(str(leg["fixture_id"])+":all_sources_unavailable")
                continue
            record={"schema":1,"source_forecast":{k:leg[k] for k in (
                "provider","fixture_id","forecast_id","input_digest",
                "home_team_id","away_team_id","kickoff_utc")},
                "provider_snapshot":provider,
                "supplemental_history_digest":evidence.get("input_digest"),
                "not_baseline_forecast_digest":True,
                "source_scope":"API-Football endpoints, not arbitrary live websites"}
            if refresh_needed:
                # Keep early source data untouched; only write a new version.
                path=root/"versions"/(path.stem+"_"+
                     provider["snapshot_sha256"][:16]+".json")
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                with path.open("x",encoding="utf-8") as file:
                    file.write(json.dumps(record,indent=2,sort_keys=True,
                                          allow_nan=False)+"\n")
                result["new_packets"].append(str(leg["fixture_id"]))
                # Freeze the already-fetched historical inputs at the SAME
                # pre-kickoff decision. This uses zero additional provider
                # requests and never rewrites existing source or pick files.
                # Only a complete as-of evidence envelope can be frozen.
                # Incomplete diagnostics/test stubs are never qualified history.
                if (isinstance(evidence, dict) and evidence.get("as_of_utc")
                        and evidence.get("input_digest")
                        and isinstance(evidence.get("history"), list)):
                    try:
                        from verified_xg_evidence_pipeline import freeze_history
                        frozen = freeze_history(fixture, evidence, path,
                                                root=archive_root,provider_root=root)
                        if frozen:
                            result.setdefault("frozen_history_inputs", []).append(str(frozen))
                            shadow=load_json(frozen,{}).get("frozen_shadow") or {}
                            status=shadow.get("status")
                            if status in ("ready","abstained"):
                                key=("frozen_shadow_ready" if status=="ready" else
                                     "frozen_shadow_abstained")
                                result[key]=result.get(key,0)+1
                    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                        result["skipped"].append(str(leg["fixture_id"]) +
                                                 ":history_freeze_" + type(exc).__name__)
                else:
                    result.setdefault("history_missing_asof_integrity", []).append(
                        str(leg["fixture_id"]))
            except FileExistsError:
                result["skipped"].append(str(leg["fixture_id"])+":race_preserved")
        except (ValueError,TypeError,KeyError,OSError):
            result["skipped"].append(str(leg.get("fixture_id"))+":unavailable")
    return finish("partial_or_complete_source_packets" if result["new_packets"]
                  else ("daily_attempt_budget_exhausted" if used>=daily_limit
                        else "no_new_source_evidence"))


if __name__=="__main__":
    print(json.dumps(capture_pair(),indent=2))
