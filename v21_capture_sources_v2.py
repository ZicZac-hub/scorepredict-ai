"""Versioned source collector; preserve V21's hashed historical implementation.

Only the capture loop is forked to keep the provider credential separate from
diagnostic counter names. Original source qualification helpers and quotas remain.
"""
import json
import os
from datetime import timedelta
from pathlib import Path
from v21_capture_sources import (
    ROOT, ORIGINAL_ROOT, PAIR_ROOT, NEAR_REFRESH_MAX_MINUTES,
    capture_clock, utc, load_json, write_json, current_pair, targeted_pair_models,
    source_packet_path, best_packet, source_availability, load_evidence,
    capture, valid_snapshot,
)

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
    target_deadline=min((utc(row["kickoff_utc"]) for row in targeted),default=None)
    if targeted:
        identities={(str(row["provider"]),str(row["fixture_id"])) for row in targeted}
        models=list(targeted)+[row for row in models if
            (str(row["provider"]),str(row["fixture_id"])) not in identities]
        result["targeted_pair_fixture_ids"]=[str(row["fixture_id"]) for row in targeted]
    upcoming = targeted_pair_models(source_root=source_root,pair_root=pair_root,
                                    now=clock,reserve_ahead=True) if not targeted else []
    result["reserved_future_pair_fixture_ids"] = [str(row["fixture_id"]) for row in upcoming]
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
            # Paired research must observe both legs before the earlier
            # kickoff. Do not spend reserved paired attempts after its window
            # closes merely because the second match starts later.
            capture_deadline=min(kickoff,target_deadline) if target else kickoff
            if not clock+timedelta(minutes=8)<capture_deadline:
                result["skipped"].append(str(leg["fixture_id"])+":common_pair_window_closed")
                continue
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
            if upcoming and not target and used>=daily_limit-reserved:
                result["skipped"].append(str(leg["fixture_id"])+
                    ":fixed_pair_common_cutoff_budget_reserved")
                continue
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
            if not clock+timedelta(minutes=8)<capture_deadline:
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
                                counter_key=("frozen_shadow_ready" if status=="ready" else
                                     "frozen_shadow_abstained")
                                result[counter_key]=result.get(counter_key,0)+1
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
