"""Bounded research-only history handoff after the normal capture work.

Use the existing provider adapter, shared cache and remaining run budget.
Never rewrite original forecasts, expand the main budget, or relabel old
observations. A new cross-competition receipt must still precede kickoff.
"""
from collections import Counter
from datetime import timedelta
from pathlib import Path
from cross_competition_capture import freeze, ROOT
from cross_competition_strength import DOMESTIC, MIN_LEAGUE, MIN_TEAM, MIN_VENUE
from history_pipeline import season_year, _date, load_evidence, build_evidence
from v37_sparse_history_research import predict

MAX_EXTRA = 20
MAX_CANDIDATES = 24


def domestic_gap(fixture, cache, cutoff):
    """Read-only diagnosis of missing contexts, using the same source gates.
    Absence from supported sources is not proof of nonparticipation.
    """
    cutoff=_date(cutoff)
    season=season_year(fixture)
    populations={}
    unavailable=[]
    for code,league in DOMESTIC.items():
        current=[]
        rows=[]
        for year in (season,season-1):
            try:
                raw,meta,error=cache[("API-Football",str(league),year)]
                if error or not meta.get("freshness_known") or meta.get("stale") is not False or _date(meta["fetched_at_utc"])>cutoff:
                    raise ValueError("unqualified_source")
                proxy={**fixture,"competition":{"id":league,"code":code,"season":year}}
                qualified=[r for r in build_evidence(proxy,raw,cutoff)["history"] if season_year(r)==year]
                rows.extend(qualified)
                if year==season:
                    current=qualified
            except (KeyError,ValueError,TypeError,AttributeError,OverflowError):
                unavailable.append(code+":"+str(year))
        populations[code]=(current,rows)
    result={"unavailable_groups":unavailable,"absence_not_proof_of_nonparticipation":True}
    for side in ("home","away"):
        team=str(fixture[side+"Team"]["id"])
        def includes(row):
            return team in (str(row["homeTeam"]["id"]),str(row["awayTeam"]["id"]))
        members=[code for code,(current,rows) in populations.items() if any(includes(r) for r in current)]
        item={"team_id":team,"observed_current_memberships":members}
        if not members:
            reason="no_membership_in_qualified_supported_leagues"
        elif len(members)>1:
            reason="ambiguous_membership"
        else:
            rows=populations[members[0]][1]
            games=[r for r in rows if includes(r)]
            venue=[r for r in games if str(r[side+"Team"]["id"])==team]
            item.update(league_matches=len(rows),team_matches=len(games),venue_matches=len(venue))
            ids=[str(r.get("id")) for r in rows]
            if any(r.get("id") is None for r in rows) or len(ids)!=len(set(ids)):
                reason="duplicate_or_missing_fixture_id"
            elif len(rows)<MIN_LEAGUE:
                reason="league_history_below_minimum"
            elif len(games)<MIN_TEAM:
                reason="team_history_below_minimum"
            elif len(venue)<MIN_VENUE:
                reason="venue_history_below_minimum"
            else:
                reason="context_counts_qualified"
        item["reason"]=reason
        result[side]=item
    return result


def membership_candidates(payload, season, observed_at):
    """Discovery only: provider league metadata is not qualified match history."""
    meta=payload["metadata"]
    if (meta.get("provider")!="API-Football" or
            meta.get("freshness_known") is not True or meta.get("stale") is not False or
            _date(meta["fetched_at_utc"])>_date(observed_at)):
        return {"status":"unqualified_source","leagues":[]}
    candidates={}
    for item in payload["data"]:
        league=item.get("league") or {}
        country=item.get("country") or {}
        identifier=league.get("id")
        if (type(identifier) is not int or identifier<=0 or league.get("type")!="League"
                or not country.get("code") or country.get("name")=="World"
                or not any(s.get("year")==season for s in item.get("seasons",[]))):
            continue
        candidates[identifier]={"league_id":identifier,"country_code":str(country["code"]),
                                "already_supported":identifier in DOMESTIC.values()}
    return {"status":"candidates_only" if candidates else "no_candidates",
            "leagues":[candidates[k] for k in sorted(candidates)]}


def discover_memberships(gaps, seasons, *, key, request, clock, can_spend):
    """Use only spare collector budget; never feed discovery into frozen models."""
    targets=set()
    for fid,gap in gaps.items():
        if gap.get("unavailable_groups")!=[] or fid not in seasons:
            continue
        for side in ("home","away"):
            item=gap.get(side,{})
            team=str(item.get("team_id",""))
            if team.isdecimal() and item.get("reason")=="no_membership_in_qualified_supported_leagues":
                targets.add((seasons[fid],int(team)))
    targets=sorted(targets)
    if targets:
        # Rotate after each six-hour bucket to avoid always querying the same IDs.
        offset=int(_date(clock()).timestamp()//21600)%len(targets)
        targets=targets[offset:]+targets[:offset]
    results={}
    for season,team in targets:
        if not can_spend(1):
            break
        try:
            payload=request(key,"/leagues",{"team":team,"season":season},with_metadata=True)
            result=membership_candidates(payload,season,clock())
        except (KeyError,ValueError,TypeError,AttributeError,OverflowError,RuntimeError,OSError):
            result={"status":"provider_or_payload_error","leagues":[]}
        results[str(season)+":"+str(team)]=result
    return {"target_count":len(targets),"attempted":len(results),"results":results,
            "research_only":True,"history_recovered":False}


def refresh(fixtures, cache, *, key, request, clock, remaining_calls,
            time_available, extra_limit=0, root=ROOT,
            loader=load_evidence, recorder=freeze, sparse_predict=predict):
    """The request wrapper is shared with production and remains authoritative."""
    cap = min(MAX_EXTRA, max(0, int(extra_limit)))
    report = {"extra_call_limit": cap, "extra_adapter_calls": 0,
              "eligible_fixtures": 0, "attempted_fixtures": 0,
              "outcomes": {}, "source_groups": {}, "source_group_states": {},
              "domestic_retry_entries": 0, "domestic_context_gaps": {},
              "no_live_model_change": True}
    outcomes = Counter()

    def can_spend(n):
        return (n <= cap-report["extra_adapter_calls"] and
                n <= max(0, remaining_calls()) and time_available())

    def bounded_request(*args, **kwargs):
        if not can_spend(1):
            raise RuntimeError("CrossCompetitionResearchBudgetExhausted")
        report["extra_adapter_calls"] += 1
        return request(*args, **kwargs)

    # Research retries are isolated to the end-of-run handoff. A failed
    # or stale cache entry must not permanently masquerade as available.
    retried=set()
    def source_state(k, cutoff):
        if k not in cache:
            return "missing"
        try:
            rows, meta, error=cache[k]
            if error:
                return "provider_error"
            if meta.get("freshness_known") is not True or meta.get("stale") is not False:
                return "unqualified_freshness"
            if _date(meta["fetched_at_utc"])>cutoff:
                return "future_observation"
            return "qualified" if rows else "qualified_empty"
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            return "invalid_metadata"

    def group(fixture, at, *, retry=False):
        comp=fixture["competition"]
        season=season_year(fixture)
        keys=[("API-Football",str(comp["id"]),year) for year in (season,season-1)]
        missing=[k for k in keys if k not in cache]
        # Do not mutate European entries after their original evidence was
        # observed. Only domestic history refresh can retry failed entries.
        retry_keys=[k for k in keys if retry and k in cache and k not in retried
                    and source_state(k,at) in ("provider_error","unqualified_freshness","invalid_metadata")]
        if missing and not can_spend(len(missing)):
            return None
        if not time_available():
            return None
        # Retry only with spare capacity after accounting for absent seasons.
        # Preserve failed entries if the bounded request cannot be made.
        if retry_keys and can_spend(len(missing)+len(retry_keys)):
            for k in retry_keys:
                retried.add(k)
                cache.pop(k)
        return loader(fixture, as_of=at, api_key=key,
                      api_request=bounded_request, cache=cache)

    now=_date(clock())
    root=Path(root)
    candidates=[]
    seen=set()
    for fixture in fixtures:
        try:
            fid=str(fixture["id"])
            kickoff=_date(fixture["utcDate"])
            if (fixture.get("provider")!="API-Football" or
                    fixture.get("competition",{}).get("code") not in ("CL","EL")
                    or fixture.get("provider_status") not in ("NS","TBD")
                    or not fid.isdecimal() or fid in seen
                    or not now+timedelta(minutes=5)<kickoff<=now+timedelta(hours=48)
                    or (root/"trials"/(fid+".json")).exists()):
                continue
            candidates.append((kickoff,fid,fixture))
            seen.add(fid)
        except (ValueError,TypeError,KeyError,AttributeError):
            outcomes["invalid_fixture"]+=1
    candidates.sort(key=lambda row:(row[0],row[1]))
    report["eligible_fixtures"]=len(candidates)
    # Preload European history only for still-future sparse candidates.
    eligible=[]
    for kickoff,fid,fixture in candidates[:MAX_CANDIDATES]:
        at=_date(clock())
        if not time_available() or at+timedelta(minutes=5)>=kickoff:
            outcomes["time_or_kickoff_limit"]+=1
            continue
        try:
            evidence=group(fixture,at)
            if evidence is None:
                outcomes["european_cache_or_budget_unavailable"]+=1
                continue
            if sparse_predict(fixture,evidence["history"],as_of=evidence["as_of_utc"]) is None:
                outcomes["sparse_control_unavailable"]+=1
                continue
            eligible.append((kickoff,fid,fixture,evidence))
        except (ValueError,TypeError,KeyError,OSError,AttributeError,OverflowError):
            outcomes["european_source_error"]+=1
    # One lookup per league-season cache miss. No per-team repeated requests.
    for season in sorted({season_year(row[2]) for row in eligible}):
        template=next(row[2] for row in eligible if season_year(row[2])==season)
        for code,league in DOMESTIC.items():
            at=_date(clock())
            if not time_available() or not any(k>at+timedelta(minutes=5) for k,_,_,_ in eligible):
                break
            proxy={**template,"competition":{"id":league,"code":code,"season":season}}
            keys=[("API-Football",str(league),year) for year in (season,season-1)]
            try:
                group(proxy,at,retry=True)
                states={str(k[2]):source_state(k,_date(clock())) for k in keys}
                group_id=code+":"+str(season)
                report["source_group_states"][group_id]=states
                report["source_groups"][group_id]=sum(
                    s in ("qualified","qualified_empty") for s in states.values())
            except (ValueError,TypeError,KeyError,OSError,AttributeError,OverflowError):
                outcomes["domestic_source_error"]+=1
    for kickoff,fid,fixture,evidence in eligible:
        if not time_available() or _date(clock())+timedelta(minutes=5)>=kickoff:
            outcomes["time_or_kickoff_limit"]+=1
            continue
        report["attempted_fixtures"]+=1
        try:
            outcome=recorder(fixture,evidence,cache,root=root,clock=clock)
            outcomes[outcome]+=1
            if outcome=="missing_or_ambiguous_domestic_context" and time_available():
                try:
                    report["domestic_context_gaps"][fid]=domestic_gap(fixture,cache,_date(clock()))
                except (KeyError,ValueError,TypeError,AttributeError,OverflowError):
                    report["domestic_context_gaps"][fid]={"reason":"diagnostic_unavailable"}
        except (ValueError,TypeError,KeyError,OSError,AttributeError,OverflowError):
            outcomes["receipt_error"]+=1
    report["domestic_membership_discovery"]=discover_memberships(
        report["domestic_context_gaps"],
        {fid:season_year(fixture) for _,fid,fixture,_ in eligible},
        key=key,request=bounded_request,clock=clock,can_spend=can_spend)
    report["domestic_retry_entries"]=len(retried)
    report["outcomes"]=dict(outcomes)
    report["candidate_limit_skipped"]=max(0,len(candidates)-MAX_CANDIDATES)
    return report
