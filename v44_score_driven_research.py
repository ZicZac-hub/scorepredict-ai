"""Independent-of-V8-rate score-driven opponent ratings, research only.

Inspired by score-driven football time series (International Journal of
Forecasting, 2019, S0169207018302048) and dynamic attack/defence research
(JRSS-C 2025, 74:717). This is a simple fixed-parameter approximation, NOT
a reproduction of either paper's fitted Bayesian or likelihood model.
Only authenticated frozen historical inputs; no provider calls or new picks.
"""
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
import json,math
from pathlib import Path
import numpy as np

from history_pipeline import build_evidence
from forecast_ledger import _forecast, LedgerValidationError
from five_double_accuracy import _final,_result_index,wilson
from professional_double_shadow import _validated
from v10_prospective import utc
import verified_xg_evidence_pipeline as archive
import v42_research_trial as sources
from v44_family_pool_research import _grid
from v44_single_calibration import _metrics
from v44_pair_transfer_research import _arm

HALF_LIFE=180.
STEP=.065
MIN_GAMES=80
MIN_TEAM=8
MIN_OTHER=30
SUPPORT=32
BLEND=.25


def _poisson(rate):
    p=np.zeros(SUPPORT)
    p[0]=math.exp(-rate)
    for i in range(1,SUPPORT):
        p[i]=p[i-1]*rate/i
    if not 1-1e-8<=p.sum()<=1+1e-10:
        raise ValueError("unrepresented_score_mass")
    return p/p.sum()


def predict(fixture,history,cutoff):
    """Compute separate attack and defensive weakness from observed goals.

    A team's residual update depends on its opponent rating and venue.
    Updates use result-availability time; simultaneous results update from
    the same pre-batch state. Fixed decay and step are not fitted to test data.
    """
    cutoff=utc(cutoff)
    if cutoff>=utc(fixture["utcDate"]):
        raise ValueError("not_pre_match")
    evidence=build_evidence(fixture,history,cutoff)
    home,away=map(str,(fixture["homeTeam"]["id"],fixture["awayTeam"]["id"]))
    grouped=defaultdict(list)
    counts=Counter()
    other=0
    for m in evidence["history"]:
        h,a=map(str,(m["homeTeam"]["id"],m["awayTeam"]["id"]))
        at=utc(m["finishedAt"]) if m.get("finishedAt") else utc(m["utcDate"])+timedelta(hours=3)
        if at>cutoff:
            raise ValueError("future_result")
        g=m["score"]["fullTime"]
        grouped[at].append((h,a,g["home"],g["away"]))
        counts.update((h,a))
        other+=int(not {h,a}&{home,away})
    n=sum(map(len,grouped.values()))
    if n<MIN_GAMES or min(counts[home],counts[away])<MIN_TEAM or other<MIN_OTHER:
        return {"status":"abstained","reason":"insufficient_league_or_team_history",
                "history_matches":n,"team_matches":[counts[home],counts[away]],"other_matches":other}
    ratings={}
    def rating(team,when):
        attack,weak,previous=ratings.get(team,(0.,0.,when))
        decay=2**(-max(0.,(when-previous).total_seconds()/86400)/HALF_LIFE)
        return attack*decay,weak*decay
    # Smoothed league base rates; no full-season hindsight.
    totals=np.array([30.,24.])
    mass=20.
    previous=None
    def rates(h,a,when):
        ah,dh=rating(h,when)
        aa,da=rating(a,when)
        baseline=np.clip(totals/mass,.5,3.)
        return np.clip(baseline*np.exp([ah+da,aa+dh]),.15,5.)
    for at,rows in sorted(grouped.items()):
        if previous is not None:
            decay=2**(-(at-previous).total_seconds()/86400/HALF_LIFE)
            totals=decay*totals+(1-decay)*np.array([30.,24.])
            mass=decay*mass+(1-decay)*20.
        previous=at
        changes=defaultdict(lambda:np.zeros(2))
        for h,a,gh,ga in sorted(rows):
            lh,la=rates(h,a,at)
            # Bounded standardized score update; prevent a single extreme score
            # from dominating ratings.
            eh,ea=np.clip([(gh-lh)/math.sqrt(lh),(ga-la)/math.sqrt(la)],-3.,3.)
            changes[h]+=STEP*np.array([eh,ea])
            changes[a]+=STEP*np.array([ea,eh])
        for team,change in changes.items():
            attack,weak=np.clip(np.array(rating(team,at))+change,-1.2,1.2)
            ratings[team]=(float(attack),float(weak),at)
        totals+=np.sum([[r[2],r[3]] for r in rows],axis=0)
        mass+=len(rows)
    lh,la=rates(home,away,cutoff)
    return {"status":"ready","grid":np.outer(_poisson(lh),_poisson(la)),
            "rates":[float(lh),float(la)],"history_matches":n,
            "team_matches":[counts[home],counts[away]],"other_matches":other}


def _identity(model):
    return str(model["provider"]),str(model["fixture_id"])


def _match(model,fixture):
    return (str(model["provider"])==fixture["provider"]
            and str(model["fixture_id"])==str(fixture["id"])
            and str(model["home_team_id"])==str(fixture["homeTeam"]["id"])
            and str(model["away_team_id"])==str(fixture["awayTeam"]["id"])
            and utc(model["kickoff_utc"])==utc(fixture["utcDate"]))


def _index(root,provider_root,source_root,clock):
    records=defaultdict(list)
    rejects=Counter()
    for path in sorted((Path(root)/"history").glob("*.json")):
        try:
            rec=archive._authenticated_record(path,provider_root)
            if rec is None or rec["capture"]>clock:
                raise ValueError("invalid_archive")
            raw=json.loads(path.read_text(encoding="utf-8"))
            packet=json.loads((Path(provider_root)/raw["snapshot_relative_path"]).read_text(encoding="utf-8"))
            ref=packet["source_forecast"]
            baseline_path=Path(source_root)/"forecasts"/str(ref["provider"])/str(ref["fixture_id"])/(str(ref["forecast_id"])+".json")
            model=_forecast(json.loads(baseline_path.read_text(encoding="utf-8")))
            if (not _match(model,rec["fixture"])
                    or not sources.qualified_packet(packet,model,now=rec["capture"])
                    or utc(model["captured_at_utc"])>rec["capture"]
                    or model["model_key"] not in ("v8","v8_near")):
                raise ValueError("unmatched_original_source")
            asof=utc(raw["as_of_utc"])
            pred=predict(rec["fixture"],rec["history"],asof)
            records[_identity(model)].append({"capture":rec["capture"],"model":model,
                        "prediction":pred,"snapshot_sha256":rec["snapshot"]["snapshot_sha256"],
                        "path":path.name})
        except (ValueError,TypeError,KeyError,OSError,AttributeError,ArithmeticError,LedgerValidationError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    return records,rejects


def _blend(base,challenger):
    base,challenger=_grid(base),_grid(challenger)
    shape=tuple(max(x,y) for x,y in zip(base.shape,challenger.shape))
    def pad(p):
        return np.pad(p,[(0,shape[i]-p.shape[i]) for i in (0,1)])
    return (1-BLEND)*pad(base)+BLEND*pad(challenger)


def _select(items,cutoff,model=None):
    available=[r for r in items if r["capture"]<=cutoff and
               (model is None or all(str(r["model"][k])==str(model[k])
                 for k in ("provider","fixture_id","home_team_id","away_team_id","kickoff_utc")))]
    return max(available,key=lambda r:(r["capture"],r["snapshot_sha256"],r["path"])) if available else None


def _summary(rows,metric):
    out={}
    for name in ("v8","ratings","blend"):
        values=[r["metrics"][name] for r in rows]
        n=len(values)
        hitkey="hit" if metric=="pairs" else "top_score_hit"
        losskey="score_log_loss" if metric=="pairs" else "exact_log_loss"
        hits=sum(v[hitkey] for v in values)
        out[name]={metric:n,"hits":hits,"hit_rate":hits/n if n else None,
                   "wilson_95":wilson(hits,n),
                   "mean_log_loss":math.fsum(v[losskey] for v in values)/n if n else None}
    return out


def evaluate(*,root=archive.ROOT,provider_root=sources.PROVIDER_ROOT,
             source_root=sources.SOURCE_ROOT,prematch_root=sources.PREMATCH_ROOT,
             pair_root=sources.PAIR_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    records,rejects=_index(root,provider_root,source_root,clock)
    finals,conflicts,result_rejects=_result_index(source_root,prematch_root)
    single,pairs=[],[]
    counts=Counter()
    for identity,items in sorted(records.items()):
        # First authenticated capture, before reading any final.
        r=min(items,key=lambda x:(x["capture"],x["snapshot_sha256"],x["path"]))
        m=r["model"]
        pred=r["prediction"]
        actual=_final(m,finals.get(identity),clock)
        if actual is None:
            counts["single_pending"]+=1
            continue
        if pred["status"]!="ready":
            counts["single_insufficient_history"]+=1
            continue
        base=_grid(m["score_matrix"])
        grids={"v8":base,"ratings":pred["grid"],"blend":_blend(base,pred["grid"])}
        single.append({"fixture_id":identity[1],"metrics":{k:_metrics(g,actual) for k,g in grids.items()}})
    seen=set()
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            raw=json.loads(path.read_text(encoding="utf-8"))
            models,choices,coverage=_validated(raw,source_root=source_root,now=clock)
            if path.stem!=raw["sydney_date"]:
                raise ValueError("filename_day_mismatch")
            ids={_identity(m) for m in models}
            if seen&ids:
                raise ValueError("repeated_fixture")
            seen.update(ids)
            cutoff=utc(raw["predicted_at_utc"])
            selected=[_select(records.get(_identity(m),[]),cutoff,m) for m in models]
            if any(r is None for r in selected):
                counts["pair_missing_pre_cutoff_history"]+=1
                continue
            if any(r["prediction"]["status"]!="ready" for r in selected):
                counts["pair_insufficient_history"]+=1
                continue
            actual=[_final(m,finals.get(_identity(m)),clock) for m in models]
            if any(s is None for s in actual):
                counts["pair_pending"]+=1
                continue
            base=[_grid(m["score_matrix"]) for m in models]
            ratings=[r["prediction"]["grid"] for r in selected]
            blended=[_blend(g,c) for g,c in zip(base,ratings)]
            arms={"v8":{"full_score_grids":base,"five_doubles":choices,"estimated_coverage":coverage},
                  "ratings":_arm(ratings),"blend":_arm(blended)}
            pairs.append({"sydney_date":path.stem,
                         "metrics":{k:sources._metric(v,actual) for k,v in arms.items()}})
        except (ValueError,TypeError,KeyError,OSError,AttributeError,ArithmeticError) as exc:
            rejects[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]+=1
    return {"study_type":"independent_rate_score_driven_history_replay",
            "authenticated_history_fixtures":len(records),"counts":dict(counts),
            "rejections":dict(rejects),"result_rejections":dict(result_rejects),
            "conflicting_result_identities":len(conflicts),
            "single_comparisons":_summary(single,"fixtures"),"pair_comparisons":_summary(pairs,"pairs"),
            "rows":{"single":single,"pairs":pairs},"external_requests":0,"automatic_promotion":False,
            "limitations":["Independently computed rates, but same historical score data may underlie V8.",
                           "Fixed parameters are research hypotheses, not tuned estimates.",
                           "Only histories actually captured before the original pair decision qualify.",
                           "Retrospective results cannot establish prospective hit rate."]}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
