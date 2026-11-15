"""ZicZac V13: fixed-weight, same-evidence statistical five-double CHALLENGER.

Combines genuine local numeric V6, V7, V8, and optionally V12 full score
distributions. V8 itself already includes V6 and fitted-model signals; these
components are correlated. Weights are conservative, declared up front and
NOT learned from the inspected 2026 results. No external AI votes, odds or
verified xG are claimed. No edits to production or frozen historical picks.
"""
from datetime import datetime, timezone
from itertools import combinations
from zoneinfo import ZoneInfo
import numpy as np
import engine
from forecast_model import predict as predict_v7
from research_ensemble_v8 import predict as predict_v8, _asof_history
from score_calibration_v12 import predict as predict_v12
from smart_double_ai import five_doubles, same_fixture
from double_score_selection import select_balanced, select_diversified, score_coverage

VERSION = "ziczac-hybrid-v13-prospective-challenger"
WEIGHTS = {"v8":0.60, "v7":0.20, "v6":0.15, "v12":0.05}
SYDNEY = ZoneInfo("Australia/Sydney")
MIN_LEAD_SECONDS = 300
MODES = ("top5", "balanced", "diversified")


def predict(fixture, history, league_history=None, *, as_of=None,
            reference_v8=None, reference_v6=None, reference_v7=None,
            reference_v12=None, max_goals=10):
    """New full-grid distribution; abstain without genuine v8 league evidence.

    The caller may supply already-calculated, same-evidence components.
    Passing post-event or differently sourced matrices is not supported.
    """
    now = engine._date(as_of) if as_of is not None else datetime.now(timezone.utc)
    kickoff = engine._date(fixture.get("utcDate"))
    if now is None or kickoff is None or (kickoff-now).total_seconds()<=MIN_LEAD_SECONDS:
        return None
    # Strictly time-qualify all numerical inputs, including V7 and V6 (which
    # otherwise only apply a kickoff cutoff, not final-score availability).
    past = _asof_history(fixture, history or [], now)
    league = _asof_history(fixture,
        history if league_history is None else league_history, now)
    if len(league)<30:
        # A genuine fitted opponent model needs a league population.
        # Thin data must not turn a weak baseline into a named hybrid.
        return None
    v8 = (reference_v8 if reference_v8 is not None else
          predict_v8(fixture, past, league_history=league,
                     as_of=now, max_goals=max_goals))
    if v8 is None or len(v8.get("component_weights") or {}) < 2:
        return None
    v6 = (reference_v6 if reference_v6 is not None else
          engine.predict(fixture,past,league_history=league,max_goals=max_goals))
    v7 = (reference_v7 if reference_v7 is not None else
          predict_v7(fixture,past,variant="reference_v7",
                     league_history=league,max_goals=max_goals))
    v12 = (reference_v12 if reference_v12 is not None else
           predict_v12(fixture,past,league_history=league,as_of=now,
                       reference_forecast=v8,max_goals=max_goals))
    components={"v8":v8}
    if v6 is not None:
        components["v6"]=v6
    if v7 is not None and not str(v7.get("model_version","")).endswith("-fallback-v6"):
        components["v7"]=v7
    if v12 is not None and v12.get("v12_diagnostics"):
        components["v12"]=v12
    if len(components)<2:
        # A lone V8 score grid is not a genuinely combined model.
        return None
    # Renormalise without making up unavailable components or giving a
    # fallback-V6 forecast a second, misleading independent model vote.
    total=sum(WEIGHTS[key] for key in components)
    weights={key:WEIGHTS[key]/total for key in WEIGHTS if key in components}
    grid=engine.mix_score_matrices(
        [np.asarray(components[key]["matrix"],dtype=float) for key in weights],
        list(weights.values()))
    ids=np.argsort(-grid.ravel(),kind="stable")[:10]
    scores=[(int(i//grid.shape[1]),int(i%grid.shape[1]),
             float(grid.ravel()[i])) for i in ids]
    home=np.arange(grid.shape[0],dtype=float)[:,None]
    away=np.arange(grid.shape[1],dtype=float)[None,:]
    return {
        "model_version":VERSION,
        "release_status":"experimental_not_production",
        "matrix":grid,"scores":scores,
        "home_xg":float((grid*home).sum()),"away_xg":float((grid*away).sum()),
        "home_win":float(np.tril(grid,-1).sum()),
        "draw":float(np.trace(grid)),
        "away_win":float(np.triu(grid,1).sum()),
        "component_weights":weights,
        "component_modal_scores":{
            name:[int(x) for x in components[name]["scores"][0][:2]]
            for name in weights},
        "component_versions":{
            name:str(components[name].get("model_version") or "unknown")
            for name in weights},
        "available_models":list(weights),
        "missing_optional_models":[k for k in WEIGHTS if k not in weights],
        "forecast_as_of":now.astimezone(timezone.utc).isoformat(),
        "llm_provider_calls":0,
        "reference_v8_probability":float(v8["scores"][0][2]),
        "independent_prospective_win_rate":None,
        "limitation":("V8, V7 and V6 are statistically correlated; mixing "
                      "cannot be presumed to improve accuracy. V12 can hurt "
                      "exact-score accuracy. Weights are predeclared, not "
                      "fitted, and need prospective tests.")
    }


def _date(row):
    return engine._date(row.get("kickoff") or
                        (row.get("fixture") or {}).get("utcDate"))


def _teams(row):
    f=row.get("fixture") or {}
    return tuple(" ".join(str((f.get(x+"Team") or {}).get("name") or
                              "").casefold().split()) for x in ("home","away"))


def valid_pair(a,b,*,now=None):
    instant=engine._date(now) if now is not None else datetime.now(timezone.utc)
    one,two=_date(a),_date(b)
    if not one or not two or not instant:
        return False
    if (min((one-instant).total_seconds(),(two-instant).total_seconds())<=
            MIN_LEAD_SECONDS):
        return False
    if one.astimezone(SYDNEY).date()!=two.astimezone(SYDNEY).date():
        return False
    if same_fixture(a,b):
        return False
    names=_teams(a)+_teams(b)
    return len(names)==4 and all(names) and len(set(names))==4


def pair(a,b,*,mode="top5",now=None):
    """Only genuine forecasts for two distinct upcoming matches."""
    if mode not in MODES:
        raise ValueError("Unsupported V13 five-double selection")
    if not valid_pair(a,b,now=now):
        return None
    top=five_doubles(a,b)
    if len(top)!=5:
        return None
    left=(a.get("result") or {}).get("scores") or []
    right=(b.get("result") or {}).get("scores") or []
    options=[{"Score 1":f"{h}-{away}","Score 2":f"{h2}-{a2}",
              "P1":float(p),"P2":float(q),"Joint":float(p*q)}
             for h,away,p in left[:10] for h2,a2,q in right[:10]]
    if mode=="top5":
        choices=top
    elif mode=="balanced":
        choices=select_balanced(options,5,diversity_weight=0.18)
    else:
        choices=select_diversified(options,5)
    if len(choices)!=5 or len({(x["Score 1"],x["Score 2"])
                               for x in choices})!=5:
        return None
    return {
        "model_version":VERSION,"selection_mode":mode,"doubles":choices,
        "estimated_joint_coverage":sum(x["Joint"] for x in choices),
        "top5_estimated_coverage":sum(x["Joint"] for x in top),
        "estimated_coverage_is_not_hit_rate":True,
        "diversity":score_coverage(choices),
        "sydney_date":one_date(a),
        "first":a,"second":b,
        "archived_originals_modified":False,
        "experimental_no_accuracy_promotion":True,
    }


def one_date(row):
    return _date(row).astimezone(SYDNEY).date().isoformat()


def best_pairs(rows,*,mode="top5",now=None,limit=1):
    """Rank only already-researched pairs: do not silently query all leagues."""
    ranked=[]
    for a,b in combinations(rows,2):
        candidate=pair(a,b,mode=mode,now=now)
        if candidate is not None:
            ranked.append(candidate)
    return sorted(ranked,
                  key=lambda x:(-x["estimated_joint_coverage"],
                      one_date(x["first"]),
                      str(x["first"]["fixture"].get("id")),
                      str(x["second"]["fixture"].get("id"))))[:limit]
