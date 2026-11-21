"""V22 paired independent evaluation of AI services versus original V8.

Configured API keys DO NOT imply the provider has supplied a forecast.
Each original historical vote must be frozen, source-matched and verified
on identical fixtures; the provider gains ZERO scoring weight by default.
"""
import json
import math
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

from persist_prematch import verified_result,load_json,write_json
from v10_prospective import read_latest,utc

PANEL_ROOT=Path("data/llm_research")
OUT=Path("data/v22_prospective/ai_status.json")
MIN_PAIRED=100
MIN_BRIER_IMPROVEMENT=.015


def _brier(chances, actual):
    probs=[chances[k] for k in ("home","draw","away")]
    if (not all(type(x) in (float,int) and math.isfinite(x) and 0<=x<=1 for x in probs) or
            abs(sum(probs)-1)>.03):
        raise ValueError("Uncalibrated model probabilities invalid")
    expected=[int(actual==k) for k in ("home","draw","away")]
    return sum((x-y)**2 for x,y in zip(probs,expected))


def _identity(panel, original, result, *, as_of):
    try:
        ids=("provider","fixture_id","home_team_id","away_team_id","kickoff_utc")
        if any(str(panel[k])!=str(original[k]) for k in ids):
            return False
        if (str(panel["source_forecast_id"])!=str(original["forecast_id"]) or
                str(panel["input_digest"])!=str(original["input_digest"])):
            return False
        kickoff=utc(original["kickoff_utc"])
        if (not utc(original["captured_at_utc"])<utc(panel["captured_at_utc"])<
                kickoff<utc(result["verified_at_utc"])<=utc(as_of)):
            return False
        if not verified_result(original,result,str(original["fixture_id"])):
            return False
        if result.get("status") not in ("FT","FINISHED","AET","PEN"):
            return False
        return True
    except (KeyError,TypeError,ValueError):
        return False


def _event(original):
    """A provider alias cannot turn one football event into another trial."""
    names=tuple(" ".join(str(original.get(side+"_team") or "").casefold().split())
                for side in ("home","away"))
    if not all(names) or names[0]==names[1]:
        raise ValueError("Known distinct team names required for event identity")
    return (utc(original["kickoff_utc"]).isoformat(),*names)


def evaluate(panels, originals, results, *, min_paired=MIN_PAIRED, as_of=None):
    clock=utc(as_of or datetime.now(timezone.utc))
    lookup={}
    for row in originals.values() if isinstance(originals,dict) else originals:
        if isinstance(row,dict) and row.get("forecast_id"):
            lookup[str(row["forecast_id"])]=row
    deltas=defaultdict(list)
    seen=set()
    seen_events=set()
    excluded=defaultdict(int)
    provider_success=defaultdict(int)
    for panel in sorted((r for r in panels if isinstance(r,dict)),
                        key=lambda x:str(x.get("captured_at_utc",""))):
        original=lookup.get(str(panel.get("source_forecast_id")))
        verified=results.get(str(panel.get("fixture_id")))
        if not original or not verified or not _identity(panel,original,verified,as_of=clock):
            excluded["unmatched_identity_result_or_time"]+=1
            continue
        actual = ("home" if verified["home_goals"]>verified["away_goals"] else
                  "away" if verified["home_goals"]<verified["away_goals"] else "draw")
        base=original.get("outcome_probabilities") or {}
        try:
            event=_event(original)
        except (TypeError,ValueError,KeyError):
            excluded["event_identity_missing"]+=1
            continue
        try:
            base_loss=_brier(base,actual)
        except (TypeError,ValueError,KeyError):
            excluded["baseline_probabilities_invalid"]+=1
            continue
        votes=panel.get("votes") or {}
        if not isinstance(votes,dict):
            excluded["invalid_provider_votes"]+=1
            continue
        for provider,vote in votes.items():
            key=(str(provider),str(original["provider"]),str(original["fixture_id"]))
            event_key=(str(provider),*event)
            if key in seen or event_key in seen_events:
                excluded["repeated_provider_fixture_or_event"]+=1
                continue
            if provider not in ("openai","claude","gemini","grok","deepseek",
                                "perplexity","mistral","qwen","groqcloud","cohere"):
                continue
            try:
                prob={"home":vote["home"],"draw":vote["draw"],"away":vote["away"]}
                vote_loss=_brier(prob,actual)
                seen.add(key)
                seen_events.add(event_key)
                deltas[provider].append(base_loss-vote_loss)
                provider_success[provider]+=1
            except (KeyError,ValueError,TypeError):
                excluded["invalid_provider_vote"]+=1
    report={}
    for provider,improvements in sorted(deltas.items()):
        n=len(improvements)
        mean=sum(improvements)/n
        variance=sum((d-mean)**2 for d in improvements)/(n-1) if n>1 else float("inf")
        stderr=math.sqrt(variance/n)
        bound=mean-1.96*stderr if math.isfinite(stderr) else float("-inf")
        qualified=n>=min_paired and mean>=MIN_BRIER_IMPROVEMENT and bound>0
        report[provider]={"source_matched_verified_fixtures":n,
                          "mean_paired_outcome_brier_advantage":mean,
                          "conservative_95pct_lower_bound":bound if math.isfinite(bound) else None,
                          "passes_statistical_shadow_gate":qualified,
                          "production_weight":0.0}
    return {"kind":"first_frozen_source_matched_AI_outcome_comparison",
            "evaluated_at_utc":clock.isoformat(),
            "provider_metrics":report,"excluded":dict(excluded),
            "min_required_verified_fixtures":min_paired,
            "configured_keys_not_automatic_success":True,
            "ai_exact_score_probability_weight":0.0,
            "automatic_promotion":False,
            "fixture_reuse_in_provider_denominator_allowed":False,
            "independence_of_distinct_events_not_established":True,
            "note":"Brier superiority on 1X2 opinions is not proof of exact-score double superiority."}


def main(*,panel_root=PANEL_ROOT,original_root=Path("data/v10_prospective"),
         out=OUT,as_of=None):
    panels=[load_json(path,{}) for path in sorted((Path(panel_root)/"forecasts").glob("*.json"))]
    records=list(read_latest(original_root).values())
    results=load_json(Path(original_root)/"verified_results.json",{})
    report=evaluate(panels,records,results,as_of=as_of)
    report["saved_provider_panel_count"]=len(panels)
    write_json(Path(out),report)
    return {k:row["source_matched_verified_fixtures"]
            for k,row in report["provider_metrics"].items()}


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
