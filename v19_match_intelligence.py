"""V19 verified match intelligence: genuine market-source paired challenger.

No external API calls or paid services. This program reads existing archived
V8 and V9/V10 pre-match probability matrices. A V19 *separate* shadow
five-exact-score set is frozen only if both legs of an original frozen V8
pair have valid timestamped market-informed archived forecast sources.

No historical V8 pairs are altered and no V19 hit is claimed retroactively.
Real lineup/xG/market-response raw feed access is NOT asserted here.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math
import hashlib

from persist_prematch import load_json, verified_result, write_json
from v10_prospective import ROOT as SOURCE_ROOT, read_latest, utc
from v16_pair_research import original_model, SAFE
from v17_confidence import wilson

ROOT = Path("data/v19_intelligence")
VERSION = "v19-original-pair-market-source-v1"
SYDNEY = ZoneInfo("Australia/Sydney")
MIN_LEAD_MINUTES = 45
MAX_LEAD_HOURS = 36
MAX_ODDS_AGE_HOURS = 12
SUPPORTED_MARKETS = ("v10", "v9")


def _time(s):
    return utc(s)


def _grid(row):
    """Strictly validate a genuine normalized full-score distribution."""
    matrix = row.get("score_matrix")
    if not isinstance(matrix, list) or not matrix or not isinstance(matrix[0], list):
        raise ValueError("missing_full_score_grid")
    width = len(matrix[0])
    if not width or len(matrix) * width > 4096:
        raise ValueError("invalid_grid_dimensions")
    if any(not isinstance(line, list) or len(line) != width for line in matrix):
        raise ValueError("non_rectangular_grid")
    values = [v for line in matrix for v in line]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid_score_probabilities")
    if abs(math.fsum(values) - 1) > 1e-6:
        raise ValueError("unnormalized_score_grid")
    return matrix


def _score_digest(row):
    """Write-once fingerprint of the complete matrix, not just a market label."""
    cells=_grid(row)
    payload=json.dumps(cells,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _original_market_file(root, row):
    """Ensure a candidate is the unchanged actual saved file, not an in-memory rewrite."""
    fields = ("provider", "fixture_id", "forecast_id")
    parts = [str(row.get(k) or "") for k in fields]
    if any(not SAFE.fullmatch(s) or s in (".", "..") for s in parts):
        return None
    saved = load_json(Path(root) / "forecasts" / parts[0] / parts[1] /
                      (parts[2] + ".json"), {})
    if not isinstance(saved, dict):
        return None
    return saved if saved == row else None


def qualified_market(source, candidate, *, root=SOURCE_ROOT, cutoff=None):
    """Return market evidence diagnostic only if independent and pre-kickoff.

    An archived model claiming to use a bookmaker is not proof that every
    raw bookmaker HTTP response was retained. The returned provenance scope
    explicitly limits the claim to archived market model metadata.
    """
    reasons = []
    if not isinstance(source, dict) or not isinstance(candidate, dict):
        return None, "missing_source_or_market_model"
    try:
        if source.get("model_key") not in ("v8", "v8_near"):
            raise ValueError("not_original_statistical_model")
        if candidate.get("model_key") not in SUPPORTED_MARKETS:
            raise ValueError("not_a_market_challenger")
        for key in ("provider", "fixture_id", "home_team_id",
                    "away_team_id", "kickoff_utc", "input_digest"):
            if str(source[key]) != str(candidate[key]):
                raise ValueError("mismatched_original_pair_source_" + key)
        if _time(source["evidence_as_of_utc"]) != _time(candidate["evidence_as_of_utc"]):
            raise ValueError("different_historical_evidence_as_of")
        if not source.get("forecast_id") or not candidate.get("forecast_id"):
            raise ValueError("missing_source_forecast_identity")
        if _original_market_file(root, candidate) is None:
            raise ValueError("market_model_not_in_original_archive")
        original=original_model(root, source)
        if (original is None or original.get("input_digest") != source["input_digest"] or
                original.get("score_matrix") != source.get("score_matrix")):
            raise ValueError("original_statistical_forecast_unverified")
        _grid(source)
        _grid(candidate)
        kickoff=_time(source["kickoff_utc"])
        observed=_time(candidate["captured_at_utc"])
        when=_time(source["captured_at_utc"])
        timestamp=_time((candidate.get("market_snapshot") or {})["snapshot_utc"])
        as_of=_time(candidate["evidence_as_of_utc"])
        if not as_of<=when<=observed<kickoff:
            raise ValueError("model_not_pre_kickoff_or_was_backdated")
        if timestamp>observed or timestamp>=kickoff:
            raise ValueError("market_odds_after_forecast_or_kickoff")
        age=observed-timestamp
        if age<timedelta(0) or age>timedelta(hours=MAX_ODDS_AGE_HOURS):
            raise ValueError("stale_market_quote")
        if cutoff is not None and (observed>_time(cutoff) or _time(cutoff)>=kickoff):
            raise ValueError("market_unavailable_at_selection")
        if candidate.get("market_used") is not True:
            raise ValueError("market_model_is_fallback")
        market=candidate.get("market_snapshot") or {}
        if market.get("settlement")!="regulation_90_minutes":
            raise ValueError("market_wrong_settlement_basis")
        vendor=str(market.get("bookmaker") or "").strip()
        if not vendor:
            raise ValueError("no_bookmaker_name")
        win=market.get("winner")
        if (not isinstance(win, list) or len(win)!=3 or
                any(type(p) not in (float, int) or not math.isfinite(p) or p<=0 or p>1
                    for p in win) or abs(math.fsum(win)-1)>1e-5):
            raise ValueError("invalid_de_vigged_1x2")
        if candidate["model_key"]=="v10":
            book_count=market.get("bookmaker_count")
            names=market.get("bookmaker_names")
            if (type(book_count) is not int or book_count<3 or
                    not isinstance(names,list) or
                    len({str(x).strip().lower() for x in names if str(x).strip()})<3):
                raise ValueError("multi_book_market_not_complete")
        return {"bookmaker":vendor, "market_model":candidate["model_key"],
                "source_forecast_id":source["forecast_id"],
                "market_forecast_id":candidate["forecast_id"],
                "input_digest":source["input_digest"],
                "market_score_matrix_sha256":_score_digest(candidate),
                "snapshot_utc":timestamp.isoformat(),
                "capture_utc":observed.isoformat(),
                "source_scope":"archived_model_and_odds_metadata_not_raw_market_api_payload",
                "raw_odds_packet_independently_replayable":False}, None
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return None, str(exc)[:110]


def top_five(a,b):
    """Exact top-five joint probability cells; only top-five from each leg needed."""
    def cells(row):
        grid=_grid(row)
        return sorted(((float(p),i,j) for i,line in enumerate(grid)
                       for j,p in enumerate(line)),key=lambda x:(-x[0],x[1],x[2]))[:5]
    ca,cb=cells(a),cells(b)
    outcomes=sorted(((p*q,i,j,k,l) for p,i,j in ca for q,k,l in cb),
                    key=lambda x:(-x[0],x[1],x[2],x[3],x[4]))[:5]
    if len(outcomes)!=5:
        raise ValueError("not_five_combinations")
    options=[{"score1":f"{i}-{j}","score2":f"{k}-{l}",
              "joint_probability":float(z)} for z,i,j,k,l in outcomes]
    mass=math.fsum(x["joint_probability"] for x in options)
    return options,mass


def _pair_sources(packet,source_root):
    a=original_model(source_root,packet["first"])
    b=original_model(source_root,packet["second"])
    if a is None or b is None:
        raise ValueError("original_pair_legs_not_verified")
    if (a["forecast_id"]!=packet["first"]["forecast_id"] or
            b["forecast_id"]!=packet["second"]["forecast_id"]):
        raise ValueError("original_leg_forecast_id_mismatch")
    if len({str(r[k]) for r in (a,b)
            for k in ("home_team_id","away_team_id")})!=4:
        raise ValueError("same_team_appears_in_both_legs")
    if a["provider"]!=b["provider"]:
        raise ValueError("mixed_football_provider")
    if (utc(a["kickoff_utc"]).astimezone(SYDNEY).date().isoformat()!=packet["sydney_date"] or
            utc(b["kickoff_utc"]).astimezone(SYDNEY).date().isoformat()!=packet["sydney_date"]):
        raise ValueError("not_same_sydney_calendar_day")
    return a,b


def choose_market_pair(packet,market_rows,source_root=SOURCE_ROOT,now=None):
    """Only two source-matched real market forecasts, no results or retrofitting."""
    clock=utc(now or datetime.now(timezone.utc))
    try:
        if (packet.get("mode")!="prospective" or
                packet.get("status")!="frozen_pre_kickoff"):
            raise ValueError("not_original_frozen_pair")
        captured=utc(packet["captured_at_utc"])
        a,b=_pair_sources(packet,source_root)
        first=min(utc(a["kickoff_utc"]),utc(b["kickoff_utc"]))
        if captured>clock or captured>=first:
            raise ValueError("original_pair_not_frozen_before_kickoff")
        if not timedelta(minutes=MIN_LEAD_MINUTES)<=first-clock<=timedelta(hours=MAX_LEAD_HOURS):
            raise ValueError("outside_pre_match_research_window")
        by_fixture={(str(c.get("provider")),str(c.get("fixture_id")),str(c.get("model_key"))):c
                    for c in market_rows if isinstance(c,dict)}
        reasons=[]
        # A v10-v10 pair is preferred only if BOTH legs have true three-book
        # evidence; otherwise fallback to TWO separately validated v9 legs.
        for mode in SUPPORTED_MARKETS:
            x=by_fixture.get((str(a["provider"]),str(a["fixture_id"]),mode))
            y=by_fixture.get((str(b["provider"]),str(b["fixture_id"]),mode))
            av,why1=qualified_market(a,x,root=source_root,cutoff=clock)
            bv,why2=qualified_market(b,y,root=source_root,cutoff=clock)
            if not (av and bv):
                reasons.append({"model":mode,"leg_reasons":[why1,why2]})
                continue
            choices,coverage=top_five(x,y)
            # Exact five-score posterior coverage is a model estimate only.
            return {"version":VERSION,"mode":"research_market_shadow",
                    "status":"frozen_pre_kickoff",
                    "sydney_date":packet["sydney_date"],
                    "captured_at_utc":clock.isoformat(),
                    "source_original_pair_captured_at_utc":packet["captured_at_utc"],
                    "market_model":mode,"first":packet["first"],
                    "second":packet["second"],"first_market_provenance":av,
                    "second_market_provenance":bv,"five_doubles":choices,
                    "estimated_coverage":coverage,
                    "original_frozen_options":packet["five_doubles"],
                    "original_frozen_estimate":packet["estimated_coverage"],
                    "unvalidated_future_hit_rate":True,
                    "pair_selection":"same_original_v8_two_matches_for_fair_comparison",
                    "independence_assumption":True},None
        return None,{"reason":"two_leg_archived_market_evidence_incomplete",
                     "candidate_details":reasons}
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        return None,{"reason":str(exc)[:110]}


def freeze(root=ROOT,source_root=SOURCE_ROOT,now=None,market_rows=None):
    clock=utc(now or datetime.now(timezone.utc))
    if market_rows is None:
        market_rows=[row for row in read_latest(source_root).values()
                     if row.get("model_key") in SUPPORTED_MARKETS]
    saved=[];dropped={}
    checked=0
    for path in sorted((Path(source_root)/"doubles").glob("*.json")):
        packet=load_json(path,{})
        if packet.get("mode")!="prospective" or packet.get("status")!="frozen_pre_kickoff":
            continue
        if (Path(root)/"doubles"/path.name).exists():
            continue
        checked+=1
        candidate,why=choose_market_pair(packet,market_rows,source_root,clock)
        if candidate is None:
            dropped[path.stem]=why
            continue
        dest=Path(root)/"doubles"/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        try:
            with dest.open("x",encoding="utf-8") as fp:
                fp.write(json.dumps(candidate,sort_keys=True,indent=2)+"\n")
            saved.append(path.stem)
        except FileExistsError:
            pass
    return {"checked_unfrozen_original_pairs":checked,
            "new_market_research_pairs":saved,
            "abstained":dropped,
            "no_paid_calls":True,
            "no_retroactive_results":True}


def _result_leg(leg,source,verified):
    original=original_model(source,leg)
    if original is None:
        return None
    result=verified.get(str(leg["fixture_id"]))
    if not isinstance(result,dict) or not verified_result(original,result,str(leg["fixture_id"])):
        return None
    try:
        if (result.get("status") not in ("FT","FINISHED","AET","PEN") or
                result.get("score_basis")!="regulation" or
                utc(result["verified_at_utc"])<=utc(original["kickoff_utc"])):
            return None
    except (TypeError,KeyError,ValueError):
        return None
    return result


def _hit(options,first,second):
    s1=f'{first["home_goals"]}-{first["away_goals"]}'
    s2=f'{second["home_goals"]}-{second["away_goals"]}'
    return any(x["score1"]==s1 and x["score2"]==s2 for x in options)


def evaluate(root=ROOT,source_root=SOURCE_ROOT,verified=None):
    verified=(load_json(Path(source_root)/"verified_results.json",{})
              if verified is None else verified)
    settled=[];pending=0;rejected=Counter();shadow_wins=baseline_wins=0
    shadow_only=baseline_only=0
    for path in sorted((Path(root)/"doubles").glob("*.json")):
        packet=load_json(path,{})
        if packet.get("version")!=VERSION or packet.get("mode")!="research_market_shadow":
            continue
        try:
            if (packet["sydney_date"]!=path.stem or
                    utc(packet["captured_at_utc"])>=min(
                        utc(packet[x]["kickoff_utc"]) for x in ("first","second"))):
                raise ValueError("market_pair_not_legitimately_pre_match")
            source_pair=load_json(Path(source_root)/"doubles"/path.name,{})
            if (source_pair.get("mode")!="prospective" or
                    source_pair.get("five_doubles")!=packet.get("original_frozen_options") or
                    source_pair.get("first")!=packet.get("first") or
                    source_pair.get("second")!=packet.get("second")):
                raise ValueError("original_pair_source_was_not_preserved")
            # Do not grade an entry if either source market challenger has
            # been tampered with since the original V19 forecast was frozen.
            originally=_pair_sources(source_pair,source_root)
            for index,side in enumerate(("first","second")):
                provenance=packet[side+"_market_provenance"]
                market_id=str(provenance.get("market_forecast_id") or "")
                if not SAFE.fullmatch(market_id) or market_id in (".",".."):
                    raise ValueError("unsafe_market_source_identity")
                mpath=(Path(source_root)/"forecasts"/
                       packet[side]["provider"]/
                       str(packet[side]["fixture_id"])/
                       (market_id+".json"))
                model=load_json(mpath,{})
                record,reason=qualified_market(originally[index],model,root=source_root,
                                               cutoff=packet["captured_at_utc"])
                if record is None or record!=provenance:
                    raise ValueError("market_evidence_not_original_or_source_changed")
            a,b=(_result_leg(packet[key],source_root,verified)
                 for key in ("first","second"))
            if a is None or b is None:
                pending+=1
                settled.append({"sydney_date":path.stem,"status":"pending"})
                continue
            hit=_hit(packet["five_doubles"],a,b)
            old=_hit(packet["original_frozen_options"],a,b)
            shadow_wins+=int(hit)
            baseline_wins+=int(old)
            shadow_only+=int(hit and not old)
            baseline_only+=int(old and not hit)
            settled.append({"sydney_date":path.stem,"status":"verified",
                            "market_hit":hit,"original_v8_hit":old,
                            "actual_scores":[f'{a["home_goals"]}-{a["away_goals"]}',
                                             f'{b["home_goals"]}-{b["away_goals"]}']})
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            rejected[str(exc)[:110]]+=1
    n=sum(x["status"]=="verified" for x in settled)
    return {"version":VERSION,"research_only":True,
            "frozen_sets":len(settled)+sum(rejected.values()),
            "verified_sets":n,"pending_sets":pending,
            "market_wins":shadow_wins,"original_same_pair_wins":baseline_wins,
            "market_observed_hit_rate":shadow_wins/n if n else None,
            "original_observed_hit_rate":baseline_wins/n if n else None,
            "market_95_wilson":wilson(shadow_wins,n),
            "market_only_winning_days":shadow_only,
            "original_only_winning_days":baseline_only,
            "excluded_integrity_failures":dict(rejected),
            "rows":settled,
            "seventy_percent_exact_double_success_not_established":True,
            "real_football_probability_accuracy_improvement_not_yet_proven":True}


def readiness(source_root=SOURCE_ROOT,root=ROOT):
    saved=load_json(Path(root)/"performance.json",{})
    read_error=None
    try:
        market_rows=[z for z in read_latest(source_root).values()
                     if z.get("model_key") in SUPPORTED_MARKETS]
    except (ValueError,OSError,TypeError) as exc:
        # Fail closed: an invalid stored forecast cannot be quietly counted
        # as genuine evidence or bring down all scheduled status reporting.
        read_error="immutable_market_forecast_ledger_invalid"
        market_rows=[]
    models=Counter(z["model_key"] for z in market_rows)
    ai=load_json(Path("data/v15_ai_double/source_audit.json"),{})
    sport=load_json(Path("data/independent_football/status.json"),{})
    v18=load_json(Path("data/v18_accuracy/accuracy_report.json"),{})
    return {"version":VERSION,"archived_prematch_market_models":dict(models),
            "qualified_source_matched_original_double_sets":(
                int(saved.get("verified_sets") or 0)+int(saved.get("pending_sets") or 0)),
            "market_source_read_error":read_error,
            "market_challenger_verified_pairs":saved.get("verified_sets",0),
            "qualified_two_match_ai_pairs":ai.get(
                "original_sets_with_at_least_two_ai_votes_on_both_legs",0),
            "sportmonks_predictions_enabled":bool(sport.get("enabled")),
            "true_shot_quality_xg_feed_integrated":False,
            "timestamped_lineup_adjustment_trained":False,
            "market_quotes_raw_http_response_audited":False,
            "original_v8_verified_singles":(v18.get("baseline_double_chance") or {}).get("verified",0),
            "data_actions_required":[
                "Confirm licensed historical shot-level xG provider and timestamps; not included in V19",
                "Archive confirmed XI and player availability before kickoff; coefficients must be trained and tested",
                "Archive and independently replay raw odds packets alongside their already-saved bookmaker metadata",
                "Capture genuine both-leg AI provider answers only with owner-approved API spending",
                "Evaluate several hundred future original and challenger forecasts before promotion"],
            "source_claim":"Market model metadata validated; raw licensed provider response not independently re-fetched"}


def main():
    capture=freeze()
    result=evaluate()
    write_json(ROOT/"performance.json",result)
    write_json(ROOT/"readiness.json",readiness())
    print(json.dumps({"new_frozen_sets":capture["new_market_research_pairs"],
                      "frozen":result["frozen_sets"],
                      "verified":result["verified_sets"],
                      "market_wins":result["market_wins"],
                      "original_wins_on_same_pairs":result["original_same_pair_wins"],
                      "no_paid_requests":True,
                      "no_70_percent_claim":True},sort_keys=True))


if __name__=="__main__":
    main()
