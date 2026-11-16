"""V25 professional-analyst research challenger: evidence gate + model robustness.

V8, Poisson, negative-binomial and correlated-Poisson distributions are
mathematical sensitivity analyses of the SAME source goal rates, NOT four
independent witnesses or evidence of improved forecast accuracy.

Only genuinely saved pre-kickoff source forecasts can create a write-once
shadow selection. The baseline uses the same selected fixtures and cutoff.
No API calls, no live-model modification, no bookmaker advice and no claims
of 70% five-double success.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from v10_prospective import ROOT as ORIGINAL_ROOT, read_latest, utc
from v21_goal_model_comparison import variants

OUT = Path("data/pro_analyst_shadow")
SYDNEY = ZoneInfo("Australia/Sydney")
VARIANTS = ("baseline_v8", "poisson_marginal",
            "negative_binomial_r3", "bivariate_poisson_c012")
MIN_HISTORY = 45
MIN_LEAD = timedelta(minutes=45)
MAX_LEAD = timedelta(hours=24)
MIN_ENVELOPE_MASS = 0.035  # research-only operating threshold, NOT validated
VERSION = "pro-analyst-v25-robust-sensitivity-shadow"


def _grid(record):
    try:
        grid = np.asarray(record["score_matrix"], dtype=float)
        if (grid.ndim != 2 or min(grid.shape) < 5 or max(grid.shape) > 25 or
                not np.isfinite(grid).all() or np.any(grid < 0) or
                not np.isclose(float(grid.sum()), 1.0, atol=1e-6)):
            return None
        return grid
    except (KeyError, TypeError, ValueError):
        return None


def _eligible(record, clock):
    """Require identifiable, immutable, pre-kickoff evidence on a future match."""
    try:
        if record.get("model_key") not in ("v8", "v8_near"):
            return None, "not_original_v8_model"
        provider = str(record["provider"])
        fixture_id = str(record["fixture_id"])
        home, away = str(record["home_team_id"]), str(record["away_team_id"])
        if (not provider or not fixture_id.isdigit() or
                not home.isdigit() or not away.isdigit() or
                home == away or not record.get("forecast_id")):
            return None, "missing_fixture_identity"
        kickoff, captured, evidence = (utc(record[k]) for k in (
            "kickoff_utc", "captured_at_utc", "evidence_as_of_utc"))
        if not (evidence <= captured <= clock < kickoff):
            return None, "invalid_source_time"
        if not (MIN_LEAD <= kickoff - clock <= MAX_LEAD):
            return None, "outside_research_freeze_window"
        if (record.get("mode") != "prospective" or
                record.get("score_basis") != "regulation" or
                not record.get("input_digest")):
            return None, "missing_verified_pre_match_origin"
        if int(record.get("history_matches") or 0) < MIN_HISTORY:
            return None, "insufficient_historical_fixtures"
        matrix = _grid(record)
        if matrix is None:
            return None, "invalid_probability_grid"
        alternatives = variants(matrix)
        if any(alt.ndim != 2 or not np.isclose(alt.sum(), 1, atol=1e-6)
               for alt in alternatives.values()):
            return None, "unusable_sensitivity_models"
        return {"record":record, "grid":matrix, "alternatives":alternatives,
                "key":(provider, fixture_id),
                "teams":{(provider, home), (provider, away)},
                "day":kickoff.astimezone(SYDNEY).date().isoformat(),
                "captured":captured, "kickoff":kickoff}, None
    except (KeyError, ValueError, TypeError, OverflowError):
        return None, "missing_or_invalid_evidence"


def _pad(grid, shape):
    output = np.zeros(shape, dtype=float)
    output[:grid.shape[0], :grid.shape[1]] = grid
    return output


def _pair(first, second):
    """Select full-support top five by lower joint model agreement.

    Lower envelopes are deliberately NOT calibrated probability lower bounds.
    The independent-game product assumption must be validated prospectively.
    """
    shapes = [tuple(max(first["alternatives"][name].shape[d],
                        second["alternatives"][name].shape[d]) for d in (0,1))
              for name in VARIANTS]
    # Ensure the SAME coordinate grid across sensitivity models and both legs.
    shape = tuple(max(shape[d] for shape in shapes) for d in (0,1))
    joint = {}
    for name in VARIANTS:
        a = _pad(first["alternatives"][name], shape)
        b = _pad(second["alternatives"][name], shape)
        joint[name] = np.outer(a.ravel(), b.ravel()).ravel()
    minimum = np.minimum.reduce([joint[name] for name in VARIANTS])
    robust = np.argsort(-minimum, kind="stable")[:5]
    baseline = np.argsort(-joint["baseline_v8"], kind="stable")[:5]
    support = shape[0] * shape[1]

    def selections(indices):
        options = []
        for cell in indices:
            left, right = divmod(int(cell), support)
            h, a = divmod(left, shape[1])
            h2, a2 = divmod(right, shape[1])
            options.append({"score1":f"{h}-{a}", "score2":f"{h2}-{a2}"})
        return options

    baseline_mass = float(np.sum(joint["baseline_v8"][baseline]))
    robust_mass = float(np.sum(minimum[robust]))
    robust_v8_mass = float(np.sum(joint["baseline_v8"][robust]))
    # Do not read match results, train parameters or change frozen originals.
    return {
        "five_doubles":selections(robust),
        "baseline_five":selections(baseline),
        "v8_optimal_five_model_mass":baseline_mass,
        "robust_five_v8_model_mass":robust_v8_mass,
        "robust_lower_envelope_model_mass":robust_mass,
        "model_disagreement_model_mass_gap":baseline_mass-robust_mass,
        "sensitivity_model_names":list(VARIANTS),
        "independence_assumption":True,
        "lower_envelope_not_a_calibrated_probability":True,
    }


def propose(forecasts, now, *, min_envelope=MIN_ENVELOPE_MASS):
    """Select at most one research set per Sydney day, or record abstention."""
    clock = utc(now)
    valid = {}
    excluded = defaultdict(int)
    for row in forecasts.values() if isinstance(forecasts, dict) else forecasts:
        item, reason = _eligible(row, clock)
        if item is None:
            excluded[reason] += 1
            continue
        key = item["key"]
        prior = valid.get(key)
        # The closest genuinely recorded model before the decision cutoff
        # is considered, but never rewritten or mislabeled as the older model.
        if prior is None or item["captured"] > prior["captured"]:
            valid[key] = item
    days = defaultdict(list)
    for value in valid.values():
        days[value["day"]].append(value)
    output = {}
    for day, candidates in sorted(days.items()):
        ranked = []
        for a, b in combinations(sorted(candidates,key=lambda v:v["key"]), 2):
            if len(a["teams"] | b["teams"]) != 4:
                continue
            metrics = _pair(a, b)
            ranked.append((metrics["robust_lower_envelope_model_mass"],
                           a["key"], b["key"], a, b, metrics))
        ranked.sort(key=lambda x:(-x[0],x[1],x[2]))
        if not ranked or ranked[0][0] < min_envelope:
            output[day] = {"decision":"ABSTAIN",
                           "reason":"no_qualified_pair_or_insufficient_model_mass",
                           "qualified_fixtures":len(candidates)}
            continue
        _,_,_,first,second,metrics = ranked[0]
        def source(item):
            r=item["record"]
            return {k:r[k] for k in (
                "provider", "fixture_id", "home_team_id","away_team_id",
                "home_team","away_team","kickoff_utc","captured_at_utc",
                "forecast_id","model_key","model_version","input_digest")}
        output[day] = {
            "decision":"FROZEN_RESEARCH_ONLY", "model_version":VERSION,
            "sydney_date":day, "selected_at_utc":clock.isoformat(),
            "first":source(first), "second":source(second),
            "qualified_fixtures":len(candidates),
            **metrics,
            "real_win_rate_not_known_at_selection":True,
            "production_model_changed":False,
        }
    return {"days":output, "eligible_fixtures":len(valid),
            "excluded":dict(excluded), "version":VERSION}


def _results(original_root, prematch_root):
    """Only verified, post-kickoff regulation-time provider results qualify."""
    accepted, conflicted = {}, set()
    for file in (Path(original_root)/"verified_results.json",
                 Path(prematch_root)/"verified_results.json"):
        try:
            source = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(source, dict):
            continue
        for row in source.values():
            if not isinstance(row, dict):
                continue
            try:
                if (row.get("evidence_version") != 2 or
                        row.get("score_basis") != "regulation" or
                        row.get("status") not in ("FT","AET","PEN","FINISHED")):
                    continue
                if not utc(row["verified_at_utc"]) > utc(row["kickoff_utc"]):
                    continue
                h,a = row["home_goals"],row["away_goals"]
                if type(h) is not int or type(a) is not int or min(h,a)<0:
                    continue
                key = (str(row["provider"]),str(row["fixture_id"]))
                signature = (str(row["home_team_id"]),str(row["away_team_id"]),
                             utc(row["kickoff_utc"]),h,a)
                if key in accepted and accepted[key][0]!=signature:
                    conflicted.add(key)
                accepted.setdefault(key,(signature,row))
            except (KeyError,ValueError,TypeError):
                continue
    return {k:v[1] for k,v in accepted.items() if k not in conflicted}, len(conflicted)


def _score_set(packet, result_index):
    actual=[]
    for side in ("first","second"):
        original = packet[side]
        result = result_index.get((str(original["provider"]),
                                   str(original["fixture_id"])))
        if result is None:
            return {"status":"pending"}
        if not all(str(result[k])==str(original[k]) for k in (
                "provider","fixture_id","home_team_id","away_team_id")):
            return {"status":"pending"}
        if utc(result["kickoff_utc"]) != utc(original["kickoff_utc"]):
            return {"status":"pending"}
        actual.append(f'{result["home_goals"]}-{result["away_goals"]}')
    wins = {name:any(item["score1"]==actual[0] and
                         item["score2"]==actual[1] for item in packet[name])
            for name in ("five_doubles","baseline_five")}
    return {"status":"verified", "actual_scores":actual,
            "robust_hit":wins["five_doubles"],"baseline_hit":wins["baseline_five"]}


def run(*, root=ORIGINAL_ROOT, out=OUT, prematch_root=Path("data/prematch"),
        now=None, forecasts=None):
    """Write new shadow sets exactly once; rescore ONLY from verified results."""
    clock = utc(now or datetime.now(timezone.utc))
    prediction = propose(read_latest(root) if forecasts is None else forecasts, clock)
    folder = Path(out)/"sets"
    folder.mkdir(parents=True, exist_ok=True)
    created = []
    for day,packet in prediction["days"].items():
        if packet["decision"] != "FROZEN_RESEARCH_ONLY":
            continue
        file = folder/(day+".json")
        try:
            with file.open("x",encoding="utf-8") as f:
                f.write(json.dumps(packet,indent=2,sort_keys=True,allow_nan=False)+"\n")
            created.append(day)
        except FileExistsError:
            pass
    results,conflicts = _results(root,prematch_root)
    outcomes=[]
    for path in sorted(folder.glob("*.json")):
        try:
            packet=json.loads(path.read_text(encoding="utf-8"))
            if packet.get("decision")!="FROZEN_RESEARCH_ONLY":
                continue
            rated=_score_set(packet,results)
            outcomes.append({"sydney_date":packet["sydney_date"],
                             "first":packet["first"]["fixture_id"],
                             "second":packet["second"]["fixture_id"], **rated})
        except (OSError,ValueError,TypeError,KeyError,UnicodeError):
            continue
    verified=[p for p in outcomes if p["status"]=="verified"]
    summary={
        "model_version":VERSION,
        "generated_at_utc":clock.isoformat(),
        "candidate_coverage":prediction,
        "new_sets":created,
        "total_frozen_sets":len(outcomes),
        "verified_sets":len(verified),
        "pending_sets":len(outcomes)-len(verified),
        "robust_wins":sum(x["robust_hit"] for x in verified),
        "baseline_wins_on_same_pairs":sum(x["baseline_hit"] for x in verified),
        "observed_robust_hit_rate":(sum(x["robust_hit"] for x in verified)/len(verified)
                                   if verified else None),
        "conflicting_verified_results_excluded":conflicts,
        "outcomes":outcomes,
        "policy":"read_only_validation_of_immutable_source_forecasts_with_strict_provider_identity",
        "model_promotion":False,
        "never_a_claim_of_70_percent_accuracy":True,
        "external_api_requests":0,
    }
    output=Path(out)/"status.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n",
                      encoding="utf-8")
    return summary


if __name__ == "__main__":
    result=run()
    print(json.dumps({k:result[k] for k in (
        "new_sets","total_frozen_sets","verified_sets","pending_sets",
        "robust_wins","baseline_wins_on_same_pairs")},sort_keys=True))
