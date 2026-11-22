"""On-demand V25 tests: real score matrices, dates, source timing and all pairs."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone

import numpy as np
import pytest

from ai_five_double_engine import pair_check
from v25_upcoming import (
    available,best_pair,generate,pair_catalog,qualified,rows_for_page,
    MAX_PAGE_PAIRS,VERSION,AUDIT_METHOD
)
from ziczac_interactive_capture import pair as audit_pair

NOW=datetime(2031,11,18,0,tzinfo=timezone.utc)
KICK="2031-11-20T03:00:00+00:00"


def fixture(fid, *, kick=KICK, captured="2031-11-14T03:00:00+00:00",
            missing_grid=False, home=None, away=None):
    h=np.array([.23,.30,.24,.13,.06,.025,.01,.005],dtype=float)
    a=np.array([.30,.35,.21,.09,.03,.015,.004,.001],dtype=float)
    grid=np.outer(h,a)
    grid/=grid.sum()
    idx=np.argsort(-grid.ravel())[:10]
    scores=[(int(i//8),int(i%8),float(grid.ravel()[i])) for i in idx]
    return {
        "fixture":{"id":fid,"provider":"API-Football","utcDate":kick,
                   "homeTeam":{"id":home if home is not None else fid*2,
                               "name":f"Team {fid} Home"},
                   "awayTeam":{"id":away if away is not None else fid*2+1,
                               "name":f"Team {fid} Away"}},
        "kickoff":kick,"archive_frozen":True,
        "frozen_at_utc":captured,
        "result":{"scores":scores,"matrix":None if missing_grid else grid.tolist(),
                  "model_version":"frozen-existing-v8","score_basis":"regulation"}
    }


def test_every_upcoming_same_day_pair_is_browsable_and_unchanged():
    fixtures=[fixture(i,kick=(NOW+timedelta(days=3,hours=i-101)).isoformat())
              for i in range(101,106)]
    originals=deepcopy(fixtures)
    groups,reasons=available(fixtures,NOW)
    assert len(groups)==1 and reasons=={}
    catalog=pair_catalog(next(iter(groups.values())),NOW)
    assert len(catalog)==10  # all five choose two combinations
    target=catalog[-1]
    report=generate(*target,NOW)
    assert report["status"]=="ready"
    assert report["method"]==VERSION
    assert report["audit_method"]==AUDIT_METHOD
    assert report["v25_five_original_v8_mass"]<=report["v8_optimal_five_estimated_mass"]+1e-10
    assert len(report["scores"])==5
    assert len({(r["score1"],r["score2"]) for r in report["scores"]})==5
    assert len(report["v8_same_pair_scores"])==5
    assert report["v25_lower_envelope_criterion"]<=report[
        "v8_optimal_five_estimated_mass"]+1e-10
    assert all(report["early_frozen_source"])
    assert report["accuracy_70pct_not_proven"] is True
    frontier = report["five_double_seventy_percent_model_feasibility"]
    assert frontier["max_five_model_mass_from_original_v8"] == pytest.approx(
        report["v8_optimal_five_estimated_mass"]
    )
    assert frontier["minimum_distinct_combinations_for_70pct_model_mass"] > 5
    assert frontier["five_original_v8_choices_reach_70pct_mass"] is False
    assert frontier["model_estimate_not_empirical_accuracy"] is True
    assert fixtures==originals


def test_missing_real_grid_is_rejected_even_if_ten_scores_exist():
    no_grid=fixture(201,missing_grid=True)
    valid=fixture(202)
    groups,reasons=available([no_grid,valid],NOW)
    assert reasons["missing_complete_original_score_grid"]==1
    assert sum(map(len,groups.values()))==1


def test_wrong_kickoff_after_start_and_future_source_are_rejected():
    before=fixture(101,kick=(NOW-timedelta(minutes=1)).isoformat())
    assert qualified(before,NOW)[1]=="invalid_or_started_fixture"
    future=fixture(102,captured=(NOW+timedelta(hours=1)).isoformat())
    assert qualified(future,NOW)[1]=="invalid_or_future_source_timestamp"
    old=fixture(103,captured=(NOW-timedelta(days=15)).isoformat())
    assert qualified(old,NOW)[1]=="source_older_than_14_days"


def test_different_sydney_day_and_shared_team_never_form_double():
    early=fixture(111)
    next_day=fixture(112,kick="2031-11-21T03:00:00+00:00")
    same_team=fixture(113,home=early["fixture"]["awayTeam"]["id"])
    candidates,_=available([early,next_day,same_team],NOW)
    day=next(c for c in candidates.values() if len(c)==2)
    assert pair_catalog(day,NOW)==[]
    assert pair_check(early,next_day,now=NOW) is not None


def test_auto_select_all_pairs_and_generate_all_pages_no_drop():
    matches=[fixture(i,kick=(NOW+timedelta(days=3,hours=i-101)).isoformat())
             for i in range(101,109)]
    byday,_=available(matches,NOW)
    catalog=pair_catalog(next(iter(byday.values())),NOW)
    assert len(catalog)==28
    auto=best_pair(catalog,NOW,limit=100)
    assert auto["status"]=="ready"
    assert auto["screened"]==28
    assert auto["total_pairs"]==28
    assert auto["shortlist_only"] is False
    first=rows_for_page(catalog,0,NOW,page_size=10)
    second=rows_for_page(catalog,1,NOW,page_size=10)
    third=rows_for_page(catalog,2,NOW,page_size=10)
    assert [x["shown_pairs"] for x in (first,second,third)]==[10,10,8]
    assert len(first["rows"])==50
    assert len(second["rows"])==50
    assert len(third["rows"])==40
    for page in (first,second,third):
        assert page["total_pairs"]==28
        assert page["pages"]==3
        assert page["not_a_bet_slip"] is True


def test_auto_research_is_bounded_and_honest_about_shortlist():
    fixtures=[fixture(i,kick=(NOW+timedelta(days=3,hours=i-101)).isoformat())
              for i in range(101,107)]
    groups,_=available(fixtures,NOW)
    catalog=pair_catalog(next(iter(groups.values())),NOW)
    report=best_pair(catalog,NOW,limit=2)
    assert report["status"]=="ready"
    assert report["total_pairs"]==15
    assert report["screened"]==2
    assert report["shortlist_only"] is True


def test_explicit_v25_generation_is_valid_for_original_immutable_audit():
    a,b=fixture(301),fixture(302,kick="2031-11-20T04:00:00+00:00")
    candidates,_=available([a,b],NOW)
    catalog=pair_catalog(next(iter(candidates.values())),NOW)
    output=generate(*catalog[0],NOW)
    record=audit_pair(output["first_fixture"],output["second_fixture"],
                      output["scores"],method=AUDIT_METHOD,now=NOW)
    assert record["source"]=="ai_five_double"
    assert record["method"]==AUDIT_METHOD
    legacy=audit_pair(output["first_fixture"],output["second_fixture"],
                      output["scores"],method="v25_professional_analyst",now=NOW)
    # Old frozen research and current on-demand output must NOT collide.
    assert record["record_id"]!=legacy["record_id"]

def test_ai_studio_v25_engine_visible_without_duplicate_streamlit_keys():
    from pathlib import Path
    app=Path("app.py").read_text(encoding="utf-8")
    studio=Path("ai_five_double_ui.py").read_text(encoding="utf-8")
    research=Path("v25_deep_research_panel.py").read_text(encoding="utf-8")
    assert app.count("render_ai_five_double_studio(st, [], {})")==1
    assert app.count("render_ai_five_double_studio(st, eligible, histories)")==1
    assert "render_v25_upcoming(st, [], {})" not in app
    assert "render_v25_upcoming(st, eligible, histories)" not in app
    assert 'key="zz_ai_five_engine_choice"' in studio
    assert studio.count("render_v25_upcoming(st,rows,histories)")==1
    assert "?view=ai_five" in research
    assert "Generate V25 Five Doubles now" in research
