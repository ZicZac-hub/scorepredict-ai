"""Every zero-save market page has auditable, quota-neutral reasons."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from v22_evidence_models import parse_markets
import v24_bulk_market_capture as bulk

NOW=datetime(2032,7,3,3,tzinfo=timezone.utc)


def item(fid=123,update=None):
    return {"fixture":{"id":fid,"date":(NOW+timedelta(hours=8)).isoformat()},
            "update":update if update is not None else NOW.isoformat(),
            "bookmakers":[{"name":"Book","bets":[
                {"name":"Match Winner","values":[
                    {"value":"Home","odd":2.25},
                    {"value":"Draw","odd":3.3},
                    {"value":"Away","odd":3.5}]},
                {"name":"Goals Over/Under","values":[
                    {"value":"Over 2.5","odd":1.85},
                    {"value":"Under 2.5","odd":2.07}]}]}]}


def test_parser_distinguishes_missing_future_stale_and_valid_sources():
    missing=item(); missing.pop("update")
    diag={}
    assert parse_markets([missing],observed_at=NOW,diagnostics=diag)=={}
    assert diag=={"missing_or_invalid_source_timestamp":2}
    diag={}
    assert parse_markets([item(update=(NOW+timedelta(seconds=1)).isoformat())],
                         observed_at=NOW,diagnostics=diag)=={}
    assert diag=={"future_source_timestamp":2}
    diag={}
    assert parse_markets([item(update=(NOW-timedelta(hours=13)).isoformat())],
                         observed_at=NOW,diagnostics=diag)=={}
    assert diag=={"stale_source_timestamp":2}
    diag={}
    prices=parse_markets([item()],observed_at=NOW,diagnostics=diag)
    assert set(prices)=={"1x2","over_2_5"}
    assert diag=={"accepted_bookmaker_market":2}


def test_zero_save_page_reports_actual_rows_and_stale_quotes(monkeypatch,tmp_path):
    monkeypatch.setattr(bulk,"_day_priority",lambda clock:[NOW.date().isoformat()])
    def request(*args):
        return {"metadata":{"fetched_at_utc":NOW.isoformat(),"stale":False},
                "data":[item(i,update=(NOW-timedelta(hours=13)).isoformat()) for i in range(10)]}
    report=bulk.capture(now=NOW,clock=lambda:NOW,enabled=True,api_key="stub",request=request,
                        root=tmp_path,original_root=tmp_path/"original",
                        source_root=tmp_path/"sources",originals=[],max_per_run=1,daily_limit=1)
    assert report["queries"]==1 and report["provider_fixture_rows"]==10
    assert not report["new_catalog_fixtures"]
    assert report["catalog_rejection_counts"]=={"missing_or_invalid_fixture_id":1,
             "no_supported_timestamped_market":9}
    assert report["quote_diagnostics"]=={"stale_source_timestamp":18}


def test_page_counts_are_local_and_binding_reports_stale_separately(monkeypatch,tmp_path):
    date=NOW.date().isoformat()
    monkeypatch.setattr(bulk,"_day_priority",lambda clock:[date])
    def request(key,endpoint,params):
        count=10 if params["page"]==1 else 2
        return {"metadata":{"fetched_at_utc":NOW.isoformat(),"stale":False},
                "data":[item(params["page"]*100+i) for i in range(count)]}
    report=bulk.capture(now=NOW,clock=lambda:NOW,enabled=True,api_key="stub",request=request,
                        root=tmp_path,original_root=tmp_path/"original",
                        source_root=tmp_path/"sources",originals=[],max_per_run=2,daily_limit=2)
    assert report["provider_fixture_rows"]==12
    markers=[json.loads(p.read_text()) for p in (tmp_path/"requests"/date).glob("*.json")]
    assert {r["page"]:r["saved_market_fixture_count"] for r in markers}=={1:10,2:2}
    market=json.loads(next((tmp_path/"fixtures"/"100").glob("*.json")).read_text())
    source={"fixture_id":"100","kickoff_utc":(NOW+timedelta(days=2)).isoformat(),"competition":"PL"}
    # Source kickoff must match before evaluating the age of prices.
    market["source_fixture_kickoff_utc"]=source["kickoff_utc"]
    market["snapshot_sha256"]=bulk._hash({k:v for k,v in market.items() if k!="snapshot_sha256"})
    assert bulk.catalog_rejection_reason(market,source,NOW) is None
    assert bulk.catalog_rejection_reason(market,source,NOW+timedelta(hours=13))=="market_quotes_stale"


def test_workflow_cannot_publish_manual_feature_branch_into_main():
    for filename,jobname in (("prematch-evidence.yml","capture"),
                             ("ziczac-unified-audit.yml","audit")):
        workflow=Path(".github/workflows",filename).read_text()
        assert "git push origin HEAD:main" in workflow
        job=workflow.split("  "+jobname+":\n",1)[1].split("    steps:",1)[0]
        assert "if: github.ref == 'refs/heads/main'" in job
