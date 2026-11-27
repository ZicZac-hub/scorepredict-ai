"""Read-only early history coverage by original forecast and pair cutoff."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json

from forecast_ledger import _forecast,LedgerValidationError
from professional_double_shadow import _validated
from v10_prospective import utc
import v42_research_trial as sources
import v44_early_history as early


def qualifies(record,model,cutoff):
    event=[str(model["provider"]),str(model["fixture_id"]),utc(model["kickoff_utc"]).isoformat(),
           str(model["home_team_id"]),str(model["away_team_id"])]
    return (record["event"]==event
            and record["history_input_digest"]==model["input_digest"]
            and utc(record["as_of_utc"])==utc(model["evidence_as_of_utc"])
            and utc(record["captured_at_utc"])<=utc(model["captured_at_utc"])<=utc(cutoff))


def evaluate(*,root=early.ROOT,source_root=sources.SOURCE_ROOT,pair_root=sources.PAIR_ROOT,now=None):
    clock=utc(now or datetime.now(timezone.utc))
    counts,rejects=Counter(),Counter()
    receipts={}
    for path in sorted((Path(root)/"receipts").glob("*.json")):
        try:
            row=early.load(path,root=root,now=clock)
            receipts[tuple(row["event"])]=row
        except (KeyError,ValueError,TypeError,OSError,AttributeError,ArithmeticError):
            rejects["invalid_receipt"]+=1
    counts["qualified_receipts"]=len(receipts)
    for path in sorted((Path(pair_root)/"sets").glob("*.json")):
        try:
            record=json.loads(path.read_text(encoding="utf-8"))
            models,_,_=_validated(record,source_root=source_root,now=clock)
            if path.stem!=record["sydney_date"]:
                raise ValueError("wrong_day")
            cutoff=utc(record["predicted_at_utc"])
            available=0
            for model in models:
                event=(str(model["provider"]),str(model["fixture_id"]),utc(model["kickoff_utc"]).isoformat(),
                       str(model["home_team_id"]),str(model["away_team_id"]))
                receipt=receipts.get(event)
                if receipt and qualifies(receipt,model,cutoff):
                    available+=1
            counts["authenticated_original_pairs"]+=1
            counts["pairs_with_"+str(available)+"_qualified_legs"]+=1
        except (KeyError,ValueError,TypeError,OSError,AttributeError,ArithmeticError,LedgerValidationError):
            rejects["invalid_original_pair"]+=1
    return {"study_type":"original_cutoff_early_history_coverage",
            "counts":dict(counts),"rejections":dict(rejects),"external_requests":0,
            "automatic_promotion":False,"historical_backfill_performed":False}


if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2))
