"""Reserve one existing heavy history-refresh slot for isolated recovery.

No new fixture selections, no extra quota, no changes to original records.
The planner inspects only authenticated original fixtures and source receipts.
"""
from datetime import timedelta
from pathlib import Path

from forecast_ledger import read_file_records
from v10_prospective import utc
from v44_refresh_priority import targets,fixture_key
from v44_refresh_expiry_audit import inspect_leg
import v44_recovery_capture as recovery
import v44_early_history as early
import v42_research_trial as sources


def plan(fixtures,awaiting,shadow,*,now,limit,root=recovery.ROOT,
         source_root=sources.SOURCE_ROOT,history_root=early.ROOT,pair_root=sources.PAIR_ROOT):
    """Return at most one recovery candidate plus the normal queue, capped.

    Preserve the first two already-prioritized normal slots. A recovery
    candidate replaces one later slot, never adds a slot. Do not retry
    fixtures already recovered, or those with usable exact receipts.
    """
    queue=list(awaiting)
    if limit<3:
        return queue,set()
    now=utc(now)
    required=targets(now=now,pair_root=pair_root,source_root=source_root)
    # Do not turn either of the protected normal market refreshes into recovery.
    protected=set()
    for row in queue[:2]:
        try:
            protected.add(fixture_key(row[1]))
        except (ValueError,KeyError,TypeError,AttributeError):
            continue
    options=[]
    for row in fixtures:
        try:
            key=fixture_key(row[1])
            if key in protected or key not in required or not timedelta(minutes=45)<key[2]-now<=timedelta(hours=48):
                continue
            near=shadow.get((*key[:2],"v8_near"))
            if near is None:
                continue
            if any(m.get("model_key")==recovery.MODEL_KEY for m in
                   read_file_records(root=Path(root)/"archive",kind="forecasts",
                                     provider=key[0],fixture_id=key[1])):
                continue
            opening=max(now,required[key]-timedelta(hours=3))
            status=inspect_leg(near,opening,now,source_root,history_root)
            if status not in ("one_shot_near_without_exact_history","one_shot_near_expired"):
                continue
            options.append((required[key],key[2],key[0],key[1],row,key))
        except (ValueError,KeyError,TypeError,OSError,AttributeError,ArithmeticError):
            continue
    if not options:
        return queue,set()
    *_,row,key=min(options,key=lambda v:v[:4])
    # Insert only after the two protected normal slots.
    queue=[item for item in queue if fixture_key(item[1])!=key]
    queue.insert(min(2,len(queue)),row)
    return queue,{key}
