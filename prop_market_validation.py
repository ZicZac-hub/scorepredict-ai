"""Historical market validation for same-game builders.

Strictly use externally verified settled outcomes and original pre-match quotes.
Never estimate joint probability by multiplying correlated leg probabilities.
"""
from datetime import datetime, timezone
from math import sqrt

def utc(value):
    try:
        t=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except (TypeError,ValueError,OverflowError):
        return None

def evaluate_market(records, min_samples=30):
    """Each record: event_id, market, selection, quoted_at, kickoff,
    quote, settled_won (bool). Each distinct event counts once per selection.
    """
    groups={}
    seen=set()
    excluded={"unverified":0,"late_quote":0,"duplicate":0,"bad_quote":0}
    for r in records:
        try:
            key=(str(r["event_id"]),str(r["market"]),str(r["selection"]))
            quote=float(r["quote"])
            won=r.get("settled_won")
            before=utc(r.get("quoted_at"))
            kickoff=utc(r.get("kickoff"))
        except (KeyError,ValueError,TypeError):
            excluded["unverified"]+=1
            continue
        if not all(key) or type(won) is not bool:
            excluded["unverified"]+=1
            continue
        if key in seen:
            excluded["duplicate"]+=1
            continue
        seen.add(key)
        if not before or not kickoff or before>=kickoff:
            excluded["late_quote"]+=1
            continue
        if not 1.01<=quote<100000:
            excluded["bad_quote"]+=1
            continue
        name=(key[1],key[2])
        groups.setdefault(name,[]).append((quote,won))
    out=[]
    for (market,selection),samples in sorted(groups.items()):
        n=len(samples)
        hits=sum(won for _,won in samples)
        rate=hits/n
        # Wilson interval, two-sided 95%, not a profitability guarantee.
        z=1.96
        denom=1+z*z/n
        centre=(rate+z*z/(2*n))/denom
        half=z*sqrt((rate*(1-rate)+z*z/(4*n))/n)/denom
        out.append({"Market":market,"Selection":selection,"Samples":n,
                    "Wins":hits,"Observed hit rate":round(rate,4),
                    "95% Wilson lower":round(max(0,centre-half),4),
                    "95% Wilson upper":round(min(1,centre+half),4),
                    "Mean quoted odds":round(sum(q for q,_ in samples)/n,3),
                    "Evidence status":"exploratory" if n<min_samples else "sample threshold met"})
    return {"markets":out,"excluded":excluded,
            "joint_probability_validated":False,
            "note":"Marginal market frequencies do not establish correlated same-game builder success."}
