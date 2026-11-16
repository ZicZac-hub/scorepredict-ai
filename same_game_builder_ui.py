"""Five same-fixture builder concepts from real individual bookmaker quotes.

The product is NOT a priced bet builder. Individual odds cannot be multiplied
into a valid correlated same-game payout; a bookmaker must confirm the bundle.
"""
from itertools import combinations
from math import prod

def builder_concepts(rows, target=80.0, limit=5):
    by_book={}
    for row in rows:
        try:
            odd=float(row["Quoted decimal odds"])
        except (ValueError,TypeError,KeyError):
            continue
        if not 1.15<=odd<=30:
            continue
        by_book.setdefault(row["Bookmaker"],[]).append(row)
    concepts=[]
    seen=set()
    for book, quotes in by_book.items():
        # One quote per market, so no conflicting over/under lines in one bet.
        distinct={}
        for row in sorted(quotes,key=lambda x:abs(x["Quoted decimal odds"]-2.2)):
            distinct.setdefault(row["Market"],row)
        pool=list(distinct.values())[:16]
        for size in (3,4,5,6):
            for legs in combinations(pool,size):
                cats={x["Category"] for x in legs}
                if len(cats)<2:
                    continue
                raw=prod(x["Quoted decimal odds"] for x in legs)
                if raw<target or raw>target*8:
                    continue
                signature=tuple(sorted((x["Market"],x["Selection"]) for x in legs))
                if signature in seen:
                    continue
                seen.add(signature)
                concepts.append({"Bookmaker":book,"Legs":list(legs),
                                 "Unadjusted product (NOT offered odds)":round(raw,2),
                                 "Market categories":len(cats)})
    return sorted(concepts,key=lambda x:(abs(x["Unadjusted product (NOT offered odds)"]-target),
                                       -x["Market categories"]))[:limit]

def render(st, rows, target=80.0):
    st.subheader("🧩 Five same-game builder concepts")
    st.warning("Accuracy status: UNVALIDATED. The app does not yet have settled "
               "historical corners/cards/shots/offsides selections linked to "
               "pre-match quotes, so these concepts are NOT statistically "
               "ranked or proven to offer value. Do not treat 80+ unadjusted "
               "multiplication as a confirmed bookmaker payout.")
    st.caption("Based on the screenshot's approach: goals, corners, cards, shots and "
               "other provider-quoted markets combined into alternative single-game "
               "concepts. Quotes must be from the SAME bookmaker. No bookmaker "
               "has confirmed the combined price.")
    concepts=builder_concepts(rows,target=target)
    if not concepts:
        st.info("Insufficient real individual market quotes to construct five "
                "illustrative 80+ combinations. No selections invented.")
        return
    st.caption("Ordering is by closeness to an illustrative 80× product, "
               "NOT by chance of winning. Five examples may be unavailable.")
    for i,concept in enumerate(concepts,1):
        with st.expander(f"Concept {i} · {concept['Bookmaker']} · unadjusted "
                         f"{concept['Unadjusted product (NOT offered odds)']:.2f}×",
                         expanded=i==1):
            for leg in concept["Legs"]:
                st.write(f"• {leg['Market']} — {leg['Selection']} "
                         f"({leg['Quoted decimal odds']:.2f} individual quote)")
            st.warning("Unadjusted multiplication only — NOT actual bet-builder odds. "
                       "Correlations, player availability and bookmaker combination "
                       "restrictions are NOT verified. Ask the bookmaker for a "
                       "real combined quote before considering this selection.")
