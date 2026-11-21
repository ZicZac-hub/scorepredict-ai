"""Read-only development replay of pinned CC0 English league results.

Source dates, not verified kickoff timestamps. Midnight UTC is solely a
calendar ordering surrogate; no prospective claim. Previously explored
2024-25 season, so this is not an untouched holdout.
"""
import hashlib
import json
from urllib.request import urlopen
from backtest_dynamic_dixon_coles import evaluate

PIN="e6744429ee395bc86f247348c6184bb08d4eb361"
HASHES={
    "2022-23":"8d09f0f9846981626cdfec48903e1f21e7c79aabe6f92535c9e092e8e79b808b",
    "2023-24":"03e13eafbf78dfe00d7e89dd3bf6643986eb6e8fd86c7664aeb8c5bc0bed88d0",
    "2024-25":"c3d18f552d83382cd4b6b21127adbbfe38e7f8618ddc979fb5404f2df07d0a75",
}
def load_matches():
    matches=[]
    missing=[]
    for season,digest in HASHES.items():
        url=f"https://raw.githubusercontent.com/openfootball/football.json/{PIN}/{season}/en.1.json"
        with urlopen(url,timeout=30) as response:
            raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError("Pinned source hash mismatch: "+season)
        for i,m in enumerate(json.loads(raw)["matches"]):
            day=m["date"]
            score=(m.get("score") or {}).get("ft")
            if not isinstance(score,list) or len(score)!=2 or any(type(g) is not int or g<0 for g in score):
                missing.append(day)
                continue
            matches.append({"id":f"{season}:{i}","provider":"openfootball-calendar-replay",
                "competition":{"id":"en.1"},"utcDate":day+"T00:00:00Z",
                "status":"FINISHED","homeTeam":{"id":m["team1"]},
                "awayTeam":{"id":m["team2"]},"score":{"fullTime":{"home":score[0],"away":score[1]}}})
    return matches,missing

def main():
    matches,missing=load_matches()
    # Exclude incomplete evaluation slates from all models equally.
    result=evaluate([m for m in matches if m["utcDate"][:10] not in missing],start="2024-08-01")
    print(json.dumps({"sources":HASHES,"source_date_surrogate":True,
        "missing_score_dates":sorted(set(missing)),"normalized_matches":len(matches),
        "development_only":True,"prospective_proof":False,"comparison":result},indent=2))
if __name__=="__main__":
    main()
