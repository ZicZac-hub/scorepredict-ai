from datetime import datetime,timedelta,timezone
import v44_refresh_priority as p

NOW=datetime(2026,1,1,tzinfo=timezone.utc)

def candidate(i,hours):
    kick=NOW+timedelta(hours=hours)
    f={"provider":"API-Football","id":i,"utcDate":kick.isoformat(),
       "homeTeam":{"id":i*2},"awayTeam":{"id":i*2+1}}
    return kick,f,"PL"

def test_reserved_slots_are_bounded_and_preserve_remaining_queue(monkeypatch):
    rows=[candidate(i,i) for i in range(1,6)]
    monkeypatch.setattr(p,"targets",lambda **kw:{p.fixture_key(r[1]):r[0] for r in rows[2:]})
    ordered,n=p.prioritize(rows,{},now=NOW,limit=4)
    assert n==2
    assert [r[1]["id"] for r in ordered]==[3,4,1,2,5]
    assert set(r[1]["id"] for r in ordered)==set(range(1,6))
    assert p.prioritize(rows,{},now=NOW,limit=1)[1]==1

def test_existing_near_forecasts_and_wrong_teams_not_prioritized(monkeypatch):
    rows=[candidate(i,i) for i in range(1,4)]
    wrong=(*p.fixture_key(rows[1][1])[:-1],"wrong")
    monkeypatch.setattr(p,"targets",lambda **kw:{wrong:rows[1][0],p.fixture_key(rows[2][1]):rows[2][0]})
    ordered,n=p.prioritize(rows,{("API-Football","3","v8_near"):{}},now=NOW,limit=2)
    assert ordered==rows and n==0

def test_original_pair_window_must_be_eligible(tmp_path,monkeypatch):
    (tmp_path/"sets").mkdir()
    (tmp_path/"sets"/"a.json").write_text("{}")
    def m(i,h):
        f=candidate(i,h)[1]
        return {"provider":f["provider"],"fixture_id":f["id"],"kickoff_utc":f["utcDate"],
                "home_team_id":f["homeTeam"]["id"],"away_team_id":f["awayTeam"]["id"]}
    monkeypatch.setattr(p.sources,"_pair",lambda *a,**kw:({},[m(1,1),m(2,2)]))
    assert len(p.targets(now=NOW,pair_root=tmp_path))==2
    monkeypatch.setattr(p.sources,"_pair",lambda *a,**kw:({},[m(1,0),m(2,2)]))
    assert p.targets(now=NOW,pair_root=tmp_path)=={}
