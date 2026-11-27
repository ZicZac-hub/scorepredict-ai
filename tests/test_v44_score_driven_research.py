from datetime import datetime,timedelta,timezone
import numpy as np
import v44_score_driven_research as model

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
FIXTURE={"provider":"API-Football","id":999,"utcDate":"2025-02-01T12:00:00Z",
         "homeTeam":{"id":1},"awayTeam":{"id":2},"competition":{"id":39,"code":"PL","season":2024}}


def history():
    rows=[]
    for i in range(100):
        h,a=[(1,3),(4,2),(3,4),(4,3),(2,3),(3,1)][i%6]
        rows.append({"provider":"API-Football","id":i+1,
                     "utcDate":(NOW-timedelta(days=100-i)).isoformat(),
                     "status":"FINISHED","homeTeam":{"id":h},"awayTeam":{"id":a},
                     "competition":{"id":39,"code":"PL","season":2024},
                     "score":{"fullTime":{"home":3 if h==1 else 0 if h==2 else 1,
                                          "away":3 if a==1 else 0 if a==2 else 1}}})
    return rows


def test_ratings_direction_and_normalized_grid():
    result=model.predict(FIXTURE,history(),NOW)
    assert result["status"]=="ready"
    assert result["rates"][0]>result["rates"][1]
    assert np.isclose(result["grid"].sum(),1)
    assert result["grid"].shape==(32,32)


def test_future_result_and_other_league_cannot_change_prediction():
    rows=history()
    a=model.predict(FIXTURE,rows,NOW)
    future={**rows[0],"id":800,"utcDate":(NOW+timedelta(minutes=1)).isoformat(),
            "score":{"fullTime":{"home":20,"away":0}}}
    other={**rows[0],"id":801,"competition":{"id":140,"season":2024}}
    b=model.predict(FIXTURE,[future,other,*reversed(rows)],NOW)
    assert np.array_equal(a["grid"],b["grid"])


def test_sparse_history_abstains_and_blend_preserves_mass():
    assert model.predict(FIXTURE,history()[:5],NOW)["status"]=="abstained"
    base=np.full((5,6),1/30)
    challenger=model.predict(FIXTURE,history(),NOW)["grid"]
    mixed=model._blend(base,challenger)
    assert mixed.shape==(32,32)
    assert np.isclose(mixed.sum(),1)
    assert np.allclose(mixed[:5,:6],.75*base+.25*challenger[:5,:6])


def test_receipt_selection_is_asof():
    past={"capture":NOW-timedelta(hours=1),"snapshot_sha256":"a","path":"a"}
    future={"capture":NOW+timedelta(seconds=1),"snapshot_sha256":"b","path":"b"}
    assert model._select([future,past],NOW)==past
    assert model._select([future],NOW) is None
