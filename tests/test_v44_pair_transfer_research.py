from datetime import datetime,timedelta,timezone
import numpy as np
import v44_pair_transfer_research as transfer

NOW=datetime(2025,2,1,10,tzinfo=timezone.utc)
MODEL={"provider":"API-Football","fixture_id":"1","home_team":"A","away_team":"B",
       "kickoff_utc":(NOW+timedelta(hours=2)).isoformat()}


def test_training_excludes_cutoff_and_duplicate_physical_events():
    roster=("anchor","tail")
    event=transfer.inventory._event(MODEL)
    def row(fid,when,event):
        return {"provider_id":("API-Football",fid),"event":event,"known":when,
                "kickoff":when-timedelta(hours=2),"roster":roster,
                "likelihood":np.array([[.1,.2]])}
    rows=[row("2",NOW-timedelta(hours=1),("earlier","C","D")),
          row("3",NOW,("same-time","E","F")),
          row("4",NOW+timedelta(hours=1),("future","G","H")),
          row("1",NOW-timedelta(hours=1),("other","I","J")),
          row("5",NOW-timedelta(hours=1),event)]
    assert len(transfer.prior_training(rows,[MODEL],NOW,roster))==1


def test_joint_five_handles_different_score_supports():
    a=np.full((5,5),1/25)
    b=np.full((7,6),1/42)
    arm=transfer._arm([a,b])
    choices=arm["five_doubles"]
    assert len(choices)==5
    assert len({(r["score1"],r["score2"]) for r in choices})==5
    assert np.isclose(arm["estimated_coverage"],5/(25*42))


def test_empty_matched_cohort_has_no_invented_accuracy():
    for report in transfer._summaries([]).values():
        assert report["pairs"]==0
        assert report["hit_rate"] is None
        assert report["score_log_loss"] is None
