"""Advance source observation time while network requests are in progress.

An explicit ``now`` anchors historical simulations without consulting today's
wall clock. A monotonic elapsed duration still prevents a slow live request
from being backdated to its start. Tests can inject a deterministic clock.
"""
from datetime import datetime, timedelta, timezone
import time

from v10_prospective import utc


def capture_clock(now=None, clock=None):
    start = utc(now if now is not None else
                (clock() if clock is not None else datetime.now(timezone.utc)))
    began = time.monotonic()

    def observed():
        instant = (utc(clock()) if clock is not None else
                   start + timedelta(seconds=max(0.0, time.monotonic() - began)))
        if instant < start:
            raise ValueError("Source observation clock moved before capture start")
        return instant

    return start, observed
