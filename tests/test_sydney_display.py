"""Sydney display tests for V25 and archived source timestamps."""
from datetime import datetime, timezone
from sydney_display import sydney_timestamp


def test_v25_screenshot_utc_converts_to_sydney_aedt():
    utc_original = "2026-11-16T22:40:23.581206+00:00"
    assert sydney_timestamp(utc_original) == "17 Nov 2026, 09:40:23 AM AEDT (Sydney)"
    assert utc_original == "2026-11-16T22:40:23.581206+00:00"


def test_winter_time_is_aest_not_aedt():
    assert sydney_timestamp("2026-06-01T22:45:00Z") == "2 Jun 2026, 08:45:00 AM AEST (Sydney)"


def test_naive_and_aware_utc_datetime_are_displayed_without_mutation():
    aware = datetime(2026, 11, 16, 22, 40, 23, tzinfo=timezone.utc)
    naive = datetime(2026, 11, 16, 22, 40, 23)
    assert sydney_timestamp(aware) == sydney_timestamp(naive)
    assert aware.utcoffset().total_seconds() == 0


def test_invalid_or_missing_time_has_no_fabricated_display():
    assert sydney_timestamp(None) == "Time unavailable"
    assert sydney_timestamp("nonsense", missing="Unknown") == "Unknown"
