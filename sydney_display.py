"""User-visible Sydney time formatting; original UTC evidence stays untouched."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SYDNEY = ZoneInfo("Australia/Sydney")


def sydney_timestamp(value, fmt="%d %b %Y, %I:%M:%S %p", *, missing="Time unavailable"):
    """Format a UTC or offset-aware instant with its Sydney AEDT/AEST designation.

    A naive datetime is assumed UTC for legacy evidence, not system-local time.
    Do not use for date-only strings.
    """
    if value is None or value == "":
        return missing
    try:
        if isinstance(value, datetime):
            instant = value
        else:
            instant = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        local = instant.astimezone(SYDNEY)
        return local.strftime(fmt).lstrip("0") + " " + local.tzname() + " (Sydney)"
    except (ValueError, TypeError, OverflowError):
        return missing
