"""Date/time parsing shared by structured fields and extracted facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time

_ISO_DATE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})")
_US_DATE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})")
_ISO_TIME = re.compile(r"[T ](\d{2}):(\d{2})")
_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    m = _ISO_DATE.match(value)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = _US_DATE.match(value)
        if not m:
            return None
        mo, d, y = map(int, m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def parse_time(value: str | None) -> time | None:
    """Parse an `HH:MM` string or the time part of an ISO datetime."""
    if not value:
        return None
    m = _HHMM.match(value) or _ISO_TIME.search(value)
    if not m:
        return None
    h, mi = map(int, m.groups())
    if not (0 <= h < 24 and 0 <= mi < 60):
        return None
    return time(h, mi)


@dataclass(frozen=True, order=True)
class Moment:
    """A point in time where the time-of-day may be unknown.

    Wall-clock times are compared as written. Structured vitals carry a `Z` suffix and
    note times are local, but the record gives no timezone for notes; we document this
    and fall back to worst-case evaluation when same-day ordering is genuinely unclear.
    """

    day: date
    clock: time | None = None

    def sort_key(self) -> tuple[date, time]:
        return (self.day, self.clock or time(0, 0))

    def as_text(self) -> str:
        return f"{self.day.isoformat()} {self.clock.strftime('%H:%M')}" if self.clock else self.day.isoformat()


def moment_from(date_text: str | None, time_text: str | None = None) -> Moment | None:
    day = parse_date(date_text)
    if day is None:
        return None
    clock = parse_time(time_text) if time_text else parse_time(date_text)
    return Moment(day, clock)


def days_between(earlier: date, later: date) -> int:
    return (later - earlier).days


def iso(d: date | datetime | None) -> str | None:
    return d.isoformat() if d else None
