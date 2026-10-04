# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
The world's major cities and the time on the clocks there.

    cities()              the table (data/cities.json): every national capital,
                          every place of 300,000 people or more, the region
                          capitals and the Antarctic stations with a clock of
                          their own - 1,584 places, each with its time zone
    search(text)          the cities matching what is typed (accents ignored;
                          English, other European and Chinese/Japanese names),
                          the best first
    parse_coords(text)    "48.86, 2.35" / "3.14N 101.69E" / "-33.87 151.21 +10"
                          -> (lat, lon, hours from UTC or None), else None
    zone_at(lat, lon)     the time zone of any point on the Earth: a city's
                          within 15 km, else the 0.1-degree grid of zones
                          (data/timezones.bin, land and sea)
    offset_s(zone, when)  the zone's offset from UTC (s) at an instant: the
                          tz database's offsets 2020-2035 are stored, so no tz
                          database is needed; outside those years zoneinfo when
                          the computer has one, else the nearest stored year
    to_utc(local, zone)   a time on the place's clock -> UTC
    utc_label(seconds)    "UTC+8", "UTC+5:30", "UTC-3"
    name_for(lat, lon)    the name of a point typed as coordinates: the city
                          it is in or near, else its coordinates
    parse_date(text, today)  a date typed in the date dropdown, else None
    parse_time(text)      an hour typed in the hour dropdown, else None

Sources: Natural Earth (public domain); time-zone boundaries of
timezone-boundary-builder 2025b (OpenStreetMap contributors, ODbL); the IANA
tz database 2026c.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
import unicodedata
import zlib
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass(frozen=True)
class City:
    label: str          # "Kuala Lumpur, Malaysia"
    name_key: str       # "kuala lumpur"
    search_key: str     # every name, folded to ASCII lower case, and the CJK names
    lat: float
    lon: float
    zone: str           # IANA name
    population: int


_DATA = None
_GRID = None
_SPLIT = re.compile(r"[^\w]+")


def _data():
    global _DATA
    if _DATA is None:
        with open(os.path.join(HERE, "cities.json"), encoding="utf-8") as f:
            d = json.load(f)
        d["city_list"] = [City(*r) for r in d["cities"]]
        d["words"] = [tuple(w for w in _SPLIT.split(c.search_key) if w) for c in d["city_list"]]
        d["starts"] = {z: [t for t, _ in seq] for z, seq in d["zones"].items()}
        _DATA = d
    return _DATA


def cities() -> list[City]:
    return _data()["city_list"]


def fold(s: str) -> str:
    """lower case, accents gone: what is typed is matched against this"""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c)).lower()


def search(text: str) -> list[City]:
    """Every city with a name (or country) beginning with each word typed -
    "san f" is San Francisco, not every San in California - or, if none,
    holding each word anywhere ("burg"); the cities whose name starts with
    what is typed first, then the rest, the larger first.  Nothing typed:
    all of them, in alphabetical order."""
    q = " ".join(fold(text).replace(",", " ").split())
    words = [w for w in _SPLIT.split(q) if w]
    if not words:
        return list(cities())
    d = _data()
    hits = [c for c, ws in zip(d["city_list"], d["words"])
            if all(any(x.startswith(w) for x in ws) for w in words)]
    if not hits:
        hits = [c for c in cities() if all(w in c.search_key for w in words)]

    def rank(c):
        if c.name_key.startswith(q):
            r = 0
        elif c.name_key.startswith(words[0]):
            r = 1
        elif any(part.startswith(words[0]) for part in c.name_key.split()):
            r = 2
        else:
            r = 3
        return (r, -c.population, c.label)
    return sorted(hits, key=rank)


def find(name: str) -> City | None:
    """The city a name means (for --city): an exact label or name first,
    else the best match of a search; None if nothing matches."""
    q = fold(name).strip()
    for c in cities():
        if fold(c.label) == q:
            return c
    exact = [c for c in cities() if c.name_key == q]
    if exact:
        return max(exact, key=lambda c: c.population)
    hits = search(name)
    return hits[0] if hits else None


# ------------------------------------------------------------- coordinates --
_NUM = r"[+-]?\d+(?:\.\d+)?"
_HEMI = re.compile(rf"^\s*({_NUM})\s*°?\s*([NSns])[\s,;]+({_NUM})\s*°?\s*([EWew])"
                   rf"(?:[\s,;]+(?:UTC|GMT)?\s*([+-]?\d+(?::\d\d|\.\d+)?))?\s*$")
_PLAIN = re.compile(rf"^\s*({_NUM})\s*°?[\s,;]+({_NUM})\s*°?"
                    rf"(?:[\s,;]+(?:UTC|GMT)?\s*([+-]?\d+(?::\d\d|\.\d+)?))?\s*$")


def _hours(s: str | None) -> float | None:
    if not s:
        return None
    if ":" in s:
        sign = -1.0 if s.strip().startswith("-") else 1.0
        h, m = s.strip().lstrip("+-").split(":")
        return sign * (int(h) + int(m) / 60.0)
    return float(s)


def parse_coords(text: str):
    """(lat, lon, hours from UTC or None) from what is typed: "48.86, 2.35",
    "48.86 2.35", "3.14N 101.69E", "33.87 S, 151.21 E", each optionally
    followed by the clock's offset ("+8", "-3", "+5:30", "UTC+10").  None if
    it is not coordinates, or not on the Earth."""
    m = _HEMI.match(text or "")
    if m:
        lat = float(m.group(1)) * (-1.0 if m.group(2) in "Ss" else 1.0)
        lon = float(m.group(3)) * (-1.0 if m.group(4) in "Ww" else 1.0)
        hrs = _hours(m.group(5))
    else:
        m = _PLAIN.match(text or "")
        if not m:
            return None
        lat, lon, hrs = float(m.group(1)), float(m.group(2)), _hours(m.group(3))
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 360.0):
        return None
    if lon > 180.0:
        lon -= 360.0
    if hrs is not None and not (-12.0 <= hrs <= 14.0):
        return None
    return lat, lon, hrs


def km(a_lat, a_lon, b_lat, b_lon) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    c = (math.sin(p1) * math.sin(p2)
         + math.cos(p1) * math.cos(p2) * math.cos(math.radians(b_lon - a_lon)))
    return 6371.0 * math.acos(max(-1.0, min(1.0, c)))


def nearest(lat: float, lon: float) -> tuple[City, float]:
    return min(((c, km(lat, lon, c.lat, c.lon)) for c in cities()), key=lambda t: t[1])


def name_for(lat: float, lon: float) -> str:
    """A name for a point typed as coordinates: the city (within 3 km),
    "near" the city (within 30 km), else the coordinates themselves."""
    c, d = nearest(lat, lon)
    if d <= 3.0:
        return c.label
    if d <= 30.0:
        return f"near {c.label}"
    return f"{lat:.3f}, {lon:.3f}"


def _grid():
    global _GRID
    if _GRID is None:
        import numpy as np
        g = _data()["grid"]
        with open(os.path.join(HERE, g["file"]), "rb") as f:
            raw = zlib.decompress(f.read())
        _GRID = np.frombuffer(raw, "<u2").reshape(g["height"], g["width"])
    return _GRID


def zone_at(lat: float, lon: float) -> tuple[str, str]:
    """(zone, how it was found) for any point: the zone of a city within
    15 km; else the grid's cell - a sea cell next to land takes the land's
    zone (territorial waters keep the coast's clock)."""
    c, d = nearest(lat, lon)
    if d <= 15.0:
        return c.zone, f"{c.label}, {d:.0f} km"
    g = _data()["grid"]
    names = g["zones"]
    grid = _grid()
    n = g["cells_per_degree"]
    w, h = g["width"], g["height"]
    i = int(math.floor((lon + 180.0) * n)) % w
    j = min(max(int(math.floor((90.0 - lat) * n)), 0), h - 1)
    z = names[int(grid[j, i])]
    if z.startswith("Etc/"):
        land = {}
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                zz = names[int(grid[min(max(j + dj, 0), h - 1), (i + di) % w])]
                if not zz.startswith("Etc/"):
                    land[zz] = land.get(zz, 0) + 1
        if land:
            z = max(sorted(land), key=lambda k: land[k])
            return z, "the coast's zone"
        return z, "the sea's zone"
    return z, "the zone map"


# ------------------------------------------------------------------ offsets --
_T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2036, 1, 1, tzinfo=timezone.utc)


def _stored(zone: str, t: float) -> int | None:
    d = _data()
    seq = d["zones"].get(zone)
    if seq is None:
        return None
    i = bisect.bisect_right(d["starts"][zone], t) - 1
    return int(seq[max(i, 0)][1])


def offset_s(zone: str | None, when: datetime) -> int | None:
    """The zone's offset from UTC at an instant, s (None for an unknown zone)."""
    if not zone:
        return None
    when = when.astimezone(timezone.utc)
    if _T0 <= when < _T1:
        return _stored(zone, when.timestamp())
    try:
        from zoneinfo import ZoneInfo
        off = when.astimezone(ZoneInfo(zone)).utcoffset()
        if off is not None:
            return int(off.total_seconds())
    except Exception:                                            # noqa: BLE001
        pass
    # no tz database here: the same day and hour of the nearest stored year
    year = 2020 if when < _T0 else 2035
    try:
        same = when.replace(year=year)
    except ValueError:                                           # 29 February
        same = when.replace(year=year, day=28)
    return _stored(zone, same.timestamp())


def to_utc(local: datetime, zone: str | None = None, fixed_s: int | None = None) -> datetime:
    """A time on the place's clock (naive) -> UTC.  An hour the clock repeats
    (summer time ending) is read as its first occurrence; an hour it skips
    (summer time starting) with the offset in force before the change."""
    t = local.replace(tzinfo=None).replace(tzinfo=timezone.utc).timestamp()
    if zone is None:
        return datetime.fromtimestamp(t - (fixed_s or 0), timezone.utc)

    def off(u):
        return offset_s(zone, datetime.fromtimestamp(u, timezone.utc))
    offs = {off(t + d) for d in (-86400, -43200, 0, 43200, 86400)} - {None}
    valid = [t - o for o in offs if off(t - o) == o]
    if valid:
        return datetime.fromtimestamp(min(valid), timezone.utc)
    return datetime.fromtimestamp(t - (off(t - 86400) or 0), timezone.utc)


def utc_label(seconds: int | float | None) -> str:
    if seconds is None:
        return "UTC"
    s = int(round(seconds / 60.0)) * 60
    if s == 0:
        return "UTC"
    sign = "-" if s < 0 else "+"
    h, m = divmod(abs(s) // 60, 60)
    return f"UTC{sign}{h}" + (f":{m:02d}" if m else "")


# ---------------------------------------------------------- dates and hours --
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _month_of(word: str) -> int | None:
    """1-12 for a month's name or its first three letters or more ("oct",
    "sept", "october"), else None"""
    w = word.rstrip(".")
    if len(w) < 3:
        return None
    for k, m in enumerate(_MONTHS):
        if m.startswith(w):
            return k + 1
    return None


def _is_weekday(word: str) -> bool:
    w = word.rstrip(".")
    return len(w) >= 3 and any(d.startswith(w) for d in _WEEKDAYS)


def _make_date(y, m, d) -> date | None:
    try:
        out = date(int(y), int(m), int(d))
    except (TypeError, ValueError):
        return None
    return out if 1900 <= out.year <= 2100 else None


def _nearest_year(month: int, day: int, today: date) -> date | None:
    """A day and month with no year: of last year, this year or next, the
    one nearest to today (a tie goes to the one to come)."""
    best = None
    for y in (today.year - 1, today.year, today.year + 1):
        d = _make_date(y, month, day)
        if d is None:
            continue
        k = (abs((d - today).days), d < today)
        if best is None or k < best[0]:
            best = (k, d)
    return None if best is None else best[1]


def parse_date(text: str, today: date) -> date | None:
    """A date typed in the date dropdown, else None:
        today, tomorrow, yesterday;  +3, -2 (days from today), in 3 days,
        3 days ago;  2026-10-05, 2026/10/05;  5/10/2026, 5-10-26, 5.10
        (day first, as in Malaysia and Europe);  5 Oct, 5th October,
        Oct 5, October 5 2026 - a weekday in front (Mon 05 Oct 2026, as the
        list shows dates) is let be.  With no year: the nearest such date."""
    q = " ".join(fold(text).replace(",", " ").split())
    if not q:
        return None
    rel = {"today": 0, "now": 0, "tomorrow": 1, "yesterday": -1}
    if q in rel:
        return today + timedelta(days=rel[q])
    m = re.fullmatch(r"([+-])\s*(\d{1,4})\s*(?:d|days?)?", q)
    if m:
        n = int(m.group(2))
        return today + timedelta(days=n if m.group(1) == "+" else -n)
    m = re.fullmatch(r"in (\d{1,4}) days?", q)
    if m:
        return today + timedelta(days=int(m.group(1)))
    m = re.fullmatch(r"(\d{1,4}) days? ago", q)
    if m:
        return today - timedelta(days=int(m.group(1)))
    words = q.split()
    if len(words) > 1 and _is_weekday(words[0]):
        q = " ".join(words[1:])
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", q)
    if m:
        return _make_date(m.group(1), m.group(2), m.group(3))
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})(?:[-/.](\d{4}|\d{2}))?", q)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        if m.group(3) is None:
            return _nearest_year(month, day, today) if 1 <= month <= 12 else None
        y = int(m.group(3))
        return _make_date(y + 2000 if y < 100 else y, month, day)
    for pat, di, mi in ((r"(\d{1,2})(?:st|nd|rd|th)?\s*([a-z]+)\.?(?:\s+(\d{4}))?", 1, 2),
                        (r"([a-z]+)\.?\s*(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?", 2, 1)):
        m = re.fullmatch(pat, q)
        if m and _month_of(m.group(mi)):
            month, day = _month_of(m.group(mi)), int(m.group(di))
            if m.group(3):
                return _make_date(m.group(3), month, day)
            return _nearest_year(month, day, today)
    return None


def parse_time(text: str) -> dtime | None:
    """An hour typed in the hour dropdown, else None: 15:30, 15.30, 15h30,
    1530, 930, 15, 3pm, 3:30 pm, 12am (midnight), 12pm, noon, midnight."""
    q = fold(text).replace(" ", "")
    if not q:
        return None
    if q in ("noon", "midday"):
        return dtime(12, 0)
    if q == "midnight":
        return dtime(0, 0)
    ampm = r"(am|pm|a\.m\.|p\.m\.|a|p)?"
    m = (re.fullmatch(r"(\d{1,2})(?:[:.h](\d{2}))?" + ampm, q)
         or re.fullmatch(r"(\d{1,2})(\d{2})" + ampm, q))
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if mi > 59:
        return None
    if ap:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if ap.startswith("p") else 0)
    elif h > 23:
        return None
    return dtime(h, mi)
