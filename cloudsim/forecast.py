# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
The weather as a function of time.

Open-Meteo answers every request with all the hours it has - three days back
and ten ahead, instantaneous values on the hour.  The program used to keep
the one hour nearest to the clock and draw the sky from it until it was
asked again, so the sky either stood still for hours or, when the weather
was fetched again, was replaced all at once.

A Forecast keeps every hour and gives the atmosphere at any moment: between
two hours each quantity is interpolated linearly in time (temperature,
humidity, the height of each pressure level, the wind by its components,
the model's cloud cover), which is what the forecast's hourly output
supports and all it supports - it says nothing about what happens inside
the hour.  The regional grid around the place is kept the same way
(RegionSeries); its cloud maps are interpolated along the wind by
region.RegionMaps.build_series.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from . import sounding
from .sounding import Level, Region, RegionPoint, Sounding

HOUR = 3600.0
PSET = set(float(p) for p in sounding.PRESSURE_LEVELS)
#: surface quantities that are categories: taken from the nearer hour
CATEGORICAL = {"weather_code", "is_day"}


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _is_surface(level: Level) -> bool:
    return float(level.p) not in PSET


def _mix(a, b, f):
    if a is None:
        return b
    if b is None:
        return a
    return a + (b - a) * f


def interpolate_soundings(a: Sounding, b: Sounding, f: float, when: datetime) -> Sounding:
    """The sounding a fraction f of the way from a to b (two neighbouring
    hours of one forecast).  Pressure levels are matched by pressure; a
    level present at only one of the hours (below the ground at the other)
    is left out.  RH and cloud cover are interpolated as they are given;
    the dew point follows from the interpolated temperature and RH."""
    f = min(max(float(f), 0.0), 1.0)
    la = {float(l.p): l for l in a.levels if not _is_surface(l)}
    lb = {float(l.p): l for l in b.levels if not _is_surface(l)}
    levels = []
    sa = next((l for l in a.levels if _is_surface(l)), None)
    sb = next((l for l in b.levels if _is_surface(l)), None)
    for p in set(la) & set(lb):
        x, y = la[p], lb[p]
        t = _mix(x.t, y.t, f)
        rh = _mix(x.rh, y.rh, f)
        if x.cc >= 0.0 and y.cc >= 0.0:
            cc = _mix(x.cc, y.cc, f)
        else:
            cc = x.cc if x.cc >= 0.0 else y.cc
        levels.append(Level(p=p, z=_mix(x.z, y.z, f), t=t, rh=rh,
                            td=sounding.dewpoint_from_rh(t, rh), u=_mix(x.u, y.u, f),
                            v=_mix(x.v, y.v, f), cc=cc))
    if sa is not None and sb is not None:
        t = _mix(sa.t, sb.t, f)
        rh = _mix(sa.rh, sb.rh, f)
        levels.append(Level(p=_mix(sa.p, sb.p, f), z=_mix(sa.z, sb.z, f), t=t, rh=rh,
                            td=sounding.dewpoint_from_rh(t, rh), u=_mix(sa.u, sb.u, f),
                            v=_mix(sa.v, sb.v, f), cc=-1.0))
    levels.sort(key=lambda l: l.z)
    if len(levels) < 6:
        return a if f < 0.5 else b
    near = a if f < 0.5 else b
    surface = {}
    for k in set(a.surface) | set(b.surface):
        va, vb = a.surface.get(k), b.surface.get(k)
        if k in CATEGORICAL or not all(isinstance(v, (int, float)) or v is None for v in (va, vb)):
            surface[k] = near.surface.get(k)
        else:
            surface[k] = _mix(va, vb, f)
    return Sounding(levels=levels, lat=a.lat, lon=a.lon, when=when, elevation=a.elevation,
                    surface=surface, source=a.source, series_hours=list(near.series_hours),
                    series=dict(near.series), region=near.region)


#: gaps in an hourly series up to this long are filled in time
MAX_GAP_H = 6


def _finite(x):
    """A number, or None for anything that is not a finite number."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clean_hourly(hourly: dict) -> tuple[dict, int]:
    """An hourly answer on a regular hourly axis, with its gaps filled.

    The program takes the answer's i-th value to be hour i after the first,
    and every value to be a number or missing.  An answer that skips an hour
    or gives one twice would put everything after it an hour off; a NaN or an
    infinity would be carried into the sky as a number.  Here the times are
    laid on the hour from the first to the last (a repeated hour keeps its
    first values), anything that is not a finite number becomes missing, and
    a gap of up to MAX_GAP_H hours inside a series is filled linearly in time
    (wind directions the shorter way round; categories such as the weather
    code from the hour before).  Returns (hourly, values filled)."""
    times_s = list(hourly.get("time") or [])
    if not times_s:
        return hourly, 0
    ts = []
    for t in times_s:
        try:
            ts.append(_utc(t))
        except (TypeError, ValueError):
            ts.append(None)
    valid = [t for t in ts if t is not None]
    if not valid:
        return hourly, 0
    t0, t1 = min(valid), max(valid)
    n = int(round((t1 - t0).total_seconds() / HOUR)) + 1
    n = max(1, min(n, 24 * 40))
    slot = []
    for t in ts:
        k = None
        if t is not None:
            x = (t - t0).total_seconds() / HOUR
            if abs(x - round(x)) < 1e-6 and 0 <= round(x) < n:
                k = int(round(x))
        slot.append(k)
    out = {"time": [(t0 + timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M") for k in range(n)]}
    filled = 0
    for key, arr in hourly.items():
        if key == "time" or not isinstance(arr, list):
            continue
        cat = key in CATEGORICAL
        col = [None] * n
        for j, k in enumerate(slot):
            if k is None or j >= len(arr) or col[k] is not None:
                continue
            col[k] = _finite(arr[j])
        good = [k for k in range(n) if col[k] is not None]
        for a, b in zip(good, good[1:]):
            gap = b - a - 1
            if gap <= 0 or gap > MAX_GAP_H:
                continue
            for k in range(a + 1, b):
                f = (k - a) / (b - a)
                if cat:
                    col[k] = col[a]
                elif key.startswith("wind_direction"):
                    d = ((col[b] - col[a] + 180.0) % 360.0) - 180.0
                    col[k] = (col[a] + f * d) % 360.0
                else:
                    col[k] = col[a] + f * (col[b] - col[a])
                filled += 1
        out[key] = col
    return out, filled


class RegionSeries:
    """The forecast on the grid around the place, every hour (request_region)."""

    VARS = sounding._REGION_VARS

    def __init__(self, offs, data, n: int = 7, spacing_km: float = 40.0,
                 source: str = "open-meteo"):
        self.n = n
        self.spacing_km = spacing_km
        self.source = source
        self.offs = list(offs)
        self.points = []            # (east, north, lat, lon, {var: [values]}) per point
        times = None
        for (e, nn, la, lo), d in zip(offs, data):
            h = d.get("hourly", {}) if isinstance(d, dict) else {}
            h, _ = clean_hourly(h)
            t = h.get("time", [])
            if not t:
                continue
            times = times or t
            if t != times:
                continue
            vals = {v: list(h.get(v) or [None] * len(t)) for v in self.VARS}
            self.points.append((e, nn, la, lo, vals))
        if not times or len(self.points) < (n * n) // 2:
            raise RuntimeError("too few region points")
        self.times = [_utc(t) for t in times]

    def index(self, when: datetime) -> float:
        dt = (when - self.times[0]).total_seconds() / HOUR
        return min(max(dt, 0.0), float(len(self.times) - 1))

    def covers(self, when: datetime) -> bool:
        return self.times[0] - timedelta(minutes=30) <= when <= self.times[-1] + timedelta(minutes=30)

    def _value(self, vals: list, x: float, categorical: bool):
        i = int(math.floor(x))
        j = min(i + 1, len(vals) - 1)
        f = x - i
        a, b = vals[i], vals[j]
        if categorical:
            v = a if f < 0.5 else b
            return v if v is not None else (b if a is None else a)
        return _mix(a, b, f)

    def region_at(self, when: datetime) -> Region:
        """The Region (as fetch_region gives it) at `when`: each quantity
        interpolated between the hours, the present weather from the nearer
        hour; 'ahead' is three hours later."""
        x = self.index(when)
        xa = min(x + 3.0, float(len(self.times) - 1))
        pts = []
        for (e, nn, la, lo, vals) in self.points:
            now = {v: self._value(vals[v], x, v in CATEGORICAL) for v in self.VARS}
            ah = {v: self._value(vals[v], xa, v in CATEGORICAL) for v in self.VARS}
            pts.append(RegionPoint(e, nn, la, lo, now, ah))
        return Region(pts, self.spacing_km, self.n, source=self.source)

    def region_hour(self, i: int) -> Region:
        return self.region_at(self.times[min(max(i, 0), len(self.times) - 1)])


class Forecast:
    """Every hour of one forecast answer; the sounding at any moment."""

    def __init__(self, data: dict | None, lat: float, lon: float,
                 static: Sounding | None = None, region: RegionSeries | None = None,
                 source: str = "open-meteo", note: str = ""):
        self.data = data
        self.lat, self.lon = lat, lon
        self.static = static
        self.region = region
        self.source = source if static is None else "synthetic"
        self.note = note
        self._hours: dict[int, Sounding] = {}
        self.filled = 0
        if data is not None:
            # hours on a regular axis, non-numbers as gaps, short gaps filled
            hourly, self.filled = clean_hourly(data["hourly"])
            self.data = data = dict(data, hourly=hourly)
            self.times = [_utc(t) for t in data["hourly"]["time"]]
        else:
            t = (static.when if static is not None and static.when else datetime.now(timezone.utc))
            self.times = [t]
        #: a new number for every fetched forecast (the sky merges by it)
        self.stamp = datetime.now(timezone.utc).timestamp()

    # ----------------------------------------------------------- building --
    @classmethod
    def fetch(cls, lat: float, lon: float, when: datetime | None = None) -> "Forecast":
        """The live forecast (and its regional grid when that answers too).
        Raises when the forecast itself cannot be had."""
        when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
        data = sounding.request_open_meteo(lat, lon, when)
        note = ""
        reg = None
        try:
            offs, rdata = sounding.request_region(lat, lon, when)
            reg = RegionSeries(offs, rdata)
            note += f", {len(reg.points)} points around"
        except Exception as e:                                   # noqa: BLE001
            note += f" (no regional grid: {type(e).__name__})"
        f = cls(data, lat, lon, region=reg, note=note)
        f.note = (f"Open-Meteo, {len(f.times)} hours"
                  + (f" ({f.filled} missing values filled in time)" if f.filled else "") + note)
        f.sounding_at(when)                     # fail now, not later, on bad data
        return f

    @classmethod
    def constant(cls, snd: Sounding, note: str = "") -> "Forecast":
        """A sounding that holds at all times (the offline profiles)."""
        return cls(None, snd.lat, snd.lon, static=snd, note=note)

    # -------------------------------------------------------------- time --
    @property
    def start(self) -> datetime:
        return self.times[0]

    @property
    def end(self) -> datetime:
        return self.times[-1]

    @property
    def is_static(self) -> bool:
        return self.data is None

    def index(self, when: datetime) -> float:
        """Fractional hour index of `when`, clamped to the forecast."""
        if self.is_static:
            return 0.0
        x = (when - self.times[0]).total_seconds() / HOUR
        return min(max(x, 0.0), float(len(self.times) - 1))

    def hour_time(self, i: int) -> datetime:
        if self.is_static:
            return self.times[0] + timedelta(hours=i)
        return self.times[0] + timedelta(hours=i)

    def hour_of(self, when: datetime) -> int:
        """The hour index at or before `when` (not clamped)."""
        return int(math.floor((when - self.times[0]).total_seconds() / HOUR))

    def covers(self, when: datetime, margin_h: float = 0.0) -> bool:
        if self.is_static:
            return True
        return (self.times[0] + timedelta(hours=margin_h) <= when
                <= self.times[-1] - timedelta(hours=margin_h))

    # --------------------------------------------------------- soundings --
    def sounding_hour(self, i: int) -> Sounding:
        """The forecast's own sounding at hour i (clamped to the forecast;
        an hour the service left empty takes the nearest complete one)."""
        if self.is_static:
            return self.static
        n = len(self.times)
        i = min(max(int(i), 0), n - 1)
        s = self._hours.get(i)
        if s is not None:
            return s
        err = None
        for k in sorted(range(n), key=lambda k: (abs(k - i), k)):
            try:
                s = sounding.sounding_from_hourly(self.data, k, self.lat, self.lon, self.times[i])
                break
            except Exception as e:                               # noqa: BLE001
                err = e
                s = None
            if abs(k - i) > 6:
                break
        if s is None:
            raise RuntimeError(f"no usable hour near {self.times[i]:%Y-%m-%d %H} UTC: {err}")
        if self.region is not None:
            s.region = self.region.region_at(self.times[i])
        self._hours[i] = s
        return s

    def sounding_at(self, when: datetime) -> Sounding:
        """The atmosphere at `when`: linear in time between the two forecast
        hours around it (at an hour, that hour's own sounding)."""
        if self.is_static:
            return self.static
        x = self.index(when)
        i = int(math.floor(x))
        f = x - i
        a = self.sounding_hour(i)
        if f < 1e-6 or i + 1 >= len(self.times):
            s = interpolate_soundings(a, a, 0.0, when)
        else:
            s = interpolate_soundings(a, self.sounding_hour(i + 1), f, when)
        if self.region is not None:
            s.region = self.region.region_at(when)
        return s


def get_forecast(lat: float, lon: float, when: datetime | None = None,
                 allow_network: bool = True, fallback_profile: str = "tropical-fair"):
    """(forecast, note).  Never raises: offline, a synthetic profile that
    holds at all times."""
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if allow_network:
        try:
            f = Forecast.fetch(lat, lon, when)
            return f, f.note
        except Exception as e:                                   # noqa: BLE001
            snd = sounding.standard_sounding(lat, lon, when, profile=fallback_profile)
            if isinstance(e, sounding.BeyondForecast):
                # not a failure of the network: no forecast reaches that far
                note = f"{e} - synthetic {fallback_profile}"
            else:
                note = f"offline ({type(e).__name__}) - synthetic {fallback_profile}"
            return Forecast.constant(snd, note), note
    snd = sounding.standard_sounding(lat, lon, when, profile=fallback_profile)
    note = f"synthetic {fallback_profile}"
    return Forecast.constant(snd, note), note
