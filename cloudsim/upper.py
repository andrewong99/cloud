# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
The clouds above the weather: noctilucent clouds in the mesosphere and
polar stratospheric clouds.

Noctilucent clouds (polar mesospheric clouds) are ice crystals of about
50 nm at the mesopause, 80-85 km up, which forms only in the summer, when
it is the coldest place on Earth (below -120 C), at latitudes of about
50-70 degrees.  Too thin to see by day, they shine after sunset because at
their height the Sun has not set: they are seen with the Sun 6-16 degrees
below the horizon, silvery blue, low over the pole-ward horizon.  Their
structure is that of the waves travelling through the mesopause, and the
four forms of Fogle & Haurwitz (1966) are the four scales of them:
type I veils (tenuous, featureless), type II bands (long streaks, the
crests of gravity waves 10-100 km apart), type III billows (short waves of
3-10 km - Kelvin-Helmholtz and other instabilities) and type IV whirls
(rings and eddies of 20-50 km).

Polar stratospheric clouds form at 15-30 km in the polar winter when the
stratosphere falls below about -78 C (nitric acid trihydrate: type I,
faint, like cirrostratus) or -85 C (ice: type II, nacreous).  Nacreous
cloud is made of nearly equal ice particles of a few micrometres, so it
diffracts sunlight into the vivid colours of mother of pearl; it is often
lenticular, standing in the lee waves over mountains.  Both are seen in
twilight, the Sun 1-6 degrees below the horizon, lit while the ground is
already in shadow.

Every one of these is rendered from the geometry of its lighting (the
Earth's shadow at its height), its particles' scattering and the wave
fields that shape it; this module decides which are present, from the
date, the latitude and the stratospheric temperatures of the sounding, and
tells the census where they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

RG = 6360000.0
NLC_H = 82500.0
PSC_H = 22000.0
NAT_H = 20000.0


def _days_from_solstice(when: datetime, lat: float) -> float:
    doy = when.timetuple().tm_yday
    sol = 172 if lat >= 0 else 355                 # 21 June / 21 December
    d = abs(doy - sol)
    return min(d, 365 - d)


@dataclass
class Upper:
    key: str
    name: str
    abbr: str
    height: float
    strength: float
    why: list = field(default_factory=list)
    data: list = field(default_factory=list)
    sun_range: tuple = (-16.0, -6.0)               # Sun elevations it is seen at
    sun_alt: float = 0.0

    def distance(self, rd) -> float:
        """Distance along a ray (eye at the ground) to this cloud's shell."""
        r0 = RG
        b = r0 * rd[1]
        c = r0 * r0 - (RG + self.height) ** 2
        return float(-b + math.sqrt(max(b * b - c, 0.0)))

    def visible_mask(self, rd: np.ndarray) -> np.ndarray:
        """Rays that see it lit against a dark enough sky."""
        lo, hi = self.sun_range
        if not (lo <= self.sun_alt <= hi) or self.strength <= 0.0:
            return np.zeros(len(rd), bool)
        elev = np.degrees(np.arcsin(np.clip(rd[:, 1], -1, 1)))
        return elev > 0.5


class UpperClouds:
    """Which clouds above the weather are there, and their render
    parameters."""

    def __init__(self):
        self.items: list[Upper] = []
        self.nlc = 0.0
        self.nlc_types = 0
        self.nacreous = 0.0
        self.lens = 0.0
        self.nat = 0.0
        self.wind = (-40.0, 0.0)          # mesospheric summer easterly
        self.psc_wind = (20.0, 0.0)
        self.forced: set = set()

    def decide(self, when: datetime, lat: float, snd=None, sun_alt: float = 0.0,
               forced: set | None = None, lon: float = 0.0):
        self.items = []
        forced = self.forced if forced is None else forced
        alat = abs(lat)
        dsol = _days_from_solstice(when, lat)
        # --- noctilucent: summer, 50-70 degrees of latitude ---------------
        nlc_season = 50.0 <= alat <= 72.0 and dsol <= 50
        self.nlc = 0.0
        self.nlc_types = 0
        if nlc_season or "nlc" in forced:
            self.nlc = 1.0 if "nlc" in forced else max(0.0, 1.0 - dsol / 50.0)
            # which of the four forms: the wave field of the night - of the
            # local night, which changes at local noon, when none is seen
            # (keyed to the UTC date the forms changed at 00 UTC, in the
            # middle of a European summer night)
            night = when + timedelta(hours=float(lon) / 15.0 - 12.0)
            seed = night.timetuple().tm_yday * 7 + int(alat)
            self.nlc_types = 1 | (2 if seed % 3 != 0 else 0) | (4 if seed % 2 == 0 else 0) \
                | (8 if seed % 5 == 0 else 0)
            if "nlc" in forced:
                self.nlc_types = 15
            forms = [n for b, n in ((1, "type I veils"), (2, "type II bands"),
                                    (4, "type III billows"), (8, "type IV whirls"))
                     if self.nlc_types & b]
            why = ["ice crystals of ~50 nm at the mesopause, 82 km up, where the summer "
                   "mesosphere falls below -120 C",
                   f"seen only in deep twilight: with the Sun {-sun_alt:.0f} deg below the "
                   f"horizon the ground and the lower sky are dark, but at 82 km the Sun has "
                   f"not set",
                   "silvery blue: the sunlight reaching them has crossed the ozone layer "
                   "edge-on, which takes out the orange, and tiny ice scatters blue best",
                   "the forms are the mesospheric waves: " + ", ".join(forms)]
            data = [f"latitude {lat:.1f}, {dsol:.0f} days from the summer solstice"
                    + (" (shown by request)" if "nlc" in forced else "")]
            self.items.append(Upper("nlc", "Noctilucent cloud (" + ", ".join(forms) + ")",
                                    "NLC", NLC_H, self.nlc, why, data, (-16.0, -6.0), sun_alt))
        # --- polar stratospheric: winter, a cold stratosphere --------------
        self.nacreous = 0.0
        self.nat = 0.0
        t_strat = None
        if snd is not None:
            hi = [lv for lv in snd.levels if lv.p <= 60.0]
            if hi:
                t_strat = min(lv.t for lv in hi)
        winter = alat >= 50.0 and _days_from_solstice(when, -lat) <= 75
        # how much of each kind there is: each grows in over 4 degrees about
        # its threshold (ice below about -85 C, nitric acid below about
        # -78 C) as the stratosphere cools, instead of appearing at once
        cold_ice = _ramp(t_strat, -83.0, -87.0) if (winter and t_strat is not None) else 0.0
        cold_nat = _ramp(t_strat, -76.0, -80.0) if (winter and t_strat is not None) else 0.0
        ice = 1.0 if "psc" in forced else cold_ice
        nat = (1.0 - ice) * (1.0 if "nat" in forced else cold_nat)
        if ice > 0.0 or nat > 0.0:
            ws = _stratospheric_wind_uv(snd) if snd is not None else None
            if ws is not None and math.hypot(*ws) > 1.0:
                self.psc_wind = ws
            if ice > 0.0:
                self.nacreous = ice
                # lenticular where the stratospheric wind carries mountain
                # waves up (over about 25 m/s), blended across 20-30 m/s
                self.lens = 1.0 if "psc" in forced else (
                    _ramp(_stratospheric_wind(snd), 20.0, 30.0) if snd is not None else 0.0)
                why = ["ice at 20-25 km, formed where the polar-winter stratosphere falls below "
                       "about -85 C",
                       "nearly equal particles a few micrometres across diffract sunlight into "
                       "mother-of-pearl colours that change with the angle from the Sun",
                       "lit from below the horizon: with the Sun a few degrees down the ground "
                       "is in shadow but 22 km up is not"]
                if self.lens >= 0.5:
                    why.append("lenticular: it forms in the crests of mountain lee waves that "
                               "reach the stratosphere, and stands still in the wind")
                data = [f"coldest stratospheric level {t_strat:.0f} C" if t_strat is not None
                        else "shown by request"]
                self.items.append(Upper("psc", "Nacreous cloud (polar stratospheric, "
                                        + ("lenticular" if self.lens >= 0.5 else "cirriform") + ")",
                                        "PSC II", PSC_H, ice, why, data, (-6.5, -0.5), sun_alt))
            if nat > 0.0:
                self.nat = nat
                why = ["nitric acid trihydrate and supercooled ternary droplets at ~20 km, "
                       "formed below about -78 C",
                       "smaller and fewer than nacreous particles: a faint milky veil like "
                       "cirrostratus, without the iridescence"]
                data = [f"coldest stratospheric level {t_strat:.0f} C" if t_strat is not None
                        else "shown by request"]
                self.items.append(Upper("nat", "Nitric acid and water polar stratospheric "
                                        "cloud", "PSC I", NAT_H, nat, why, data,
                                        (-6.5, -0.5), sun_alt))
        return self.items

    def uniforms(self, when_s: float) -> dict:
        return {"uNlc": float(self.nlc), "uNlcTypes": int(self.nlc_types),
                "uNacreous": float(self.nacreous), "uPscLens": float(self.lens),
                "uNat": float(self.nat), "uUpperWind": tuple(float(x) for x in self.wind),
                "uPscWind": tuple(float(x) for x in self.psc_wind), "uUpperTime": float(when_s)}


def _ramp(x: float, x0: float, x1: float) -> float:
    """0 at x0, 1 at x1, smooth between (either order)."""
    t = min(max((float(x) - x0) / (x1 - x0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _stratospheric_wind_uv(snd):
    """The wind (u, v) at the sounding's highest level at or above 60 hPa."""
    hi = [lv for lv in snd.levels if lv.p <= 60.0]
    if not hi:
        return None
    lv = min(hi, key=lambda l: l.p)
    return (float(lv.u), float(lv.v))


def _stratospheric_wind(snd) -> float:
    hi = [lv for lv in snd.levels if lv.p <= 60.0]
    return max((lv.wind_speed for lv in hi), default=0.0)
