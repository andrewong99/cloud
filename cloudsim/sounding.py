# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
Atmospheric sounding: fetch, thermodynamics, wind analysis, and the diagnosis
of which Atlas cloud types the atmosphere actually supports.

Data source: Open-Meteo (https://open-meteo.com), free and key-less for
non-commercial use.  Pressure-level fields give temperature, relative
humidity, cloud cover, wind and geopotential height on 19 levels from
1000 hPa to 30 hPa, which is a real sounding in everything but name.

If there is no network the module falls back to a synthetic sounding built
from the International Standard Atmosphere plus a chosen humidity profile,
so the simulator always runs.

Every diagnosis rule carries the threshold it used and a one-line reason, so
the sky the program draws can be argued with.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

# ------------------------------------------------------------- constants ----
RD = 287.058          # J/(kg K) dry air
RV = 461.5            # J/(kg K) water vapour
EPS = RD / RV         # 0.622
CPD = 1005.7          # J/(kg K)
LV0 = 2.501e6         # J/kg at 0 C
G0 = 9.80665
P0 = 1013.25          # hPa

PRESSURE_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500,
                   400, 300, 250, 200, 150, 100, 70, 50, 30]


def esat_hpa(t_c: float) -> float:
    """Saturation vapour pressure over water, hPa (Bolton 1980, eq. 10)."""
    return 6.112 * math.exp(17.67 * t_c / (t_c + 243.5))


def esat_ice_hpa(t_c: float) -> float:
    """Saturation vapour pressure over ice, hPa (Goff-Gratch simplified)."""
    return 6.1115 * math.exp(22.452 * t_c / (t_c + 272.55))


def dewpoint_from_rh(t_c: float, rh_pct: float) -> float:
    rh = max(min(rh_pct, 100.0), 0.1) / 100.0
    e = rh * esat_hpa(t_c)
    ln = math.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln)


def mixing_ratio(p_hpa: float, e_hpa: float) -> float:
    e = min(e_hpa, p_hpa * 0.999)
    return EPS * e / (p_hpa - e)


def lv(t_c: float) -> float:
    return LV0 - 2340.0 * t_c


def theta(t_c: float, p_hpa: float) -> float:
    return (t_c + 273.15) * (P0 / p_hpa) ** (RD / CPD)


def theta_e(t_c: float, td_c: float, p_hpa: float) -> float:
    """Equivalent potential temperature, Bolton 1980 eq. 43 (approximate)."""
    tk = t_c + 273.15
    e = esat_hpa(td_c)
    r = mixing_ratio(p_hpa, e)
    tl = 56.0 + 1.0 / (1.0 / (td_c + 273.15 - 56.0) + math.log(tk / (td_c + 273.15)) / 800.0)
    th = tk * (1000.0 / p_hpa) ** (0.2854 * (1 - 0.28 * r))
    return th * math.exp((3036.0 / tl - 1.78) * r * (1 + 0.448 * r))


def moist_lapse_dtdp(t_k: float, p_hpa: float) -> float:
    """dT/dp along a saturated pseudo-adiabat, K/hPa."""
    t_c = t_k - 273.15
    rs = mixing_ratio(p_hpa, esat_hpa(t_c))
    l = lv(t_c)
    num = RD * t_k + l * rs
    den = CPD + (l * l * rs * EPS) / (RD * t_k * t_k)
    return (1.0 / p_hpa) * num / den


def lcl_bolton(t_c: float, td_c: float, p_hpa: float) -> tuple[float, float, float]:
    """LCL temperature (C), pressure (hPa) and height above the parcel (m).

    Temperature: Bolton 1980 eq. 15.  Height: dry-adiabatic, (T-Tlcl)/9.8 K/km.
    """
    tk, tdk = t_c + 273.15, td_c + 273.15
    tl = 1.0 / (1.0 / (tdk - 56.0) + math.log(tk / tdk) / 800.0) + 56.0
    pl = p_hpa * (tl / tk) ** (CPD / RD)
    z = (tk - tl) * CPD / G0
    return tl - 273.15, pl, max(z, 0.0)


# ------------------------------------------------- RH and cloud fraction ---

#: The critical relative humidity of Open-Meteo's pressure-level cloud cover,
#: which is Sundqvist et al. (1989), cc = 1 - sqrt((1 - RH) / (1 - RHc)).
#: Inverted from its own (RH, cc) pairs: Kuala Lumpur (3.13 N 101.54 E),
#: January-June 2025, hourly, 550-3000 invertible pairs per level; the spread
#: (10th-90th percentile) is within +-0.02 at every level.  Used only where a
#: level's own pair cannot be inverted (cc 0 or 1).
RHC_OPEN_METEO = [(1000.0, 0.889), (975.0, 0.875), (950.0, 0.849), (925.0, 0.828),
                  (900.0, 0.811), (850.0, 0.771), (800.0, 0.741), (700.0, 0.709),
                  (600.0, 0.703), (500.0, 0.703), (400.0, 0.702), (300.0, 0.697)]


def rhc_open_meteo(p_hpa: float) -> float:
    """RHC_OPEN_METEO interpolated in pressure (0.70 above 300 hPa)."""
    tab = RHC_OPEN_METEO
    if p_hpa >= tab[0][0]:
        return tab[0][1]
    for (pa, ra), (pb, rb) in zip(tab, tab[1:]):
        if pb <= p_hpa <= pa:
            f = (pa - p_hpa) / (pa - pb)
            return ra + f * (rb - ra)
    return 0.70


def cloud_fraction(rh: float, rhc: float, form: str = "sundqvist") -> float:
    """Grid-box cloud fraction from relative humidity (0..1).
    'sundqvist': Sundqvist et al. (1989), 1 - sqrt((1 - RH) / (1 - RHc)),
    what Open-Meteo's pressure-level cloud cover is.  'sqrt': this program's
    own relation for soundings without a model cloud fraction,
    sqrt((RH - RHc) / (1 - RHc))."""
    if rh <= rhc:
        return 0.0
    if rh >= 1.0:
        return 1.0
    if form == "sundqvist":
        return 1.0 - math.sqrt((1.0 - rh) / (1.0 - rhc))
    return min(1.0, math.sqrt((rh - rhc) / (1.0 - rhc)))


@dataclass
class CloudLayer:
    """A cloudy layer of the sounding (Sounding.cloud_layers).  Heights are
    metres above sea level."""
    base: float          # the cloud fraction first reaches half its peak
    top: float           # ... and last does
    cover: float         # the peak cloud fraction, 0..1 (the model's, at a level)
    z_peak: float        # the level of the peak
    edge_base: float     # the fraction first exceeds 5 % (the layer's faint edge)
    edge_top: float
    n_levels: int        # model levels with more than 5 % cloud

    @property
    def depth(self) -> float:
        return self.top - self.base


# ------------------------------------------------------------------ level ---

@dataclass
class Level:
    p: float          # hPa
    z: float          # m above sea level
    t: float          # C
    rh: float         # %
    td: float         # C
    u: float          # m/s (east)
    v: float          # m/s (north)
    cc: float         # model cloud cover at this level, 0..1 (may be -1 if absent)

    @property
    def wind_speed(self) -> float:
        return math.hypot(self.u, self.v)

    @property
    def wind_dir_from(self) -> float:
        """Meteorological direction the wind blows FROM, degrees."""
        return (math.degrees(math.atan2(-self.u, -self.v))) % 360.0


@dataclass
class Sounding:
    levels: list[Level]
    lat: float
    lon: float
    when: datetime
    elevation: float = 0.0
    surface: dict = field(default_factory=dict)
    source: str = "synthetic"
    #: surface variables hour by hour around the chosen time: hours relative
    #: to it, and {variable: [value per hour]} - what the trend reasoning
    #: (pressure tendency, a storm on the way) reads
    series_hours: list = field(default_factory=list)
    series: dict = field(default_factory=dict)
    #: the weather around the place (fetch_region), or None
    region: object = None

    def series_at(self, var: str, hour: int):
        """The value of a surface variable `hour` hours from the chosen
        time, or None."""
        if var not in self.series or hour not in self.series_hours:
            return None
        return self.series[var][self.series_hours.index(hour)]

    # -- interpolation -----------------------------------------------------
    def _interp(self, z: float, attr: str) -> float:
        ls = self.levels
        if z <= ls[0].z:
            return getattr(ls[0], attr)
        for a, b in zip(ls, ls[1:]):
            if a.z <= z <= b.z:
                f = (z - a.z) / max(b.z - a.z, 1e-6)
                return getattr(a, attr) * (1 - f) + getattr(b, attr) * f
        return getattr(ls[-1], attr)

    def t_at(self, z): return self._interp(z, "t")
    def rh_at(self, z): return self._interp(z, "rh")
    def td_at(self, z): return self._interp(z, "td")
    def p_at(self, z): return self._interp(z, "p")
    def cc_at(self, z): return self._interp(z, "cc")

    def wind_at(self, z) -> tuple[float, float]:
        return self._interp(z, "u"), self._interp(z, "v")

    def shear_at(self, z: float, dz: float = 500.0) -> tuple[float, float]:
        u1, v1 = self.wind_at(max(z - dz * 0.5, self.levels[0].z))
        u2, v2 = self.wind_at(z + dz * 0.5)
        return (u2 - u1) / dz, (v2 - v1) / dz

    def bulk_shear(self, z0: float, z1: float) -> float:
        u0, v0 = self.wind_at(z0)
        u1, v1 = self.wind_at(z1)
        return math.hypot(u1 - u0, v1 - v0)

    # -- derived quantities ------------------------------------------------
    def lapse_rate(self, z0: float, z1: float) -> float:
        """Environmental lapse rate, K/km (positive = temperature falling)."""
        return -(self.t_at(z1) - self.t_at(z0)) / max((z1 - z0) / 1000.0, 1e-6)

    def moist_adiabatic_lapse(self, z: float) -> float:
        """Saturated adiabatic lapse rate at this height, K/km."""
        p = self.p_at(z)
        tk = self.t_at(z) + 273.15
        dtdp = moist_lapse_dtdp(tk, p)
        rho = p * 100.0 / (RD * tk)
        return dtdp * rho * G0 / 100.0 * 1000.0

    def freezing_level(self) -> float | None:
        for a, b in zip(self.levels, self.levels[1:]):
            if a.t >= 0.0 > b.t:
                f = a.t / max(a.t - b.t, 1e-6)
                return a.z + f * (b.z - a.z)
        return None

    def level_of_temperature(self, t_c: float) -> float | None:
        for a, b in zip(self.levels, self.levels[1:]):
            if a.t >= t_c > b.t:
                f = (a.t - t_c) / max(a.t - b.t, 1e-6)
                return a.z + f * (b.z - a.z)
        return None

    def tropopause(self) -> float:
        """WMO definition: lowest level where the lapse rate falls to 2 K/km
        and stays below that on average through the next 2 km."""
        for lo in self.levels:
            if lo.z < 5000.0:
                continue
            if lo.z > 20000.0:
                break
            if self.lapse_rate(lo.z, min(lo.z + 2000.0, self.levels[-1].z)) < 2.0:
                return lo.z
        return 12000.0 if abs(self.lat) > 30 else 16000.0

    def inversions(self, min_strength: float = 0.3) -> list[tuple[float, float, float]]:
        """(base, top, strength K) of layers where temperature rises with height."""
        out = []
        z0 = None
        for a, b in zip(self.levels, self.levels[1:]):
            if b.z > 12000.0:
                break
            rising = b.t > a.t - 1e-6
            if rising and z0 is None:
                z0, t0 = a.z, a.t
            elif not rising and z0 is not None:
                if a.t - t0 >= min_strength:
                    out.append((z0, a.z, a.t - t0))
                z0 = None
        return out

    def parcel(self, z_start: float | None = None, mixed_depth: float = 0.0):
        """Lift a parcel and return (CAPE, CIN, LCL_z, LFC_z, EL_z, profile).

        profile is a list of (z, T_parcel_C).  Virtual-temperature corrected.
        mixed_depth > 0 averages the lowest metres first (mixed-layer parcel).
        """
        base = self.levels[0].z if z_start is None else z_start
        if mixed_depth > 0:
            n = 12
            ts = [self.t_at(base + i * mixed_depth / n) for i in range(n)]
            tds = [self.td_at(base + i * mixed_depth / n) for i in range(n)]
            t0 = sum(ts) / n
            td0 = sum(tds) / n
        else:
            t0, td0 = self.t_at(base), self.td_at(base)
        p0 = self.p_at(base)

        tl_c, p_lcl, dz_lcl = lcl_bolton(t0, td0, p0)
        z_lcl = base + dz_lcl

        prof = []
        cape = cin = 0.0
        lfc = el = None
        z = base
        tp = t0
        prev_buoy = None
        prev_z = base
        step = 60.0
        top = min(self.levels[-1].z, 20000.0)
        while z < top:
            z += step
            p = self.p_at(z)
            if z <= z_lcl:
                tp = t0 - 9.8 * (z - base) / 1000.0
                r = mixing_ratio(p0, esat_hpa(td0))
            else:
                pm = self.p_at(z - step)
                dp = p - pm
                tp = tp + moist_lapse_dtdp(tp + 273.15, max(pm, 1.0)) * dp
                r = mixing_ratio(p, esat_hpa(tp))
            te = self.t_at(z)
            re = mixing_ratio(p, esat_hpa(self.td_at(z)))
            tvp = (tp + 273.15) * (1 + 0.608 * r)
            tve = (te + 273.15) * (1 + 0.608 * re)
            buoy = G0 * (tvp - tve) / tve
            prof.append((z, tp))
            if prev_buoy is not None:
                dzs = z - prev_z
                if buoy > 0:
                    if lfc is None and z > z_lcl:
                        lfc = z          # first positive crossing above the LCL
                    cape += buoy * dzs
                else:
                    if lfc is None:
                        cin += buoy * dzs
                    elif prev_buoy > 0:
                        el = z           # keep the LAST + to - crossing
            prev_buoy, prev_z = buoy, z
        if lfc is not None and el is None:
            el = top
        return cape, cin, z_lcl, lfc, el, prof

    def layer_parcel(self, z0: float, span: float = 4000.0) -> tuple[float, float | None]:
        """A saturated parcel from a cloud layer at height z0, lifted along the
        pseudo-adiabat (what a turret rising from the layer is): its CAPE,
        J/kg - the positive buoyancy (virtual temperature) over the next
        `span` metres, in 50 m steps - and the top of its first buoyant
        stretch (None if it is never buoyant)."""
        tp = self.t_at(z0)
        z, pm = z0, self.p_at(z0)
        top = min(z0 + span, self.levels[-1].z)
        cape, el, was_pos = 0.0, None, False
        while z < top:
            z += 50.0
            p = self.p_at(z)
            tp += moist_lapse_dtdp(tp + 273.15, max(pm, 1.0)) * (p - pm)
            pm = p
            te, tde = self.t_at(z), self.td_at(z)
            tvp = (tp + 273.15) * (1 + 0.608 * mixing_ratio(p, esat_hpa(tp)))
            tve = (te + 273.15) * (1 + 0.608 * mixing_ratio(p, esat_hpa(tde)))
            b = G0 * (tvp - tve) / tve
            if b > 0:
                cape += b * 50.0
                was_pos = True
            elif was_pos and el is None:
                el = z - 50.0
        if was_pos and el is None:
            el = top
        return cape, el

    # -- moist layers ------------------------------------------------------
    def _level_clouds(self, use_model_cloud: bool = True):
        """Per level up to 18 km: height, RH (0..1), cloud fraction, the
        critical RH of the RH -> cloud-fraction relation, and which form of
        that relation the fraction came from.

        The model's own cloud fraction is preferred.  Open-Meteo's
        pressure-level cloud cover is Sundqvist et al. (1989),
        cc = 1 - sqrt((1 - RH) / (1 - RHc)); inverting the model's own
        (RH, cc) pairs gives RHc at each level where the pair is invertible
        (0 < cc < 1), and RHC_OPEN_METEO where it is not.  Without the
        model's fraction the program's own relation is used, as before."""
        rows = []
        model = use_model_cloud and any(l.cc >= 0.0 for l in self.levels)
        for l in self.levels:
            if l.z > 18000.0:
                break
            rh = min(max(l.rh / 100.0, 0.0), 1.0)
            if model and l.cc < 0.0:
                # the surface row (2 m): the model's fractions are its own
                # levels'; the screen-level humidity makes no cloud short of
                # saturation (fog is the forecast's visibility) - from 83-99 %
                # it made 41 of the 54 Stratus layers in four recorded
                # forecasts, one based under the ground at noon in 24 km
                # visibility
                rows.append((l.z, rh, 0.0, 0.995, "sundqvist"))
                continue
            if use_model_cloud and l.cc >= 0.0:
                cc = min(max(l.cc, 0.0), 1.0)
                if 0.02 < cc < 0.98 and rh < 0.995:
                    rhc = 1.0 - (1.0 - rh) / ((1.0 - cc) ** 2)
                    rhc = min(max(rhc, 0.30), 0.99)
                else:
                    rhc = rhc_open_meteo(l.p)
                rows.append((l.z, rh, cc, rhc, "sundqvist"))
            else:
                # RH -> cloud fraction, Sundqvist-style.  The critical RH is
                # height dependent: near the surface a grid box has to be very
                # nearly saturated before any of it is cloud, aloft less so.
                s = min(1.0, max(0.0, (l.z - self.levels[0].z) / 6000.0))
                rhc = 0.95 - 0.13 * s
                rows.append((l.z, rh, cloud_fraction(rh, rhc, "sqrt"), rhc, "sqrt"))
        return rows

    @staticmethod
    def _cover_between(rows, z: float) -> float:
        """The cloud fraction at height z: RH and the critical RH are
        interpolated linearly in height between the levels (RH is the
        continuous variable; the fraction is not) and put through the same
        relation the levels' fractions came from.  At a level it returns
        that level's own fraction."""
        if z <= rows[0][0]:
            return rows[0][2]
        for a, b in zip(rows, rows[1:]):
            if a[0] <= z <= b[0]:
                f = (z - a[0]) / max(b[0] - a[0], 1e-6)
                if f <= 0.0:
                    return a[2]
                if f >= 1.0:
                    return b[2]
                rh = a[1] + f * (b[1] - a[1])
                rhc = a[3] + f * (b[3] - a[3])
                return cloud_fraction(rh, rhc, a[4])
        return rows[-1][2]

    def cloud_layers(self, use_model_cloud: bool = True) -> list[CloudLayer]:
        """Contiguous cloudy layers, with their depth taken from the cloudy
        part only.

        A layer is a run of levels whose cloud fraction exceeds 5 %.  Its
        base and top are where the fraction between the levels reaches half
        its peak (the full width at half maximum of the cloud-fraction
        profile), the fraction between levels coming from interpolated RH
        (_cover_between).  Pressure levels are 1-2 km apart in the middle
        troposphere; interpolating the fraction itself from one cloudy level
        to its clear neighbours spread a thin layer over 2-3 km."""
        rows = self._level_clouds(use_model_cloud)
        n = len(rows)
        out = []
        i = 0
        while i < n:
            if rows[i][2] <= 0.05:
                i += 1
                continue
            j = i
            while j + 1 < n and rows[j + 1][2] > 0.05:
                j += 1
            peak_k = max(range(i, j + 1), key=lambda k: rows[k][2])
            peak = rows[peak_k][2]
            z_lo = rows[i - 1][0] if i > 0 else rows[i][0]
            z_hi = rows[j + 1][0] if j + 1 < n else rows[j][0] + 300.0
            half = 0.5 * peak
            steps = max(int((z_hi - z_lo) / 10.0), 1)
            zs = [z_lo + (z_hi - z_lo) * s / steps for s in range(steps + 1)]
            cs = [self._cover_between(rows, z) if z <= rows[-1][0] else rows[j][2] for z in zs]
            inside = [z for z, c in zip(zs, cs) if c >= half - 1e-9]
            edge = [z for z, c in zip(zs, cs) if c > 0.05]
            base, top = (inside[0], inside[-1]) if inside else (rows[peak_k][0], rows[peak_k][0])
            if top - base < 100.0:              # never thinner than 100 m
                mid = 0.5 * (base + top)
                base, top = mid - 50.0, mid + 50.0
            out.append(CloudLayer(base=base, top=top, cover=peak, z_peak=rows[peak_k][0],
                                  edge_base=edge[0] if edge else base,
                                  edge_top=edge[-1] if edge else top,
                                  n_levels=j - i + 1))
            i = j + 1
        return out

    def moist_layers(self, rh_threshold: float = 80.0,
                     use_model_cloud: bool = True) -> list[tuple[float, float, float]]:
        """Contiguous cloudy layers as (base_z, top_z, peak_cover 0..1): the
        cloudy part of each layer (cloud_layers), at least 200 m deep.

        Prefers the model's own per-level cloud cover; falls back to a
        relative-humidity threshold, which is how cloud fraction has been
        diagnosed since Smagorinsky (1960)."""
        return [(L.base, max(L.top, L.base + 200.0), L.cover)
                for L in self.cloud_layers(use_model_cloud)]

    # -- wave / turbulence diagnostics --------------------------------------
    def brunt_vaisala(self, z: float, dz: float = 400.0) -> float:
        """N in s^-1; imaginary (returned negative) if the layer is unstable."""
        z0, z1 = max(z - dz / 2, self.levels[0].z), z + dz / 2
        th0 = theta(self.t_at(z0), self.p_at(z0))
        th1 = theta(self.t_at(z1), self.p_at(z1))
        n2 = G0 / ((th0 + th1) / 2) * (th1 - th0) / (z1 - z0)
        return math.sqrt(n2) if n2 > 0 else -math.sqrt(-n2)

    def richardson(self, z: float, dz: float = 400.0) -> float:
        """Gradient Richardson number.  Ri < 0.25 permits Kelvin-Helmholtz
        billows, which the Atlas calls fluctus."""
        n = self.brunt_vaisala(z, dz)
        du, dv = self.shear_at(z, dz)
        s2 = du * du + dv * dv
        if s2 < 1e-12:
            return 999.0
        return (n * abs(n)) / s2

    def scorer_parameter(self, z: float, dz: float = 800.0) -> float:
        """l^2 = N^2/U^2 - (d2U/dz2)/U.  Trapped lee waves need l^2 to fall
        with height; the wavelength of the trapped wave is 2*pi/l."""
        n = self.brunt_vaisala(z, dz)
        u, v = self.wind_at(z)
        spd = max(math.hypot(u, v), 1.0)
        u0 = math.hypot(*self.wind_at(max(z - dz, self.levels[0].z)))
        u2 = math.hypot(*self.wind_at(z + dz))
        d2u = (u2 - 2 * spd + u0) / (dz * dz)
        return (n * abs(n)) / (spd * spd) - d2u / spd

    def lee_wavelength(self, z: float) -> float | None:
        l2 = self.scorer_parameter(z)
        if l2 <= 1e-9:
            return None
        return 2.0 * math.pi / math.sqrt(l2)

    def storm_motion(self) -> tuple[float, float]:
        """Bunkers right-mover, a decent proxy for how a Cb travels."""
        zs = self.levels[0].z
        us, vs = [], []
        z = zs
        while z <= zs + 6000.0:
            u, v = self.wind_at(z)
            us.append(u)
            vs.append(v)
            z += 250.0
        mu, mv = sum(us) / len(us), sum(vs) / len(vs)
        u0, v0 = self.wind_at(zs + 250.0)
        u6, v6 = self.wind_at(zs + 6000.0)
        du, dv = u6 - u0, v6 - v0
        m = math.hypot(du, dv)
        if m < 1e-6:
            return mu, mv
        return mu + 7.5 * dv / m, mv - 7.5 * du / m


def inhibition_text(cin: float, lfc: float | None, unit: bool = True) -> str:
    """CIN as shown: the negative area a parcel crosses up to its level of
    free convection.  With no such level the parcel never rises freely and
    there is no CIN to overcome - the area summed to the top of the profile
    read as -14,000 to -18,500 J/kg in the panel."""
    if lfc is None:
        return "no LFC"
    return f"CIN {cin:.0f}" + (" J/kg" if unit else "")


# ------------------------------------------------------------- synthesise ---

def standard_sounding(lat: float = 3.14, lon: float = 101.69,
                      when: datetime | None = None,
                      surface_t: float = 31.0, surface_td: float = 24.0,
                      profile: str = "tropical-fair",
                      wind_dir: float = 240.0, wind_speed: float = 4.0,
                      jet_speed: float = 22.0) -> Sounding:
    """A synthetic but physically consistent sounding, for offline use.

    profile:
      tropical-fair   shallow moist layer, trade inversion near 2 km
      humid-deep      deep moist column, high CAPE
      stable-stratus  surface-based saturation under a strong inversion
      dry             very dry, cirrus only
    """
    when = when or datetime.now(timezone.utc)
    levels = []
    # each profile carries its own surface state, otherwise a "stable stratus"
    # column built on a 31 C surface comes out wildly unstable
    _sfc = {"tropical-fair": (31.0, 24.0), "humid-deep": (30.0, 25.5),
            "stable-stratus": (16.0, 15.5), "dry": (33.0, 8.0)}
    if profile in _sfc and surface_t == 31.0:
        surface_t, surface_td = _sfc[profile]
    t_sfc = surface_t
    # tropopause height by latitude: ~16.5 km at the equator, ~9 km at the pole
    z_trop = 16500.0 - 7500.0 * (abs(lat) / 90.0) ** 1.4

    def smooth_step(x, a, b):
        t = min(1.0, max(0.0, (x - a) / max(b - a, 1e-6)))
        return t * t * (3 - 2 * t)

    # finer than the 19 standard levels, so inversions and shallow moist
    # layers survive; a real fetch keeps the model's own levels
    heights = ([z for z in range(0, 5000, 250)] + [z for z in range(5000, 14000, 500)]
               + [14000, 15000, 16000, 17000, 18000, 20000, 22000])
    for z in heights:
        p = P0 * (1.0 - z / 44330.0) ** (1.0 / 0.1903)
        if z < z_trop:
            t = t_sfc - 6.5 * z / 1000.0
        else:
            t = t_sfc - 6.5 * z_trop / 1000.0 + 1.5 * (z - z_trop) / 1000.0
        if profile == "tropical-fair":
            rh = (82.0 - 14.0 * smooth_step(z, 700, 1100)
                  - 40.0 * smooth_step(z, 1900, 2600)
                  + 27.0 * smooth_step(z, 5500, 7000)
                  - 30.0 * smooth_step(z, 10500, 12500))
            t += 3.0 * (smooth_step(z, 1850, 2050) - smooth_step(z, 2250, 2500))
        elif profile == "humid-deep":
            rh = (88.0 - 10.0 * smooth_step(z, 1000, 1500)
                  - 8.0 * smooth_step(z, 6000, 8000)
                  - 40.0 * smooth_step(z, 12000, 14000))
        elif profile == "stable-stratus":
            rh = 97.0 - 62.0 * smooth_step(z, 850, 1100) - 10.0 * smooth_step(z, 3500, 4500)
            # the inversion's warmth stays above it (it was taken back over
            # 1300-1600 m: 21 K/km there, steeper than the dry adiabat, which
            # gave the layer CAPE and turrets - Stratocumulus castellanus)
            t += 5.0 * smooth_step(z, 880, 1050)
        else:
            rh = 30.0 + 35.0 * smooth_step(z, 6500, 8000) - 45.0 * smooth_step(z, 11500, 13000)
        rh = min(max(rh, 3.0), 100.0)
        # explicit cloud cover for the offline demonstration profiles, so the
        # fallback sky is a definite sky rather than whatever an RH threshold
        # happens to produce
        if profile == "tropical-fair":
            cc = 0.35 if 9800 < z < 12200 else 0.0
        elif profile == "humid-deep":
            cc = (0.55 if 3800 < z < 6200 else (0.75 if 9500 < z < 13000 else 0.0))
        elif profile == "stable-stratus":
            cc = 0.95 if 100 < z < 900 else 0.0
        else:
            cc = 0.18 if 9000 < z < 11500 else 0.0
        spd = wind_speed + (jet_speed - wind_speed) * min(1.0, max(0.0, (z - 200) / 11000.0))
        dirn = wind_dir + 40.0 * min(1.0, z / 10000.0)
        u = -spd * math.sin(math.radians(dirn))
        v = -spd * math.cos(math.radians(dirn))
        levels.append(Level(p=p, z=z, t=t, rh=rh, td=dewpoint_from_rh(t, rh),
                            u=u, v=v, cc=cc))
    levels.sort(key=lambda l: l.z)
    # present weather and cover for the demonstration profiles (what a
    # forecast model would report for such a column)
    wx = {"tropical-fair": (2, 30.0, 0.0, 35.0, 0.0, 20000.0),
          "humid-deep": (95, 45.0, 55.0, 75.0, 6.0, 8000.0),
          "stable-stratus": (3, 95.0, 0.0, 0.0, 0.0, 9000.0),
          "dry": (1, 0.0, 0.0, 18.0, 0.0, 40000.0)}.get(profile)
    surface = {"t": surface_t, "td": surface_td, "profile": profile}
    if wx:
        code, lo, mi, hi, pr, vis = wx
        surface.update({"weather_code": float(code), "cloud_cover_low": lo,
                        "cloud_cover_mid": mi, "cloud_cover_high": hi,
                        "cloud_cover": max(lo, mi, hi), "precipitation": pr,
                        "visibility": vis, "temperature_2m": surface_t,
                        "dew_point_2m": surface_td,
                        "relative_humidity_2m": 100.0 * esat_hpa(surface_td) / esat_hpa(surface_t)})
    return Sounding(levels=levels, lat=lat, lon=lon, when=when,
                    surface=surface, source=f"synthetic:{profile}")


# ----------------------------------------------------------- open-meteo -----

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
#: Open-Meteo's forecast reaches 16 days - today and 15 more, by the UTC date
#: (its forecast_days is at most 16, past_days at most 92); its archive of past
#: forecasts, the historical-forecast API, starts around 2022 (some models earlier)
FORECAST_DAYS_MAX = 16
ARCHIVE_FROM = date(2022, 1, 1)


class BeyondForecast(RuntimeError):
    """No forecast exists for the time asked: further ahead than the
    forecast reaches."""


def forecast_reach(now: datetime | None = None) -> datetime:
    """The last hour Open-Meteo forecasts: 23:00 UTC of the 16th day."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    d = now.date() + timedelta(days=FORECAST_DAYS_MAX - 1)
    return datetime(d.year, d.month, d.day, 23, tzinfo=timezone.utc)


def _day(d: date, hour: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)


def request_plan(when: datetime | None = None, now: datetime | None = None):
    """(endpoint, window, first hour, last hour): what request_open_meteo
    asks Open-Meteo for around `when`, and the hours the answer holds.
    Three days back to nine ahead: the forecast three days back and ten
    ahead (the usual case).  Further ahead, as far as the forecast reaches
    (its last hour, and half an hour past it, which takes that hour): the
    forecast a day back and sixteen ahead.  Further back: the archived
    forecast from the day before to the day after.  Further ahead than the
    forecast reaches: BeyondForecast."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    when = (when or now).astimezone(timezone.utc)
    today = now.date()
    edge = timedelta(minutes=30)
    # chosen by the hours each window holds (by days from now, a time nine
    # days ahead late in the UTC day fell past the usual window's last hour)
    first, last = _day(today - timedelta(days=3)), _day(today + timedelta(days=9), 23)
    if first <= when <= last + edge:
        return FORECAST_API, {"past_days": "3", "forecast_days": "10"}, first, last
    reach = forecast_reach(now)
    if when > last:
        if when > reach + edge:
            raise BeyondForecast(f"no forecast for {when:%Y-%m-%d %H:%M} UTC: Open-Meteo "
                                 f"forecasts {FORECAST_DAYS_MAX} days ahead, to "
                                 f"{reach:%Y-%m-%d %H:%M} UTC")
        return (FORECAST_API, {"past_days": "1", "forecast_days": str(FORECAST_DAYS_MAX)},
                _day(today - timedelta(days=1)), reach)
    a, b = when.date() - timedelta(days=1), when.date() + timedelta(days=1)
    return (HISTORICAL_API, {"start_date": a.strftime("%Y-%m-%d"),
                             "end_date": b.strftime("%Y-%m-%d")}, _day(a), _day(b, 23))


def region_plan(when: datetime | None = None, now: datetime | None = None):
    """(endpoint, window) of request_region: a day back and five ahead in the
    usual case; as many days either way as it takes to hold `when` (and a
    day beyond) up to three days back and as far ahead as the forecast
    reaches; further back, the archived forecast from the day before to the
    day after."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    when = (when or now).astimezone(timezone.utc)
    ahead = (when - now).total_seconds() / 86400.0
    if ahead >= -3.0:
        if when > forecast_reach(now) + timedelta(minutes=30):
            raise BeyondForecast(f"no forecast for {when:%Y-%m-%d %H:%M} UTC")
        back = min(max(math.ceil(-ahead + 0.5), 1), 92)
        days = min(max(math.ceil(ahead + 1.5), 5), FORECAST_DAYS_MAX)
        return FORECAST_API, {"past_days": str(back), "forecast_days": str(days)}
    a, b = when.date() - timedelta(days=1), when.date() + timedelta(days=1)
    return HISTORICAL_API, {"start_date": a.strftime("%Y-%m-%d"), "end_date": b.strftime("%Y-%m-%d")}

#: Always requested - every Open-Meteo model provides these.
_CORE_SURFACE = ["temperature_2m", "dew_point_2m", "relative_humidity_2m",
                 "surface_pressure", "cloud_cover", "cloud_cover_low",
                 "cloud_cover_mid", "cloud_cover_high", "weather_code",
                 "wind_speed_10m", "wind_direction_10m", "precipitation"]
#: Nice to have, but not offered by every model - dropped on a 400.
_EXTRA_SURFACE = ["cape", "lifted_index", "convective_inhibition",
                  "freezing_level_height", "boundary_layer_height", "visibility",
                  "pressure_msl", "showers", "soil_temperature_0cm", "is_day"]
#: The ground's state: lying snow (m) and root-zone soil water (m3/m3).  Asked
#: for in a tier of their own, so a model without them keeps the rest.
_LAND_SURFACE = ["snow_depth", "soil_moisture_9_to_27cm"]
#: kept hour by hour, SERIES_SPAN hours either side of the chosen time
_SERIES_VARS = ["weather_code", "precipitation", "cape", "surface_pressure",
                "pressure_msl", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
                "cloud_cover_high", "temperature_2m", "dew_point_2m",
                "wind_speed_10m", "wind_direction_10m", "visibility"]
SERIES_SPAN = 12
_SURFACE_VARS = _CORE_SURFACE + _EXTRA_SURFACE


def snow_cover(snd) -> float:
    """Fraction of the ground under lying snow, from the forecast's snow depth
    (m): full cover from 10 cm, as in the ECMWF land surface (Dutra et al.
    2010, eq. 8).  0 when the forecast gives none."""
    sf = getattr(snd, "surface", None) or {}
    d = sf.get("snow_depth")
    try:
        d = float(d)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(d) or d <= 0.0:
        return 0.0
    return min(1.0, d / 0.10)


def ground_dryness(snd) -> float:
    """How dry the ground looks, 0 (green) .. 1 (bare sand), from the
    forecast's root-zone soil water: green from 0.20 m3/m3 down, bare at the
    wilting point (0.066, Noah's loam).  A forecast has no land cover; this
    is the program's proxy for it.  0 when the forecast gives none."""
    sf = getattr(snd, "surface", None) or {}
    try:
        th = float(sf.get("soil_moisture_9_to_27cm"))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(th) or th < 0.0:
        return 0.0
    x = min(max((0.20 - th) / (0.20 - 0.066), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def station_elevation(snd) -> float:
    """The ground's height above sea level, m (0 when unknown)."""
    try:
        e = float(getattr(snd, "elevation", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return e if math.isfinite(e) else 0.0


def _request(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.load(r)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data.get("reason", "Open-Meteo error")))
    return data


def request_open_meteo(lat: float, lon: float, when: datetime | None = None,
                       timeout: float = 15.0) -> dict:
    """The forecast service's whole hourly answer around `when`: three days
    back and ten ahead (or, further ahead, a day back and sixteen ahead; far
    back, the historical forecast from the day before to the day after -
    request_plan).  Raises on any network or parse failure, and
    BeyondForecast for a time no forecast reaches."""
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    endpoint, window, _, _ = request_plan(when)

    plevel = [f"{v}_{p}hPa" for p in PRESSURE_LEVELS
              for v in ("temperature", "relative_humidity", "cloud_cover",
                        "wind_speed", "wind_direction", "geopotential_height")]

    base = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "timezone": "UTC", "wind_speed_unit": "ms", "cell_selection": "nearest"}
    base.update(window)

    data = None
    last = None
    for surface in (_SURFACE_VARS + _LAND_SURFACE, _SURFACE_VARS, _CORE_SURFACE):
        q = dict(base)
        q["hourly"] = ",".join(surface + plevel)
        try:
            data = _request(endpoint + "?" + urllib.parse.urlencode(q), timeout)
            break
        except Exception as e:                                   # noqa: BLE001
            last = e
    if data is None:
        raise last if last else RuntimeError("Open-Meteo request failed")
    if not data.get("hourly", {}).get("time"):
        raise RuntimeError("Open-Meteo returned no hours")
    return data


def hour_index(times: list, when: datetime) -> int:
    """The index of the hour in an Open-Meteo time list nearest to `when`."""
    target = when.strftime("%Y-%m-%dT%H:00")
    if target in times:
        return times.index(target)
    return min(range(len(times)),
               key=lambda k: abs(datetime.fromisoformat(times[k]).replace(
                   tzinfo=timezone.utc) - when))


def fetch_open_meteo(lat: float, lon: float, when: datetime | None = None,
                     timeout: float = 15.0) -> Sounding:
    """Fetch a real sounding: the forecast hour nearest to `when`.  Raises
    on any network or parse failure so the caller can decide whether to
    fall back."""
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    data = request_open_meteo(lat, lon, when, timeout)
    i = hour_index(data["hourly"]["time"], when)
    return sounding_from_hourly(data, i, lat, lon, when)


def sounding_from_hourly(data: dict, i: int, lat: float, lon: float,
                         when: datetime) -> Sounding:
    """The sounding of hour i of an Open-Meteo answer (request_open_meteo)."""
    times = data["hourly"]["time"]
    h = data["hourly"]

    def val(key, default=None):
        arr = h.get(key)
        if not arr or i >= len(arr) or arr[i] is None:
            return default
        try:
            x = float(arr[i])
        except (TypeError, ValueError):
            return default
        # a NaN or an infinity is a gap, never a number
        return x if math.isfinite(x) else default

    elev = float(data.get("elevation", 0.0))
    levels = []
    for p in PRESSURE_LEVELS:
        t = val(f"temperature_{p}hPa")
        rh = val(f"relative_humidity_{p}hPa")
        z = val(f"geopotential_height_{p}hPa")
        if t is None or rh is None or z is None:
            continue
        spd = val(f"wind_speed_{p}hPa", 0.0)
        drc = val(f"wind_direction_{p}hPa", 0.0)
        cc = val(f"cloud_cover_{p}hPa")
        u = -spd * math.sin(math.radians(drc))
        v = -spd * math.cos(math.radians(drc))
        levels.append(Level(p=float(p), z=z, t=t, rh=rh,
                            td=dewpoint_from_rh(t, rh), u=u, v=v,
                            cc=(cc / 100.0 if cc is not None else -1.0)))
    # pressure levels below the ground (the forecast model extrapolates them
    # at a high station) are not air the sky is made of
    t2 = val("temperature_2m")
    levels = [l for l in levels if l.z > elev + (30.0 if t2 is not None else -1.0)]
    if len(levels) < 6:
        raise RuntimeError("Open-Meteo returned too few pressure levels")

    # splice in the surface as the lowest level
    if t2 is not None:
        rh2 = val("relative_humidity_2m", 70.0)
        sp = val("surface_pressure", P0)
        spd = val("wind_speed_10m", 0.0)
        drc = val("wind_direction_10m", 0.0)
        levels.insert(0, Level(p=sp, z=elev + 2.0, t=t2, rh=rh2,
                               td=dewpoint_from_rh(t2, rh2),
                               u=-spd * math.sin(math.radians(drc)),
                               v=-spd * math.cos(math.radians(drc)), cc=-1.0))
    levels.sort(key=lambda l: l.z)

    surface = {k: val(k) for k in _SURFACE_VARS + _LAND_SURFACE}
    surface["elevation"] = elev
    hours = [dh for dh in range(-SERIES_SPAN, SERIES_SPAN + 1) if 0 <= i + dh < len(times)]
    series = {}
    for var in _SERIES_VARS:
        arr = h.get(var)
        if not arr:
            continue
        series[var] = [(float(arr[i + dh]) if i + dh < len(arr) and arr[i + dh] is not None
                        else None) for dh in hours]
    return Sounding(levels=levels, lat=lat, lon=lon, when=when, elevation=elev,
                    surface=surface, source="open-meteo", series_hours=hours, series=series)


# ------------------------------------------------------------ the region ----
_REGION_VARS = ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "precipitation",
                "weather_code", "cape", "pressure_msl", "temperature_2m",
                "wind_speed_700hPa", "wind_direction_700hPa",
                "wind_speed_850hPa", "wind_direction_850hPa"]


@dataclass
class RegionPoint:
    east_km: float          # offset from the observer
    north_km: float
    lat: float
    lon: float
    values: dict            # variable -> value at the chosen hour
    ahead: dict             # variable -> value three hours later


@dataclass
class Region:
    """The forecast on an n x n grid around the observer: where the cloud
    is, as a satellite would see it, from the same forecast model."""
    points: list
    spacing_km: float
    n: int
    source: str = "open-meteo"

    def grid(self, var: str, ahead: bool = False):
        """n x n array [north index][east index] of a variable (NaN where
        missing), rows from south to north."""
        import numpy as _np
        out = _np.full((self.n, self.n), _np.nan)
        for p in self.points:
            i = int(round(p.north_km / self.spacing_km)) + self.n // 2
            j = int(round(p.east_km / self.spacing_km)) + self.n // 2
            v = (p.ahead if ahead else p.values).get(var)
            if v is not None and 0 <= i < self.n and 0 <= j < self.n:
                out[i, j] = v
        return out


def region_offsets(n: int, spacing_km: float):
    half = n // 2
    return [((j - half) * spacing_km, (i - half) * spacing_km)
            for i in range(n) for j in range(n)]


def fetch_region(lat: float, lon: float, when: datetime | None = None, n: int = 7,
                 spacing_km: float = 40.0, timeout: float = 20.0) -> Region:
    """The forecast on an n x n grid spacing_km apart centred on the place,
    in one request: Open-Meteo's multi-location form (comma-separated
    coordinates; the answer is a list, one structure per location)."""
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    offs, data = request_region(lat, lon, when, n, spacing_km, timeout)
    return region_from_hourly(offs, data, when, n, spacing_km)


def request_region(lat: float, lon: float, when: datetime | None = None, n: int = 7,
                   spacing_km: float = 40.0, timeout: float = 20.0):
    """(offsets, answer): the n x n grid's positions (east km, north km, lat,
    lon) and Open-Meteo's hourly answer for each, a day back and five
    ahead - or as many days as hold `when`, or the historical forecast
    around it (region_plan)."""
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    lats, lons, offs = [], [], []
    for (e, nn) in region_offsets(n, spacing_km):
        la = lat + nn / 111.2
        lo = lon + e / (111.2 * max(math.cos(math.radians(lat)), 0.05))
        # a point carried past a pole lies on the far meridian
        if la > 90.0:
            la, lo = 180.0 - la, lo + 180.0
        elif la < -90.0:
            la, lo = -180.0 - la, lo + 180.0
        lo = ((lo + 180.0) % 360.0) - 180.0
        lats.append(f"{la:.4f}")
        lons.append(f"{lo:.4f}")
        offs.append((e, nn, la, lo))
    q = {"latitude": ",".join(lats), "longitude": ",".join(lons), "timezone": "UTC",
         "wind_speed_unit": "ms", "cell_selection": "nearest",
         "hourly": ",".join(_REGION_VARS)}
    endpoint, window = region_plan(when)
    q.update(window)
    data = _request(endpoint + "?" + urllib.parse.urlencode(q), timeout)
    if isinstance(data, dict):
        data = [data]
    return offs, data


def region_from_hourly(offs, data, when: datetime, n: int = 7,
                       spacing_km: float = 40.0) -> Region:
    """The Region at the hour nearest `when` (and three hours later) from a
    request_region answer."""
    target = when.strftime("%Y-%m-%dT%H:00")
    pts = []
    for (e, nn, la, lo), d in zip(offs, data):
        h = d.get("hourly", {})
        times = h.get("time", [])
        if not times:
            continue
        if target in times:
            i = times.index(target)
        else:
            # the nearest hour, and only if it is the chosen hour's neighbour
            gaps = [abs((datetime.fromisoformat(t).replace(tzinfo=timezone.utc) - when)
                        .total_seconds()) for t in times]
            i = min(range(len(times)), key=gaps.__getitem__)
            if gaps[i] > 5400.0:
                continue
        vals, ahead = {}, {}
        for var in _REGION_VARS:
            arr = h.get(var) or []
            vals[var] = float(arr[i]) if i < len(arr) and arr[i] is not None else None
            k = i + 3
            ahead[var] = float(arr[k]) if k < len(arr) and arr[k] is not None else None
        pts.append(RegionPoint(e, nn, la, lo, vals, ahead))
    if len(pts) < (n * n) // 2:
        raise RuntimeError("Open-Meteo returned too few region points")
    return Region(pts, spacing_km, n)


def get_sounding(lat: float, lon: float, when: datetime | None = None,
                 allow_network: bool = True, fallback_profile: str = "tropical-fair"):
    """(sounding, note).  Never raises."""
    if allow_network:
        try:
            s = fetch_open_meteo(lat, lon, when)
            note = f"Open-Meteo, {len(s.levels)} levels"
            try:
                s.region = fetch_region(lat, lon, when)
                note += f", {len(s.region.points)} points around"
            except Exception as e:                               # noqa: BLE001
                note += f" (no regional grid: {type(e).__name__})"
            return s, note
        except Exception as e:                                   # noqa: BLE001
            return (standard_sounding(lat, lon, when, profile=fallback_profile),
                    f"offline ({type(e).__name__}) - synthetic {fallback_profile}")
    return (standard_sounding(lat, lon, when, profile=fallback_profile),
            f"synthetic {fallback_profile}")
