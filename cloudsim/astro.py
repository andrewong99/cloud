# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
Sun, Moon and star-field geometry.

Pure Python + math, no ephemeris dependency.  Sun follows Meeus chapter 25
(low precision, |error| < 0.01 deg for 1800-2200); Moon follows Meeus
chapter 47 truncated to the largest periodic terms (|error| ~ 10 arcmin,
which is a third of the Moon's own diameter - fine for lighting and for a
visually correct phase, not for occultation work).

Everything here is checked against pyephem in selftest.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

#: Solar constant at 1 au, W m^-2 (WMO/IPCC value).
SOLAR_IRRADIANCE = 1361.0
#: Mean angular diameter of the Sun and Moon, degrees.
SUN_ANG_DIAM_DEG = 0.5334
MOON_ANG_DIAM_DEG = 0.5181


def julian_day(dt: datetime) -> float:
    """Julian Day from a timezone-aware datetime (converted to UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    d = (dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond * 1e-6) / 60.0) / 60.0) / 24.0)
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def _norm360(x: float) -> float:
    return x - 360.0 * math.floor(x / 360.0)


def obliquity(jd: float) -> float:
    """Mean obliquity of the ecliptic, degrees (Meeus 22.2)."""
    t = (jd - 2451545.0) / 36525.0
    return (23.0 + 26.0 / 60.0 + 21.448 / 3600.0
            - (46.8150 * t + 0.00059 * t * t - 0.001813 * t ** 3) / 3600.0)


def gmst_deg(jd: float) -> float:
    """Greenwich mean sidereal time in degrees (Meeus 12.4)."""
    t = (jd - 2451545.0) / 36525.0
    theta = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
             + 0.000387933 * t * t - t ** 3 / 38710000.0)
    return _norm360(theta)


def lst_deg(jd: float, lon_deg: float) -> float:
    """Local mean sidereal time, degrees (east longitude positive)."""
    return _norm360(gmst_deg(jd) + lon_deg)


# ------------------------------------------------------------------- Sun ----

def sun_ecliptic(jd: float) -> tuple[float, float]:
    """Apparent ecliptic longitude (deg) and radius vector (au)."""
    t = (jd - 2451545.0) / 36525.0
    l0 = _norm360(280.46646 + 36000.76983 * t + 0.0003032 * t * t)
    m = _norm360(357.52911 + 35999.05029 * t - 0.0001537 * t * t)
    e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t
    mr = m * D2R
    c = ((1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(mr)
         + (0.019993 - 0.000101 * t) * math.sin(2 * mr)
         + 0.000289 * math.sin(3 * mr))
    true_long = l0 + c
    v = m + c
    r = (1.000001018 * (1 - e * e)) / (1 + e * math.cos(v * D2R))
    omega = 125.04 - 1934.136 * t
    lam = true_long - 0.00569 - 0.00478 * math.sin(omega * D2R)   # apparent
    return _norm360(lam), r


def sun_equatorial(jd: float) -> tuple[float, float, float]:
    """Apparent right ascension (deg), declination (deg), distance (au)."""
    lam, r = sun_ecliptic(jd)
    t = (jd - 2451545.0) / 36525.0
    omega = 125.04 - 1934.136 * t
    eps = obliquity(jd) + 0.00256 * math.cos(omega * D2R)
    lr, er = lam * D2R, eps * D2R
    ra = math.atan2(math.cos(er) * math.sin(lr), math.cos(lr))
    dec = math.asin(math.sin(er) * math.sin(lr))
    return _norm360(ra * R2D), dec * R2D, r


# ------------------------------------------------------------------ Moon ----

# Meeus 47.A, the 24 largest terms in longitude/distance and 18 in latitude.
_LR = [
    (0, 0, 1, 0, 6288774, -20905355), (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968), (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888), (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158), (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733), (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620), (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755), (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0), (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782), (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636), (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824), (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675), (2, -1, 1, 0, 4036, -12831),
]
_B = [
    (0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237), (2, 0, -1, 1, 55413), (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573), (0, 0, 2, 1, 17198), (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822), (2, -1, 0, -1, 8216), (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200), (2, 1, 0, -1, -3359), (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211), (2, -1, -1, -1, 2065), (0, 1, -1, -1, -1870),
]


def moon_ecliptic(jd: float) -> tuple[float, float, float]:
    """Geocentric ecliptic longitude (deg), latitude (deg), distance (km)."""
    t = (jd - 2451545.0) / 36525.0
    lp = _norm360(218.3164477 + 481267.88123421 * t - 0.0015786 * t * t
                  + t ** 3 / 538841.0 - t ** 4 / 65194000.0)
    d = _norm360(297.8501921 + 445267.1114034 * t - 0.0018819 * t * t
                 + t ** 3 / 545868.0 - t ** 4 / 113065000.0)
    m = _norm360(357.5291092 + 35999.0502909 * t - 0.0001536 * t * t + t ** 3 / 24490000.0)
    mp = _norm360(134.9633964 + 477198.8675055 * t + 0.0087414 * t * t
                  + t ** 3 / 69699.0 - t ** 4 / 14712000.0)
    f = _norm360(93.2720950 + 483202.0175233 * t - 0.0036539 * t * t
                 - t ** 3 / 3526000.0 + t ** 4 / 863310000.0)
    e = 1.0 - 0.002516 * t - 0.0000074 * t * t

    sl = sr = sb = 0.0
    for cd, cm, cmp_, cf, cl, cr in _LR:
        arg = (cd * d + cm * m + cmp_ * mp + cf * f) * D2R
        ecc = e ** abs(cm)
        sl += cl * ecc * math.sin(arg)
        sr += cr * ecc * math.cos(arg)
    for cd, cm, cmp_, cf, cb in _B:
        arg = (cd * d + cm * m + cmp_ * mp + cf * f) * D2R
        sb += cb * (e ** abs(cm)) * math.sin(arg)

    lon = _norm360(lp + sl / 1e6)
    lat = sb / 1e6
    dist = 385000.56 + sr / 1000.0
    return lon, lat, dist


def moon_equatorial(jd: float) -> tuple[float, float, float]:
    lon, lat, dist = moon_ecliptic(jd)
    eps = obliquity(jd) * D2R
    lr, br = lon * D2R, lat * D2R
    ra = math.atan2(math.sin(lr) * math.cos(eps) - math.tan(br) * math.sin(eps), math.cos(lr))
    dec = math.asin(math.sin(br) * math.cos(eps) + math.cos(br) * math.sin(eps) * math.sin(lr))
    return _norm360(ra * R2D), dec * R2D, dist


def moon_phase(jd: float) -> tuple[float, float]:
    """(phase angle in degrees, illuminated fraction 0..1).

    Phase angle 0 = full, 180 = new (Meeus 48).
    """
    sra, sdec, sr_au = sun_equatorial(jd)
    mra, mdec, mdist = moon_equatorial(jd)
    sr_km = sr_au * 149597870.7
    psi = math.acos(max(-1.0, min(1.0, math.sin(sdec * D2R) * math.sin(mdec * D2R)
                                  + math.cos(sdec * D2R) * math.cos(mdec * D2R)
                                  * math.cos((sra - mra) * D2R))))
    i = math.atan2(sr_km * math.sin(psi), mdist - sr_km * math.cos(psi))
    return i * R2D, (1.0 + math.cos(i)) / 2.0


def moon_bright_limb_angle(jd: float) -> float:
    """Position angle of the Moon's bright limb, degrees east of north (Meeus 48.5)."""
    sra, sdec, _ = sun_equatorial(jd)
    mra, mdec, _ = moon_equatorial(jd)
    dra = (sra - mra) * D2R
    y = math.cos(sdec * D2R) * math.sin(dra)
    x = (math.sin(sdec * D2R) * math.cos(mdec * D2R)
         - math.cos(sdec * D2R) * math.sin(mdec * D2R) * math.cos(dra))
    return _norm360(math.atan2(y, x) * R2D)


# ------------------------------------------------------- coordinate change --

def equatorial_to_horizontal(ra_deg: float, dec_deg: float, jd: float,
                             lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Return (azimuth deg from North through East, altitude deg), geometric."""
    h = (lst_deg(jd, lon_deg) - ra_deg) * D2R
    phi, dec = lat_deg * D2R, dec_deg * D2R
    sin_alt = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(h)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(math.sin(h), math.cos(h) * math.sin(phi) - math.tan(dec) * math.cos(phi))
    return _norm360(az * R2D + 180.0), alt * R2D


def parallax_alt_correction_deg(alt_deg: float, dist_km: float) -> float:
    """Diurnal parallax: how much lower a body at `dist_km` appears from the
    surface than from the Earth's centre.  ~0.95 deg at the horizon for the
    Moon, negligible for the Sun.  Returns a value to ADD to the altitude."""
    sin_pi = 6378.14 / dist_km
    return -math.degrees(math.asin(max(-1.0, min(1.0, sin_pi * math.cos(alt_deg * D2R)))))


def refraction_deg(alt_deg: float, p_hpa: float = 1013.25, t_c: float = 15.0) -> float:
    """Bennett's refraction formula, degrees to add to a geometric altitude.
    Below the horizon it is tapered to nothing between -1 and -3 degrees
    (smoothstep): there is no line of sight to refract there, and the
    formula was cut off at -2 degrees instead - the Sun's direction then
    dropped 0.73 degrees in an instant, and the twilight sky with it (4.6
    times darker in one frame, at every sunset and sunrise)."""
    if alt_deg <= -3.0:
        return 0.0
    r = 1.02 / math.tan((alt_deg + 10.3 / (alt_deg + 5.11)) * D2R) / 60.0
    if alt_deg < -1.0:
        t = (alt_deg + 3.0) / 2.0
        r *= t * t * (3.0 - 2.0 * t)
    return r * (p_hpa / 1010.0) * (283.0 / (273.0 + t_c))


def unit_from_azalt(az_deg: float, alt_deg: float) -> tuple[float, float, float]:
    """East-North-Up unit vector.  x=East, y=Up, z=North (renderer convention)."""
    a, e = az_deg * D2R, alt_deg * D2R
    return (math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a))


@dataclass
class SkyGeometry:
    jd: float
    sun_az: float
    sun_alt: float
    sun_alt_refracted: float
    sun_dist_au: float
    moon_az: float
    moon_alt: float
    moon_dist_km: float
    moon_phase_angle: float
    moon_illum: float
    moon_limb_angle: float
    lst: float
    lat: float
    lon: float

    @property
    def sun_dir(self):
        return unit_from_azalt(self.sun_az, self.sun_alt_refracted)

    @property
    def moon_dir(self):
        return unit_from_azalt(self.moon_az, self.moon_alt)

    @property
    def sun_ang_radius_deg(self) -> float:
        return 0.5 * SUN_ANG_DIAM_DEG / self.sun_dist_au

    @property
    def moon_ang_radius_deg(self) -> float:
        return 0.5 * MOON_ANG_DIAM_DEG * (385000.56 / self.moon_dist_km)


def sky_geometry(dt: datetime, lat_deg: float, lon_deg: float) -> SkyGeometry:
    jd = julian_day(dt)
    sra, sdec, sdist = sun_equatorial(jd)
    saz, salt = equatorial_to_horizontal(sra, sdec, jd, lat_deg, lon_deg)
    mra, mdec, mdist = moon_equatorial(jd)
    maz, malt = equatorial_to_horizontal(mra, mdec, jd, lat_deg, lon_deg)
    malt += parallax_alt_correction_deg(malt, mdist)      # topocentric
    malt += refraction_deg(malt)
    pa, illum = moon_phase(jd)
    return SkyGeometry(
        jd=jd, sun_az=saz, sun_alt=salt, sun_alt_refracted=salt + refraction_deg(salt),
        sun_dist_au=sdist, moon_az=maz, moon_alt=malt, moon_dist_km=mdist,
        moon_phase_angle=pa, moon_illum=illum, moon_limb_angle=moon_bright_limb_angle(jd),
        lst=lst_deg(jd, lon_deg), lat=lat_deg, lon=lon_deg)
