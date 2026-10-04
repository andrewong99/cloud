# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
Which sky the weather data calls for, and why.

This reads the Open-Meteo data the program fetched - the pressure-level
profile, the surface fields, their hour-by-hour series and the grid of
points around the place - and decides which physical regime is making the
clouds over the observer: a thunderstorm (here or on its way), showers,
fair-weather cumulus, a stratocumulus deck, an altocumulus layer, virga,
fog, a high veil, or nothing.

It then builds the cloud-resolving model for that regime FROM THE SAME
PROFILE: the model's temperature, humidity and wind are the forecast's, so
its clouds form at the forecast's heights and drift, lean and trail their
virga with the forecast's wind.

Every step keeps the number that drove it.  `Regime.lines()` gives the
argument in plain words - which data, what value, what it means - and the
alternatives that were considered and why they lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import crm
from .sounding import Sounding, inhibition_text, lcl_bolton, theta

OMEGA = 7.2921e-5

# WMO code table 4677 as Open-Meteo reports it
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snowfall", 73: "moderate snowfall", 75: "heavy snowfall", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(deg: float) -> str:
    return COMPASS[int((deg % 360.0) / 22.5 + 0.5) % 16]


def wind_from(u: float, v: float) -> float:
    """Meteorological direction the wind blows FROM, degrees."""
    return math.degrees(math.atan2(-u, -v)) % 360.0


def code_text(code) -> str:
    if code is None:
        return "no weather code"
    c = int(code)
    return f"code {c} - {WMO_CODES.get(c, 'unlisted')}"


@dataclass
class Evidence:
    topic: str          # weather code / cloud cover / instability / pressure / ...
    fact: str           # the data, with its numbers
    meaning: str        # what it means for the sky
    weight: int = 0     # +1 supports the choice, -1 argues against it, 0 context

    def line(self) -> str:
        mark = {1: "+", -1: "-", 0: "·"}[self.weight]
        return f"{mark} {self.topic}: {self.fact} -> {self.meaning}"


@dataclass
class Regime:
    key: str                       # thunderstorm, storm_coming, showers, cumulus, ...
    title: str
    scenario: object | None        # crm.Scenario, or None when nothing is to be modelled
    evidence: list = field(default_factory=list)
    decision: str = ""
    rejected: list = field(default_factory=list)       # (title, why not)
    also: list = field(default_factory=list)           # other layers present
    spinup_s: float = 0.0
    facts: dict = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [self.decision] if self.decision else []
        out += [e.line() for e in self.evidence]
        if self.also:
            out.append("also in the sky: " + "; ".join(self.also))
        for t, why in self.rejected:
            out.append(f"not {t}: {why}")
        return out


# ------------------------------------------------------------ analysis ----
def _num(x, default=None):
    """A number from the data, or the default for a missing, malformed or
    non-finite one (a forecast gap is null; NaN or infinity is never data)."""
    try:
        v = float(x) if x is not None else default
    except (TypeError, ValueError):
        return default
    if v is not None and not math.isfinite(v):
        return default
    return v


def analyse(snd: Sounding, sun_alt_deg: float = 45.0) -> dict:
    """Every quantity the decision reads, from the data, with no judgement
    in it yet."""
    s = snd.surface or {}
    a: dict = {}
    elev = float(snd.elevation or 0.0)
    a["elev"] = elev
    a["code"] = _num(s.get("weather_code"))
    a["low"] = _num(s.get("cloud_cover_low"))
    a["mid"] = _num(s.get("cloud_cover_mid"))
    a["high"] = _num(s.get("cloud_cover_high"))
    a["total"] = _num(s.get("cloud_cover"))
    a["precip"] = _num(s.get("precipitation"), 0.0)
    a["showers"] = _num(s.get("showers"), 0.0)
    a["vis"] = _num(s.get("visibility"))
    a["blh"] = _num(s.get("boundary_layer_height"))
    a["fzl"] = _num(s.get("freezing_level_height"))
    a["cape_model"] = _num(s.get("cape"))
    a["li"] = _num(s.get("lifted_index"))
    a["cin_model"] = _num(s.get("convective_inhibition"))
    a["tsfc_skin"] = _num(s.get("soil_temperature_0cm"))
    lv0 = snd.levels[0]
    a["t2"] = _num(s.get("temperature_2m"), lv0.t)
    a["td2"] = _num(s.get("dew_point_2m"), lv0.td)
    a["rh2"] = _num(s.get("relative_humidity_2m"), lv0.rh)
    a["p_sfc"] = _num(s.get("surface_pressure"), lv0.p)
    a["p_msl"] = _num(s.get("pressure_msl"))
    a["wind10"] = _num(s.get("wind_speed_10m"), math.hypot(lv0.u, lv0.v))
    a["wdir10"] = _num(s.get("wind_direction_10m"), wind_from(lv0.u, lv0.v))
    # the forecast's own synthetic-profile covers, when the data are offline
    if a["low"] is None or a["mid"] is None or a["high"] is None:
        cov = {"low": 0.0, "mid": 0.0, "high": 0.0}
        for (b, t, c) in snd.moist_layers():
            et = "low" if b - elev < 2000 else ("mid" if b - elev < 6000 else "high")
            cov[et] = max(cov[et], c * 100.0)
        for k in cov:
            if a[k] is None:
                a[k] = cov[k]
        a["covers_from_profile"] = True
    # the parcel, from the profile itself
    cape, cin, zlcl, lfc, el, _ = snd.parcel(mixed_depth=400)
    a["cape"], a["cin"], a["zlcl"], a["lfc"], a["el"] = cape, cin, zlcl, lfc, el
    tl, pl, zl = lcl_bolton(a["t2"], a["td2"], a["p_sfc"])
    a["lcl_sfc"] = zl
    a["spread"] = a["t2"] - a["td2"]
    a["trop"] = snd.tropopause()
    a["fz_profile"] = snd.freezing_level()
    z0 = snd.levels[0].z
    a["shear06"] = snd.bulk_shear(z0, z0 + 6000.0)
    a["shear01"] = snd.bulk_shear(z0, z0 + 1000.0)
    u850, v850 = snd.wind_at(z0 + 1500.0)
    u700, v700 = snd.wind_at(z0 + 3000.0)
    u500, v500 = snd.wind_at(z0 + 5600.0)
    a["w850"] = (u850, v850)
    a["w700"] = (u700, v700)
    a["w500"] = (u500, v500)
    a["storm_motion"] = snd.storm_motion()
    a["layers"] = [(b - elev, t - elev, c) for (b, t, c) in snd.moist_layers()]
    a["inversions"] = [(z - elev, dz, dt) for (z, dz, dt) in snd.inversions()]
    a["sun_alt"] = sun_alt_deg
    # trends from the hourly series
    def ser(var, h):
        return snd.series_at(var, h)
    p_now = ser("pressure_msl", 0) or ser("surface_pressure", 0)
    p_3 = ser("pressure_msl", -3) or ser("surface_pressure", -3)
    a["p_tend3"] = (p_now - p_3) if (p_now is not None and p_3 is not None) else None
    nxt = []
    for h in range(1, 7):
        c = ser("weather_code", h)
        if c is not None:
            nxt.append((h, int(c), ser("precipitation", h) or 0.0, ser("cape", h)))
    a["next"] = nxt
    prv = []
    for h in range(-12, 0):
        c = ser("weather_code", h)
        if c is not None:
            prv.append((h, int(c)))
    a["prev"] = prv
    # the region: storms and cloud around
    a["region_storms"] = []
    a["region"] = snd.region
    if snd.region is not None:
        for p in snd.region.points:
            c = p.values.get("weather_code")
            if c is not None and c >= 95:
                d = math.hypot(p.east_km, p.north_km)
                if d < 1.0:
                    continue
                spd = p.values.get("wind_speed_700hPa")
                drc = p.values.get("wind_direction_700hPa")
                a["region_storms"].append((d, p.east_km, p.north_km, int(c), spd, drc))
        a["region_storms"].sort()
    return a


def _storm_eta(a):
    """The nearest thunderstorm around, and when the 700 hPa wind that
    steers it would bring it overhead: (distance km, bearing deg, eta h or
    None, text)."""
    if not a["region_storms"]:
        return None
    d, e, n, c, spd, drc = a["region_storms"][0]
    bearing = math.degrees(math.atan2(e, n)) % 360.0
    if spd is None or drc is None:
        su, sv = a["storm_motion"]
    else:
        su = -spd * math.sin(math.radians(drc))
        sv = -spd * math.cos(math.radians(drc))
    # closing speed: component of the storm's motion toward the observer
    to_obs = (-e / d, -n / d)
    closing = (su * to_obs[0] + sv * to_obs[1]) * 3.6          # km/h
    eta = d / closing if closing > 5.0 else None
    txt = (f"{WMO_CODES.get(c, 'thunderstorm')} {d:.0f} km to the {compass(bearing)}, "
           f"steered by the {math.hypot(su, sv):.0f} m/s wind at 3 km"
           + (f" toward you: overhead in about {eta:.1f} h" if eta else ", not toward you"))
    return d, bearing, eta, txt


# ------------------------------------------------------------ decision ----
def choose(snd: Sounding, sun_alt_deg: float = 45.0, size: str = "standard",
           lat: float | None = None) -> Regime:
    """Decide the regime and build its model.  The order of the tests is the
    order in which one regime hides or overrides another in a real sky:
    a storm over everything, then fog at the ground, a low deck over the
    cloud above it, and so on."""
    a = analyse(snd, sun_alt_deg)
    lat = snd.lat if lat is None else lat
    ev: list[Evidence] = []
    rej: list = []
    code = a["code"]
    ci = int(code) if code is not None else None
    day = sun_alt_deg > 5.0

    # ---- what the data say, in the order a forecaster reads them --------
    ev.append(Evidence("weather code", code_text(code) + " (Open-Meteo, this hour)",
                       _code_meaning(ci)))
    cov_note = " (from the profile's humidity - no cover in the data)" \
        if a.get("covers_from_profile") else ""
    ev.append(Evidence("cloud cover",
                       f"low {a['low']:.0f}%  mid {a['mid']:.0f}%  high {a['high']:.0f}%" + cov_note,
                       _cover_meaning(a)))
    ev.append(Evidence("instability",
                       f"CAPE {a['cape']:.0f} J/kg (forecast model "
                       + (f"{a['cape_model']:.0f}" if a['cape_model'] is not None else "n/a")
                       + f"), {inhibition_text(a['cin'], a['lfc'])}, LCL {a['zlcl']:.0f} m"
                       + (f", tops could reach {a['el']/1000:.1f} km" if a['el'] else ""),
                       _instability_meaning(a)))
    if a["p_tend3"] is not None:
        ev.append(Evidence("pressure",
                           f"{(a['p_msl'] or a['p_sfc']):.1f} hPa, {a['p_tend3']:+.1f} hPa in 3 h",
                           _pressure_meaning(a["p_tend3"])))
    else:
        ev.append(Evidence("pressure", f"{(a['p_msl'] or a['p_sfc']):.1f} hPa",
                           "no hourly series, so no tendency"))
    ev.append(Evidence("temperature",
                       f"{a['t2']:.1f} C, dew point {a['td2']:.1f} C (spread {a['spread']:.1f} K)",
                       f"air lifted from the ground saturates at about {a['lcl_sfc']:.0f} m"
                       + (", i.e. at the surface: fog or drizzle weather" if a['lcl_sfc'] < 150 else "")))
    u8, v8 = a["w850"]
    u5, v5 = a["w500"]
    ev.append(Evidence("wind",
                       f"{a['wind10']:.0f} m/s from {compass(a['wdir10'])} at 10 m, "
                       f"{math.hypot(u8, v8):.0f} m/s from {compass(wind_from(u8, v8))} at 1.5 km, "
                       f"{math.hypot(u5, v5):.0f} m/s from {compass(wind_from(u5, v5))} at 5.6 km; "
                       f"0-6 km shear {a['shear06']:.0f} m/s",
                       _wind_meaning(a)))
    storm = _storm_eta(a)
    nxt_thunder = next(((h, c) for (h, c, pr, cp) in a["next"] if c >= 95), None)
    nxt_rain = sum(pr for (h, c, pr, cp) in a["next"])
    if nxt_thunder:
        ev.append(Evidence("next hours", f"thunderstorm forecast in {nxt_thunder[0]} h "
                           f"({WMO_CODES.get(nxt_thunder[1], '')})",
                           "a storm is on its way", 1 if (ci or 0) < 95 else 0))
    elif a["next"]:
        worst = max(a["next"], key=lambda t: t[1])
        ev.append(Evidence("next hours",
                           f"worst in the next 6 h: {WMO_CODES.get(worst[1], worst[1])} "
                           f"(+{worst[0]} h), {nxt_rain:.1f} mm in total",
                           "no storm on the way" if worst[1] < 80 else "showers on the way"))
    if storm:
        ev.append(Evidence("around", storm[3],
                           "a storm approaching" if storm[2] else "a storm nearby"))
    elif a["region"] is not None:
        ev.append(Evidence("around", f"no thunderstorm within "
                           f"{a['region'].spacing_km * (a['region'].n // 2):.0f} km", "none coming"))

    # ---- the decision ----------------------------------------------------
    deep_possible = a["cape"] >= 300 and a["el"] and (a["el"] - a["zlcl"]) >= 4000
    storm_here = ci is not None and ci >= 95
    storm_coming = (not storm_here) and (bool(nxt_thunder and nxt_thunder[0] <= 3)
                                         or bool(storm and storm[2] is not None and storm[2] <= 3))
    showers = ci is not None and ci in (80, 81, 82, 85, 86)
    fog = (ci in (45, 48)) or (a["vis"] is not None and a["vis"] < 1000.0)
    stratiform_rain = ci is not None and (51 <= ci <= 67 or 71 <= ci <= 77) and not deep_possible
    low_layers = [L for L in a["layers"] if L[0] < 2000.0]
    mid_layers = [L for L in a["layers"] if 2000.0 <= L[0] < 7000.0]

    # convection has to be in the data, not only possible: a showery code
    # now or within 2 h, or convective rain falling
    soon_showers = any(c in (80, 81, 82, 85, 86) for (h, c, pr, cp) in a["next"] if h <= 2)
    convective_rain = a["showers"] >= 0.3 or (a["precip"] >= 1.0 and deep_possible)
    key = None
    if storm_here or (deep_possible and a["cape"] >= 800 and a["precip"] >= 1.0):
        key = "thunderstorm"
    elif storm_coming and deep_possible:
        key = "storm_coming"
    elif showers or ((soon_showers or convective_rain) and a["cape"] >= 150):
        key = "showers"
    elif fog:
        key = "fog"
    elif stratiform_rain:
        key = "rain_layer"
    elif a["low"] >= 50.0 and low_layers:
        key = "stratocumulus"
    elif day and a["cape"] >= 20.0 and a["zlcl"] <= 3000.0 and a["low"] >= 5.0:
        key = "cumulus"
    elif a["mid"] >= 20.0 and mid_layers:
        key = "altocumulus"
    elif a["low"] >= 20.0 and low_layers:
        key = "stratocumulus"
    elif a["high"] >= 10.0:
        key = "cirrus"
    else:
        key = "clear"

    # the alternatives, and why each lost
    if key != "thunderstorm":
        rej.append(("a thunderstorm", "no thunder in the data" if not deep_possible else
                    f"the atmosphere could (CAPE {a['cape']:.0f} J/kg) but nothing has set it off here yet"))
    if key not in ("showers", "thunderstorm", "storm_coming"):
        rej.append(("showers / towering cumulus",
                    f"CAPE {a['cape']:.0f} J/kg is too little" if a["cape"] < 150
                    else ("it is night: no surface heating to start them" if not day
                          else f"the buoyancy is there (CAPE {a['cape']:.0f} J/kg) but the data "
                               f"report no showers now or in the next 2 h - the model is left to "
                               f"grow them if the thermals can")))
    if key != "fog" and not fog:
        rej.append(("fog", f"visibility {a['vis']/1000:.0f} km" if a["vis"] else
                    f"dew-point spread {a['spread']:.1f} K"))
    if key not in ("stratocumulus",):
        rej.append(("a stratocumulus deck", f"low cover {a['low']:.0f}%" if a["low"] < 50
                    else "a higher-priority regime"))
    if key not in ("altocumulus",):
        rej.append(("an altocumulus layer", f"mid cover {a['mid']:.0f}%" if a["mid"] < 20
                    else "hidden by the lower cloud / a higher-priority regime"))

    reg = _build(key, a, snd, size, lat, day, storm, nxt_thunder)
    reg.evidence = ev + reg.evidence
    reg.rejected = rej
    # what else is up there, drawn as Atlas layers
    for (b, t, c) in a["layers"]:
        et = "low" if b < 2000 else ("middle" if b < 7000 else "high")
        if reg.facts.get("layer") and abs(b - reg.facts["layer"][0]) < 300:
            continue
        reg.also.append(f"{et} layer {b:.0f}-{t:.0f} m, {c*100:.0f}% (Atlas layer)")
    reg.facts.update(a)
    return reg


def _code_meaning(ci):
    if ci is None:
        return "no present-weather report"
    if ci >= 95:
        return "deep convection is happening here now"
    if ci in (80, 81, 82, 85, 86):
        return "convective showers: cumulus congestus or cumulonimbus"
    if ci in (45, 48):
        return "cloud at the ground"
    if 51 <= ci <= 67 or 71 <= ci <= 77:
        return "steady precipitation from a layer cloud (stratus / nimbostratus)"
    if ci == 3:
        return "the sky is covered"
    if ci in (1, 2):
        return "some cloud, with clear sky between"
    return "little or no cloud"


def _cover_meaning(a):
    lo, mi, hi = a["low"], a["mid"], a["high"]
    if max(lo, mi, hi) < 5:
        return "no cloud layer anywhere in the column"
    parts = []
    if lo >= 50:
        parts.append("a low deck dominates the view")
    elif lo >= 5:
        parts.append("broken low cloud")
    if mi >= 20:
        parts.append("a middle layer" + (" (seen through the gaps)" if lo >= 50 else ""))
    if hi >= 10:
        parts.append("a high ice veil" + (" above it all" if lo + mi > 60 else ""))
    return "; ".join(parts)


def _instability_meaning(a):
    if a["cape"] < 20:
        return "stable: nothing rises by itself, no convective cloud"
    if a["cape"] < 300:
        return "weak buoyancy: fair-weather cumulus at most"
    if a["cape"] < 1000:
        return "moderate buoyancy: towering cumulus and showers once triggered"
    if a["cape"] < 2500:
        return "strong buoyancy: thunderstorms once triggered"
    return "extreme buoyancy: severe thunderstorms once triggered"


def _pressure_meaning(dp):
    if dp <= -3.0:
        return "falling fast: a front or a storm system is closing in"
    if dp <= -1.0:
        return "falling: weather on the way"
    if dp >= 3.0:
        return "rising fast: something has just passed, clearing behind it"
    if dp >= 1.0:
        return "rising: settling down"
    return "steady: no change coming from the large scale"


def _wind_meaning(a):
    u8, v8 = a["w850"]
    spd = math.hypot(u8, v8)
    to = (wind_from(u8, v8) + 180.0) % 360.0
    s = f"low clouds drift toward {compass(to)} at {spd * 3.6:.0f} km/h"
    if a["shear06"] >= 20 and a["cape"] >= 1000:
        s += "; enough shear for rotating (supercell) updraughts"
    elif spd >= 8:
        s += "; strong enough to line boundary-layer thermals up into streets"
    return s


# ------------------------------------------------------------ builders ----
def _coriolis(lat):
    return 2.0 * OMEGA * math.sin(math.radians(lat))


def _build(key, a, snd, size, lat, day, storm, nxt_thunder) -> Regime:
    if key in ("thunderstorm", "storm_coming", "showers"):
        sc = crm.scenario_weather(snd, size, sun_up=day)
        sc.coriolis = _coriolis(lat)
        title = {"thunderstorm": "Thunderstorm", "storm_coming": "Thunderstorm on its way",
                 "showers": "Showers and towering cumulus"}[key]
        sc.name = title + " (from your sounding)"
        why = {"thunderstorm": "the data report thunder here and the sounding has the buoyancy "
                               "to build cumulonimbus: the model grows them from the ground up",
               "storm_coming": "thunder is forecast within 3 h or a storm is heading here; the "
                               "sounding has the buoyancy, so the model builds the storms it will bring",
               "showers": "the sounding has the buoyancy for towering cumulus and showers"}[key]
        sc.reasons = [why]
        sc.shows = [("Cumulonimbus / Cumulus congestus",
                     f"surface-heated thermals rising through CAPE {a['cape']:.0f} J/kg "
                     f"to about {(a['el'] or 0)/1000:.0f} km")]
        spin = 45 * 60.0 if day else 20 * 60.0
        return Regime(key, title, sc, decision=f"Chosen: {title.lower()} - {why}.",
                      spinup_s=spin, evidence=_deep_extra(a))
    if key == "cumulus":
        sc = scenario_cumulus(snd, size, a, lat)
        why = (f"daytime heating with CAPE {a['cape']:.0f} J/kg and condensation at "
               f"{a['zlcl']:.0f} m: thermals rise, cap at {_cu_top(a):.0f} m and make "
               f"cumulus humilis/mediocris")
        return Regime(key, "Fair-weather cumulus", sc, decision="Chosen: fair-weather cumulus - " + why + ".",
                      spinup_s=40 * 60.0)
    if key in ("stratocumulus", "fog", "rain_layer", "altocumulus"):
        layer = _pick_layer(key, a)
        kind = key
        if key == "altocumulus" and _virga_likely(snd, a, layer):
            kind = "virga"
        sc = scenario_layer(snd, layer[0], layer[1], kind, size, lat, a)
        title = {"stratocumulus": "Stratocumulus deck" if layer[0] > 300 else "Stratus deck",
                 "fog": "Fog", "rain_layer": "Nimbostratus / stratus with rain",
                 "altocumulus": "Altocumulus layer", "virga": "Altocumulus with virga"}[kind]
        reg = Regime(kind, title, sc, spinup_s=30 * 60.0)
        reg.facts["layer"] = layer
        reg.decision = (f"Chosen: {title.lower()} - the data put a {layer[2]*100:.0f}% layer "
                        f"between {layer[0]:.0f} and {layer[1]:.0f} m; the model builds it as a "
                        f"well-mixed layer under its inversion, overturned by the cooling of its "
                        f"own top" + (" and snowing into the dry air below" if kind == "virga" else "")
                        + ".")
        return reg
    if key == "cirrus":
        return Regime(key, "High ice cloud only", None,
                      decision=f"Chosen: high cloud only - {a['high']:.0f}% high cover and no low "
                               f"or middle layer; the ice veil is drawn as Atlas layers.")
    return Regime(key, "Clear sky", None,
                  decision="Chosen: clear sky - no cloud layer in the data and "
                           + ("no buoyancy for convection." if a["cape"] < 20
                              else "thermals too dry to reach their condensation level."))


def _deep_extra(a):
    ev = []
    if a["shear06"] >= 20 and a["cape"] >= 1500:
        ev.append(Evidence("organisation", f"0-6 km shear {a['shear06']:.0f} m/s with CAPE "
                           f"{a['cape']:.0f} J/kg", "supercells possible: rotating updraughts, "
                           "wall cloud (murus), beaver tail (flumen)", 1))
    elif a["shear06"] >= 10:
        ev.append(Evidence("organisation", f"0-6 km shear {a['shear06']:.0f} m/s",
                           "multicell lines with gust fronts: arcus (shelf cloud) likely", 1))
    else:
        ev.append(Evidence("organisation", f"0-6 km shear {a['shear06']:.0f} m/s",
                           "single-cell pulse storms that rain into their own updraught"))
    if a["el"] and a["el"] >= a["trop"] - 1500:
        ev.append(Evidence("anvil", f"parcels reach {a['el']/1000:.1f} km, tropopause "
                           f"{a['trop']/1000:.1f} km", "tops flatten into an incus; the "
                           "strongest overshoot it (overshooting top)", 1))
    return ev


def _cu_top(a):
    for (z, dz, dt) in a["inversions"]:
        if z > a["zlcl"]:
            return z
    return (a["el"] or a["zlcl"] + 1000.0)


def _pick_layer(key, a):
    layers = a["layers"]
    if key == "fog":
        top = 150.0
        return (0.0, top, 1.0)
    if key in ("stratocumulus", "rain_layer"):
        low = [L for L in layers if L[0] < 2000.0] or layers
        if low:
            b, t, c = max(low, key=lambda L: L[2])
            if key == "stratocumulus":
                t = min(max(t, b + 200.0), b + 1200.0)
            return (max(b, 60.0), t, c)
        return (max(a["lcl_sfc"], 200.0), max(a["lcl_sfc"], 200.0) + 400.0, a["low"] / 100.0)
    mid = [L for L in layers if 2000.0 <= L[0] < 7000.0] or layers
    b, t, c = max(mid, key=lambda L: L[2])
    return (b, min(max(t, b + 200.0), b + 800.0), c)


def _virga_likely(snd, a, layer):
    """Precipitation from a middle layer evaporates before the ground when
    the air below it is dry (Atlas: virga 'evaporates before reaching the
    ground'); ice makes it fall far enough to see."""
    b = layer[0] + a["elev"]
    t_top = snd.t_at(layer[1] + a["elev"])
    rh_below = sum(snd.rh_at(b - f * 1500.0) for f in (0.2, 0.5, 0.8)) / 3.0
    return t_top < -8.0 and rh_below < 60.0


# ------------------------------------------------------------ the models --
def scenario_cumulus(snd, size, a, lat):
    """Fair-weather cumulus from the sounding: the 'shallow' grid (LES
    spacing), the ground heated by the Sun, the Bowen ratio from how dry
    the air near the ground is (this program's assumption: humid tropics
    0.3 ... desert 2)."""
    grid = crm.make_grid("shallow", size)
    top = grid.top
    th_f, qv_f, u_f, v_f, psfc = crm.sounding_profiles(snd, top)
    base = crm.BaseState.build(grid, th_f, qv_f, u_f, v_f, psfc)
    rh = a["rh2"] / 100.0
    bowen_rh = float(np.clip(2.2 - 2.4 * rh, 0.25, 2.0))
    # the forecast's soil water and snow, where it has them
    bowen, absorb, ground = crm.surface_state(snd, default_bowen=bowen_rh)
    um, vm = u_f(0.5 * top), v_f(0.5 * top)
    sc = crm.Scenario(
        "cumulus", "Fair-weather cumulus (from your sounding)",
        "Thermals from the sun-warmed ground reach their condensation level and "
        "grow into cumulus until the inversion caps them.",
        grid, base, micro_ice=True, random_th=0.3, random_qv=2.0e-5, random_depth=400.0,
        flux_mode="sun", bowen=bowen, sw_absorb=absorb, heterogeneity=0.3, z0=0.1,
        coriolis=_coriolis(lat), translate=(um, vm), sponge_depth=min(800.0, 0.25 * top),
        tile=True)
    sc.reasons = [f"the ground: {ground}" if "Bowen" in ground else
                  f"Bowen ratio {bowen:.2f} from the {a['rh2']:.0f}% surface humidity"
                  + (f"; {ground}" if ground else "")]
    sc.shows = [("Cumulus humilis / mediocris",
                 f"thermals from the heated ground condensing at {a['zlcl']:.0f} m; spacing set "
                 f"by the depth of the mixed layer")]
    return sc


LAYER_FLUXES = {
    # (F0, F1) W/m2: DYCOMS-II's values for a low deck; larger cloud-top
    # cooling for a middle layer under a drier, colder sky; weaker for fog
    # (this program's scaling of the DYCOMS-II parameterisation)
    "stratocumulus": (70.0, 22.0), "rain_layer": (70.0, 22.0),
    "altocumulus": (90.0, 30.0), "virga": (90.0, 30.0), "fog": (60.0, 8.0),
}


def scenario_layer(snd, base, top, kind, size, lat, a):
    """A cloud layer from the sounding: the layer the data report between
    `base` and `top` (m above the ground) is made one well-mixed, cloud-
    topped layer - theta_l and q_t constant, q_t set so that the cloud base
    is where the data put it - capped by the sounding above (at least 1 K
    warmer: 25-50 hPa forecast levels smear the capping inversion), with the
    sounding's own air below and the sounding's winds throughout."""
    thick = max(top - base, 150.0)
    fam = {"stratocumulus": "layer", "rain_layer": "fall", "altocumulus": "layer",
           "virga": "fall", "fog": "fog"}[kind]
    n, dx, nz, dz = FOG_PRESETS[size] if fam == "fog" else crm.GRID_PRESETS[fam][size]
    ground = kind in ("stratocumulus", "fog", "rain_layer") and base < 1500.0
    if ground:
        z_b, z0 = 0.0, 0.0
    else:
        z_b = max(base - 0.35 * thick, base - 150.0 if thick < 400 else base - 0.35 * thick)
        below = 2500.0 if kind == "virga" else 300.0
        z0 = max(0.0, z_b - below)
    need = top + (350.0 if fam != "fog" else 3.0 * thick) - z0
    dz = max(dz, need / nz)
    grid = crm.Grid(n, n, nz, dx, dx, dz, z0)
    th_f, qv_f, u_f, v_f, psfc = crm.sounding_profiles(snd, grid.top)
    th_layer = th_f(0.5 * top) if ground else th_f(z_b)

    def thl(z):
        if z < z_b:
            return th_f(z)
        if z <= top:
            return th_layer
        return max(th_f(z), th_layer + 1.0 + 2.0e-3 * (z - top))

    col = crm._DryColumn(thl, psfc, grid.top + 200.0)
    zc_b = max(base, 5.0)
    q_layer = float(crm.qsat_liq(np.float64(th_layer * float(np.interp(zc_b, col.z, col.pi))),
                                 col.p(zc_b)))

    def qt(z):
        if z_b <= z <= top:
            return q_layer
        return min(qv_f(z), 0.95 * col.qsat(z))

    t_top = th_layer * float(np.interp(top, col.z, col.pi))
    ice = t_top < crm.T0K
    base_state, qc, qi = crm.BaseState.from_liquid(grid, thl, qt, u_f, v_f, psfc, ice=ice)
    f0, f1 = LAYER_FLUXES[kind]
    code = a.get("code")
    drizzle = code is not None and 51 <= int(code) <= 67
    zm = 0.5 * (base + top)
    um, vm = u_f(zm), v_f(zm)
    sponge = min(250.0 if fam != "fall" else 400.0, 0.25 * (grid.top - top) + 100.0)
    rh2 = a.get("rh2", 70.0) / 100.0
    bowen, absorb, _ground = crm.surface_state(
        snd, default_bowen=float(np.clip(2.2 - 2.4 * rh2, 0.25, 2.0)))
    sc = crm.Scenario(
        kind, {"stratocumulus": "Stratocumulus", "rain_layer": "Stratiform rain layer",
               "altocumulus": "Altocumulus", "virga": "Altocumulus with virga",
               "fog": "Fog"}[kind] + " (from your sounding)",
        f"The data's layer at {base:.0f}-{top:.0f} m built as a cloud-topped mixed layer.",
        grid, base_state, micro_ice=ice, graupel=False, random_th=0.1, random_qv=2.5e-5,
        random_zmin=z_b, random_depth=top, init_qc=qc, init_qi=qi,
        flux_mode="sun" if ground else "none", bowen=bowen, sw_absorb=absorb,
        heterogeneity=0.15, drag=ground, z0=0.1,
        coriolis=_coriolis(lat), lw_f0=f0, lw_f1=f1, lw_kappa=85.0,
        q_auto=0.5e-3 if (drizzle or kind == "rain_layer") else crm.Q_AUTO,
        qi_auto=crm.QI_AUTO_LAYER if kind == "virga" else crm.QI_AUTO,
        nudge_tau=900.0, nudge_zmin=max(z_b - 50.0, grid.z0), nudge_zmax=top + 200.0,
        translate=(um, vm), sponge_depth=sponge, tile=True, layer_depth=top - z_b)
    wd = wind_from(um, vm)
    sc.shows = [({"stratocumulus": "Stratocumulus stratiformis" if base > 300 else "Stratus nebulosus",
                  "rain_layer": "Nimbostratus praecipitatio",
                  "altocumulus": "Altocumulus stratiformis",
                  "virga": "Altocumulus stratiformis virga",
                  "fog": "Stratus nebulosus (fog)"}[kind],
                 {"stratocumulus": "the layer under the inversion radiates to space from its "
                                   "top; the chilled air sinks in sheets and the layer overturns "
                                   "in cells about twice as wide as it is deep",
                  "rain_layer": "a deep saturated layer whose drops grow big enough to fall "
                                "out as steady rain",
                  "altocumulus": "a thin moist layer at middle height, overturned in small "
                                 "cells by the long-wave cooling of its own top",
                  "virga": "a mixed-phase layer: ice grows at the droplets' expense, falls as "
                           "snow and sublimates in the dry air below",
                  "fog": "air at the ground cooled to its dew point; the fog's top radiates to "
                         "space and deepens it"}[kind])]
    sc.reasons = [f"layer {base:.0f}-{top:.0f} m, theta_l {th_layer:.1f} K, q_t {q_layer*1e3:.2f} g/kg "
                  f"(saturated from {base:.0f} m), cloud-top {t_top - crm.T0K:.0f} C "
                  f"({'ice and water' if ice else 'water'})",
                  f"long-wave cooling at its top {f0:.0f} W/m2, warming at its base {f1:.0f} W/m2",
                  "the layer's MEAN temperature and moisture are held to the forecast's (15-min "
                  "relaxation); its cells, gaps and bands are the model's own",
                  f"wind {math.hypot(um, vm):.0f} m/s from {compass(wd)} carries it toward "
                  f"{compass(wd + 180.0)}"]
    return sc


# fog: 20 m horizontal, 6-8 m vertical spacing (this program's choice)
FOG_PRESETS = {"fast": (64, 25.0, 40, 8.0), "standard": (128, 20.0, 48, 6.0),
               "fine": (256, 12.5, 64, 5.0)}
