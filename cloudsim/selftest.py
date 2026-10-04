# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
Self-checks.  Run with:  python cloud_sim.py --selftest

Every check is a rule with a name and a count of the individual assertions it
made, so a failure says which rule broke and how badly.  The rules cover the
Atlas taxonomy, the one numeric criterion the Atlas actually states (element
apparent width), the thermodynamics, the astronomy, and the geometry the
renderer depends on.  Nothing here needs a graphics card except the last
group, which is skipped if no OpenGL context can be made.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from . import astro, atlas, atmosphere, noise, scene, sounding
from .atlas import CloudSpec

RESULTS = []


class Rule:
    def __init__(self, name):
        self.name = name
        self.n = 0
        self.fails = []

    def check(self, ok, detail=""):
        self.n += 1
        if not ok:
            self.fails.append(detail or f"assertion {self.n}")
        return ok

    def close(self):
        RESULTS.append((self.name, self.n, self.fails))


def near(a, b, tol):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- taxonomy --

def test_taxonomy():
    r = Rule("A1 Atlas inventory is complete")
    for name, table, n in (("genera", atlas.GENERA, 10), ("species", atlas.SPECIES, 15),
                           ("varieties", atlas.VARIETIES, 9),
                           ("supplementary features", atlas.SUPPLEMENTARY, 11),
                           ("accessory clouds", atlas.ACCESSORY, 4),
                           ("special clouds", atlas.SPECIAL, 5)):
        r.check(len(table) == n, f"{name}: {len(table)} not {n}")
    r.check(len(atlas.MOTHER["genitus"]) == 11, "11 genitus forms")
    r.check(len(atlas.MOTHER["mutatus"]) == 10, "10 mutatus forms")
    r.close()

    r = Rule("A2 genus tables invert exactly")
    for table, key in ((atlas.SPECIES, "species"), (atlas.VARIETIES, "varieties"),
                       (atlas.SUPPLEMENTARY, "supplementary"),
                       (atlas.ACCESSORY, "accessory")):
        for abbr, item in table.items():
            for g in item["genera"]:
                r.check(abbr in atlas.GENERA[g][key],
                        f"{abbr} lists {g} but {g}'s {key} does not list {abbr}")
        for g in atlas.GENUS_ORDER:
            for abbr in atlas.GENERA[g][key]:
                r.check(g in table[abbr]["genera"],
                        f"{g}'s {key} lists {abbr} but {abbr} does not list {g}")
    r.close()

    r = Rule("A3 stated combination rules are enforced")
    for g in atlas.GENUS_ORDER:
        if "tr" in atlas.GENERA[g]["varieties"] and "op" in atlas.GENERA[g]["varieties"]:
            s = CloudSpec(g, None, ["tr", "op"])
            r.check(bool(s.validate()[0]), f"{g} translucidus+opacus should be rejected")
            r.check(not CloudSpec(g, None, ["tr"]).validate()[0], f"{g} translucidus alone")
            r.check(not CloudSpec(g, None, ["op"]).validate()[0], f"{g} opacus alone")
        if "pe" in atlas.GENERA[g]["varieties"] and "tr" in atlas.GENERA[g]["varieties"]:
            r.check(not CloudSpec(g, None, ["pe", "tr"]).validate()[0],
                    f"{g} perlucidus with translucidus is permitted")
    r.check(bool(CloudSpec("Cu", "hum", [], [], [], ("Ac", "genitus"), "flgen")
                 .validate()[0]), "special and mother together must be rejected")
    r.check(bool(CloudSpec("Ci", None, ["un", "un"]).validate()[0]),
            "a repeated variety must be rejected")
    r.close()

    r = Rule("A4 names assemble and parse back")
    samples = [CloudSpec("Ac", "len", ["un", "du"]),
               CloudSpec("Cb", "cap", [], ["inc", "mam", "pra"], ["pan"]),
               CloudSpec("Ci", "unc", ["ra"]),
               CloudSpec("Sc", "str", ["pe", "op"], ["vir"]),
               CloudSpec("St", "fra", ["op"], [], [], ("Ns", "genitus"))]
    for s in samples:
        txt = s.latin()
        r.check(txt.startswith(atlas.GENERA[s.genus]["name"]), f"{txt} starts with genus")
        back = atlas.parse(s.abbrev().split(" " + s.abbrev().split()[-1])[0]
                           if s.mother else s.abbrev())
        r.check(back.genus == s.genus, f"{s.abbrev()} round trip genus")
        r.check(back.species == s.species, f"{s.abbrev()} round trip species")
        r.check(sorted(back.varieties) == sorted(s.varieties),
                f"{s.abbrev()} round trip varieties")
    r.close()

    r = Rule("A5 mother-cloud graph is well formed")
    for t in atlas.TRANSITIONS:
        r.check(t["from"] in atlas.GENERA and t["to"] in atlas.GENERA,
                f"{t} names real genera")
        r.check(t["form"] in ("genitus", "mutatus"), f"{t} form")
    r.check(len(atlas.TRANSITIONS) >= 40,
            f"only {len(atlas.TRANSITIONS)} transitions")
    r.close()

    r = Rule("A6 every tabulated combination builds a deck")
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    n = 0
    for g in atlas.GENUS_ORDER:
        o = atlas.options_for(g)
        for sp in [None] + o["species"]:
            for v in [[]] + [[x] for x in o["varieties"]]:
                for f in [[]] + [[x] for x in o["supplementary"]]:
                    s = CloudSpec(g, sp, list(v), list(f))
                    if s.validate()[0]:
                        continue
                    d = scene.deck_from_spec(s, snd)
                    n += 1
                    r.check(d.top_m > d.base_m and d.sigma_e > 0 and d.element_m > 0,
                            f"{s.abbrev()} produced a degenerate deck")
    r.check(n > 400, f"only {n} combinations exercised")
    r.close()


# ------------------------------------------------------- element geometry ---

def test_elements():
    r = Rule("B1 element apparent width obeys the Atlas bands, from 30 degrees up to overhead")
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    for g in ("Cc", "Ac", "Sc"):
        lo, hi = atlas.ELEMENT_WIDTH_DEG[g]
        blo, bhi = atlas.etage_range_m(g, 3.14)
        for base in np.linspace(max(blo, 300), bhi, 7):
            d = scene.deck_from_spec(CloudSpec(g, "str" if g != "Ci" else None),
                                     snd, base_m=float(base))
            w = d.element_deg
            ok = (w >= lo * 0.99) and (w <= hi * 1.01 or hi >= 11.0)
            r.check(ok, f"{g} at {base:.0f} m: {w:.2f}° outside {lo}-{hi}°")
            # the same element overhead, at half the distance (it was 6.6
            # degrees for a Paris altocumulus sized at 30 degrees alone)
            h = max(d.base_m - sounding.station_elevation(snd), 150.0)
            w90 = atlas.metres_to_angular_width(d.element_w_m, h, 90.0)
            r.check(w90 <= hi * 1.01 or hi >= 11.0,
                    f"{g} at {base:.0f} m: {w90:.2f}° overhead, over {hi}°")
            r.check(abs(atlas.metres_to_angular_width(d.element_w_m, h, 30.0) - w) < 1e-6,
                    f"{g} at {base:.0f} m: element_deg is not the width seen 30° up")
    r.close()

    r = Rule("B2 generated maps have the spacing they were asked for")
    for elem, extent in ((150.0, 6000.0), (400.0, 12000.0), (900.0, 24000.0),
                         (1600.0, 40000.0)):
        m = noise.deck_map(512, extent, elem, 0.6, cellularity=0.9, warp=0.0, seed=5)
        meas = noise.measured_element_spacing_m(m[..., 0], extent)
        r.check(abs(meas - elem) / elem < 0.12,
                f"asked {elem:.0f} m, measured {meas:.0f} m")
    r.close()

    r = Rule("C1 cloud cover is exactly what was requested")
    for cov in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        m = noise.deck_map(384, 12000.0, 500.0, cov, cellularity=0.7, warp=0.0, seed=9)
        got = float((m[..., 0] > 0.5).mean())
        r.check(abs(got - cov) < 0.02, f"asked {cov:.3f}, got {got:.3f}")
    r.close()

    r = Rule("C2 optical depth is in the observed range for each genus")
    # published typical ranges of visible optical depth
    expect = {"Ci": (0.05, 4.0), "Cc": (0.1, 4.0), "Cs": (0.2, 6.0),
              "Ac": (2.0, 60.0), "As": (2.0, 60.0), "Ns": (20.0, 400.0),
              "Sc": (5.0, 90.0), "St": (4.0, 60.0), "Cu": (5.0, 200.0),
              "Cb": (100.0, 5000.0)}
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    for g, (lo, hi) in expect.items():
        d = scene.deck_from_spec(CloudSpec(g), snd)
        r.check(lo <= d.optical_depth <= hi,
                f"{g}: optical depth {d.optical_depth:.1f} outside {lo}-{hi}")
    r.close()

    r = Rule("C3 opacus is thicker than translucidus")
    for g in ("Ac", "As", "Sc", "St"):
        a = scene.deck_from_spec(CloudSpec(g, None, ["tr"]), snd)
        b = scene.deck_from_spec(CloudSpec(g, None, ["op"]), snd)
        r.check(b.optical_depth > a.optical_depth * 1.5,
                f"{g}: opacus {b.optical_depth:.1f} vs translucidus {a.optical_depth:.1f}")
    r.close()


# -------------------------------------------------------- thermodynamics ----

def test_thermo():
    r = Rule("D1 saturation vapour pressure matches the tables")
    for t, e in ((0.0, 6.11), (10.0, 12.28), (20.0, 23.39), (30.0, 42.47),
                 (-10.0, 2.86)):
        got = sounding.esat_hpa(t)
        r.check(abs(got - e) / e < 0.01, f"esat({t}) = {got:.3f}, table {e}")
    r.close()

    r = Rule("D2 dewpoint inverts relative humidity")
    for t in (-20, 0, 15, 30):
        for rh in (20, 50, 80, 99):
            td = sounding.dewpoint_from_rh(t, rh)
            back = 100.0 * sounding.esat_hpa(td) / sounding.esat_hpa(t)
            r.check(abs(back - rh) < 0.5, f"T={t} RH={rh} -> {back:.2f}")
    r.close()

    r = Rule("D3 LCL agrees with the 125 m per K rule")
    for t, td in ((30, 24), (20, 10), (10, 9), (35, 15)):
        _, _, z = sounding.lcl_bolton(t, td, 1000.0)
        rule = 125.0 * (t - td)
        r.check(abs(z - rule) < max(120.0, 0.18 * rule),
                f"T={t} Td={td}: Bolton {z:.0f} m, rule of thumb {rule:.0f} m")
    r.close()

    r = Rule("D4 moist adiabatic lapse rate is physical")
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    for z, lo, hi in ((0.0, 3.5, 5.5), (5000.0, 4.5, 7.5), (9000.0, 6.0, 9.5)):
        g = snd.moist_adiabatic_lapse(z)
        r.check(lo <= g <= hi, f"{z:.0f} m: {g:.2f} K/km outside {lo}-{hi}")
    r.close()

    r = Rule("D5 a neutral sounding has no CAPE, an unstable one has some")
    dry = sounding.standard_sounding(3.14, 101.69, profile="dry")
    cape_dry = dry.parcel(mixed_depth=400)[0]
    r.check(cape_dry < 50.0, f"dry profile CAPE {cape_dry:.0f} should be ~0")
    wet = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    cape_wet = wet.parcel(mixed_depth=400)[0]
    r.check(500.0 < cape_wet < 8000.0, f"humid profile CAPE {cape_wet:.0f} implausible")
    cape2, _, zl, lfc, el, _ = wet.parcel(mixed_depth=400)
    r.check(lfc is not None and el is not None and el > lfc > zl,
            f"LCL {zl:.0f} < LFC {lfc} < EL {el} ordering")
    r.close()

    r = Rule("D6 freezing level and tropopause are where the profile says")
    for prof in ("tropical-fair", "humid-deep", "stable-stratus", "dry"):
        s = sounding.standard_sounding(3.14, 101.69, profile=prof)
        fz = s.freezing_level()
        r.check(fz is not None and abs(s.t_at(fz)) < 0.3,
                f"{prof}: T at freezing level is {s.t_at(fz):.2f} C")
        tp = s.tropopause()
        r.check(5000 < tp < 20000, f"{prof}: tropopause {tp:.0f} m")
        r.check(s.lapse_rate(tp, tp + 2000) < 2.01,
                f"{prof}: lapse above tropopause {s.lapse_rate(tp, tp+2000):.2f} K/km")
    r.close()

    r = Rule("D7 wind, shear and Richardson number are consistent")
    s = sounding.standard_sounding(48.0, 2.0, profile="stable-stratus",
                                   wind_speed=5.0, jet_speed=40.0)
    for z in (500, 2000, 6000, 10000):
        u, v = s.wind_at(z)
        du, dv = s.shear_at(z)
        u2, v2 = s.wind_at(z + 500)
        r.check(near(du * 500, u2 - u, 0.6), f"shear u at {z} m")
        r.check(near(dv * 500, v2 - v, 0.6), f"shear v at {z} m")
        n = s.brunt_vaisala(z)
        ri = s.richardson(z)
        sh = math.hypot(du, dv)
        if sh > 1e-6 and n > 0:
            r.check(near(ri, (n * n) / (sh * sh), abs(ri) * 0.02 + 1e-6),
                    f"Ri definition at {z} m")
    r.close()

    r = Rule("D8 the diagnosis puts each genus in its own étage")
    for prof in ("tropical-fair", "humid-deep", "stable-stratus", "dry"):
        s = sounding.standard_sounding(3.14, 101.69, profile=prof)
        for d in scene.diagnose(s):
            g = d.spec.genus
            lo, hi = atlas.etage_range_m(g, s.lat)
            if g in ("Cu", "Cb"):
                r.check(d.base_m <= hi + 200, f"{prof}: {g} base {d.base_m:.0f} m")
            elif g == "Ns":
                r.check(True)
            else:
                r.check(lo - 200 <= d.base_m <= hi + 200,
                        f"{prof}: {g} base {d.base_m:.0f} m outside {lo}-{hi} m")
            e, w = d.spec.validate()
            r.check(not e, f"{prof}: {d.spec.abbrev()} invalid: {e}")
    r.close()


# ------------------------------------------------------------- astronomy ----

def test_astro():
    r = Rule("E1 solar position against an independent ephemeris")
    # fixtures produced from pyephem (apparent topocentric, no refraction)
    fixtures = [
        # (utc, lat, lon, az, alt)
        ("2026-06-21T12:00:00", 51.4778, -0.0015, 179.11, 61.96),
        ("2026-12-21T12:00:00", 51.4778, -0.0015, 180.46, 15.08),
        ("2026-03-20T04:00:00", 3.1390, 101.6869, 98.97, 69.54),
        ("2026-09-10T06:00:00", -33.8688, 151.2093, 290.70, 19.96),
        ("2026-01-01T00:00:00", 40.7128, -74.0060, 261.08, -25.95),
    ]
    for iso, lat, lon, az, alt in fixtures:
        t = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
        jd = astro.julian_day(t)
        ra, dec, _ = astro.sun_equatorial(jd)
        a, e = astro.equatorial_to_horizontal(ra, dec, jd, lat, lon)
        r.check(abs(((a - az + 180) % 360) - 180) * max(math.cos(math.radians(e)), 0.05)
                < 0.05, f"{iso} azimuth {a:.2f} vs {az}")
        r.check(abs(e - alt) < 0.05, f"{iso} altitude {e:.2f} vs {alt}")
    r.close()

    r = Rule("E2 the Sun crosses the meridian at the right altitude")
    for lat in (-60, -23.44, 0, 23.44, 51.5):
        for iso in ("2026-06-21", "2026-12-21", "2026-03-20"):
            best, bestalt, bestdec = None, -999, 0
            for m in range(0, 1440, 2):
                t = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc) \
                    + timedelta(minutes=m)
                jd = astro.julian_day(t)
                ra, dec, _ = astro.sun_equatorial(jd)
                a, e = astro.equatorial_to_horizontal(ra, dec, jd, lat, 0.0)
                if e > bestalt:
                    bestalt, bestdec, best = e, dec, t
            expect = 90.0 - abs(lat - bestdec)
            r.check(abs(bestalt - expect) < 0.25,
                    f"lat {lat} {iso}: noon altitude {bestalt:.2f} vs {expect:.2f}")
    r.close()

    r = Rule("E3 lunar phase at known syzygies")
    for iso, frac, tol in (("2026-03-03T11:37", 1.0000, 0.005),
                           ("2026-01-03T10:02", 0.9986, 0.005),
                           ("2026-02-01T22:09", 0.9996, 0.005),
                           ("2026-01-18T19:51", 0.0009, 0.005),
                           ("2026-02-17T12:01", 0.0001, 0.005)):
        jd = astro.julian_day(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc))
        pa, illum = astro.moon_phase(jd)
        r.check(abs(illum - frac) < tol, f"{iso}: illuminated {illum:.3f} vs {frac}")
    r.close()

    r = Rule("E6 the Moon's position seen from the ground (topocentric) against an independent ephemeris")
    # fixtures from pyephem 4.2 (ELP2000; apparent topocentric, no refraction)
    for iso, lat, lon, az, alt in (("2026-10-04T12:00:00", 3.139, 101.687, 357.679, -62.956),
                                   ("2026-10-04T22:00:00", 48.857, 2.352, 40.134, -9.417),
                                   ("2027-03-15T06:30:00", -33.869, 151.209, 16.210, 25.802),
                                   ("2027-07-01T03:00:00", 64.147, -21.942, 59.844, 11.972),
                                   ("2028-01-20T18:00:00", 41.878, -87.63, 240.750, -4.005),
                                   ("2026-12-24T00:00:00", 35.69, 139.692, 324.070, -17.859)):
        jd = astro.julian_day(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc))
        mra, mdec, md = astro.moon_equatorial(jd)
        a, e = astro.equatorial_to_horizontal(mra, mdec, jd, lat, lon)
        e += astro.parallax_alt_correction_deg(e, md)
        c = (math.sin(math.radians(e)) * math.sin(math.radians(alt)) + math.cos(math.radians(e))
             * math.cos(math.radians(alt)) * math.cos(math.radians(a - az)))
        sep = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        r.check(sep < 0.10, f"{iso} at {lat},{lon}: {sep:.3f} deg from the ephemeris")
    r.close()

    r = Rule("E7 sunrise, sunset, moonrise and moonset (upper limb on the horizon, standard "
             "refraction) against an independent ephemeris: the Sun within 30 s, the Moon 2 min")

    def upper_limb(t, lat, lon, body):
        g = astro.sky_geometry(t, lat, lon)
        return (g.sun_alt_refracted + g.sun_ang_radius_deg) if body == "sun" \
            else (g.moon_alt + g.moon_ang_radius_deg)

    def event_near(t_ref, lat, lon, body, kind):
        best = None
        t = t_ref - timedelta(hours=3)
        f0 = upper_limb(t, lat, lon, body)
        while t < t_ref + timedelta(hours=3):
            t2 = t + timedelta(minutes=5)
            f2 = upper_limb(t2, lat, lon, body)
            if (f0 < 0) != (f2 < 0) and (f2 > 0) == (kind == "rise"):
                lo, hi, flo = t, t2, f0
                for _ in range(24):
                    mid = lo + (hi - lo) / 2
                    fm = upper_limb(mid, lat, lon, body)
                    if (fm < 0) == (flo < 0):
                        lo, flo = mid, fm
                    else:
                        hi = mid
                if best is None or abs((lo - t_ref).total_seconds()) < abs((best - t_ref).total_seconds()):
                    best = lo
            t, f0 = t2, f2
        return best
    for name, lat, lon, body, kind, iso in (
            ("Kuala Lumpur", 3.139, 101.687, "sun", "rise", "2026-10-04T22:59:16"),
            ("Kuala Lumpur", 3.139, 101.687, "sun", "set", "2026-10-04T11:04:33"),
            ("Kuala Lumpur", 3.139, 101.687, "moon", "rise", "2026-10-04T18:14:04"),
            ("Kuala Lumpur", 3.139, 101.687, "moon", "set", "2026-10-04T05:55:56"),
            ("Paris", 48.857, 2.352, "sun", "rise", "2026-10-04T05:54:00"),
            ("Paris", 48.857, 2.352, "sun", "set", "2026-10-04T17:23:45"),
            ("Paris", 48.857, 2.352, "moon", "rise", "2026-10-04T23:15:00"),
            ("Paris", 48.857, 2.352, "moon", "set", "2026-10-04T14:49:53"),
            ("Reykjavik", 64.147, -21.942, "sun", "rise", "2026-06-21T02:53:47"),
            ("Reykjavik", 64.147, -21.942, "sun", "set", "2026-06-21T00:05:10"),
            ("Reykjavik", 64.147, -21.942, "moon", "rise", "2026-06-21T13:13:04"),
            ("Reykjavik", 64.147, -21.942, "moon", "set", "2026-06-21T01:11:47"),
            ("Sydney", -33.869, 151.209, "sun", "rise", "2026-12-21T18:40:47"),
            ("Sydney", -33.869, 151.209, "sun", "set", "2026-12-21T09:05:44"),
            ("Sydney", -33.869, 151.209, "moon", "rise", "2026-12-21T06:05:31"),
            ("Sydney", -33.869, 151.209, "moon", "set", "2026-12-21T16:13:20")):
        t_ref = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
        got = event_near(t_ref, lat, lon, body, kind)
        tol = 30.0 if body == "sun" else 120.0
        r.check(got is not None and abs((got - t_ref).total_seconds()) <= tol,
                f"{name} {body}{kind}: {got} vs {t_ref} (ephemeris)")
    r.close()

    r = Rule("E8 the illuminated fraction of the Moon at its quarters against an independent ephemeris")
    for iso, frac in (("2026-01-10T15:48", 0.5011), ("2026-01-26T04:47", 0.5027),
                      ("2026-02-09T12:43", 0.5015), ("2026-02-24T12:27", 0.5020),
                      ("2026-03-11T09:38", 0.5017)):
        jd = astro.julian_day(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc))
        illum = astro.moon_phase(jd)[1]
        r.check(abs(illum - frac) < 0.005, f"{iso}: illuminated {illum:.4f} vs {frac}")
    r.close()

    r = Rule("E4 sidereal time advances by 360.9856 degrees a day")
    jd0 = astro.julian_day(datetime(2026, 5, 1, tzinfo=timezone.utc))
    for k in range(1, 6):
        d = (astro.gmst_deg(jd0 + k) - astro.gmst_deg(jd0)) % 360.0
        want = (360.98564736629 * k) % 360.0
        r.check(min(abs(d - want), 360 - abs(d - want)) < 0.01,
                f"day {k}: {d:.4f} vs {want:.4f}")
    r.close()

    r = Rule("E5 horizon and equatorial frames are consistent")
    try:
        from .gl_sky import SkyRenderer
    except Exception as e:                                       # noqa: BLE001
        RESULTS.append((r.name, 0, [f"skipped, moderngl not installed ({type(e).__name__})"]))
        return
    for lat in (-70, -20, 0, 20, 70):
        for lst in (0, 90, 200, 300):
            class G:
                pass
            g = G()
            g.lat, g.lst = lat, lst
            m = SkyRenderer._hor_to_equ(g)
            r.check(np.allclose(m @ m.T, np.eye(3), atol=1e-9),
                    f"lat {lat} lst {lst}: not orthonormal")
            zen = m[1]                       # the "up" basis vector
            dec = math.degrees(math.asin(max(-1, min(1, zen[2]))))
            r.check(abs(dec - lat) < 1e-6,
                    f"lat {lat}: zenith declination {dec:.6f}")
            ra = math.degrees(math.atan2(zen[1], zen[0])) % 360.0
            r.check(abs(((ra - lst + 180) % 360) - 180) < 1e-6,
                    f"lst {lst}: zenith right ascension {ra:.6f}")
    r.close()


# -------------------------------------------------------------- geometry ----

def test_geometry():
    r = Rule("F1 camera basis stays orthonormal and right handed")
    try:
        from .gl_sky import Camera
    except Exception as e:                                       # noqa: BLE001
        for name in ("F1 camera basis stays orthonormal and right handed",
                     "F2 deck parameters pack and unpack",
                     "F3 the wind actually moves the decks"):
            RESULTS.append((name, 0,
                            [f"skipped, moderngl not installed ({type(e).__name__})"]))
        return
    for az in range(0, 360, 23):
        for alt in (-88, -45, -5, 0, 5, 45, 88):
            c = Camera(az, alt, 60.0)
            right, up, fwd = c.basis()
            for a, b in ((right, up), (up, fwd), (fwd, right)):
                r.check(abs(float(np.dot(a, b))) < 1e-5, f"az {az} alt {alt}: not orthogonal")
            for v in (right, up, fwd):
                r.check(abs(float(np.linalg.norm(v)) - 1.0) < 1e-5, "not unit length")
            r.check(float(np.dot(np.cross(right, up), fwd)) > 0.99,
                    f"az {az} alt {alt}: left handed")
            # looking north means +z, looking east means +x
            if alt == 0:
                want = (math.sin(math.radians(az)), 0.0, math.cos(math.radians(az)))
                r.check(np.allclose(fwd, want, atol=1e-6), f"az {az} forward vector")
    r.close()

    r = Rule("F2 deck parameters pack and unpack")
    from . import gl_sky        # noqa: F401  (import proved above)
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    decks = [scene.deck_from_spec(CloudSpec(g), snd) for g in ("Ci", "Ac", "Cu", "Cb")]
    p = gl_sky.pack_deck_params(decks, 0.0)
    for i, d in enumerate(decks):
        r.check(near(float(p[i, 0, 0]), d.base_m, 1.0), "base")
        r.check(near(float(p[i, 0, 1]), d.top_m, 1.0), "top")
        # a layer of rounded masses: raised so an element's middle has the
        # deck's optical depth (gl_sky.DOME_COLUMN)
        k = 1.0 + d.cell_dome * (gl_sky.DOME_COLUMN - 1.0)
        r.check(near(float(p[i, 0, 2]), d.sigma_e * k, 1e-6 * k), "extinction")
        r.check(near(float(p[i, 16, 3]), d.coverage, 1e-6), "cover (row 16.w)")
        r.check(near(float(p[i, 2, 0]), gl_sky.deck_map_extent(d), 1.0), "map extent")
        r.check(float(p[i, 2, 3]) == (1.0 if d.ice else 0.0), "ice flag")
    r.check(np.isfinite(p).all(), "parameters contain no NaN")
    r.close()

    r = Rule("F3 the wind actually moves the decks")
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    for g in ("Ci", "Ac", "Sc", "Cu"):
        d = scene.deck_from_spec(CloudSpec(g), snd)
        u, v = snd.wind_at((d.base_m + d.top_m) * 0.5)
        r.check(near(d.wind_u, u, 1e-6) and near(d.wind_v, v, 1e-6),
                f"{g}: deck wind does not match the sounding at its height")
        speed = math.hypot(d.wind_u, d.wind_v)
        r.check(speed > 0.1, f"{g}: deck is not moving at all")
        ex = 24 * 3600.0 * speed
        r.check(ex > 1000.0, f"{g}: travels only {ex:.0f} m in a day")
    r.close()



# ----------------------------------------------------- the cloud model -----

def _small_storm_scenario(n=24, nz=24, lateral=False, ice=True, bubble=True):
    from . import crm
    g = crm.Grid(n, n, nz, 1000.0, 1000.0, 600.0)
    base = crm.wk82_base(g)
    bubbles = [(n * 500.0, n * 500.0, 1500.0, 6000.0, 1500.0, 3.0)] if bubble else []
    return crm.Scenario("t", "test", "", g, base, micro_ice=ice, random_th=0.0,
                        bubbles=bubbles, flux_mode="none", drag=False,
                        lateral_sponge=lateral, tile=True)


def test_model():
    from . import crm
    r = Rule("H1 reference state is CM1's Weisman-Klemp sounding")
    g = crm.Grid(8, 8, 200, 1000.0, 1000.0, 100.0)
    b = crm.wk82_base(g)
    # CM1 algorithm values at 50 m steps from the research brief (base.F port)
    for z, th, p_hpa, qv in ((1050.0, 302.05, 886.7, 14.00), (4050.0, 311.06, 616.9, 4.28),
                             (8050.0, 326.11, 362.9, 0.53), (12050.0, 343.79, 200.5, 0.02)):
        k = int(round(z / 100.0 - 0.5))
        r.check(near(float(b.th0[k]), th, 0.15), f"theta at {z:.0f} m: {b.th0[k]:.2f} not {th}")
        r.check(near(float(b.p0[k]) / 100.0, p_hpa, 1.5), f"p at {z:.0f} m: {b.p0[k]/100:.1f} hPa not {p_hpa}")
        r.check(near(float(b.qv0[k]) * 1000.0, qv, max(0.08, 0.03 * qv)),
                f"qv at {z:.0f} m: {b.qv0[k]*1000:.2f} g/kg not {qv}")
    # hydrostatic: d(pi)/dz = -g / (cp thv) between levels
    thv = b.th0 * (1 + b.qv0 / crm.EPS) / (1 + b.qv0)
    dpi = np.diff(b.pi0) / 100.0
    want = -crm.G / (crm.CP * 0.5 * (thv[1:] + thv[:-1]))
    r.check(np.allclose(dpi, want, rtol=1e-10), "Exner function not hydrostatic")
    cape, cin, zl, ze = b.cape()
    r.check(1700.0 < cape < 2150.0, f"surface-parcel CAPE {cape:.0f} J/kg (CM1: ~1877)")
    r.check(900.0 < zl < 1150.0, f"LCL {zl:.0f} m (CM1: ~1008 m)")
    r.close()

    r = Rule("H2 the pressure solve makes the flow exactly non-divergent")
    sc = _small_storm_scenario(bubble=False)
    m = crm.CPUModel(sc)
    rng = np.random.default_rng(4)
    for trial in range(3):
        u = rng.standard_normal(m.u.shape)
        v = rng.standard_normal(m.v.shape)
        w = rng.standard_normal(m.w.shape)
        w[0] = w[-1] = 0.0
        before = np.abs(m._divergence(u, v, w)).max()
        m.project(u, v, w, 1.0)
        after = np.abs(m._divergence(u, v, w)).max()
        r.check(after < 1e-12 * max(before, 1.0), f"max |div| {after:.2e} after projection")
    r.close()

    r = Rule("H3 an atmosphere at rest stays at rest")
    sc = _small_storm_scenario(bubble=False)
    m = crm.CPUModel(sc)
    for _ in range(6):
        m.step(10.0)
    r.check(float(np.abs(m.w).max()) < 1e-12, f"w grew to {np.abs(m.w).max():.2e} m/s")
    r.check(float(np.abs(m.th).max()) < 1e-10, f"theta' grew to {np.abs(m.th).max():.2e} K")
    r.check(float(np.abs(m.qc).max()) == 0.0, "cloud appeared from nothing")
    r.close()

    r = Rule("H4 fifth-order upwind fluxes")
    one = np.ones(8)
    for vel in (-3.0, 0.5, 7.0):
        r.check(near(float(crm._flux5(*(one * 2.5)[:6], vel)), 2.5 * vel, 1e-12),
                "flux of a constant is velocity x constant")
    # a step: upwind value (2*1 - 13*1 + 47*1 + 27*0 - 3*0)/60 = 0.6
    r.check(near(float(crm._flux5(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)), 0.6, 1e-12),
            "5th-order upwind interpolation of a step")
    r.check(near(float(crm._flux3(1.0, 1.0, 0.0, 0.0, 1.0)), 4.0 / 6.0, 1e-12),
            "3rd-order upwind interpolation of a step")
    # Finite-volume exactness: from the cell averages of a polynomial of
    # degree <= 4 the upwind reconstruction returns its exact value at the
    # face, whichever way the wind blows.
    centres = np.arange(-2.5, 3.0, 1.0)            # cells m-3 .. m+2, face at 0
    for deg in range(5):
        avg = ((centres + 0.5) ** (deg + 1) - (centres - 0.5) ** (deg + 1)) / (deg + 1)
        face = 1.0 if deg == 0 else 0.0
        for vel in (1.0, -1.0):
            f = float(crm._flux5(*avg, vel))
            r.check(near(f, vel * face, 1e-12), f"degree {deg}, wind {vel:+.0f}: {f:.3e}")
    r.close()

    r = Rule("H5 rising air loses th' at w dth0/dz")
    g = crm.Grid(8, 8, 40, 1000.0, 1000.0, 250.0)
    base = crm.BaseState.build(g, lambda z: 300.0 + 0.004 * z + 2e-7 * z * z,
                               lambda z: 0.0, lambda z: 0.0, lambda z: 0.0, 100000.0)
    sc = crm.Scenario("s", "s", "", g, base, micro_ice=False, random_th=0.0,
                      flux_mode="none", drag=False)
    m = crm.CPUModel(sc)
    st = {k: v.copy() for k, v in m.state().items()}
    st["u"][:] = 0.0
    st["v"][:] = 0.0
    st["w"][:] = 1.0
    st["w"][0] = st["w"][-1] = 0.0
    m.damp_c[:] = 0.0
    m.damp_f[:] = 0.0
    t = m.tendencies(st)["th"][:, 0, 0]
    exact = -(0.004 + 4e-7 * m.g.zc)
    for k in range(3, m.g.nz - 3):
        r.check(near(float(t[k]), float(exact[k]), 0.01 * abs(float(exact[k]))),
                f"level {k}: {t[k]:.6f} vs {exact[k]:.6f} K/s")
    r.close()

    r = Rule("H6 total water is conserved to round-off")
    sc = _small_storm_scenario()
    m = crm.CPUModel(sc)
    rho = m.b.rho0.reshape(-1, 1, 1)

    def total():
        q = m.qv + m.qc + m.qr + m.qi + m.qs
        return float((rho * q).sum() * m.g.dx * m.g.dy * m.g.dz + m.rain.sum() * m.g.dx * m.g.dy)
    w0 = total()
    for _ in range(40):
        m.step(m.max_dt(0.9))
    w1 = total()
    r.check(abs(w1 / w0 - 1.0) < 1e-12, f"relative change {w1 / w0 - 1.0:.2e}")
    r.check(float(m.qc.max()) > 1e-4, "the bubble made no cloud")
    r.check(float(m.w.max()) > 5.0, f"the bubble did not rise (w {m.w.max():.1f} m/s)")
    r.close()

    r = Rule("H7 saturation adjustment conserves energy and saturates exactly")
    sc = _small_storm_scenario(bubble=False)
    m = crm.CPUModel(sc)
    rng = np.random.default_rng(1)
    m.qv = m.qv * (1 + 0.3 * rng.random(m.qv.shape))
    m.qc = 1e-3 * rng.random(m.qc.shape)
    m.qi = 1e-3 * rng.random(m.qc.shape) * (m.g.zc[:, None, None] > 5000)
    pi0 = m.b.pi0.reshape(-1, 1, 1)
    th0 = m.b.th0.reshape(-1, 1, 1)
    p = m.b.p0.reshape(-1, 1, 1)
    t1 = (th0 + m.th) * pi0
    lv = crm.latent_vap(t1)
    lf = crm.latent_sub(t1) - lv
    h1 = crm.CP * t1 - lv * m.qc - (lv + lf) * m.qi
    q1 = m.qv + m.qc + m.qi
    m._saturation_adjust()
    t2 = (th0 + m.th) * pi0
    h2 = crm.CP * t2 - lv * m.qc - (lv + lf) * m.qi
    fl = crm.liquid_fraction(t2)
    qsm = fl * crm.qsat_liq(t2, p) + (1 - fl) * crm.qsat_ice(t2, p)
    cl = (m.qc + m.qi) > 0
    r.check(float(np.abs(h2 - h1).max()) / crm.CP < 1e-8, "energy not conserved")
    r.check(float(np.abs(m.qv + m.qc + m.qi - q1).max()) < 1e-15, "water not conserved")
    r.check(float(np.abs(m.qv[cl] / qsm[cl] - 1).max()) < 1e-9, "cloudy air not exactly saturated")
    r.check(float((m.qv[~cl] / qsm[~cl]).max()) <= 1.0 + 1e-12, "clear air supersaturated")
    r.close()

    r = Rule("H8 microphysical rates match their sources")
    # CM1 Kessler terminal velocity (research table), m/s
    for rq, rho_, want in ((1e-4, 1.15, 4.15), (1e-3, 1.15, 5.66), (5e-3, 1.15, 7.03),
                           (1e-3, 0.50, 8.58)):
        v = float(crm.rain_fall_speed(np.array(rq / rho_), np.array(rho_)))
        r.check(near(v, want, 0.02), f"rain {rq*1e3:.1f} g/m3 at rho {rho_}: {v:.2f} m/s not {want}")
    # SAM1MOM snow and graupel mass-weighted fall speeds (research table)
    t_cold = np.array(223.16)       # graupel fraction 0: pure snow
    t_warm = np.array(283.16)       # graupel fraction 1
    vs = float(crm.ice_precip_fall_speed(np.array(1e-3 / 1.15), np.array(1.15), t_cold))
    vg = float(crm.ice_precip_fall_speed(np.array(1e-3 / 1.15), np.array(1.15), t_warm))
    r.check(near(vs, 1.26, 0.03), f"snow fall speed {vs:.2f} m/s not 1.26")
    r.check(near(vg, 5.0, 0.08), f"graupel fall speed {vg:.2f} m/s not 5.0")
    # saturation over water and ice (research table, g/kg)
    for t, pp, wl, wi in ((300.0, 1e5, 22.79, None), (273.15, 7e4, 5.479, 5.479),
                          (253.15, 5e4, 1.568, 1.282), (233.15, 3e4, 0.393, 0.262)):
        r.check(near(float(crm.qsat_liq(np.array(t), np.array(pp))) * 1e3, wl, 0.01 * wl),
                f"qsat water at {t} K")
        if wi:
            r.check(near(float(crm.qsat_ice(np.array(t), np.array(pp))) * 1e3, wi, 0.01 * wi),
                    f"qsat ice at {t} K")
    r.close()

    r = Rule("H9 conversions conserve water and moist enthalpy")
    sc = _small_storm_scenario(bubble=False)
    m = crm.CPUModel(sc)
    rng = np.random.default_rng(9)
    m.qc = 2e-3 * rng.random(m.qc.shape)
    m.qr = 2e-3 * rng.random(m.qc.shape)
    m.qi = 1e-3 * rng.random(m.qc.shape)
    m.qs = 2e-3 * rng.random(m.qc.shape)
    m.qv = m.qv * (0.6 + 0.5 * rng.random(m.qv.shape))
    pi0 = m.b.pi0.reshape(-1, 1, 1)
    th0 = m.b.th0.reshape(-1, 1, 1)
    t1 = (th0 + m.th) * pi0
    lv = crm.latent_vap(t1)
    ls = crm.latent_sub(t1)

    def hh():
        t = (th0 + m.th) * pi0
        return crm.CP * t - lv * (m.qc + m.qr) - ls * (m.qi + m.qs)
    h1 = hh()
    w1 = m.qv + m.qc + m.qr + m.qi + m.qs
    m._conversions(20.0)
    r.check(float(np.abs(m.qv + m.qc + m.qr + m.qi + m.qs - w1).max()) < 1e-15, "water")
    r.check(float(np.abs(hh() - h1).max()) / crm.CP < 1e-9, "enthalpy")
    for q in ("qv", "qc", "qr", "qi", "qs"):
        r.check(float(getattr(m, q).min()) >= -1e-18, f"{q} went negative")
    r.close()

    r = Rule("H10 a dry thermal rises and stays symmetric")
    g = crm.Grid(64, 4, 40, 250.0, 250.0, 250.0)
    base = crm.BaseState.build(g, lambda z: 300.0, lambda z: 0.0, lambda z: 0.0,
                               lambda z: 0.0, 100000.0)
    sc = crm.Scenario("dry", "dry", "", g, base, micro_ice=False, random_th=0.0,
                      flux_mode="none", drag=False, sponge_depth=1e-3)
    m = crm.CPUModel(sc)
    x = (np.arange(g.nx) + 0.5) * g.dx
    z = g.zc
    beta = np.sqrt((x[None, :] - 8000.0) ** 2 / 2000.0 ** 2 + (z[:, None] - 2000.0) ** 2 / 2000.0 ** 2)
    m.th[:] = np.where(beta < 1, 2.0 * np.cos(0.5 * np.pi * beta) ** 2, 0.0)[:, None, :]
    while m.time < 1000.0 - 1e-9:
        m.step(min(m.max_dt(0.9), 1000.0 - m.time, 4.0))
    th = m.th[:, 0, :]
    r.check(float(np.abs(th - th[:, ::-1]).max()) < 1e-9, "lost its left-right symmetry")
    wmax = float(m.w.max())
    r.check(10.0 < wmax < 17.0, f"w max {wmax:.1f} m/s after 1000 s (125 m grid: 14.4)")
    k = np.where((th > 0.1).any(axis=1))[0]
    top = float(z[k[-1]]) if k.size else 0.0
    r.check(6000.0 < top < 9500.0, f"thermal top {top:.0f} m after 1000 s")
    r.close()


def test_model_gpu():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context(require=430, backend="egl")
    except Exception:
        try:
            import moderngl
            ctx = moderngl.create_standalone_context(require=430)
        except Exception as e:                                  # noqa: BLE001
            RESULTS.append(("H11 GPU model agrees with the NumPy model", 0,
                            [f"skipped, no OpenGL 4.3 context ({type(e).__name__})"]))
            return
    from . import crm, crm_gpu
    r = Rule("H11 GPU model agrees with the NumPy model")
    sc = _small_storm_scenario(n=32, nz=24, lateral=True)
    cpu = crm.CPUModel(sc, dtype="f8")
    gpu = crm_gpu.GPUModel(ctx, sc, init=crm.CPUModel(sc, dtype="f8"))
    for _ in range(30):
        dt = cpu.max_dt(0.9)
        cpu.step(dt)
        gpu.step(dt)
    st = gpu.download_state()
    for name, tol in (("u", 2e-3), ("v", 2e-3), ("w", 2e-3), ("th", 2e-3), ("qv", 1e-3),
                      ("qc", 5e-3), ("qr", 5e-3), ("qi", 1e-2), ("qs", 1e-2)):
        a = getattr(cpu, name)
        b = st[name]
        scale = max(float(np.abs(a).max()), 1e-12)
        err = float(np.abs(a - b).max()) / scale
        r.check(err < tol, f"{name}: max difference {err:.1e} of its range")
    dc, dg = cpu.diagnostics(), gpu.diagnostics()
    r.check(near(dg["w_max"], dc["w_max"], 0.05 + 0.01 * abs(dc["w_max"])), "w max")
    r.check(near(dg["water"], dc["water"], 1e-5 * dc["water"]), "total water")
    gpu.release()
    r.close()

    r = Rule("H12 ... and with the layer-cloud physics")
    for key in ("stratocumulus", "altocumulus_ra", "streets", "virga", "asperitas", "squall"):
        sc = _small(crm.scenario_by_key(key, "fast"), 16)
        cpu = crm.CPUModel(sc, dtype="f8")
        gpu = crm_gpu.GPUModel(ctx, sc, init=crm.CPUModel(sc, dtype="f8"))
        for _ in range(20):
            dt = min(cpu.max_dt(0.8), 4.0)
            cpu.step(dt)
            gpu.step(dt)
        st = gpu.download_state()
        for name, tol in (("u", 3e-3), ("v", 5e-3), ("w", 5e-3), ("th", 1e-2), ("qv", 1e-3),
                          ("qc", 1e-2), ("qs", 2e-2)):
            if name == "qs" and not sc.micro_ice:
                continue
            a = getattr(cpu, name)
            b = st[name]
            scale = max(float(np.abs(a).max()), 1e-12)
            err = float(np.abs(a - b).max()) / scale
            r.check(err < tol, f"{key} {name}: max difference {err:.1e} of its range")
        gpu.release()
    r.close()


# ------------------------------------------------------- layer-cloud physics --

def _small(sc, n=16):
    """The same case on an n x n grid (same levels)."""
    import copy
    import dataclasses
    from . import crm
    g = sc.grid
    g2 = crm.Grid(n, n, g.nz, g.dx, g.dy, g.dz, g.z0)
    b2 = copy.copy(sc.base)
    b2.grid = g2
    return dataclasses.replace(sc, grid=g2, base=b2)


def test_layer_physics():
    import dataclasses
    from . import crm
    r = Rule("J1 an elevated domain sits on the same hydrostatic column")
    thf = lambda z: 300.0 + 3.5e-3 * z
    qvf = lambda z: 0.006 * math.exp(-z / 2500.0)
    zero = lambda z: 0.0
    gg = crm.Grid(4, 4, 160, 100.0, 100.0, 25.0)
    gu = crm.Grid(4, 4, 48, 100.0, 100.0, 25.0, z0=2700.0)
    b1 = crm.BaseState.build(gg, thf, qvf, zero, zero, 101300.0)
    b2 = crm.BaseState.build(gu, thf, qvf, zero, zero, 101300.0)
    r.check(near(gu.zc[0], 2712.5, 1e-9) and near(gu.top, 3900.0, 1e-9), "heights above the ground")
    r.check(float(np.abs(b2.p0 - b1.p0[108:156]).max()) < 0.5, "pressure at matching levels")
    r.check(float(np.abs(b2.th0 - b1.th0[108:156]).max()) < 1e-9, "theta at matching levels")
    r.close()

    r = Rule("J2 a cloud layer from theta_l and q_t is in saturation equilibrium")
    sc = crm.scenario_stratocumulus("fast")
    b = sc.base
    qc = np.asarray(sc.init_qc)
    cl = qc > 0
    r.check(cl.sum() >= 5, f"{int(cl.sum())} cloudy levels")
    qs = crm.qsat_liq(b.t0, b.p0)
    r.check(float(np.abs(b.qv0[cl] / qs[cl] - 1.0).max()) < 1e-5, "vapour at saturation in cloud")
    tl = 289.0 * b.pi0[cl]
    thl_back = b.th0[cl] - crm.latent_vap(tl) * qc[cl] / (crm.CP * b.pi0[cl])
    r.check(float(np.abs(thl_back - 289.0).max()) < 1e-6, "theta_l recovered to 1e-6 K")
    zc = sc.grid.zc
    r.check(near(zc[cl][-1], 840.0, sc.grid.dz), f"cloud top {zc[cl][-1]:.0f} m at the 840 m inversion")
    r.check(560.0 < zc[cl][0] < 640.0, f"cloud base {zc[cl][0]:.0f} m (RF01: about 600 m)")
    r.close()

    r = Rule("J3 long-wave radiation cools the cloud top and conserves flux")
    m = crm.CPUModel(_small(sc, 8), dtype="f8")
    heat = m.lw_heating()[:, 0, 0]
    top = int(np.nonzero(m.qc[:, 0, 0] > 0)[0][-1])
    bot = int(np.nonzero(m.qc[:, 0, 0] > 0)[0][0])
    r.check(heat[top] * 3600.0 < -2.0, f"cloud-top heating {heat[top] * 3600:.1f} K/h")
    r.check(heat[bot] > 0.0, f"cloud-base heating {heat[bot] * 3600:.2f} K/h")
    # the column integral of rho cp dT/dt is the flux convergence F(bottom) - F(top)
    g = m.g
    rho = m.b.rho0
    ql = np.maximum(m.qc[:, 0, 0], 0)
    q = 85.0 * rho * ql * g.dz
    zi = float(m.inversion_height()[0, 0])
    def flux(k):
        above, below = q[k:].sum(), q[:k].sum()
        f = 70.0 * math.exp(-above) + 22.0 * math.exp(-below)
        z = g.zf[k]
        if z > zi:
            d = z - zi
            f += 1.13 * crm.CP * 3.75e-6 * (d ** (4 / 3) / 4 + zi * d ** (1 / 3))
        return f
    integral = float((rho * crm.CP * heat * g.dz).sum())
    r.check(near(integral, flux(0) - flux(g.nz), 1e-6 * abs(flux(g.nz)) + 1e-6),
            f"column {integral:.4f} W/m2 vs flux convergence {flux(0) - flux(g.nz):.4f}")
    r.check(near(zi, 840.0, g.dz), f"zi {zi:.0f} m from the 8 g/kg isoline")
    r.close()

    r = Rule("J4 above the inversion radiation balances subsidence (DYCOMS-II)")
    sc = crm.scenario_stratocumulus("standard")
    m = crm.CPUModel(_small(sc, 8), dtype="f8")
    g = m.g
    heat = m.lw_heating()[:, 0, 0]
    zi = float(m.inversion_height()[0, 0])
    th = m.b.th0 + m.th[:, 0, 0]
    wls = np.array([sc.subsidence(z) for z in g.zc])
    grad = np.zeros_like(th)
    grad[:-1] = (th[1:] - th[:-1]) / g.dz
    sub = -wls * grad                               # K/s of theta
    rad = heat / m.b.pi0
    sel = (g.zc > zi + 100.0) & (g.zc < g.top - sc.sponge_depth)
    rel = np.abs(rad[sel] + sub[sel]) / np.maximum(np.abs(sub[sel]), 1e-12)
    r.check(sel.sum() >= 3, "levels to compare")
    r.check(float(rel.max()) < 0.15, f"imbalance up to {rel.max() * 100:.0f}% of the subsidence warming")
    r.close()

    r = Rule("J5 the Coriolis force turns a wind excess inertially")
    gi = crm.Grid(8, 8, 4, 500.0, 500.0, 200.0)
    bi = crm.BaseState.build(gi, lambda z: 300.0, lambda z: 0.0, lambda z: 10.0, zero, 100000.0)
    sci = crm.Scenario("i", "inertial", "", gi, bi, micro_ice=False, random_th=0.0,
                       flux_mode="none", drag=False, coriolis=1.0e-3, tile=True,
                       sponge_depth=0.0)
    mi = crm.CPUModel(sci, dtype="f8")
    mi.damp_c[:] = 0.0
    mi.damp_f[:] = 0.0
    mi.u += 2.0
    for _ in range(100):
        mi.step(10.0)
    up = float(mi.u.mean() - mi.u0.mean())
    vp = float(mi.v.mean() - mi.v0.mean())
    r.check(near(up, 2.0 * math.cos(1.0), 0.01), f"u' {up:.4f} vs {2 * math.cos(1.0):.4f}")
    r.check(near(vp, -2.0 * math.sin(1.0), 0.01), f"v' {vp:.4f} vs {-2 * math.sin(1.0):.4f}")
    r.close()

    r = Rule("J6 bulk fluxes from a warm sea")
    sct = _small(crm.scenario_streets("fast"), 8)
    mt = crm.CPUModel(sct, dtype="f8")
    wth, wqv = mt.surface_fluxes()
    um, vm = sct.translate
    uc = 0.5 * (mt.u[0] + np.roll(mt.u[0], -1, 1)) + um
    vc = 0.5 * (mt.v[0] + np.roll(mt.v[0], -1, 0)) + vm
    spd = np.sqrt(uc ** 2 + vc ** 2 + 1.0)
    ch = (0.4 / math.log((0.5 * sct.grid.dz + sct.z0) / sct.z0)) ** 2
    ths = sct.sst * (crm.P00 / sct.base.psfc) ** (crm.RD / crm.CP)
    want = ch * spd * (ths - (mt.b.th0[0] + mt.th[0])) * mt.b.pi0[0]
    r.check(float(np.abs(wth - want).max()) < 1e-12, "sensible heat flux formula")
    r.check(float(wth.min()) > 0.0 and float(wqv.min()) > 0.0, "cold air over a warm sea gains heat and vapour")
    h = float(wth.mean()) * float(mt.b.rho0[0]) * crm.CP
    r.check(100.0 < h < 400.0, f"sensible heat flux {h:.0f} W/m2 (cold-air outbreak range)")
    r.close()

    r = Rule("J7 nudging relaxes the layer's mean at its time scale")
    sca = dataclasses.replace(_small(crm.scenario_altocumulus("fast"), 8), lw_f0=0.0, lw_f1=0.0)
    ma = crm.CPUModel(sca, dtype="f8")
    k = np.nonzero(ma.nudge_mask)[0]
    ma.th[k] += 1.0
    thl0, qt0 = ma.mean_thl_qt()
    ma._forcing(10.0)
    thl1, qt1 = ma.mean_thl_qt()
    want = -10.0 / sca.nudge_tau * (thl0[k] - ma.thl_target[k])
    r.check(float(np.abs((thl1[k] - thl0[k]) - want).max()) < 1e-9, "theta_l relaxation")
    r.check(float(np.abs(qt1[k] - qt0[k] - (-10.0 / sca.nudge_tau * (qt0[k] - ma.qt_target[k]))).max()) < 1e-12,
            "q_t relaxation")
    out = np.nonzero(ma.nudge_mask == 0)[0]
    r.check(float(np.abs(thl1[out] - thl0[out]).max()) < 1e-12, "nothing outside the layer")
    r.close()


def test_regimes():
    from . import regimes
    r = Rule("K1 the offline weathers choose their regimes")
    when = datetime(2026, 9, 10, 6, tzinfo=timezone.utc)
    want = {"tropical-fair": "cumulus", "humid-deep": "thunderstorm",
            "stable-stratus": "stratocumulus", "dry": "cirrus"}
    for prof, key in want.items():
        snd = sounding.standard_sounding(3.14, 101.69, when, profile=prof)
        reg = regimes.choose(snd, sun_alt_deg=55.0, size="fast")
        r.check(reg.key == key, f"{prof}: {reg.key} (expected {key})")
        text = " ".join(reg.lines())
        for topic in ("weather code", "cloud cover", "instability", "pressure", "temperature", "wind"):
            r.check(topic in text, f"{prof}: the reasons do not mention {topic}")
        if reg.scenario is not None:
            r.check(bool(reg.scenario.shows), f"{prof}: the model's cloud has no name")
    r.close()

    r = Rule("K2 a layer built from the data has its base where the data put it")
    snd = sounding.standard_sounding(3.14, 101.69, when, profile="stable-stratus")
    reg = regimes.choose(snd, sun_alt_deg=55.0, size="fast")
    sc = reg.scenario
    base_data = reg.facts["layer"][0]
    qc = np.asarray(sc.init_qc) + (np.asarray(sc.init_qi) if sc.init_qi is not None else 0.0)
    zc = sc.grid.zc
    first = zc[np.nonzero(qc > 0)[0][0]]
    r.check(abs(first - base_data) <= 1.5 * sc.grid.dz, f"model base {first:.0f} m, data {base_data:.0f} m")
    r.check(sc.nudge_tau > 0, "the layer is held to the forecast's mean state")
    r.close()

    r = Rule("K3 a storm on its way is recognised from the regional grid")
    from .sounding import Region, RegionPoint
    snd = sounding.standard_sounding(3.14, 101.69, when, profile="humid-deep")
    snd.surface["weather_code"] = 2.0
    snd.surface["precipitation"] = 0.0
    pts = []
    for (e, n) in sounding.region_offsets(7, 40.0):
        storm = (e == -80.0 and n == 0.0)
        pts.append(RegionPoint(e, n, 0.0, 0.0, {"weather_code": 95.0 if storm else 2.0,
                                                 "wind_speed_700hPa": 15.0,
                                                 "wind_direction_700hPa": 270.0,
                                                 "cloud_cover_low": 40.0, "cloud_cover_mid": 30.0,
                                                 "cloud_cover_high": 50.0}, {}))
    snd.region = Region(pts, 40.0, 7)
    reg = regimes.choose(snd, sun_alt_deg=55.0, size="fast")
    text = " ".join(reg.lines())
    r.check(reg.key == "storm_coming", f"regime {reg.key}")
    r.check("toward you" in text and "80 km" in text, "the approach is explained")
    r.close()


def test_region_and_view():
    from . import crm, gl_sky, region, viewinfo
    from .sounding import Region, RegionPoint
    r = Rule("L1 regional maps: continuous, zero displacement here, forecast cover")
    pts = []
    for (e, n) in sounding.region_offsets(7, 40.0):
        cov = max(0.0, 80.0 - (e + 120.0) / 240.0 * 80.0 * (1.0 if e > 0 else 0.0))
        pts.append(RegionPoint(e, n, 0.0, 0.0, {"cloud_cover_low": 80.0 if e <= 0 else cov,
                                                 "cloud_cover_mid": 0.0, "cloud_cover_high": 0.0,
                                                 "wind_speed_850hPa": 8.0 + e / 40.0,
                                                 "wind_direction_850hPa": 270.0}, {}))
    snd = type("S", (), {"region": Region(pts, 40.0, 7)})()
    rm = region.RegionMaps().build(snd, 800.0, 4000.0)
    n = rm.n
    w, c = rm.sample(np.array([0.0]), np.array([0.0]))
    r.check(float(np.abs(w).max()) < 1.0, f"displacement here {float(np.abs(w).max()):.2f} m")
    r.check(near(float(c[0, 0]), 1.0, 0.05), f"cover ratio here {float(c[0, 0]):.2f}")
    w2, c2 = rm.sample(np.array([130000.0]), np.array([0.0]))
    r.check(float(c2[0, 3]) < 0.2, f"cover ratio at the clear edge {float(c2[0, 3]):.2f}")
    r.check(float(np.abs(np.diff(rm.cover[..., 3], axis=1)).max()) < 0.2, "the cover map is smooth")
    tab = rm.set_tau_table(np.linspace(0.0, 20.0, 400))
    r.check(all(a >= b - 1e-9 for a, b in zip(tab, tab[1:])), "tau table falls as the cover rises")
    r.check(tab[-1] == 0.0, "full cover keeps every column")
    r.close()

    r = Rule("M1 the in-view census names what the camera sees")
    sc = _small(crm.scenario_stratocumulus("fast"), 16)
    m = crm.CPUModel(sc, dtype="f8")
    drv = type("D", (), {})()
    drv.model, drv.sc, drv.gpu_ok, drv.diag, drv.model_time = m, sc, False, m.diagnostics(), 3600.0
    v = gl_sky.SimView()
    g = sc.grid
    v.size, v.cell, v.z0 = (g.lx, g.ly, g.nz * g.dz), (g.dx, g.dy, g.dz), g.z0
    v.obs, v.tile, v.zlo, v.zhi = (0.5 * g.lx, 0.5 * g.ly), 3, 400.0, 1000.0
    drv.view = v
    cen = viewinfo.Census(nu=12, nv=8, samples=80)
    up = cen.update(gl_sky.Camera(az=0.0, alt=80.0, fov=60.0), 1.5, 1.7, drv, None, [], None, [], 0.0)
    r.check(bool(up) and up[0].name.startswith("Stratocumulus"), f"looking up: {[s.name for s in up]}")
    r.check(bool(up) and up[0].share > 0.9, "the deck fills the view overhead")
    r.check(bool(up) and 500.0 < up[0].heights[0] < 900.0, "at the deck's height")
    down = cen.update(gl_sky.Camera(az=0.0, alt=-60.0, fov=40.0), 1.5, 1.7, drv, None, [], None, [], 0.0)
    r.check(not down and cen.ground_share > 0.9, "looking down from the ground sees the ground")
    r.close()

    r = Rule("N1 the clouds above the weather appear when and where they should")
    from . import upper
    uc = upper.UpperClouds()
    june = datetime(2026, 6, 25, 21, tzinfo=timezone.utc)
    dec = datetime(2026, 12, 25, 21, tzinfo=timezone.utc)
    r.check(any(i.key == "nlc" for i in uc.decide(june, 58.0)), "noctilucent at 58N in June")
    r.check(not any(i.key == "nlc" for i in uc.decide(dec, 58.0)), "none at 58N in December")
    r.check(not any(i.key == "nlc" for i in uc.decide(june, 20.0)), "none at 20N")
    r.check(any(i.key == "nlc" for i in uc.decide(dec, -60.0)), "noctilucent at 60S in December")
    snd = sounding.standard_sounding(69.0, 19.0, datetime(2027, 1, 10, 12, tzinfo=timezone.utc))
    for lv in snd.levels:
        if lv.p <= 60.0:
            lv.t = -88.0
    items = uc.decide(datetime(2027, 1, 10, 12, tzinfo=timezone.utc), 69.0, snd)
    r.check(any(i.key == "psc" for i in items), "nacreous with a -88 C stratosphere")
    for lv in snd.levels:
        if lv.p <= 60.0:
            lv.t = -80.0
    items = uc.decide(datetime(2027, 1, 10, 12, tzinfo=timezone.utc), 69.0, snd)
    r.check(any(i.key == "nat" for i in items) and not any(i.key == "psc" for i in items),
            "only nitric acid PSC at -80 C")
    r.close()


# ---------------------------------------------- fall streaks, waves, keep --

def test_fall_and_waves():
    from . import crm, region, viewinfo, mie
    r = Rule("O1 in layer clouds the frozen precipitation is snow, not graupel")
    rho, t = np.array([0.75]), np.array([263.0])
    q = np.array([2e-5])
    v_g = float(crm.ice_precip_fall_speed(q, rho, t, 1.0)[0])
    v_s = float(crm.ice_precip_fall_speed(q, rho, t, 0.0)[0])
    r.check(v_s < 0.7 * v_g, f"snow falls at {v_s:.2f} m/s, SAM's -10 C mixture at {v_g:.2f}")
    sc = _small(crm.scenario_virga("fast"), 8)
    r.check(not sc.graupel and crm.scenario_weather(
        sounding.standard_sounding(profile="humid-deep"), "fast").graupel,
            "layer cases all snow, deep convection keeps graupel")
    m = crm.CPUModel(sc, dtype="f8")
    m.qs[:] = 0.0
    m.qs[10] = 3e-5
    o = m.optics()
    rq = m.b.rho0[10] * 3e-5
    r.check(near(float(o["snow"][10, 0, 0]), float(crm.extinction_mp(np.array(rq), crm.RHO_S,
                                                                     crm.N0S)), 1e-9),
            "its optics are the snow spectrum's")
    r.check(sc.qi_auto == crm.QI_AUTO_LAYER < crm.QI_AUTO, "ice turns to snow from 0.01 g/kg")
    r.close()

    r = Rule("O2 precipitation counts as there once it can be seen")
    m.qc[:] = 0.0
    m.qi[:] = 0.0
    m.qc[40] = 2e-4
    m.qs[:] = 0.0
    m.qs[12] = 2e-7
    zb = m.diagnostics()["precip_bottom"]
    r.check(near(zb, sc.grid.z0 + 12 * sc.grid.dz, 1e-6), f"0.2 mg/kg of snow counts ({zb:.0f} m)")
    m.qs[12] = 5e-8
    m.qs[20] = 2e-7
    zb = m.diagnostics()["precip_bottom"]
    r.check(near(zb, sc.grid.z0 + 20 * sc.grid.dz, 1e-6), "0.05 mg/kg (extinction below 1e-5/m) "
            "does not")
    r.close()

    r = Rule("O3 the trails lean the way the winds say")
    dg = {"cloud_base": 5550.0, "precip_bottom": 4600.0}
    t1 = viewinfo._trail_lean(crm.scenario_virga("fast"), dg) or ""
    t2 = viewinfo._trail_lean(crm.scenario_virga("fast", shear=-0.006), dg) or ""
    t3 = viewinfo._trail_lean(crm.scenario_virga("fast", shear=0.0), dg) or ""
    r.check("toward the W," in t1 and t1.endswith("toward the W"),
            "wind slowing downward: trails hang back to the west, upwind")
    r.check("toward the E" in t2, "wind quickening downward: they run ahead to the east")
    r.check("straight down" in t3, "no shear: straight down")
    r.close()

    r = Rule("O4 the forecast's cover leaves out whole cells, not columns")
    n = 32
    yy, xx = np.mgrid[0:n, 0:n]
    blob = lambda cx, cy, a: a * np.clip(1.0 - np.hypot(xx - cx, yy - cy) / 4.0, 0.0, 1.0)
    tau = blob(8, 8, 20.0) + blob(24, 22, 8.0)
    km = region.keep_tau_map(tau, 100.0, radius_m=300.0)
    rm = region.RegionMaps()
    tab = rm.set_tau_table(tau, 100.0)
    th = tab[8]                                      # the forecast has half the cover here
    a_cells = tau > 1.0
    in_a = a_cells & (np.hypot(xx - 8, yy - 8) < 4)
    in_b = a_cells & (np.hypot(xx - 24, yy - 22) < 4)
    r.check(bool((km[in_a] > th).all()), "the thicker cell is kept whole")
    r.check(bool((km[in_b] <= th).all()), "the thinner one is left out whole")
    r.check(float(np.abs(region.keep_tau_map(np.roll(tau, 5, 1), 100.0) - np.roll(km, 5, 1)).max())
            < 1e-12, "on the periodic domain")
    r.close()

    r = Rule("O5 a reference case never takes the forecast's cover")
    snd = type("S", (), {"region": "forecast"})()
    sc_v = crm.scenario_virga("fast")
    sc_a = crm.scenario_altocumulus("fast")
    r.check(region.source_for("auto", sc_a, snd) is snd, "the sky chosen from the data: forecast")
    r.check(region.source_for("altocumulus", sc_a, snd) is None, "a reference case: the same "
            "everywhere")
    src = region.source_for("virga", sc_v, snd)
    rm = region.RegionMaps().build(src, 5400.0, sc_v.grid.lx)
    r.check(rm.has_data and "composition" in rm.note, "or its own, labelled, composition")
    r.close()

    r = Rule("O6 asperitas: a sheared, nearly saturated stable layer under the base")
    sc = crm.scenario_asperitas("fast")
    b, zc = sc.base, sc.grid.zc
    k = int(np.argmin(np.abs(zc - 1500.0)))
    dthdz = (b.th0[k + 1] - b.th0[k - 1]) / (zc[k + 1] - zc[k - 1])
    n2 = crm.G / b.th0[k] * dthdz
    s2 = ((b.u0[k + 1] - b.u0[k - 1]) ** 2 + (b.v0[k + 1] - b.v0[k - 1]) ** 2) \
        / (zc[k + 1] - zc[k - 1]) ** 2
    ri = n2 / s2
    r.check(0.05 < ri < 0.25, f"Richardson number {ri:.2f} under the base")
    turn = math.degrees(math.atan2(b.v0[-1], b.u0[-1]) - math.atan2(b.v0[0], b.u0[0]))
    r.check(turn > 30.0, f"the wind turns {turn:.0f} deg across it")
    kb = int(np.argmin(np.abs(zc - 1580.0)))
    rh = b.qv0[kb] / float(crm.qsat_liq(b.t0[kb], b.p0[kb]))
    r.check(rh > 0.95, f"{rh * 100:.0f}% saturated just under the base")
    r.check(float(np.asarray(sc.init_qc).max()) > 3e-4, "a thick cloud above")
    r.close()

    r = Rule("O7 nacreous colours are the Mie colours of equal ice spheres")
    lut = mie.psc_phase_lut()
    th = mie.lut_theta(lut.shape[1])
    w = 2.0 * math.pi * np.sin(th) * np.gradient(th)
    r.check(float(np.abs((lut * w[None, :, None]).sum(1) - 1.0).max()) < 2e-3,
            "every row normalised")
    i15 = int(np.argmin(np.abs(np.degrees(th) - 15.0)))
    c1 = lut[3, i15] / lut[3, i15].sum()
    c2 = lut[9, i15] / lut[9, i15].sum()
    r.check(float(np.abs(c1 - c2).max()) > 0.1, f"15 deg from the Sun the colour depends on the "
            f"size: {np.round(c1, 2)} vs {np.round(c2, 2)}")
    r.close()


# ------------------------------------------- regressions from the stress test --

def test_stress_regressions():
    from . import crm, regimes, region, gl_sky
    from .sounding import Level, Sounding
    r = Rule("P1 forecast data: a high station, a pole, a missing hour")
    when = datetime(2026, 9, 25, 6, tzinfo=timezone.utc)

    def fake(lat, lon, elev, with_t2):
        times = [(when + timedelta(hours=k - 12)).strftime("%Y-%m-%dT%H:00") for k in range(24)]
        h = {"time": times}
        for p in sounding.PRESSURE_LEVELS:
            z = 44330.0 * (1.0 - (p / 1013.25) ** 0.1903)
            for var, val in (("temperature", 15.0 - 6.5 * z / 1000.0), ("relative_humidity", 50.0),
                             ("cloud_cover", 0.0), ("wind_speed", 5.0), ("wind_direction", 270.0),
                             ("geopotential_height", z)):
                h[f"{var}_{p}hPa"] = [val] * 24
        h["temperature_2m"] = [(-5.0 if with_t2 else None)] * 24
        h["surface_pressure"] = [650.0] * 24
        return {"elevation": elev, "hourly": h}

    orig = sounding._request
    try:
        sounding._request = lambda url, t: fake(29.65, 91.1, 3650.0, False)
        s = sounding.fetch_open_meteo(29.65, 91.1, when)
        r.check(s.levels[0].z >= 3650.0 - 1.0, f"no 2 m temperature: lowest level {s.levels[0].z:.0f} m "
                f"is not below the 3650 m ground")
        seen = {}

        def grid_req(url, t):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(url).query)
            lats = [float(x) for x in q["latitude"][0].split(",")]
            lons = [float(x) for x in q["longitude"][0].split(",")]
            seen["lats"], seen["lons"] = lats, lons
            off = seen.get("offset_h", 0)
            times = [(when + timedelta(hours=k - 12 + off)).strftime("%Y-%m-%dT%H:00") for k in range(24)]
            return [{"hourly": {"time": times, **{v: [1.0] * 24 for v in sounding._REGION_VARS}}}
                    for _ in lats]
        sounding._request = grid_req
        reg = sounding.fetch_region(89.5, 10.0, when)
        r.check(all(-90.0 <= x <= 90.0 for x in seen["lats"]) and
                all(-180.0 <= x <= 180.0 for x in seen["lons"]) and len(reg.points) == 49,
                f"the grid around 89.5 N stays on the globe ({max(seen['lats']):.2f} N max)")
        seen["offset_h"] = -40
        try:
            sounding.fetch_region(3.1, 101.7, when)
            r.check(False, "an answer without the chosen hour was taken")
        except RuntimeError:
            r.check(True, "an answer without the chosen hour is refused")
    finally:
        sounding._request = orig
    r.close()

    r = Rule("P2 non-finite forecast values are gaps, never numbers")
    r.check(regimes._num(float("nan"), 7.0) == 7.0 and regimes._num(float("inf")) is None
            and regimes._num("abc", 1.0) == 1.0, "NaN, infinity and text read as missing")
    snd = sounding.standard_sounding(profile="tropical-fair")
    snd.surface.update({"weather_code": float("nan"), "temperature_2m": float("inf"),
                        "pressure_msl": float("nan"), "cloud_cover_low": float("nan")})
    reg = regimes.choose(snd, 30.0, "fast", snd.lat)
    lines = " ".join(reg.lines()).lower()
    r.check(" nan" not in lines and " inf" not in lines, f"the explanation has no NaN ({reg.key})")
    r.close()

    r = Rule("P3 every model grid can run on the GPU (power-of-two sides)")
    bad = [(f, sz) for f, d in crm.GRID_PRESETS.items() for sz, v in d.items() if v[0] & (v[0] - 1)]
    bad += [("fog", sz) for sz, v in regimes.FOG_PRESETS.items() if v[0] & (v[0] - 1)]
    r.check(not bad, f"non-power-of-two grids: {bad}")
    r.close()

    r = Rule("P4 gaps in the regional maps and a black frame cannot poison the view")
    rm = region.RegionMaps()
    rm.build(None, 3000.0, 4000.0)
    rm.warp_a[:] = np.nan
    rm.raw_a[:] = np.nan
    rm.set_time(0.0)
    w, c = rm.sample(np.array([0.0, 50000.0]), np.array([0.0, -30000.0]))
    r.check(bool(np.isfinite(w).all() and np.isfinite(c).all()), "sampled maps are finite")

    class Tex:
        def __init__(self, v):
            self.v, self.width, self.height = v, 4, 4

        def read(self):
            return np.full((4, 4, 4), self.v, "f2").tobytes()

    for v, what in ((np.nan, "a NaN frame"), (0.0, "a black frame")):
        dummy = type("R", (), {})()
        dummy._bloom = [(Tex(v), None)]
        dummy._ev_used, dummy._ev_adapted, dummy.exposure_bias, dummy.adapt_rate = 2.8, 2.8, 0.0, 1.0
        dummy._ev_limits = (1.0, 8.0)
        gl_sky.SkyRenderer.meter(dummy)
        r.check(dummy._ev_adapted == 2.8, f"{what} leaves the exposure as it was ({dummy._ev_adapted})")
    r.close()

    r = Rule("P5 a model that blows up is stopped and reported")
    from . import simdriver
    dummy = type("D", (), {})()
    dummy.error, dummy.running = "", True
    ok = simdriver.SimDriver._check_health(dummy, {"w_max": float("nan"), "w_min": 0.0,
                                                   "water": 1.0, "cloud_top": 0.0, "time": 600.0})
    r.check(not ok and not dummy.running and "unstable" in dummy.error, dummy.error or "not stopped")
    dummy.error, dummy.running = "", True
    ok = simdriver.SimDriver._check_health(dummy, {"w_max": 12.0, "w_min": -6.0, "water": 1.0,
                                                   "cloud_top": 9000.0, "time": 600.0})
    r.check(ok and dummy.running, "a healthy storm runs on")
    r.close()

    r = Rule("P6 the forecast's ground: snow and soil water, in a tier of their own")
    orig = sounding._request
    asked = []
    try:
        def ground_req(url, t, refuse_land=False):
            asked.append(url)
            if refuse_land and "snow_depth" in url:
                raise RuntimeError("HTTP Error 400: Bad Request")
            d = fake(69.65, 18.96, 50.0, True)
            n = len(d["hourly"]["time"])
            d["hourly"].update({"snow_depth": [0.25] * n, "soil_moisture_9_to_27cm": [0.30] * n,
                                "visibility": [42000.0] * n})
            return d
        sounding._request = ground_req
        s = sounding.fetch_open_meteo(69.65, 18.96, when)
        r.check(s.surface.get("snow_depth") == 0.25 and s.surface.get("soil_moisture_9_to_27cm") == 0.30,
                "snow depth and root-zone soil water are read")
        r.check(sounding.snow_cover(s) == 1.0, f"25 cm of snow covers the ground ({sounding.snow_cover(s)})")
        sounding._request = lambda url, t: ground_req(url, t, refuse_land=True)
        s = sounding.fetch_open_meteo(69.65, 18.96, when)
        r.check(s.surface.get("visibility") == 42000.0,
                "a model without the ground's variables keeps the others (visibility)")
    finally:
        sounding._request = orig
    for depth, want in ((0.0, 0.0), (0.05, 0.5), (0.3, 1.0), (float("nan"), 0.0), ("x", 0.0), (None, 0.0)):
        got = sounding.snow_cover(type("S", (), {"surface": {"snow_depth": depth}})())
        r.check(near(got, want, 1e-9), f"snow depth {depth!r} m -> cover {got} (want {want})")
    r.close()

    r = Rule("P7 a desert heats its air and does not moisten it; snow reflects the sunlight")
    bw_dry, ab_dry, note_dry = crm.surface_state(type("S", (), {"surface": {"soil_moisture_9_to_27cm": 0.03}})())
    bw_wet, ab_wet, _ = crm.surface_state(type("S", (), {"surface": {"soil_moisture_9_to_27cm": 0.35}})())
    bw_none, ab_none, note_none = crm.surface_state(type("S", (), {"surface": {}})())
    _, ab_snow, _ = crm.surface_state(type("S", (), {"surface": {"snow_depth": 0.2}})())
    r.check(bw_dry > 10.0, f"dry soil (0.03 m3/m3): Bowen ratio {bw_dry:.1f}")
    r.check(near(bw_wet, 0.45, 0.01) and near(bw_none, 0.45, 1e-9) and note_none == "",
            f"moist soil and no data: Bowen ratio 0.45 ({bw_wet:.3f}, {bw_none:.3f})")
    r.check(near(ab_none, 0.77, 1e-9) and near(ab_snow, 0.36, 1e-9) and near(ab_dry, 0.65, 1e-9)
            and near(ab_wet, 0.77, 1e-9),
            f"absorbed sunlight 0.77 moist, 0.65 bare sand, 0.36 under snow ({ab_wet}, {ab_dry}, {ab_snow})")
    dry_look = [sounding.ground_dryness(type("S", (), {"surface": {"soil_moisture_9_to_27cm": v}})())
                for v in (0.03, 0.066, 0.133, 0.20, 0.35, None, float("nan"))]
    r.check(dry_look[0] == 1.0 and dry_look[1] == 1.0 and near(dry_look[2], 0.5, 0.01) and
            dry_look[3] == 0.0 and dry_look[4] == 0.0 and dry_look[5] == 0.0 and dry_look[6] == 0.0,
            f"the ground's look from its soil water: {[round(x, 2) for x in dry_look]}")
    from .sounding import standard_sounding
    fluxes = []
    for sm in (0.03, 0.35):
        snd_g = standard_sounding(profile="dry")
        snd_g.surface["soil_moisture_9_to_27cm"] = sm
        m = crm.CPUModel(crm.scenario_weather(snd_g, "fast"), dtype="f8")
        m.sun_elev = 1.2
        wth, wqv = m.surface_fluxes()
        fluxes.append((float(np.mean(wth)), float(np.mean(wqv))))
    (h_dry, q_dry), (h_wet, q_wet) = fluxes
    r.check(q_dry < 0.1 * q_wet and h_dry > 1.5 * h_wet,
            f"the model's surface: vapour flux {q_dry:.2e} vs {q_wet:.2e} m/s, heat {h_dry:.3f} vs {h_wet:.3f} K m/s")
    r.close()

    r = Rule("P8 a still frame averages over its whole pixel")
    pts = np.array([gl_sky.pixel_jitter(k) for k in range(16)])
    r.check(bool((pts >= -0.5).all() and (pts < 0.5).all()), "offsets inside the pixel")
    r.check(bool(abs(pts.mean(0)).max() < 0.06), f"offsets centred ({pts.mean(0).round(3)})")
    quads = {(bool(x >= 0.0), bool(y >= 0.0)) for x, y in pts}
    r.check(len(quads) == 4, "every quarter of the pixel is visited")
    r.close()

    r = Rule("P11 lightning is drawn where it struck, whatever the eye's height")
    from . import simdriver
    p_ground = simdriver.world_point(3000.0, -4000.0, 6000.0)
    for eye in (1.7, 8000.0, 22000.0):
        q = gl_sky.eye_frame(p_ground, eye)
        # its height above the ground below it, seen from this eye
        rg = 6360000.0
        h = math.sqrt(q[0] ** 2 + (q[1] + rg + eye) ** 2 + q[2] ** 2) - rg
        r.check(near(h, 6000.0, 2.0), f"eye {eye:.0f} m: the flash {h:.0f} m above the ground (6000)")
    r.close()

    r = Rule("P9 the ground under the sky: snow, height, and which decks make optics")
    dummy = type("R", (), {})()
    dummy.ground_albedo, dummy.snow = 0.13, 0.0
    r.check(near(gl_sky.SkyRenderer.effective_albedo(dummy), 0.13, 1e-9), "bare ground albedo 0.13")
    dummy.snow = 1.0
    r.check(near(gl_sky.SkyRenderer.effective_albedo(dummy), 0.64, 1e-9), "snow-covered ground 0.64")
    dummy.snow = float("nan")
    r.check(near(gl_sky.SkyRenderer.effective_albedo(dummy), 0.13, 1e-9), "a NaN snow cover is none")
    dummy.snow, dummy.dryness = 0.0, 1.0
    r.check(near(gl_sky.SkyRenderer.effective_albedo(dummy), 0.35, 1e-9), "bare sand 0.35")
    dummy.dryness = 0.0
    for e, want in ((5364.0, 5364.0), (float("nan"), 0.0), (-430.0, 0.0), (20000.0, 9000.0)):
        dummy.ground_elev = e
        got = gl_sky.SkyRenderer.ground_elevation(dummy)
        r.check(near(got, want, 1e-9), f"station height {e} m -> {got} m (want {want})")
    snd_o = sounding.standard_sounding(profile="humid-deep")
    decks = [scene.deck_from_spec(CloudSpec("Cs", "neb"), snd_o, base_m=9000.0, coverage=0.9),
             scene.deck_from_spec(CloudSpec("Ac", "str"), snd_o, base_m=4000.0, coverage=0.6),
             scene.deck_from_spec(CloudSpec("St", "neb"), snd_o, base_m=500.0, coverage=1.0)]
    pk = gl_sky.pack_deck_params(decks)
    got = [(float(pk[i, 9, 2]), float(pk[i, 9, 3])) for i in range(3)]
    want = [tuple(float(np.float32(v)) for v in gl_sky.deck_optics(d)) for d in decks]
    r.check(got == want, f"each deck carries its own halo/corona strength {got} (want {want})")
    r.check(want[0][0] > 0.0 and want[0][1] == 0.0 and want[2] == (0.0, 0.0),
            "the cirrostratus makes a halo, the stratus nothing")
    r.check(gl_sky.optical_phenomena(decks) == tuple(max(gl_sky.deck_optics(d)[k] for d in decks)
                                                      for k in (0, 1)), "strengths are the decks' strongest")
    r.close()


def test_zero_low_cloud():
    import copy
    r = Rule("P12 a forecast of 0 % low cloud is a value, not a gap: the convective layer "
             "then has no cover (it read as missing and drew Cb at 4/8)")
    for prof in ("tropical-fair", "humid-deep"):
        base_snd = sounding.standard_sounding(3.14, 101.69, profile=prof)
        got = {}
        for low in (None, 0.0, 2.0, 30.0):
            s12 = copy.deepcopy(base_snd)
            if low is None:
                s12.surface.pop("cloud_cover_low", None)
            else:
                s12.surface["cloud_cover_low"] = low
            conv = [d for d in scene.diagnose(s12) if d.spec.genus in ("Cb", "Cu")]
            got[low] = conv[0].coverage if conv else None
        r.check(got[0.0] is not None and near(got[0.0], 0.0, 1e-9),
                f"{prof}: 0 % low cloud -> convective cover {got[0.0]} (want 0)")
        r.check(got[2.0] is not None and near(got[2.0], 0.02, 1e-9),
                f"{prof}: 2 % low cloud -> {got[2.0]} (want 0.02)")
        r.check(got[30.0] is not None and near(got[30.0], 0.30, 1e-9),
                f"{prof}: 30 % low cloud -> {got[30.0]} (want 0.30)")
        r.check(got[None] is not None and got[None] in (0.5, 0.22),
                f"{prof}: no low-cloud figure -> the default {got[None]} (0.5 for Cb, 0.22 for Cu)")
    r.close()

def test_app_logic():
    from . import app, crm, timeline
    r = Rule("P10 over the squall-line model the weather's veils stay; only Cb and Cu make way")
    snd = sounding.standard_sounding(profile="humid-deep")
    decks = [scene.deck_from_spec(CloudSpec(g), snd, seed=i + 1)
             for i, g in enumerate(("Cb", "Cu", "Cs", "As"))]
    d = type("C", (), {})()
    d.sim_key, d.mode, d.selected, d.status = "squall", "auto", 0, ""
    d.sim = type("S", (), {"sc": crm.scenario_by_key("squall", "fast")})()
    d.sky = timeline.Sky(manual=decks, threaded=False, n=64)
    d.sky.update(datetime(2026, 9, 10, 4, tzinfo=timezone.utc), 0.0, sync=True)
    for fade in (1.0, 0.4, 0.0):
        d.sim_fade = fade
        app.CloudSim._set_model_shares(d)
        d.sky.update(datetime(2026, 9, 10, 4, tzinfo=timezone.utc), 0.0, sync=True)
        cov = {s.deck.spec.genus: s.deck.coverage for s in d.sky.states}
        want = {s.deck.spec.genus: s.base_cover for s in d.sky.states}
        r.check(all(near(cov[g], want[g] * (1.0 - fade), 1e-6) for g in ("Cb", "Cu")),
                f"model shown {fade}: the storms' layers give way as much ({cov})")
        r.check(all(near(cov[g], want[g], 1e-6) for g in ("Cs", "As")),
                f"model shown {fade}: the cirrostratus and altostratus stay ({cov})")
    r.close()


# ------------------------------------------------------------- transitions --

class _HandForecast:
    """A forecast of n hours for timeline.Sky whose layers are made by hand
    (decks_at(i)): the forecast's own hours, nothing between them."""

    def __init__(self, t0: datetime, n: int):
        self.times = [t0 + timedelta(hours=i) for i in range(n)]
        self.is_static = False
        self.region = None

    @property
    def start(self):
        return self.times[0]

    def sounding_hour(self, i: int) -> int:
        return i


def _evolving_decks(snd):
    """Ten hours of a sky that does everything a sky can do between two
    forecast hours: a stratocumulus that rises through the 2 km boundary of
    the étages and thins out; cirrus that thickens into cirrostratus (a
    halo) and then altostratus (none); an altocumulus that grows turrets
    and loses them; a cumulus field that forms, lives three hours with its
    wind turning round, and dies; perlucidus gaps coming and going."""
    ci = [("Ci", "fib"), ("Ci", "fib"), ("Cs", "fib"), ("Cs", "neb"), ("Cs", "neb"),
          ("As", None), ("As", None), ("As", None), ("As", None), ("As", None)]

    def decks_at(i: int) -> list:
        out = []
        if i <= 7:
            d = scene.deck_from_spec(CloudSpec("Sc", "str", ["pe"] if i in (2, 3) else []), snd,
                                     base_m=1400.0 + 150.0 * i, coverage=max(0.05, 0.8 - 0.1 * i),
                                     thickness_m=500.0, seed=11)
            out.append(d)
        g, sp = ci[i]
        out.append(scene.deck_from_spec(CloudSpec(g, sp), snd, base_m=9000.0 - 550.0 * i,
                                        coverage=min(0.3 + 0.1 * i, 0.95),
                                        thickness_m=900.0 + 250.0 * i, seed=22))
        if 3 <= i <= 6:
            cas = i in (4, 5)
            out.append(scene.deck_from_spec(CloudSpec("Ac", "cas" if cas else "str"), snd,
                                            base_m=4500.0, coverage=0.3, thickness_m=600.0, seed=33,
                                            turret_rise_m=800.0 if cas else None))
        if 2 <= i <= 5:
            d = scene.deck_from_spec(CloudSpec("Cu", "hum"), snd, base_m=900.0, coverage=0.25,
                                     thickness_m=700.0, seed=44)
            d.wind_u, d.wind_v = 6.0 * (3.5 - i), 2.0
            out.append(d)
        if 6 <= i <= 8:
            # a thin cirrocumulus with waves, born on the hour: its corona and
            # its waves must grow with it
            out.append(scene.deck_from_spec(CloudSpec("Cc", "str", ["un"]), snd, base_m=7600.0,
                                            coverage=0.35, thickness_m=200.0, seed=55))
        return out
    return decks_at


def _sky_hands(n_hours: int = 10, n: int = 64):
    from . import timeline
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    t0 = datetime(2026, 9, 10, 0, tzinfo=timezone.utc)
    fc = _HandForecast(t0, n_hours)
    return timeline.Sky(fc, lat=3.14, threaded=False, n=n, seed=5,
                        diagnose=_evolving_decks(snd)), t0


def _band_series():
    """Four hours of a regional grid with a band of low cloud some 50 km wide
    carried east by a 10 m/s wind (36 km an hour), middle cloud 50 %."""
    from . import forecast
    offs = sounding.region_offsets(7, 40.0)
    t0r = datetime(2026, 9, 10, 0, tzinfo=timezone.utc)
    times = [(t0r + timedelta(hours=k)).strftime("%Y-%m-%dT%H:00") for k in range(4)]
    data = []
    for (e, nn) in offs:
        low = [100.0 * math.exp(-((e - (-60.0 + 36.0 * k)) / 30.0) ** 2) for k in range(4)]
        data.append({"hourly": {"time": times, "cloud_cover_low": low,
                                "cloud_cover_mid": [50.0] * 4, "cloud_cover_high": [0.0] * 4,
                                "wind_speed_850hPa": [10.0] * 4, "wind_direction_850hPa": [270.0] * 4,
                                "wind_speed_700hPa": [10.0] * 4, "wind_direction_700hPa": [270.0] * 4}})
    return forecast.RegionSeries([(e, nn, 0.0, 0.0) for (e, nn) in offs], data), t0r


def _probe_state(sky, when):
    from . import gl_sky
    sky.update(when, 0.0, sync=True)
    p = gl_sky.pack_states(sky.states)
    optics = gl_sky.optical_phenomena([s.deck for s in sky.states if s.deck.coverage > 0.0])
    return p, optics, {s.group: s.track for s in sky.states}


def _jumps(pa, pb, ta, tb, big):
    """What changes by more than a continuous quantity can over the shorter
    of two intervals: (group, row, component, change) per jump.  big: the
    changes over the longer interval, the same keys."""
    out = []
    for g in set(ta) & set(tb):
        if ta[g] != tb[g]:
            continue
        a, b = pa[g], pb[g]
        for r_ in range(a.shape[0]):
            for c in range(4):
                slot = (c if (r_ == 11 and c < 3) or (r_ in (14, 15) and c < 3) else
                        (c // 2 if r_ == 12 else (2 if (r_ == 13 and c < 2) else None)))
                if slot is not None and (a[10, slot] <= 1e-6 or b[10, slot] <= 1e-6):
                    continue
                x, y = float(a[r_, c]), float(b[r_, c])
                if r_ == 10 and c == 3:
                    x, y = float(noise.norm_cdf(-x)), float(noise.norm_cdf(-y))
                dv = abs(y - x)
                if r_ in (12, 13) and slot is not None:
                    ex = float(a[11, slot])
                    dv = abs((y - x + 0.5 * ex) % ex - 0.5 * ex)
                if r_ == 16 and c < 2 and a[16, 2] > 0:
                    ex = 0.55 * float(a[16, 2])
                    dv = abs((y - x + 0.5 * ex) % ex - 0.5 * ex)
                key = (g, r_, c)
                if big is not None and key in big:
                    if dv > 1e-5 * max(abs(x), abs(y), 1e-3) + 1e-7 and dv > 0.5 * big[key]:
                        out.append((g, r_, c, dv, x, y))
                elif big is None:
                    out.append((key, dv))
    return out


def test_transitions():
    from . import forecast, gl_sky, region, timeline, upper

    r = Rule("T1 the pattern's blend: squares sum to one, a steady rate, each at rest at its ends")
    th = np.linspace(0.0, 9.0, 3601)
    s2 = [sum(w * w for w in timeline.weights(t).values()) for t in th]
    r.check(max(abs(v - 1.0) for v in s2) < 1e-12, f"sum of squares 1 within {max(abs(v - 1.0) for v in s2):.1e}")
    h = 1e-6
    sp = [sum(((timeline.weights(t + h).get(j, 0.0) - timeline.weights(t - h).get(j, 0.0)) / (2 * h)) ** 2
              for j in timeline.weights(t)) for t in th[1:-1:7]]
    r.check(max(sp) - min(sp) < 1e-6 * max(sp), f"the pattern changes at a steady rate "
            f"(speed^2 {min(sp):.6f} .. {max(sp):.6f})")
    r.check(all(abs(timeline.weights(float(j)).get(j, 0.0)) < 1e-15 for j in range(5)),
            "a realization begins with weight zero")
    w_end = [timeline.weights(j + 3.0 - 1e-6).get(j, 0.0) for j in range(3)]
    r.check(max(w_end) < 1e-10, f"and ends at rest ({max(w_end):.1e})")
    r.check(near(timeline.pattern_correlation(timeline.HALF_LIFE_HOPS), 0.5, 1e-6),
            f"half the pattern renewed after {timeline.HALF_LIFE_HOPS:.4f} hops")
    r.check(timeline.pattern_correlation(3.0) < 1e-12, "nothing of it left after three hops")
    r.close()

    r = Rule("T2 the blend is standard normal: a layer covers what the forecast says, at any moment")
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    t0 = datetime(2026, 9, 10, 0, tzinfo=timezone.utc)
    worst, waves = 0.0, False
    for g, sp_, v_, cov in (("Sc", "str", [], 0.35), ("Ac", "str", [], 0.7), ("Cu", "hum", [], 0.12),
                            ("Cc", None, [], 0.5), ("St", "neb", ["un"], 0.16), ("Sc", "str", ["un"], 0.6)):
        d = scene.deck_from_spec(CloudSpec(g, sp_, v_), snd, coverage=cov, seed=7)
        sky = timeline.Sky(manual=[d], threaded=False, n=256, seed=3)
        for k in range(4):
            sky.update(t0 + timedelta(minutes=17 * k), 0.0, sync=True)
            st = sky.states[0]
            ex = max(st.extents)
            # random points (none on a texel's edge) over a tile, and over
            # many wavelengths of undulatus waves
            rng = np.random.default_rng(k)
            span = max(ex, 12 * 2 * math.pi / max(math.hypot(st.wave[1], st.wave[2]), 1e-9)
                       if st.wave[0] > 0 else ex)
            X, Y = rng.uniform(0, span, 60000), rng.uniform(0, span, 60000)
            area = float((sky.frac_at(st, X, Y) > 0.5).mean())
            worst = max(worst, abs(area - st.deck.coverage))
            waves = waves or st.wave[0] > 0
    r.check(worst < 0.03, f"the area over the cover's threshold is the cover within {worst:.3f}")
    r.check(waves, "undulatus waves were among the layers tried")
    r.close()

    sky, t0 = _sky_hands()
    r = Rule("T3 no teleport: everything the renderer reads is continuous across the hours")
    found = []
    probes = [(float(k), "hour") for k in range(1, 9)] + [(k + 0.5, "half hour") for k in range(9)]
    for x, what in probes:
        res = {}
        for eps in (2.0, 0.2):
            hh = eps / 2.0 / 3600.0
            pa, oa, ta = _probe_state(sky, t0 + timedelta(hours=x - hh))
            pb, ob, tb = _probe_state(sky, t0 + timedelta(hours=x + hh))
            res[eps] = (pa, pb, ta, tb, oa, ob)
        big = {k: dv for k, dv in _jumps(*res[2.0][:4], None)}
        for g, rr, c, dv, a_, b_ in _jumps(*res[0.2][:4], big):
            found.append(f"{what} {x:g}: group {g} row {rr}.{c} {a_:.4g} -> {b_:.4g}")
        for k_, nm in enumerate(("halo", "corona")):
            ds = abs(res[0.2][5][k_] - res[0.2][4][k_])
            db = abs(res[2.0][5][k_] - res[2.0][4][k_])
            if ds > 1e-7 and ds > 0.5 * db:
                found.append(f"{what} {x:g}: {nm} {res[0.2][4][k_]:.4g} -> {res[0.2][5][k_]:.4g}")
    r.check(not found, "; ".join(found[:6]) or "")
    names = set()
    for k in range(10):
        sky.update(t0 + timedelta(hours=k), 0.0, sync=True)
        names |= {s.deck.spec.abbrev() for s in sky.states}
    r.check({"Ci fib", "As", "Cu hum"} <= names and any("cas" in n for n in names)
            and any(n.startswith("Cc") and "un" in n for n in names),
            f"the hours hold what the test is about: {sorted(names)}")
    r.close()

    r = Rule("T4 each number passes through the hours smoothly and stays between them")
    worst_kink, worst_over, n_nodes = 0.0, 0.0, 0
    hh = 1e-3
    for tr in list(sky.tracks.values()):
        for i in range(tr.first - 1, tr.last + 2):
            dm = [tr.deck_at(i - 2 * hh), tr.deck_at(i - hh), tr.deck_at(float(i))]
            dp_ = [tr.deck_at(float(i)), tr.deck_at(i + hh), tr.deck_at(i + 2 * hh)]
            k0, k1, k2 = tr.key(i - 1), tr.key(i), tr.key(i + 1)
            if None in dm or None in dp_ or None in (k0, k1, k2):
                continue
            n_nodes += 1
            for f_ in ("coverage", "base_m", "top_m", "sigma_e", "wind_u", "element_m"):
                a0, a1, a2 = (getattr(d, f_) for d in dm)
                b0, b1, b2 = (getattr(d, f_) for d in dp_)
                sl = (3 * a2 - 4 * a1 + a0) / (2 * hh)          # one-sided, second order
                sr = (-3 * b0 + 4 * b1 - b2) / (2 * hh)
                y0, y1, y2 = (getattr(k, f_) for k in (k0, k1, k2))
                sc = max(abs(y1 - y0), abs(y2 - y1), 1e-6 * max(abs(y1), 1.0))
                worst_kink = max(worst_kink, abs(sl - sr) / sc)
        for i in range(tr.first - 1, tr.last + 1):
            a, b = tr.key(i), tr.key(i + 1)
            if a is None or b is None:
                continue
            for f in np.linspace(0.0, 1.0, 21):
                d = tr.deck_at(i + f)
                for f_ in ("coverage", "base_m", "top_m"):
                    lo, hi = sorted((getattr(a, f_), getattr(b, f_)))
                    v = getattr(d, f_)
                    worst_over = max(worst_over, lo - v, v - hi)
    r.check(n_nodes > 20 and worst_kink < 1e-3,
            f"left and right rates agree at {n_nodes} hours (worst {worst_kink:.1e} of the hour's change)")
    r.check(worst_over <= 1e-9, f"no overshoot between two hours ({worst_over:.1e})")
    r.close()

    r = Rule("T5 drift and renewal are integrals of the wind and the rate: smooth through the hours")
    bad = []
    for tr in list(sky.tracks.values()):
        for i in range(tr.first, tr.last + 1):
            # 1e-9 h is 3.6 us: 0.2 mm of drift at 50 m/s
            a0, a1 = tr.theta_off(i - 1e-9), tr.theta_off(i + 1e-9)
            if abs(a1[0] - a0[0]) > 1e-6 or max(abs(a1[1] - a0[1]), abs(a1[2] - a0[2])) > 1e-3:
                bad.append(f"track {tr.id} hour {i}: {a0} -> {a1}")
            x = i + 0.37
            dt = 1e-3
            p, q = tr.theta_off(x - dt), tr.theta_off(x + dt)
            d = tr.deck_at(x) or tr.keys[tr.first]
            vx = (q[1] - p[1]) / (2 * dt * 3600.0)
            still = 1.0 - min(max(max(d.lens, d.roll), 0.0), 1.0)
            want = tr._wind(math.floor(x))[0] * (1 - 0.37) + tr._wind(math.floor(x) + 1)[0] * 0.37
            if abs(vx - want) > 1e-3 * max(abs(want), 1.0):
                bad.append(f"track {tr.id}: drift {vx:.3f} m/s, wind {want:.3f} (still {still:.2f})")
    r.check(not bad, "; ".join(bad[:4]))
    r.close()

    r = Rule("T6 the same moment gives the same sky, however the clock got there")
    target = t0 + timedelta(hours=4, minutes=23)
    a, _ = _sky_hands()
    a.update(target, 0.0, sync=True)
    b, _ = _sky_hands()
    for m in range(0, 7 * 60, 20):
        b.update(t0 + timedelta(hours=7) - timedelta(minutes=m), 0.0, sync=True)
    b.update(target, 0.0, sync=True)

    def summary(s):
        return sorted((round(st.deck.base_m, 6), round(st.deck.coverage, 9), round(st.theta, 9),
                       tuple(round(o, 6) for off in st.offsets for o in off),
                       tuple(round(w, 9) for w in st.weights)) for st in s.states)
    r.check(summary(a) == summary(b), "straight there, or down from three hours later: the same layers, "
            "patterns and drift")
    r.close()

    r = Rule("T7 a new forecast that says the same carries the sky on unchanged")
    s1, _ = _sky_hands()
    when = t0 + timedelta(hours=3, minutes=40)
    s1.update(when, 0.0, sync=True)
    before = summary(s1)
    groups = sorted((st.group, round(st.theta, 9)) for st in s1.states)
    s1.set_source(forecast=s1.forecast, merge=True)
    s1.update(when, 0.0, sync=True)
    r.check(summary(s1) == before, "the same layers, patterns, drift and cover")
    r.check(sorted((st.group, round(st.theta, 9)) for st in s1.states) == groups and not s1.fading,
            "in the same renderer slots, nothing fading")
    s2, _ = _sky_hands()
    later = when + timedelta(hours=3, minutes=10)
    s1.update(later, 0.0, sync=True)
    s2.update(later, 0.0, sync=True)
    r.check(summary(s1) == summary(s2), "and three hours on, the layers born since are the ones "
            "the forecast would have had anyway")
    r.close()

    r = Rule("T8 the regional maps carry a band of cloud along the wind between the hours")
    ser, t0r = _band_series()
    rm = region.RegionMaps()
    xs = np.linspace(-130000.0, 130000.0, 521)

    def centre(when):
        rm.build_series(ser, when, None, 5000.0, high_wind=lambda k: (0.0, 0.0))
        rm.set_time(rm.f)
        raw = (1.0 - rm.f) * rm._bil(rm.raw_a[..., 0], xs - rm.shift_a[0, 0], 0.0 * xs) + \
            rm.f * rm._bil(rm.raw_b[..., 0], xs - rm.shift_b[0, 0], 0.0 * xs)
        return float(xs[int(np.argmax(raw))]), float(raw.max())
    c0, m0 = centre(t0r + timedelta(hours=1))
    c5, m5 = centre(t0r + timedelta(hours=1, minutes=30))
    c1, m1 = centre(t0r + timedelta(hours=2))
    r.check(near(c5, 0.5 * (c0 + c1), 2500.0), f"half way between the hours the band is half way "
            f"({c0 / 1000:.0f}, {c5 / 1000:.0f}, {c1 / 1000:.0f} km)")
    r.check(m5 > 0.8 * 0.5 * (m0 + m1), f"one band, not two faded copies (peak {m5:.0f}% vs {m0:.0f}%, {m1:.0f}%)")
    e1, e2 = rm.sample(np.array([0.0]), np.array([0.0]))
    r.check(near(float(e2[0, 0]), 1.0, 1e-6) and near(float(e2[0, 1]), 1.0, 1e-6),
            "the cover ratio here is one")
    gx = np.linspace(-100000.0, 100000.0, 41)
    GX, GY = np.meshgrid(gx, gx)
    rm.build_series(ser, t0r + timedelta(hours=2) - timedelta(seconds=0.5), None, 5000.0,
                    high_wind=lambda k: (0.0, 0.0))
    wa, ca = rm.sample(GX, GY)
    rm.build_series(ser, t0r + timedelta(hours=2) + timedelta(seconds=0.5), None, 5000.0,
                    high_wind=lambda k: (0.0, 0.0))
    wb, cb = rm.sample(GX, GY)
    r.check(float(np.abs(ca - cb).max()) < 2e-3 and float(np.abs(wa - wb).max()) < 1.0,
            f"continuous through the hour: cover ratio {float(np.abs(ca - cb).max()):.1e}, "
            f"warp {float(np.abs(wa - wb).max()):.2f} m in a second")
    r.close()

    r = Rule("T9 the atmosphere between two hours is the hours' mix; categories from the nearer")
    t9 = datetime(2026, 9, 25, 5, tzinfo=timezone.utc)
    hq = {"time": [t9.strftime("%Y-%m-%dT%H:00"), (t9 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")],
          "temperature_2m": [31.9, 33.1], "dew_point_2m": [23.2, 22.0], "relative_humidity_2m": [60, 52],
          "surface_pressure": [1004.5, 1003.9], "wind_speed_10m": [0.86, 2.0],
          "wind_direction_10m": [80, 120], "weather_code": [3, 61], "visibility": [20000.0, 8000.0]}
    for p, (t, rh, cc, ws_, wd, z) in KL_0925_05.items():
        later = (t + 1.5, max(rh - 12, 0), cc, ws_ + 2.0, wd + 40, z + 15)
        for name, va, vb in zip(("temperature", "relative_humidity", "cloud_cover", "wind_speed",
                                 "wind_direction", "geopotential_height"), (t, rh, cc, ws_, wd, z), later):
            hq[f"{name}_{p}hPa"] = [va, vb]
    fc9 = forecast.Forecast({"elevation": 62.0, "hourly": hq}, 3.14, 101.69)
    sa, sb = fc9.sounding_hour(0), fc9.sounding_hour(1)
    m = fc9.sounding_at(t9 + timedelta(minutes=18))
    pa_ = {float(l.p): l for l in sa.levels}
    pb_ = {float(l.p): l for l in sb.levels}
    errs, n9 = [], 0
    for l in m.levels:
        p = float(l.p)
        if p in pa_ and p in pb_ and p in forecast.PSET:
            n9 += 1
            for k in ("t", "rh", "z", "u", "v"):
                want = 0.7 * getattr(pa_[p], k) + 0.3 * getattr(pb_[p], k)
                if abs(getattr(l, k) - want) > 1e-6 * max(1.0, abs(want)):
                    errs.append(f"{p} {k}")
    r.check(n9 >= 15 and not errs, f"linear in time on {n9} pressure levels ({errs[:4]})")
    r.check(m.surface.get("weather_code") == 3 and near(m.surface.get("visibility", 0.0), 16400.0, 1e-6),
            f"weather code from the nearer hour, visibility mixed "
            f"({m.surface.get('weather_code')}, {m.surface.get('visibility')})")
    r.close()

    r = Rule("T15 a broken forecast answer: hours stay in place, non-numbers are gaps, short gaps "
             "are filled")
    t15 = datetime(2026, 9, 25, 0, tzinfo=timezone.utc)
    hours = [0, 1, 2, 4, 5, 5, 6, 7]                 # hour 3 skipped, hour 5 given twice
    hb = {"time": [(t15 + timedelta(hours=k)).strftime("%Y-%m-%dT%H:00") for k in hours],
          "temperature_2m": [31.9] * 8, "dew_point_2m": [23.2] * 8, "relative_humidity_2m": [60] * 8,
          "surface_pressure": [1004.5] * 8, "wind_speed_10m": [0.9] * 8, "wind_direction_10m": [80] * 8,
          "weather_code": [3, 3, 61, 61, 2, 2, 2, 2]}
    for p, (t, rh, cc, ws_, wd, z) in KL_0925_05.items():
        for name, v in (("temperature", t), ("relative_humidity", rh), ("cloud_cover", cc),
                        ("wind_speed", ws_), ("wind_direction", wd), ("geopotential_height", z)):
            col = [v + (0.5 * k if name == "temperature" else 0.0) for k in hours]
            if name == "wind_speed":
                col[2] = float("inf")                # hour 2: infinite winds
            if name == "temperature" and p == 500:
                col[6] = float("nan")                # hour 6: a NaN at 500 hPa
            hb[f"{name}_{p}hPa"] = col
    fc15 = forecast.Forecast({"elevation": 62.0, "hourly": hb}, 3.14, 101.69)
    r.check(len(fc15.times) == 8 and all(fc15.times[k] == t15 + timedelta(hours=k) for k in range(8)),
            f"eight hours, each in its place ({len(fc15.times)})")
    t500 = []
    for k in range(8):
        lv = [l for l in fc15.sounding_hour(k).levels if l.p == 500.0]
        t500.append(lv[0].t if lv else None)
    want = [KL_0925_05[500][0] + 0.5 * k for k in range(8)]
    r.check(all(v is not None and abs(v - w) < 1e-9 for v, w in zip(t500, want)),
            f"the skipped hour and the NaN are filled in time: 500 hPa {t500}")
    finite = all(math.isfinite(x) for k in range(8) for l in fc15.sounding_hour(k).levels
                 for x in (l.t, l.rh, l.z, l.u, l.v))
    r.check(finite and fc15.filled > 0, f"every value a number ({fc15.filled} filled)")
    r.close()

    r = Rule("T16 a tile of a few large elements: the cover threshold is set for the blend's own "
             "spread over the tile")
    snd16 = sounding.standard_sounding(49.0, 2.5, profile="humid-deep")
    t16 = datetime(2026, 9, 30, 0, tzinfo=timezone.utc)
    errs16, sd_err = [], 0.0
    # a held sky counts its hours from the wall clock's hour, so the patterns
    # it shows depend on when the test runs (the statistic ranged 0.0066 to
    # 0.0125 over a day): six fixed epochs, the same whenever it runs
    for g, sp_, cov, seed, ep in [(*c, ep) for ep in range(6)
                                  for c in (("Cs", "neb", 0.7, 3), ("Ns", None, 0.8, 6),
                                            ("As", None, 0.4, 5), ("St", "neb", 0.6, 8))]:
        d = scene.deck_from_spec(CloudSpec(g, sp_), snd16, base_m=3000.0, coverage=cov,
                                 thickness_m=900.0, seed=seed)
        sky16 = timeline.Sky(manual=[d], threaded=False, n=256, seed=seed)
        sky16._epoch0 = t16.timestamp() - 3600.0 * (1 + 4 * ep) - 1234.0
        for k in range(8):
            sky16.update(t16 + timedelta(minutes=13 * k), 0.0, sync=True)
            st = sky16.states[0]
            tr = sky16.tracks[st.track]
            showing = [s for s in range(3) if st.weights[s] > 1e-7]
            ex, n = st.extents[showing[0]], sky16.n
            # the blend itself over its tile (the realizations of a layer
            # that does not change share one)
            z = sum(w * sky16.cache[(tr.seed, j)]["pack"][..., 0].astype("f8")
                    for j, w in timeline.weights(st.theta).items() if w > 1e-7)
            sd_err = max(sd_err, abs(float(z.std()) - st.spread))
            # the share of the tile that is cloud, at every texel's centre
            c = (np.arange(n) + 0.5) / n * ex
            X, Y = np.meshgrid(c, c)
            ox, oy = st.offsets[showing[0]]
            f = sky16.frac_at(st, X + ox, Y + oy)
            errs16.append(abs(float((f > 0.5).mean()) - st.deck.coverage))
    r.check(sd_err < 1e-3, f"the spread is the blend's standard deviation over the tile within "
            f"{sd_err:.1e}")
    r.check(float(np.mean(errs16)) < 0.012,
            f"the share of each tile that is cloud is the cover within {np.mean(errs16):.4f} on "
            f"average, worst {max(errs16):.3f} (a threshold for spread 1: 0.018, worst 0.077)")
    # castellanus realizations share the rows their turrets stand on - by
    # design, and not as a normal variable would: that is left out
    d = scene.deck_from_spec(CloudSpec("Ac", "cas"), snd16, base_m=4400.0, coverage=0.35,
                             thickness_m=600.0, seed=7, turret_rise_m=900.0)
    sky16 = timeline.Sky(manual=[d], threaded=False, n=256, seed=3)
    sky16.update(t16, 0.0, sync=True)
    r.check(abs(sky16.states[0].spread - 1.0) < 1e-12 and any(sky16.states[0].has_turret),
            f"castellanus: the shared rows are not counted ({sky16.states[0].spread})")
    r.close()

    r = Rule("T10 perlucidus: the gaps lie on the walls between the elements")
    f = noise.deck_fields(256, 12000.0, 1000.0, cellularity=1.0, gaps="perlucidus", warp=0.0, seed=9)
    net = f["gap"] < noise.gap_star("perlucidus")
    r.check(near(float(net.mean()), noise.GAP_SHARE["perlucidus"], 0.01),
            f"the net takes {100 * net.mean():.1f}% of the area")
    zin, zout = float(f["z"][net].mean()), float(f["z"][~net].mean())
    r.check(zin < zout - 0.5, f"between the elements, where the pattern is low (z {zin:.2f} vs {zout:.2f})")
    r.close()

    r = Rule("T11 past the forecast's last hour the sky holds; it does not empty")
    sky2, t0h = _sky_hands()
    sky2.update(t0h + timedelta(hours=9), 0.0, sync=True)
    at_end = sorted((s.deck.spec.abbrev(), round(s.deck.coverage, 9)) for s in sky2.states)
    sky2.update(t0h + timedelta(hours=14, minutes=30), 0.0, sync=True)
    later = sorted((s.deck.spec.abbrev(), round(s.deck.coverage, 9)) for s in sky2.states)
    r.check(at_end and later == at_end, f"the last hour's layers stay ({at_end} -> {later})")
    th_a = [s.theta for s in sky2.states]
    r.check(all(t > 0 for t in th_a), "and go on living")
    r.close()

    r = Rule("T12 halos and coronae fade as a layer changes its name; none switches on or off")
    snd_o = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    cs = scene.deck_from_spec(CloudSpec("Cs", "neb"), snd_o, base_m=8500.0, coverage=0.9, seed=3)
    as_ = scene.deck_from_spec(CloudSpec("As", None), snd_o, base_m=6500.0, coverage=0.9,
                               thickness_m=2500.0, seed=3)
    for d in (cs, as_):
        d.halo_w, d.corona_w, d.icecorona_w = d.optic_kind()
    hs = [gl_sky.deck_optics(timeline.interp_deck(cs, as_, f))[0] for f in np.linspace(0, 1, 2001)]
    step = max(abs(a - b) for a, b in zip(hs, hs[1:]))
    r.check(hs[0] > 0.05 and hs[-1] == 0.0 and step < 2e-3,
            f"cirrostratus to altostratus: halo {hs[0]:.3f} -> {hs[-1]:.3f}, "
            f"largest change in 1/2000 h {step:.1e}")
    taus = np.linspace(0.0, 10.0, 4001)
    d = scene.deck_from_spec(CloudSpec("Cs", "neb"), snd_o, base_m=8500.0, coverage=0.9, seed=3)
    vals = []
    for tau in taus:
        d.sigma_e = tau / d.thickness
        vals.append(gl_sky.deck_optics(d)[0])
    r.check(max(abs(a - b) for a, b in zip(vals, vals[1:])) < 1e-3,
            "a thickening veil loses its halo gradually")
    r.close()

    r = Rule("T13 a layer passing an étage boundary takes the new étage's cover gradually")
    ws = [region.etage_weights(z) for z in np.linspace(0.0, 12000.0, 12001)]
    r.check(all(near(sum(w), 1.0, 1e-12) and min(w) >= 0.0 for w in ws), "the weights sum to one")
    r.check(max(max(abs(a - b) for a, b in zip(u, v)) for u, v in zip(ws, ws[1:])) < 5e-3,
            "and change by little for a metre of height")
    r.check(ws[1000] == (1.0, 0.0, 0.0) and ws[4000] == (0.0, 1.0, 0.0) and ws[9000] == (0.0, 0.0, 1.0),
            "each étage's own cover inside it")
    r.close()

    r = Rule("T14 above the weather nothing pops: PSC grow as the stratosphere cools, NLC forms "
             "hold through the night")
    uc = upper.UpperClouds()
    snd_p = sounding.standard_sounding(69.0, 19.0, datetime(2027, 1, 10, 12, tzinfo=timezone.utc))
    amounts = []
    for t_s in np.linspace(-80.0, -90.0, 201):
        for lv in snd_p.levels:
            if lv.p <= 60.0:
                lv.t = float(t_s)
        uc.decide(datetime(2027, 1, 10, 12, tzinfo=timezone.utc), 69.0, snd_p)
        amounts.append((uc.nacreous, uc.nat))
    r.check(max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(amounts, amounts[1:])) < 0.06,
            "ice and nitric acid clouds change by little for 0.05 C")
    r.check(amounts[-1][0] == 1.0 and amounts[0] == (0.0, 1.0), "all nitric acid at -80 C, all ice at -90 C")
    types = []
    for hour in (19, 21, 23, 1, 3):
        day = 25 if hour > 12 else 26
        uc.decide(datetime(2026, 6, day, hour, tzinfo=timezone.utc), 58.0, None, lon=0.0)
        types.append(uc.nlc_types)
    r.check(len(set(types)) == 1, f"the same noctilucent forms all night ({types})")
    r.close()


# ----------------------------------------------------------------- species --

#: Open-Meteo at Kuala Lumpur (3.14 N 101.69 E, grid 62 m), 25 Sep 2026
#: 05 UTC, as the program fetched it: p: (T C, RH %, cloud cover %, wind m/s,
#: direction deg, geopotential height m).  The sky that drew a dense field of
#: Altocumulus castellanus opacus 1.6 km deep.
KL_0925_05 = {
    1000: (32.4, 52, 0, 1.65, 166, 99), 975: (30, 58, 0, 1.68, 158, 325.47),
    950: (27.6, 64, 0, 1.73, 150, 557), 925: (25.3, 71, 0, 1.77, 145, 793),
    900: (22.9, 79, 0, 1.73, 138, 1034), 850: (19.9, 70, 0, 1.62, 39, 1530),
    800: (15.9, 84, 21, 1.19, 339, 2050), 700: (10.8, 76, 9, 3.89, 257, 3177),
    600: (5, 64, 0, 3.53, 285, 4449), 500: (-5.6, 84, 28, 4.48, 312, 5910),
    400: (-15.6, 18, 0, 4.07, 125, 7628.4), 300: (-30, 58, 0, 11.08, 97, 9740.32),
    250: (-40, 49, 0, 13.11, 83, 11011.43), 200: (-52.5, 38, 0, 18.68, 55, 12493.02),
    150: (-67, 54, 0, 23.63, 74, 14289.55), 100: (-76.5, 20, 0, 6.75, 237, 16662.5),
    70: (-69.5, 4, 0, 3.07, 311, 18748.64), 50: (-64, 0, 0, 15.09, 239, 20779.31),
    30: (-54, 0, 0, 13.48, 279, 23990.66)}


def _kl_sounding():
    when = datetime(2026, 9, 25, 5, tzinfo=timezone.utc)
    h = {"time": [when.strftime("%Y-%m-%dT%H:00")], "temperature_2m": [31.9],
         "dew_point_2m": [23.2], "relative_humidity_2m": [60], "surface_pressure": [1004.5],
         "wind_speed_10m": [0.86], "wind_direction_10m": [80]}
    for p, (t, rh, cc, ws, wd, z) in KL_0925_05.items():
        for name, val in (("temperature", t), ("relative_humidity", rh), ("cloud_cover", cc),
                          ("wind_speed", ws), ("wind_direction", wd), ("geopotential_height", z)):
            h[f"{name}_{p}hPa"] = [val]
    orig = sounding._request
    sounding._request = lambda url, timeout: {"elevation": 62.0, "hourly": h}
    try:
        return sounding.fetch_open_meteo(3.14, 101.69, when)
    finally:
        sounding._request = orig


def test_species():
    kl = _kl_sounding()
    r = Rule("Q1 a layer is as deep as its cloudy part, not the gap to the clear levels")
    # One cloudy level (500 hPa, 28%) between clear ones 1.5 and 1.7 km away:
    # interpolating the cloud fraction itself spread it over 4710-7322 m.
    mid = [L for L in kl.cloud_layers() if 4000 < L.z_peak < 7000]
    r.check(len(mid) == 1, f"one middle layer ({len(mid)})")
    if mid:
        L = mid[0]
        r.check(L.base < 5910.0 < L.top and L.depth < 1000.0,
                f"middle layer {L.base:.0f}-{L.top:.0f} m round the 5910 m level (old: 4710-7322)")
        rows = kl._level_clouds()
        for z in (L.base, L.top):
            c = kl._cover_between(rows, z)
            r.check(abs(c - 0.5 * L.cover) < 0.02, f"at {z:.0f} m the fraction is {c:.3f}, half "
                    f"the peak {L.cover:.2f}")
    r.close()

    r = Rule("Q2 Open-Meteo's level cloud cover is Sundqvist's from RH, as the program reads it")
    # (RH, cloud cover) pairs as Open-Meteo served them, Kuala Lumpur 2025
    for p, rh, cc in ((500, 0.84, 0.28), (500, 0.97, 0.69), (500, 0.93, 0.51), (500, 0.79, 0.15),
                      (800, 0.84, 0.21), (700, 0.76, 0.09), (500, 0.70, 0.0), (600, 0.64, 0.0)):
        got = sounding.cloud_fraction(rh, sounding.rhc_open_meteo(p), "sundqvist")
        r.check(abs(got - cc) < 0.025, f"{p} hPa, RH {rh:.2f}: {got:.3f} not {cc:.2f}")
    r.close()

    r = Rule("Q3 castellanus/floccus only as often as observers report it")
    decks = scene.diagnose(kl)
    ac = [d for d in decks if d.spec.genus == "Ac"]
    r.check(len(ac) == 1 and ac[0].spec.species not in ("cas", "flo"),
            f"KL 25 Sep 05 UTC: {[d.spec.abbrev() for d in ac]} (was Ac cas op)")
    m = scene.SPECIES_DATA["Ac cas/flo"]
    xs = [-1.5, -0.5, 0.0, 0.5, 1.0, 2.0]
    for clim in ("tropical", "temperate summer", "temperate winter"):
        ps = [scene.reported_odds(m, x, clim) for x in xs]
        r.check(all(0.0 <= p <= 1.0 for p in ps), f"{clim}: probabilities in 0..1")
        r.check(all(b > a for a, b in zip(ps, ps[1:])), f"{clim}: more likely the less stable the "
                f"layer ({[round(p, 4) for p in ps]})")
    r.check(scene.reported_odds(m, 1.0, "tropical") < scene.reported_odds(m, 1.0, "temperate summer"),
            "rarer in the tropics than in a temperate summer, as reported")
    r.check(scene.climate_of(3.1, 9) == "tropical" and scene.climate_of(50.0, 7) == "temperate summer"
            and scene.climate_of(-35.0, 7) == "temperate winter", "the climates")
    cc = scene.SPECIES_DATA["Cc"]
    pc = [scene.reported_odds(cc, t, "temperate summer") for t in (-55.0, -40.0, -25.0, -12.0)]
    r.check(all(b > a for a, b in zip(pc, pc[1:])) and pc[-1] < 0.1,
            f"Cirrocumulus likelier the warmer the ice layer, never likely ({[round(p, 4) for p in pc]})")
    r.check(scene.reported_odds(cc, -25.0, "tropical") < pc[2], "and rarer in the tropics")
    r.check(bool(ac) and ("castellanus or floccus" in ac[0].odds),
            "the panel shows the odds")
    r.close()

    r = Rule("Q4 opacus only for an extensive sheet")
    r.check(bool(ac) and "op" not in ac[0].spec.varieties,
            f"a 28% field of elements is not opacus ({ac[0].spec.abbrev() if ac else '-'})")
    spec = CloudSpec("Ac", "str")
    ctx = dict(tau=30.0, ri=5.0, wind=5.0, shear=2.0, cover=0.9, sub_rh=70.0, thick=900.0,
               t_base=-5.0, dir_shear=5.0)
    reasons = []
    scene._attach_features(spec, kl, ctx, reasons)
    r.check("op" in spec.varieties, f"a 90% sheet of optical depth 30 is opacus ({spec.abbrev()})")
    r.close()

    r = Rule("Q5 no Stratus far above the ground")
    low = [d for d in decks if d.base_m < 2000.0 and d.spec.genus in ("St", "Sc")]
    r.check(bool(low) and all(d.spec.genus == "Sc" for d in low),
            f"the 1.9 km layer is {[d.spec.abbrev() for d in low]} (was St fra op at 1.6 km)")
    r.check(scene.stratus_share(100.0) > 0.1 and scene.stratus_share(1500.0) < 0.02,
            "Stratus is reported for low bases, hardly ever above 1 km")
    z0 = kl.levels[0].z

    def low(agl, lapse_in, cover=0.9):
        L = sounding.CloudLayer(base=z0 + agl, top=z0 + agl + 300.0, cover=cover,
                                z_peak=z0 + agl + 150.0, edge_base=z0 + agl, edge_top=z0 + agl + 300.0,
                                n_levels=1)
        ph = dict(lapse_in=lapse_in, lapse_above=-1.0, cape=0.0, el=None, dthetae=1.0, rh_below=95.0)
        return scene._low_species(kl, L, ph, 3.0, cover)[0].genus
    r.check(low(80.0, -0.5) == "St", "saturation at the ground under a stable layer is Stratus")
    r.check(low(1500.0, -0.5) == "Sc" and low(700.0, -0.5, 0.3) == "Sc",
            "the same layer 0.7 or 1.5 km up is Stratocumulus")
    r.close()

    r = Rule("Q6 castellanus turrets stand on the common layer, on lines along the shear")
    d = scene.deck_from_spec(CloudSpec("Ac", "cas"), kl, base_m=5400.0, coverage=0.3,
                             thickness_m=650.0, seed=7, turret_rise_m=900.0)
    from . import gl_sky
    mp = gl_sky.build_deck_map(d)
    frac, tops = mp[..., 0], mp[..., 1]
    tb = d.turret
    turret = tops > tb + 1e-6
    # the common layer: the 650 m asked, as deep as its elements allow (half
    # their width); the turrets keep their tops, 900 m above the layer asked
    layer6 = min(650.0, max(0.5 * d.element_w_m, 50.0))
    r.check(abs(d.thickness - 1550.0) < 1.0 and abs(tb - layer6 / 1550.0) < 1e-3,
            f"layer {layer6:.0f} m + turrets to 1550 m: depth {d.thickness:.0f} m, layer top {tb:.3f}")
    r.check(0.002 < float(turret.mean()) < 0.2, f"turrets on {100 * turret.mean():.1f}% of the map")
    r.check(bool((frac[turret] > 0.5).all()), "every turret stands on cloud")
    r.check(bool((np.abs(tops[~turret] - tb) < 1e-6).all()), "elsewhere the layer's top is common")
    r.check(float(tops.max()) <= 1.0 + 1e-6, f"no turret above the deck ({tops.max():.3f})")
    ex = gl_sky.deck_map_extent(d)
    lines = noise.line_mask(mp.shape[0], ex, 4.0 * d.element_m, d.aniso_angle, (d.seed & 0x7FFFFFFF) + 61)
    on = float((lines[turret] > 0.3).mean())
    r.check(on > 0.9, f"{100 * on:.0f}% of the turrets' area is on the lines")
    str_d = scene.deck_from_spec(CloudSpec("Ac", "str"), kl, base_m=5400.0, coverage=0.3,
                                 thickness_m=650.0, seed=7)
    r.check(str_d.turret == 0.0
            and abs(str_d.thickness - min(650.0, max(0.5 * str_d.element_w_m, 50.0))) < 1.0,
            f"stratiformis has no turrets and the layer's own depth (as its elements allow): "
            f"{str_d.thickness:.0f} m")
    r.close()


def test_twilight():
    from . import gl_sky

    r = Rule("T17 through twilight the light does not jump: refraction, the exposure and the "
             "Moon's light are continuous in the Sun's and the Moon's elevation")
    fine = np.arange(-20.0, 10.0, 0.001)
    rf = np.array([astro.refraction_deg(a) for a in np.arange(-5.0, 3.0, 0.001)])
    r.check(float(np.abs(np.diff(rf)).max()) < 0.002,
            f"refraction changes by at most {np.abs(np.diff(rf)).max():.4f} deg per 0.001 deg")
    R17 = gl_sky.SkyRenderer.__new__(gl_sky.SkyRenderer)
    R17.exposure_bias, R17.adaptive = 0.0, True
    for moon_alt in (-30.0, 20.0):
        worst, at = 0.0, None
        prev = None
        for s in fine:
            R17._ev_adapted = None
            base = R17.auto_exposure(float(s), moon_alt, 1.0)
            # an exposure metered far from the day's (a dim twilight frame)
            R17._ev_adapted = base * 20.0
            ev = math.log(R17.auto_exposure(float(s), moon_alt, 1.0))
            if prev is not None and abs(ev - prev) > worst:
                worst, at = abs(ev - prev), float(s)
            prev = ev
        r.check(worst < 0.01, f"Moon at {moon_alt:g} deg: the exposure changes by at most "
                f"{100 * (math.exp(worst) - 1):.1f} % per 0.001 deg of the Sun (at {at:.3f})")

    class G17:
        pass
    sw_bad, prev_src = [], None
    for moon_alt in (-5.0, 0.5, 20.0):
        for s in np.arange(-14.0, 2.0, 0.001):
            g = G17()
            g.sun_alt, g.moon_alt, g.moon_illum = float(s), moon_alt, 1.0
            g.sun_dir, g.moon_dir = (0.0, math.sin(math.radians(s)), 1.0), (0.0, 0.5, -1.0)
            d_, col, is_sun = gl_sky.SkyRenderer.light_source(g)
            if prev_src is not None and is_sun != prev_src[0]:
                # the Moon's side of the switch: its light, and the Sun there
                moon_col, s_moon = (max(col), float(s)) if not is_sun else (prev_src[1], prev_src[2])
                if moon_col > 1e-6 or s_moon > -6.0:
                    sw_bad.append(f"Sun {s_moon:.3f}, Moon {moon_alt:g}: the Moon's light {moon_col:.1e}")
            prev_src = (is_sun, max(col), float(s))
        prev_src = None
    r.check(not sw_bad, "the clouds' light passes from the Sun to the Moon only where the Sun "
            "lights no cloud and the Moon's light is nothing yet: " + "; ".join(sw_bad[:3]))
    r.close()


# ------------------------------------------------------------------ optics --

def test_optics():
    from . import mie, gl_sky
    r = Rule("I1 Mie series (BHMIE) against reference values")
    # reference values from miepython 3.3 (independent implementation)
    for x, m, qs_ref, g_ref in ((5.213, 1.55, 3.104996, 0.633104),
                                (100.0, 1.333, 2.119968, 0.875782),
                                (5000.0, 1.3436, 2.00817, 0.87989)):
        mu = np.cos(np.linspace(0.0, math.pi, 3001))
        s1, s2, qsca, g = mie.bhmie(x, complex(m, 0.0), mu)
        r.check(near(qsca, qs_ref, 3e-4 * qs_ref), f"x={x}: Qsca {qsca:.6f} not {qs_ref}")
        r.check(near(g, g_ref, 3e-4), f"x={x}: g {g:.6f} not {g_ref}")
    th = np.linspace(0.0, math.pi, 200001)
    s1, s2, qsca, g = mie.bhmie(100.0, complex(1.333, 0.0), np.cos(th))
    p = mie.phase_from_s(s1, s2, 100.0, qsca)
    w = 2 * math.pi * np.sin(th)
    r.check(near(float(np.trapezoid(p * w, th)), 1.0, 1e-5), "phase function normalised")
    r.check(near(float(np.trapezoid(p * np.cos(th) * w, th)), g, 1e-5), "g from the phase function")
    r.check(near(float(mie.water_index(0.5893)), 1.3333, 0.0005), "water index at the sodium D line")
    r.close()

    r = Rule("I2 the rain phase function holds the rainbows")
    lut = mie.rain_phase_lut()
    thd = np.radians(np.arange(120.0, 180.0, 0.02))
    pr = mie.lookup(lut, thd)
    prim, sec = [], []
    for c in range(3):
        sel = (thd > np.radians(135)) & (thd < np.radians(142))
        prim.append(180.0 - np.degrees(thd[np.argmax(np.where(sel, pr[:, c], 0))]))
        sel = (thd > np.radians(123)) & (thd < np.radians(132))
        sec.append(180.0 - np.degrees(thd[np.argmax(np.where(sel, pr[:, c], 0))]))
    r.check(40.3 < prim[2] < prim[1] < prim[0] < 42.8,
            f"primary bow radii R {prim[0]:.2f} G {prim[1]:.2f} B {prim[2]:.2f}: red outside")
    r.check(50.0 < sec[0] < sec[1] < sec[2] < 54.0,
            f"secondary bow R {sec[0]:.2f} G {sec[1]:.2f} B {sec[2]:.2f}: order reversed")
    band = (thd > np.radians(131.0)) & (thd < np.radians(136.5))
    inside = (thd > np.radians(142.0)) & (thd < np.radians(150.0))
    ratio = float(pr[band, 1].mean() / pr[inside, 1].mean())
    r.check(ratio < 0.25, f"Alexander's dark band is {ratio:.2f} of the sky inside the bow")
    tt = mie.lut_theta()
    wq = 2 * math.pi * np.sin(tt) * np.gradient(tt)
    for c in range(3):
        r.check(near(float((lut[:, c] * wq).sum()), 1.0, 2e-3), f"channel {c} normalised")
    r.close()

    r = Rule("I3 Jendersie-d'Eon droplet phase function")
    th = np.linspace(0.0, math.pi, 400001)
    c = np.cos(th)
    w = 2 * math.pi * np.sin(th)

    def hg(c, g):
        den = (1 - g) ** 2 + 2 * g * (1 - c)
        return (1 - g * g) / (4 * math.pi * den * np.sqrt(den))

    def dr(c, g, a):
        den = (1 - g) ** 2 + 2 * g * (1 - c)
        return (1 - g * g) * (1 + a * c * c) / (4 * math.pi * (1 + a * (1 + 2 * g * g) / 3) * den * np.sqrt(den))
    # mean cosines of the fit, from the research brief
    for d, gref in ((5.0, 0.8434), (10.0, 0.8642), (20.0, 0.8793), (50.0, 0.8912)):
        P = gl_sky.jde_params(d / 2.0)
        p = (1 - P[3]) * hg(c, P[0]) + P[3] * dr(c, P[1], P[2])
        r.check(near(float(np.trapezoid(p * w, th)), 1.0, 1e-5), f"d={d}: not normalised")
        r.check(near(float(np.trapezoid(p * c * w, th)), gref, 0.0005), f"d={d}: mean cosine")
    r.close()

    r = Rule("I4 display and haze")
    for v in (2000.0, 10000.0, 30000.0):
        b = gl_sky.haze_from_visibility(v)
        r.check(near(3.912 / (b + 1.8e-5), v, 1.0), f"Koschmieder round trip at {v:.0f} m")
    r.check(gl_sky.visibility_guess(40) > gl_sky.visibility_guess(95), "haze thickens with humidity")
    x = np.linspace(0.0, 1.0, 1001)
    x2 = x * x
    x4 = x2 * x2
    agx = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
    r.check(bool(np.all(np.diff(agx) > -1e-6)), "AgX contrast curve is monotonic")
    r.check(abs(float(agx[0])) < 0.01 and abs(float(agx[-1]) - 1.0) < 0.02, "AgX curve spans 0..1")
    r.close()

# ------------------------------------------------------------------- GL -----

def test_gl():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context(backend="egl", require=330)
    except Exception:
        try:
            import moderngl
            ctx = moderngl.create_standalone_context(require=330)
        except Exception as e:                                  # noqa: BLE001
            RESULTS.append(("G1 renderer smoke test", 0,
                            [f"skipped, no OpenGL context ({type(e).__name__})"]))
            return
    from . import gl_sky
    r = Rule("G1 every genus renders to a non-blank frame")
    rend = gl_sky.SkyRenderer(ctx, 200, 120, quality="low")
    snd = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    when = datetime(2026, 9, 10, 2, tzinfo=timezone.utc)
    geom = astro.sky_geometry(when, 3.14, 101.69)
    cam = gl_sky.Camera(200.0, 25.0, 80.0)
    clear = None
    for g in atlas.GENUS_ORDER:
        d = scene.deck_from_spec(CloudSpec(g), snd, coverage=0.6)
        rend.set_decks([d])
        img = rend.read_image(rend.render(cam, geom, 0.0))
        r.check(np.isfinite(img).all(), f"{g}: non-finite pixels")
        r.check(img.std() > 1.0, f"{g}: frame is flat (std {img.std():.2f})")
        if clear is None:
            rend.set_decks([])
            clear = rend.read_image(rend.render(cam, geom, 0.0)).astype(float)
        diff = abs(img.astype(float) - clear).mean()
        r.check(diff > 1.0, f"{g}: indistinguishable from a clear sky (mean diff {diff:.2f})")
    r.close()

    r = Rule("G2 night is dark and noon is bright")
    rend.set_decks([])
    noon = rend.read_image(rend.render(cam, astro.sky_geometry(
        datetime(2026, 9, 10, 4, tzinfo=timezone.utc), 3.14, 101.69), 0.0)).mean()
    night = rend.read_image(rend.render(cam, astro.sky_geometry(
        datetime(2026, 9, 10, 18, tzinfo=timezone.utc), 3.14, 101.69), 0.0)).mean()
    # (a clear sky with the multiple scattering of Hillaire (2020) done right
    # is a deep blue; the frame's mean is under mid-grey)
    r.check(noon > 70, f"midday mean brightness {noon:.1f}")
    r.check(night < 40, f"midnight mean brightness {night:.1f}")
    r.check(noon > night * 3, "day is not clearly brighter than night")
    r.close()

    def hdr(rd):
        t = rd._fbos["scene"][0]
        return np.frombuffer(t.read(), t.dtype).reshape(t.height, t.width, -1)[..., :3].astype("f8")

    def angles_from(cam, rd, direction):
        """Angle (deg) between each pixel's ray and `direction`."""
        right, up, fwd = cam.basis()
        th = math.tan(math.radians(cam.fov) * 0.5)
        ys, xs = np.mgrid[0:rd.rh, 0:rd.rw]
        nx = (xs + 0.5) / rd.rw * 2.0 - 1.0
        ny = (ys + 0.5) / rd.rh * 2.0 - 1.0
        v = (np.asarray(fwd)[None, None, :] + np.asarray(right)[None, None, :] * (nx * th * rd.rw / rd.rh)[..., None]
             + np.asarray(up)[None, None, :] * (ny * th)[..., None])
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return np.degrees(np.arccos(np.clip(v @ np.asarray(direction), -1.0, 1.0)))

    kl = astro.sky_geometry(datetime(2026, 9, 25, 2, tzinfo=timezone.utc), 3.14, 101.69)
    snd_k = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    st = scene.deck_from_spec(CloudSpec("St", "neb"), snd_k, base_m=500.0, coverage=1.0)
    ci = scene.deck_from_spec(CloudSpec("Ci", "fib"), snd_k, base_m=9500.0, coverage=0.6)
    cs = scene.deck_from_spec(CloudSpec("Cs", "neb"), snd_k, base_m=9000.0, coverage=0.9)
    ns = scene.deck_from_spec(CloudSpec("Ns"), snd_k, base_m=900.0, coverage=1.0)

    r = Rule("G3 the nearest deck is in front, whatever the order of the list")
    rend2 = gl_sky.SkyRenderer(ctx, 160, 90, quality="low")
    rend2.observer_alt = 12000.0
    rend2.adaptive = False
    down = gl_sky.Camera(0.0, -35.0, 80.0)
    frames = {}
    for name, dl in (("St,Ci", [st, ci]), ("Ci,St", [ci, st]), ("St", [st])):
        rend2.set_decks(dl)
        rend2.frame = 0                       # the same dither for each
        rend2.render(down, kl, 0.0)
        frames[name] = hdr(rend2)
    same = float(abs(frames["St,Ci"] - frames["Ci,St"]).mean() / max(frames["St"].mean(), 1e-9))
    seen = float(abs(frames["St,Ci"] - frames["St"]).mean() / max(frames["St"].mean(), 1e-9))
    r.check(same < 1e-3, f"from 12 km the two orders differ by {100 * same:.2f}%")
    r.check(seen > 0.01, f"the cirrus above the stratus is seen from above ({100 * seen:.1f}% of the frame)")
    r.close()

    r = Rule("G4 no halo through a stratus overcast, no Sun through a nimbostratus")
    rend3 = gl_sky.SkyRenderer(ctx, 200, 120, quality="low")
    rend3.adaptive = False
    at_sun = gl_sky.Camera(kl.sun_az, kl.sun_alt, 70.0)
    ring = {}
    for name, dl in (("Cs", [cs]), ("St+Cs", [st, cs])):
        rend3.set_decks(dl)
        for _ in range(3):
            rend3.render(at_sun, kl, 0.0, accumulate=True)
        img = hdr(rend3).mean(-1)
        ang = angles_from(at_sun, rend3, kl.sun_dir)
        on = img[(ang > 21.8) & (ang < 23.4)].mean()
        inside = img[(ang > 17.5) & (ang < 20.5)].mean()
        ring[name] = on / max(inside, 1e-9) - 1.0
    r.check(ring["Cs"] > 0.05, f"a cirrostratus alone shows its halo (ring +{100 * ring['Cs']:.1f}%)")
    r.check(abs(ring["St+Cs"]) < 0.02, f"under a stratus (tau {st.optical_depth:.0f}) the ring is "
            f"{100 * ring['St+Cs']:+.1f}%")
    rend3.set_decks([ns])
    at_sun_n = gl_sky.Camera(kl.sun_az, kl.sun_alt, 20.0)
    for _ in range(2):
        rend3.render(at_sun_n, kl, 0.0, accumulate=True)
    img = hdr(rend3).mean(-1)
    ang = angles_from(at_sun_n, rend3, kl.sun_dir)
    disc = float(img[ang < 0.6].max()) if (ang < 0.6).any() else 0.0
    r.check(disc < 1.1 * float(np.median(img)), f"the Sun's disc behind tau {ns.optical_depth:.0f}: "
            f"{disc / max(float(np.median(img)), 1e-9):.2f} x the cloud")
    r.close()

    r = Rule("G5 a station at 5364 m has half the air above it: a darker, bluer sky")
    geo = astro.sky_geometry(datetime(2026, 9, 25, 5, tzinfo=timezone.utc), 28.0, 86.85)
    cam_up = gl_sky.Camera((geo.sun_az + 150.0) % 360.0, 70.0, 2.0)
    sky = {}
    for elev in (0.0, 5364.0):
        rend4 = gl_sky.SkyRenderer(ctx, 16, 9, quality="low")
        rend4.adaptive, rend4.haze, rend4.ground_elev = False, 0.0, elev
        rend4.set_decks([])
        rend4.render(cam_up, geo, 0.0)
        sky[elev] = hdr(rend4).mean(axis=(0, 1)) / rend4._ev_used
    lum = sky[5364.0].mean() / sky[0.0].mean()
    blue = (sky[5364.0][2] / sky[5364.0][0]) / (sky[0.0][2] / sky[0.0][0])
    ref = atmosphere.sky_radiance(rend4.tr_lut, rend4.ms_lut, np.array([cam_up.basis()[2]]),
                                  np.array(geo.sun_dir), 5364.0 + 1.7)[0]
    r.check(0.35 < lum < 0.75, f"sky at 70 deg: {lum:.2f} x as bright as at sea level")
    r.check(blue > 1.05, f"and bluer: blue/red {blue:.2f} x")
    r.check(bool(np.allclose(sky[5364.0], ref, rtol=0.08)), f"the shader's sky at 5364 m {sky[5364.0].round(4)} "
            f"matches the reference {ref.round(4)}")
    r.close()

    r = Rule("G6 lying snow whitens the ground; a desert is sand")
    gnd = {}
    for snow in (0.0, 1.0, "dry"):
        rend5 = gl_sky.SkyRenderer(ctx, 64, 36, quality="low")
        rend5.adaptive = False
        if snow == "dry":
            rend5.dryness = 1.0
        else:
            rend5.snow = snow
        rend5.set_decks([])
        rend5.render(gl_sky.Camera(0.0, -30.0, 40.0), kl, 0.0)
        gnd[snow] = hdr(rend5).mean(axis=(0, 1))
    r.check(gnd[1.0].mean() > 2.5 * gnd[0.0].mean(), f"snow {gnd[1.0].mean():.3f} vs bare {gnd[0.0].mean():.3f}")
    r.check(gnd[1.0][2] >= gnd[1.0][1] >= gnd[1.0][0] * 0.95, f"and white, not green ({gnd[1.0].round(3)})")
    sand = gnd["dry"]
    r.check(sand[0] > sand[1] > sand[2] and sand.mean() > 1.8 * gnd[0.0].mean(),
            f"bone-dry ground is sand, not grass ({sand.round(3)} vs {gnd[0.0].round(3)})")
    r.close()

    r = Rule("G7 snow under a stratus overcast: grey-white and nearly as bright as the cloud base")
    pole = astro.sky_geometry(datetime(2026, 6, 21, 12, tzinfo=timezone.utc), 89.99, 0.0)
    for where, geo, want_ratio, want_tint in (("Kuala Lumpur, Sun 44 deg", kl, 0.6, 1.08),
                                             ("the pole, Sun 23 deg", pole, 0.0, 1.15)):
        look = {}
        for name, cam in (("ground", gl_sky.Camera(0.0, -30.0, 40.0)),
                          ("base", gl_sky.Camera(0.0, 45.0, 40.0))):
            rend6 = gl_sky.SkyRenderer(ctx, 64, 36, quality="low")
            rend6.adaptive, rend6.snow = False, 1.0
            rend6.set_decks([st])
            for _ in range(2):
                rend6.render(cam, geo, 0.0, accumulate=True)
            t6 = rend6._fbos["accA"][0]
            look[name] = np.frombuffer(t6.read(), t6.dtype).reshape(t6.height, t6.width, -1)[..., :3].astype(
                "f8").mean(axis=(0, 1))
        ratio = look["ground"].mean() / max(look["base"].mean(), 1e-9)
        tint = look["ground"][2] / max(look["ground"][0], 1e-9)
        if want_ratio:
            r.check(ratio > want_ratio, f"{where}: snow {ratio:.2f} x the cloud base's radiance (albedo 0.75, "
                    "and the light going back and forth between them)")
        r.check(tint < want_tint, f"{where}: the snow's blue/red {tint:.2f} (the overcast must dim the blue "
                "sky's light as well)")
    r.close()

    r = Rule("G8 from 22 km straight down, a thin layer of the model's is not stepped over")
    import time as _time
    from . import crm, region, simdriver

    def thin_layer_frame(with_layer):
        sc_ac = crm.scenario_altocumulus("fast")
        m_ac = crm.CPUModel(sc_ac, dtype="f4")
        # a layer one level (25 m) thick, broken into 250 m squares: seen
        # from a satellite, a step of t/50 (400 m and more) jumped it
        k0 = int(np.argmin(abs(sc_ac.grid.zc - 3200.0)))
        jj, ii = np.mgrid[0:sc_ac.grid.ny, 0:sc_ac.grid.nx]
        for f in ("qc", "qi", "qr", "qs"):
            getattr(m_ac, f)[:] = 0.0
        if with_layer:
            m_ac.qc[k0] = np.where(((ii // 4) + (jj // 4)) % 2 == 0, 3.0e-4, 0.0).astype(m_ac.qc.dtype)
        else:
            m_ac.qc[k0, 0, 0] = 3.0e-4           # one cell: the same heights to march
        drv = simdriver.SimDriver(ctx, prefer_gpu=False)
        drv.sc, drv.model, drv.gpu_ok = sc_ac, m_ac, False
        drv._make_cpu_textures()
        drv._upload_cpu(first=True)
        drv.running = False
        drv.diag = m_ac.diagnostics()
        view = drv.frame(m_ac.time, _time.time())
        rend7 = gl_sky.SkyRenderer(ctx, 160, 90, quality="medium")
        rend7.observer_alt, rend7.adaptive = 22000.0, False
        rend7.set_sim(view)
        for _ in range(3):
            rend7.render(gl_sky.Camera(45.0, -89.0, 100.0), kl, 0.0, accumulate=True)
        t7 = rend7._fbos["accA"][0]
        return np.frombuffer(t7.read(), t7.dtype).reshape(t7.height, t7.width, -1)[..., :3].astype(
            "f8").mean(-1)

    with_l, without = thin_layer_frame(True), thin_layer_frame(False)
    H, W = with_l.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / math.sqrt(2.0)
    veil = float(((with_l - without) / np.maximum(without, 1e-9))[rr > 0.75].mean())
    # the layer itself brightens the view 19 % here (its shadow on the hazy
    # air below takes 1 % back); stepped over, it shows nothing (-1 %).  (The
    # bound was 22 % while a cloud's shadow brightened the air seen from
    # above - rule Y9 - and added 11 % of its own, 14 % even stepped over.)
    r.check(veil > 0.10, f"toward the frame's edge the layer brightens the view by {100 * veil:.0f}% "
            "(stepped over it shows nothing)")
    r.close()

    r = Rule("G9 regional maps full of NaN give a sky, not a black frame")
    sc9 = crm.scenario_altocumulus("fast")
    m9 = crm.CPUModel(sc9, dtype="f4")
    drv9 = simdriver.SimDriver(ctx, prefer_gpu=False)
    drv9.sc, drv9.model, drv9.gpu_ok = sc9, m9, False
    drv9._make_cpu_textures()
    drv9._upload_cpu(first=True)
    drv9.running = False
    drv9.diag = m9.diagnostics()
    rend8 = gl_sky.SkyRenderer(ctx, 96, 54, quality="low")
    rend8.set_sim(drv9.frame(m9.time, _time.time()))
    n8 = 16
    rm8 = region.RegionMaps(n8).build(None, 3000.0, 4000.0)
    rm8.has_data = True
    rm8.warp_a = rm8.warp_b = np.full((n8, n8, 2), np.nan, "f4")
    rm8.raw_a = rm8.raw_b = np.full((n8, n8, 3), np.nan, "f4")
    rm8.winds = np.full((3, 2), np.nan)
    rm8.set_time(0.5)
    rend8.set_region(rm8, tau_th=[0.0] * 17)
    for _ in range(4):
        img8 = rend8.read_image(rend8.render(gl_sky.Camera(90.0, 25.0, 80.0), kl, 0.0, accumulate=True))
    t8 = rend8._fbos["scene"][0]
    raw8 = np.frombuffer(t8.read(), t8.dtype)
    r.check(bool(np.isfinite(raw8).all()), "no NaN in the frame")
    r.check(float(img8.mean()) > 40.0, f"the frame's mean brightness {img8.mean():.0f} (0 was black)")
    r.check(math.isfinite(rend8._ev_used) and rend8._ev_used < 10.0, f"exposure {rend8._ev_used:.3g}")
    r.close()

    r = Rule("G10 castellanus turrets are solid domes on their layer, not stacks of discs")
    # Seen side-on from the layer's height, a turret is thinner along a ray
    # than the ray's stride through empty air; a sample dithered within one
    # step of a three-step stride hit it on some rows and missed it on the
    # next, and each turret came out as horizontal stripes.  Stripes make
    # the brightness change far more from row to row than from column to
    # column: 1.50 with that sampling, 1.28 with the stride covered.
    snd10 = sounding.standard_sounding(3.14, 101.69, profile="tropical-fair")
    d10 = scene.deck_from_spec(CloudSpec("Ac", "cas"), snd10, base_m=5400.0, coverage=0.3,
                               thickness_m=650.0, seed=4242, turret_rise_m=900.0)
    want10 = min(650.0, max(0.5 * d10.element_w_m, 50.0)) / 1550.0
    r.check(abs(d10.turret - want10) < 1e-3 and 0.1 < d10.turret < 0.6,
            f"the common layer is {d10.turret:.2f} of the deck's depth ({want10:.2f})")
    rend10 = gl_sky.SkyRenderer(ctx, 320, 180, quality="medium")
    rend10.observer_alt = 5000.0
    rend10.set_decks([d10])
    g10 = astro.sky_geometry(datetime(2026, 9, 10, 4, tzinfo=timezone.utc), 3.14, 101.69)
    for _ in range(6):
        img10 = rend10.read_image(rend10.render(gl_sky.Camera(300.0, 4.0, 14.0), g10, 0.0,
                                                accumulate=True))
    lum = img10.astype("f8").mean(axis=2)
    # on the turrets: the lower middle rows at the sides of the frame.  (The
    # element right overhead, seen 400 m under its base at a grazing angle,
    # is foreshortened by perspective into bands across the frame; shaded as
    # a rounded mass, 4 Oct 2026, it took the whole frame's ratio to 1.58
    # with the turrets themselves at 1.21 and 1.24.)
    H10, W10 = lum.shape
    for cols in (slice(0, int(0.3 * W10)), slice(int(0.7 * W10), W10)):
        part = lum[int(0.35 * H10):int(0.8 * H10), cols]
        dy = np.abs(np.diff(part, axis=0))[:, :-1]
        dx = np.abs(np.diff(part, axis=1))[:-1, :]
        cloud = part[:-1, :-1] > np.percentile(lum, 40)
        ratio = float(dy[cloud].mean() / max(dx[cloud].mean(), 1e-6))
        r.check(ratio < 1.40, f"row-to-row over column-to-column change on the turrets {ratio:.2f} "
                              f"(stripes: 1.50)")
    r.close()

    r = Rule("G11 on the screen nothing jumps at the hour: what changes in a second changes ten "
             "times less in a tenth")
    from . import region as _region, timeline
    sky_h, t0h = _sky_hands(n=128)
    rend11 = gl_sky.SkyRenderer(ctx, 192, 108, quality="low")
    rend11.adaptive = False
    geo11 = astro.sky_geometry(datetime(2026, 9, 25, 3, tzinfo=timezone.utc), 3.14, 101.69)
    cam11 = gl_sky.Camera(120.0, 32.0, 110.0)
    holder = {"sky": sky_h}

    def shot(when, rm=None):
        if holder["sky"] is not None:
            holder["sky"].update(when, 0.0, sync=True)
            rend11.set_deck_states(holder["sky"].states, holder["sky"].take_uploads())
        if rm is not None:
            rend11.set_region(rm, None, rm.tau_th)
        rend11.frame = 0
        rend11.accum_frames = 0
        for _ in range(2):
            rend11.render(cam11, geo11, 0.0, accumulate=True, animating=True)
        return hdr(rend11)

    def change(a, b):
        return float(np.abs(a - b).mean() / max(0.5 * (a.mean() + b.mean()), 1e-9))

    def jump(fn, c):
        """(change over 1 s about c, over 0.1 s): a continuous picture changes
        about ten times less over the shorter time, a jump as much."""
        d1 = change(fn(c - timedelta(seconds=0.5)), fn(c + timedelta(seconds=0.5)))
        d01 = change(fn(c - timedelta(seconds=0.05)), fn(c + timedelta(seconds=0.05)))
        return d1, d01

    shot(t0h + timedelta(hours=1))                  # the renderer's first frames settle
    bad, small, ten = [], [], []
    for hr in (2, 3, 4, 5, 6, 7):
        c = t0h + timedelta(hours=hr)
        d1, d01 = jump(shot, c)
        small.append(d01)
        if d01 > 1e-5 and d01 > 0.5 * d1:
            bad.append(f"hour {hr}: {d01:.1e} in 0.1 s, {d1:.1e} in 1 s")
        ten.append(change(shot(c), shot(c + timedelta(minutes=10))))
    r.check(not bad, "; ".join(bad))
    r.check(min(ten) > 5.0 * max(small),
            f"ten minutes change the picture by {min(ten):.1e} or more (the test sees change)")
    # the regional maps: a band of low cloud crossing the grid, over a layer
    ser, t0r = _band_series()
    rm = _region.RegionMaps()
    snd11 = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    sk = timeline.Sky(manual=[scene.deck_from_spec(CloudSpec("Sc", "str"), snd11, base_m=1000.0,
                                                   coverage=0.9, thickness_m=500.0, seed=5)],
                      threaded=False, n=128)
    sk.update(t0r, 0.0, sync=True)
    rend11.set_deck_states(sk.states, sk.take_uploads())
    holder["sky"] = None                            # the layer holds; only the maps move

    def rshot(when):
        rm.build_series(ser, when, None, 5000.0, high_wind=lambda k: (0.0, 0.0))
        return shot(when, rm)
    c = t0r + timedelta(hours=2)
    rshot(c)
    d1, d01 = jump(rshot, c)
    moved = change(rshot(c), rshot(c + timedelta(minutes=20)))
    r.check(d01 <= 0.5 * d1 + 1e-6 and moved > 20.0 * d1,
            f"the regional maps across the hour: {d01:.1e} in 0.1 s, {d1:.1e} in 1 s; "
            f"{moved:.1e} in twenty minutes")
    r.close()

    r = Rule("G12 through twilight nothing jumps on the screen: what the Sun's setting changes in "
             "24 s it changes ten times less in 2.4 s")
    rend12 = gl_sky.SkyRenderer(ctx, 160, 90, quality="low")
    rend12.adapt_rate = 0.0                 # the metered exposure held where it is put
    snd12 = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
    rend12.set_decks([scene.deck_from_spec(CloudSpec("Cs", "neb"), snd12, base_m=8500.0,
                                           coverage=0.9, thickness_m=700.0, seed=4),
                      scene.deck_from_spec(CloudSpec("Ac", "str"), snd12, base_m=4500.0,
                                           coverage=0.35, seed=5)])
    cam12 = gl_sky.Camera(265.0, 14.0, 100.0)          # towards the setting Sun
    day12 = datetime(2026, 9, 26, tzinfo=timezone.utc)  # the full Moon rises at dusk
    lat12, lon12 = 3.14, 101.69

    def sun_when(alt, refracted=False):
        a, b = day12 + timedelta(hours=10, minutes=30), day12 + timedelta(hours=12, minutes=40)
        for _ in range(48):
            m = a + (b - a) / 2
            g = astro.sky_geometry(m, lat12, lon12)
            if (g.sun_alt_refracted if refracted else g.sun_alt) > alt:
                a = m
            else:
                b = m
        return a

    def shot12(w, adapted):
        g = astro.sky_geometry(w, lat12, lon12)
        rend12._ev_adapted = adapted
        rend12.frame = 0
        rend12.accum_frames = 0
        for _ in range(2):
            rend12.render(cam12, g, 0.0, accumulate=True, animating=True)
        return hdr(rend12)                  # the scene buffer is radiance x exposure

    bad12, seen12 = [], []
    for s0, refr, what in ((4.0, False, "the day's exposure range"), (0.0, True, "the halo at sunset"),
                           (-2.0, False, "refraction"), (-6.0, False, "the Moon's exposure"),
                           (-10.0, False, "the Moon's light"), (-14.0, False, "the night's range")):
        c = sun_when(s0, refr)
        g0 = astro.sky_geometry(c, lat12, lon12)
        rend12._ev_adapted = None
        adapted = 20.0 * rend12.auto_exposure(g0.sun_alt, g0.moon_alt, g0.moon_illum)
        d_big = change(shot12(c - timedelta(seconds=12), adapted), shot12(c + timedelta(seconds=12), adapted))
        d_small = change(shot12(c - timedelta(seconds=1.2), adapted),
                         shot12(c + timedelta(seconds=1.2), adapted))
        seen12.append(d_big)
        if d_small > 1e-5 and d_small > 0.5 * d_big:
            bad12.append(f"{what} (Sun {s0:g} deg): {d_small:.1e} in 2.4 s, {d_big:.1e} in 24 s")
    r.check(not bad12, "; ".join(bad12))
    r.check(min(seen12) > 1e-4, f"the Sun's setting changes the picture ({min(seen12):.1e} in 24 s at "
            f"least: the test sees change)")
    irr = [float(rend12.sky_irradiance(float(s))[0][1]) for s in np.arange(-12.0, -1.99, 0.01)]
    step_ = max(max(a / b, b / a) for a, b in zip(irr[:-1], irr[1:]))
    r.check(step_ < 1.03, f"the sky's light on the clouds changes by at most "
            f"{100 * (step_ - 1):.1f} % per 0.01 deg of the Sun")
    r.close()

    r = Rule("G13 the panels still draw after the sky: the renderer keeps off pyglet's "
             "uniform-block binding 0")
    # pyglet draws the panels' shapes and text with programs that read their
    # projection from the uniform block at binding point 0 (WindowBlock),
    # bound once when the window is made; the sky is drawn first every frame
    import moderngl as _mgl
    prog13 = ctx.program(
        vertex_shader="""#version 330
        in vec2 pos;
        layout(std140) uniform WindowBlock { mat4 projection; mat4 view; } window;
        void main() { gl_Position = window.projection * window.view * vec4(pos, 0.0, 1.0); }""",
        fragment_shader="""#version 330
        out vec4 colour;
        void main() { colour = vec4(1.0, 0.0, 0.0, 1.0); }""")
    prog13["WindowBlock"].binding = 0
    eye4 = np.eye(4, dtype="f4").ravel()
    ubo13 = ctx.buffer(np.concatenate([eye4, eye4]).tobytes())
    ubo13.bind_to_uniform_block(0)
    vbo13 = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
    vao13 = ctx.vertex_array(prog13, [(vbo13, "2f", "pos")])
    rend13 = gl_sky.SkyRenderer(ctx, 96, 54, quality="low")
    geo13 = astro.sky_geometry(datetime(2026, 9, 25, 3, tzinfo=timezone.utc), 3.14, 101.69)
    rend13.render(gl_sky.Camera(120.0, 30.0, 90.0), geo13, 0.0)
    fbo13 = ctx.simple_framebuffer((8, 8))
    fbo13.use()
    fbo13.clear(0.0, 0.0, 0.0, 1.0)
    vao13.render(_mgl.TRIANGLE_STRIP)
    px13 = np.frombuffer(fbo13.read(components=3), np.uint8).reshape(8, 8, 3)
    red = float((px13[..., 0] > 200).mean())
    r.check(red > 0.99, f"a panel drawn after the sky covers {100 * red:.0f}% of its area "
            f"(the layers' uniform block at binding {getattr(rend13, 'deck_binding', 0)})")
    r.check(getattr(rend13, "deck_binding", 0) != 0, "the layers' block is not on binding 0")
    r.close()



def test_panels():
    """G14: the panels' text, drawn by pyglet in a window of its own."""
    name = ("G14 the panels' text: every line of the help is drawn, and wrapped text stays in "
            "its own place, before and after a scroll")
    try:
        import pyglet
        win = pyglet.window.Window(width=640, height=400, visible=False)
        from . import app, ui
    except ModuleNotFoundError:
        raise
    except Exception as e:                                       # noqa: BLE001
        RESULTS.append((name, 0, [f"skipped, no window can be made here ({type(e).__name__})"]))
        return
    import moderngl as _mgl
    r = Rule(name)
    try:
        win.switch_to()
        ctx = _mgl.create_context()
        fbo = ctx.simple_framebuffer((640, 400))
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        # the help box, drawn as the program draws it (h): each of its lines must
        # leave light pixels in its own band (pyglet deletes a label nothing holds)
        stub = type("A", (), {})()
        stub.win = win
        app.CloudSim.draw_help(stub)
        px = np.frombuffer(fbo.read(components=3), np.uint8).reshape(400, 640, 3).astype(int)
        n_lines = len(app.CloudSim.HELP_LINES)
        # the box: its dark fill over the black clear colour; the text: light pixels
        box = (np.abs(px - np.array([9, 13, 19])).max(axis=2) <= 3)
        ys, xs = np.nonzero(box)
        light = px.sum(axis=2) > 360
        if len(xs) < 1000:
            r.check(False, "the help box is not drawn")
        else:
            bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
            h_ = 24 * n_lines + 30
            drawn = 0
            for i in range(n_lines):
                base = by0 + h_ - 30 - i * 24                 # the line's baseline, from the bottom
                band = light[max(base - 4, 0):base + 12, bx0:bx1 + 1]
                drawn += int(band.sum() >= 20)
            r.check(drawn == n_lines, f"help lines with their text on the screen: {drawn} of {n_lines}")
            out = light.copy()
            out[by0:by1 + 1, bx0:bx1 + 1] = False
            r.check(not out.any(), f"help text outside its box: {int(out.sum())} pixels "
                                   f"(box {bx1 - bx0 + 1} px wide)")

        def bounds(w):
            lab = w._labels[0]
            if hasattr(lab, "top") and hasattr(lab, "bottom"):
                return float(lab.top), float(lab.bottom)
            if lab.anchor_y == "top":
                return float(lab.y), float(lab.y - lab.content_height)
            f = pyglet.font.load(ui.FONT, w.size, weight="bold" if w.bold else "normal")
            return float(lab.y + f.ascent), float(lab.y + f.ascent - lab.content_height)

        texts = [("Thunderstorm (from your sounding): 128x128x72 cells, 250 m x 250 m, 32.0 km "
                  "domain; CAPE 2759 J/kg (GPU)", 9, False, 0),
                 ("Cumulonimbus capillatus praecipitatio  (Cb cap pra)", 10, True, 0),
                 ("why: equilibrium level 14097 m at -66 C, tropopause 16667 m", 9, False, 4),
                 ("the weather data call for: Fair-weather cumulus (the reasons are in the panel "
                  "on the right)", 9, False, 0),
                 ("· wind: 1 m/s from W at 10 m, 2 m/s from S at 1.5 km, 3 m/s from WNW at 5.6 km; "
                  "0-6 km shear 1 m/s -> low clouds drift toward N at 8 km/h", 9, False, 4),
                 ("one line", 9, False, 0),
                 ("", 9, False, 0),
                 ("a paragraph\nand a second one, long enough to wrap onto a line of its own "
                  "in the narrower panel", 9, False, 8)]
        for width in (352, 400):
            p = ui.Panel(width=width)
            p.begin(400, x0=0)
            for t, size, bold, ind in texts:
                p.text(t, colour=ui.DIM, size=size, bold=bold, wrap=True, indent=ind)
                p.buttons([("next", None, False, True)], height=18)
            p.end()
            for when in ("as built", "after a scroll"):
                if when == "after a scroll":
                    p.on_scroll(-1)                           # a wheel notch down
                    p.end()                                   # the same widgets placed again
                bad = []
                for k, w in enumerate(p.widgets):
                    if not isinstance(w, ui.Text):
                        continue
                    top, bot = bounds(w)
                    nxt = p.widgets[k + 1]
                    if top > w.y + w.h + 0.5 or bot < w.y - 0.5 or bot < nxt.y + nxt.h - 0.5:
                        bad.append(f"{w.text[:30]!r}: lines {bot:.0f}..{top:.0f}, box {w.y:.0f}.."
                                   f"{w.y + w.h:.0f}, next widget's top {nxt.y + nxt.h:.0f}")
                r.check(not bad, f"panel {width} px, {when}: " + "; ".join(bad[:3]))
    finally:
        r.close()
        win.close()

    # G15: the program itself, paused and with no model running (as under a
    # high ice veil, where the model has nothing to grow): an hour's time
    # shift is played through, and the panel's clock must then read the new time
    import time as _time
    r = Rule("G15 the panel's clock follows a time shift played through while paused, "
             "with no model running")
    cs = None
    real_window = app._make_window

    def hidden_window(width, height):
        """app._make_window's window, kept out of sight"""
        from pyglet.gl import Config
        last = None
        for major, minor in ((4, 3), (3, 3)):
            try:
                cfg = Config(major_version=major, minor_version=minor, forward_compatible=True,
                             double_buffer=True, depth_size=0, sample_buffers=0)
                return pyglet.window.Window(width=width, height=height, config=cfg,
                                            visible=False)
            except Exception as e:                           # noqa: BLE001
                last = e
        raise last

    try:
        app._make_window = hidden_window
        try:
            cs = app.CloudSim(width=320, height=200, quality="low", offline=True, sim="none")
        finally:
            app._make_window = real_window
        pyglet.clock.unschedule(cs.update)
        t0 = cs.when
        cs.shift_time(3600.0)
        cs.update(1 / 30)
        r.check(cs._anim is not None, "the shift is played through, not jumped")
        t_end = _time.time() + 10.0
        while cs._anim is not None and _time.time() < t_end:
            _time.sleep(1 / 30)
            cs.update(1 / 30)
        for _ in range(4):
            cs.update(0.2)                                    # half a second and more after it
        r.check(cs._anim is None and abs((cs.when - t0).total_seconds() - 3600.0) < 1e-3,
                f"the clock moved {(cs.when - t0).total_seconds():.3f} s")
        shown = [getattr(w, "text", "") for w in cs.panel.widgets]
        want = cs.local.strftime("%Y-%m-%d  %H:%M:%S")
        r.check(any(t.startswith(want) for t in shown),
                f"the panel reads {[t for t in shown if t[:2] == want[:2]][:1]}, the clock {want}")
        sun = f"sun {cs.geom.sun_alt:+.1f}°"
        r.check(any(t.startswith(sun) for t in shown), f"the panel's Sun line is not {sun!r}")
        r.check(not cs.playing, "still paused")
    finally:
        r.close()

    # G16, the same program: who starts the clock
    r = Rule("G16 a model restart the program makes by itself leaves a paused clock paused; "
             "a click on a model starts the clock, as before")
    try:
        if cs is None:
            raise RuntimeError("the program did not start")
        cs.sim_size = "fast"
        cs.playing = False
        cs.start_sim("bomex")
        r.check(cs.playing and cs.sim_key == "bomex", "a click on a model starts the clock")
        cs.playing = False
        cs._request_restart(cs.sim_key)          # as when the weather changes or the clock went back
        cs.sim_fade = 0.0
        cs._update_model_fade(0.1)
        r.check(cs._restart_pending is None and cs.sim_key == "bomex",
                "the program restarted the model")
        r.check(not cs.playing, "a restart the program made by itself started the paused clock")
        cs.sim_fade = 0.6                        # the model's clouds showing
        cs.start_sim("streets")
        r.check(cs._restart_pending is not None and cs.sim_key == "bomex" and not cs.playing,
                "a click while the model's clouds show hands them over first")
        cs.sim_fade = 0.0
        cs._update_model_fade(0.1)
        r.check(cs.sim_key == "streets" and cs.playing,
                "after the hand-over the clicked model runs, and the clock with it")
    except Exception as e:                                       # noqa: BLE001
        r.check(False, f"{type(e).__name__}: {e}")
    finally:
        r.close()
        if cs is not None:
            cs.sim.stop()
            cs.sky.close()
            cs.win.close()


# ------------------------------------------------- places, clocks and dates --

def test_places():
    """W1-W8: the world's cities, their clocks, the dates and hours typed in
    the dropdowns, and the forecast windows asked for."""
    from datetime import date, time as dtime
    from . import places

    r = Rule("W1 the city table: 1,584 places, each once, on the Earth, with a time zone whose "
             "offsets are stored; Kuala Lumpur where the program has always put it")
    try:
        cs = places.cities()
        d = places._data()
        r.check(len(cs) == 1584, f"{len(cs)} cities")
        labels = [c.label for c in cs]
        r.check(len(set(labels)) == len(labels), "a label twice")
        bad = [c.label for c in cs if not (-90 <= c.lat <= 90 and -180 <= c.lon <= 180)]
        r.check(not bad, f"off the Earth: {bad[:3]}")
        bad = [c.label for c in cs if c.zone not in d["zones"]]
        r.check(not bad, f"no stored offsets: {bad[:3]}")
        bad = [c.label for c in cs if c.name_key not in c.search_key or c.population < 0]
        r.check(not bad, f"name not searchable, or a negative population: {bad[:3]}")
        r.check([c.label for c in cs] == sorted(labels, key=places.fold),
                "the table is not in alphabetical order (the list opens in it)")
        kl = places.find("Kuala Lumpur")
        r.check(kl is not None and (kl.lat, kl.lon, kl.zone) == (3.1390, 101.6869, "Asia/Kuala_Lumpur"),
                f"Kuala Lumpur: {kl}")
        g = d["grid"]
        r.check(places._grid().shape == (g["height"], g["width"]) == (1800, 3600),
                f"zone grid {places._grid().shape}")
        r.check(int(places._grid().max()) < len(g["zones"]), "a grid cell names no zone")
        r.check("2026c" in d["source"] and "ODbL" in d["source"], "the sources are not named")
        for z, seq in d["zones"].items():
            ts = [t for t, _ in seq]
            if ts != sorted(ts) or not all(-12 * 3600 <= o <= 14 * 3600 for _, o in seq):
                r.check(False, f"{z}: offsets out of order or out of range")
                break
    except Exception as e:                                       # noqa: BLE001
        r.check(False, f"{type(e).__name__}: {e}")
    r.close()

    r = Rule("W2 the clocks: standard and summer time, half and quarter hours, and the instants "
             "summer time begins and ends (2026)")
    U = timezone.utc
    known = [("Asia/Kuala_Lumpur", datetime(2026, 1, 15, tzinfo=U), 8.0),
             ("Asia/Kuala_Lumpur", datetime(2026, 7, 15, tzinfo=U), 8.0),
             ("Europe/Paris", datetime(2026, 1, 15, tzinfo=U), 1.0),
             ("Europe/Paris", datetime(2026, 7, 15, tzinfo=U), 2.0),
             ("Australia/Sydney", datetime(2026, 1, 15, tzinfo=U), 11.0),
             ("Australia/Sydney", datetime(2026, 7, 15, tzinfo=U), 10.0),
             ("Australia/Adelaide", datetime(2026, 1, 15, tzinfo=U), 10.5),
             ("Asia/Kolkata", datetime(2026, 7, 15, tzinfo=U), 5.5),
             ("Asia/Kathmandu", datetime(2026, 7, 15, tzinfo=U), 5.75),
             ("America/New_York", datetime(2026, 1, 15, tzinfo=U), -5.0),
             ("America/New_York", datetime(2026, 7, 15, tzinfo=U), -4.0),
             ("Asia/Tehran", datetime(2026, 7, 15, tzinfo=U), 3.5),
             ("America/Sao_Paulo", datetime(2026, 1, 15, tzinfo=U), -3.0),
             ("Pacific/Honolulu", datetime(2030, 7, 15, tzinfo=U), -10.0),
             # the instants of the change, a second before and on it
             ("Europe/Paris", datetime(2026, 3, 29, 0, 59, 59, tzinfo=U), 1.0),
             ("Europe/Paris", datetime(2026, 3, 29, 1, 0, 0, tzinfo=U), 2.0),
             ("Europe/Paris", datetime(2026, 10, 25, 0, 59, 59, tzinfo=U), 2.0),
             ("Europe/Paris", datetime(2026, 10, 25, 1, 0, 0, tzinfo=U), 1.0),
             ("America/New_York", datetime(2026, 3, 8, 7, 0, 0, tzinfo=U), -4.0),
             ("America/New_York", datetime(2026, 11, 1, 6, 0, 0, tzinfo=U), -5.0),
             ("Australia/Sydney", datetime(2026, 4, 4, 15, 59, 59, tzinfo=U), 11.0),
             ("Australia/Sydney", datetime(2026, 4, 4, 16, 0, 0, tzinfo=U), 10.0),
             ("Australia/Sydney", datetime(2026, 10, 3, 16, 0, 0, tzinfo=U), 11.0)]
    for z, t, h in known:
        got = places.offset_s(z, t)
        r.check(got == round(h * 3600), f"{z} at {t:%Y-%m-%d %H:%M:%S} UTC: {got} s, not {h} h")
    r.check(places.offset_s(None, known[0][1]) is None and places.offset_s("Nowhere/X", known[0][1]) is None,
            "an unknown zone has an offset")
    for s_, want in ((8 * 3600, "UTC+8"), (19800, "UTC+5:30"), (20700, "UTC+5:45"), (-3 * 3600, "UTC-3"),
                     (-9000, "UTC-2:30"), (0, "UTC"), (None, "UTC")):
        r.check(places.utc_label(s_) == want, f"utc_label({s_}) = {places.utc_label(s_)}")
    # where this computer has a tz database, the stored offsets are its own
    try:
        from zoneinfo import ZoneInfo
        rng = np.random.default_rng(7)
        zones = sorted(places._data()["zones"])
        ZoneInfo(zones[0])
        n = bad = 0
        t0 = datetime(2020, 1, 1, tzinfo=U).timestamp()
        span = datetime(2036, 1, 1, tzinfo=U).timestamp() - t0
        for z in zones[::3]:
            zi = ZoneInfo(z)
            for x in rng.random(12):
                t = datetime.fromtimestamp(t0 + x * span, U)
                n += 1
                if places.offset_s(z, t) != int(t.astimezone(zi).utcoffset().total_seconds()):
                    bad += 1
        # the stored years are the tz database of 2026; a computer's own may be newer
        r.check(bad <= max(2, n // 200), f"{bad} of {n} instants differ from this computer's tz database")
    except Exception:                                            # noqa: BLE001
        pass                                    # no tz database here: the stored years are the clock
    r.close()

    r = Rule("W3 a time on the place's clock to UTC: an hour summer time skips takes the offset "
             "before it, an hour it repeats is its first, a fixed clock as typed")
    cases = [(datetime(2026, 7, 1, 12, 0), "Europe/Paris", None, datetime(2026, 7, 1, 10, 0, tzinfo=U)),
             (datetime(2026, 3, 29, 2, 30), "Europe/Paris", None, datetime(2026, 3, 29, 1, 30, tzinfo=U)),
             (datetime(2026, 10, 25, 2, 30), "Europe/Paris", None, datetime(2026, 10, 25, 0, 30, tzinfo=U)),
             (datetime(2026, 10, 25, 3, 0), "Europe/Paris", None, datetime(2026, 10, 25, 2, 0, tzinfo=U)),
             (datetime(2026, 10, 4, 2, 30), "Australia/Sydney", None, datetime(2026, 10, 3, 16, 30, tzinfo=U)),
             (datetime(2026, 10, 3, 12, 0), "Asia/Kuala_Lumpur", None, datetime(2026, 10, 3, 4, 0, tzinfo=U)),
             (datetime(2026, 10, 3, 12, 0), "Asia/Kathmandu", None, datetime(2026, 10, 3, 6, 15, tzinfo=U)),
             (datetime(2026, 10, 3, 12, 0), None, 19800, datetime(2026, 10, 3, 6, 30, tzinfo=U)),
             (datetime(2026, 10, 3, 12, 0), None, -10800, datetime(2026, 10, 3, 15, 0, tzinfo=U)),
             (datetime(2040, 7, 1, 12, 0), "Europe/Paris", None, datetime(2040, 7, 1, 10, 0, tzinfo=U))]
    for local, z, fx, want in cases:
        got = places.to_utc(local, z, fx)
        r.check(got == want, f"{local} {z or fx}: {got}, not {want}")
    # every instant of a year and back, away from the repeated hours
    rng = np.random.default_rng(3)
    bad = []
    for z in ("Europe/Paris", "America/New_York", "Australia/Sydney", "Asia/Kolkata", "America/Santiago"):
        for x in rng.random(60):
            u = datetime(2026, 1, 1, tzinfo=U) + timedelta(seconds=int(x * 365 * 86400))
            off = places.offset_s(z, u)
            local = (u + timedelta(seconds=off)).replace(tzinfo=None)
            back = places.to_utc(local, z)
            again = places.offset_s(z, u - timedelta(hours=1)) != off or \
                places.offset_s(z, u + timedelta(hours=1)) != off
            if back != u and not again:
                bad.append(f"{z} {u}: {back}")
    r.check(not bad, f"local and back: {bad[:3]}")
    r.close()

    r = Rule("W4 the time zone of a point typed by hand: a city's within 15 km, else the zone map, "
             "at sea the coast's or the sea's own")
    for lat, lon, want in ((3.139, 101.687, "Asia/Kuala_Lumpur"), (48.86, 2.35, "Europe/Paris"),
                           (27.0, 85.5, "Asia/Kathmandu"), (0.0, -150.0, "Etc/GMT+10"),
                           (-25.0, 134.0, "Australia/Darwin"), (40.0, -105.5, "America/Denver"),
                           (64.0, -20.0, "Atlantic/Reykjavik"), (-33.87, 151.21, "Australia/Sydney")):
        z, how = places.zone_at(lat, lon)
        r.check(z == want, f"({lat}, {lon}): {z} ({how}), not {want}")
    z, how = places.zone_at(2.9, 101.0)              # the Strait of Malacca, off Klang
    r.check(z == "Asia/Kuala_Lumpur", f"off the Malaysian coast: {z} ({how})")
    for c in places.cities()[::40]:
        z, how = places.zone_at(c.lat, c.lon)
        r.check(z == c.zone, f"{c.label}: {z} ({how}), not {c.zone}")
    r.check(places.name_for(3.139, 101.687) == "Kuala Lumpur, Malaysia"
            and places.name_for(3.07, 101.52).startswith("near ")
            and places.name_for(0.0, -150.0) == "0.000, -150.000", "names of typed points")
    r.close()

    r = Rule("W5 the city search: a name's start, a word's start, accents and other alphabets, "
             "the larger first; nothing typed, every city in order")
    for q, first in (("kuala", "Kuala Lumpur, Malaysia"), ("san f", "San Francisco, United States"),
                     ("sao paulo", "São Paulo, Brazil"), ("São Paulo", "São Paulo, Brazil"),
                     ("tokyo", "Tokyo, Japan"), ("new york", "New York, United States"),
                     ("london", "London, United Kingdom"), ("kathm", "Kathmandu, Nepal"),
                     ("paris france", "Paris, France"), ("johor", "Johor Bahru, Malaysia")):
        hits = places.search(q)
        r.check(bool(hits) and hits[0].label == first,
                f"{q!r}: {[h.label for h in hits[:3]]}, not {first} first")
    r.check(len(places.search("")) == 1584 and places.search("") == places.cities(), "nothing typed")
    r.check(places.search("xyzzy") == [], "a name no city has")
    r.check("San Diego, United States" not in [h.label for h in places.search("san f")],
            "'san f' finds San Diego (the f of California): words are matched from their start")
    r.check(any(h.label == "Pittsburgh, United States" for h in places.search("burg")),
            "inside a word, when no word begins so")
    f = places.find("Paris, France")
    r.check(f is not None and f.label == "Paris, France", "an exact label")
    r.check(places.find("Xyzzyville") is None, "find: no such city")
    r.close()

    r = Rule("W6 latitude and longitude typed: decimal, N/S/E/W, an offset after them; nothing "
             "that is not on the Earth")
    for text, want in (("48.86, 2.35", (48.86, 2.35, None)), ("48.86 2.35", (48.86, 2.35, None)),
                       ("3.14N 101.69E", (3.14, 101.69, None)), ("33.87 S, 151.21 E", (-33.87, 151.21, None)),
                       ("-33.87,151.21 +10", (-33.87, 151.21, 10.0)), ("40.7, -74.0 UTC-5", (40.7, -74.0, -5.0)),
                       ("27.7, 85.3 +5:45", (27.7, 85.3, 5.75)), ("10, 200", (10.0, -160.0, None)),
                       ("91, 0", None), ("0, 400", None), ("10, 20 +15", None), ("paris", None),
                       ("48.86", None), ("", None)):
        got = places.parse_coords(text)
        ok = (got is None and want is None) or (got is not None and want is not None and
                                                all(abs(a - b) < 1e-9 if b is not None else a is None
                                                    for a, b in zip(got, want)))
        r.check(ok, f"{text!r}: {got}, not {want}")
    r.close()

    r = Rule("W7 a date and an hour typed in the dropdowns (day first; no year: the nearest date)")
    today = date(2026, 10, 3)
    for text, want in (("today", date(2026, 10, 3)), ("Tomorrow", date(2026, 10, 4)),
                       ("yesterday", date(2026, 10, 2)), ("+3", date(2026, 10, 6)), ("-2", date(2026, 10, 1)),
                       ("in 3 days", date(2026, 10, 6)), ("3 days ago", date(2026, 9, 30)),
                       ("2026-10-05", date(2026, 10, 5)), ("2026/1/5", date(2026, 1, 5)),
                       ("5/10/2026", date(2026, 10, 5)), ("5-10-26", date(2026, 10, 5)), ("5.10", date(2026, 10, 5)),
                       ("5 Oct", date(2026, 10, 5)), ("5th October", date(2026, 10, 5)), ("Oct 5", date(2026, 10, 5)),
                       ("October 5, 2026", date(2026, 10, 5)), ("Mon 05 Oct 2026", date(2026, 10, 5)),
                       ("2 jan", date(2027, 1, 2)), ("25 dec", date(2026, 12, 25)), ("3 apr", date(2027, 4, 3)),
                       ("31/2/2026", None), ("13/13/2026", None), ("oct", None), ("sat", None), ("5", None),
                       ("hello", None), ("", None), ("1 jan 1850", None)):
        got = places.parse_date(text, today)
        r.check(got == want, f"date {text!r}: {got}, not {want}")
    for text, want in (("15:30", dtime(15, 30)), ("15.30", dtime(15, 30)), ("15h30", dtime(15, 30)),
                       ("1530", dtime(15, 30)), ("930", dtime(9, 30)), ("15", dtime(15)), ("3pm", dtime(15)),
                       ("3:30 pm", dtime(15, 30)), ("12am", dtime(0)), ("12pm", dtime(12)),
                       ("9 p.m.", dtime(21)), ("noon", dtime(12)), ("midnight", dtime(0)), ("0", dtime(0)),
                       ("24", None), ("2400", None), ("15:60", None), ("13pm", None), ("0am", None),
                       ("abc", None), ("", None)):
        got = places.parse_time(text)
        r.check(got == want, f"time {text!r}: {got}, not {want}")
    r.close()

    r = Rule("W8 the forecast asked for holds the time asked for (to its last hour and half an "
             "hour past it): the usual window unchanged, 16 days ahead at most, the archive for the past")
    now = datetime(2026, 10, 3, 1, 37, tzinfo=U)
    reach = sounding.forecast_reach(now)
    r.check(reach == datetime(2026, 10, 18, 23, tzinfo=U), f"the forecast reaches {reach}")
    for days in (-400, -40, -3.5, -3.0, -1.0, 0.0, 0.5, 3.6, 7.0, 9.0, 9.2, 12.0, 15.5, 15.9):
        w = now + timedelta(days=days)
        try:
            ep, win, a, b = sounding.request_plan(w, now)
        except sounding.BeyondForecast as e:
            r.check(False, f"{days} days: {e}")
            continue
        r.check(a <= w <= b + timedelta(minutes=30), f"{days} days: {a}..{b} does not hold {w}")
        if -3.0 <= days <= 9.0:
            r.check(ep == sounding.FORECAST_API and win == {"past_days": "3", "forecast_days": "10"},
                    f"{days} days: the usual request changed: {win}")
        if days < -3.0:
            r.check(ep == sounding.HISTORICAL_API, f"{days} days: not the archive")
        rep_, rwin = sounding.region_plan(w, now)
        if rep_ == sounding.FORECAST_API:
            pd, fd = int(rwin["past_days"]), int(rwin["forecast_days"])
            r.check(1 <= pd <= 92 and 5 <= fd <= 16, f"{days} days: region window {rwin}")
            ra = datetime(2026, 10, 3 - pd, tzinfo=U) if pd < 3 else datetime(2026, 10, 3, tzinfo=U) - timedelta(days=pd)
            rb = datetime(2026, 10, 3, 23, tzinfo=U) + timedelta(days=fd - 1)
            r.check(ra <= w <= rb + timedelta(minutes=30),
                    f"{days} days: the region {rwin} does not hold {w}")
        if days == 0.0:
            r.check(rwin == {"past_days": "1", "forecast_days": "5"}, f"the usual region request changed: {rwin}")
    # late in the UTC day, nine days ahead is past the usual window's last hour
    late = datetime(2026, 1, 5, 23, 49, tzinfo=U)
    w8 = late + timedelta(days=8.99)
    ep8, win8, a8, b8 = sounding.request_plan(w8, late)
    r.check(a8 <= w8 <= b8 + timedelta(minutes=30), f"{w8} asked from {late}: {win8} holds {a8}..{b8}")
    for days in (16.0, 30.0):
        try:
            sounding.request_plan(now + timedelta(days=days), now)
            r.check(False, f"{days} days ahead: asked for a forecast that does not exist")
        except sounding.BeyondForecast:
            r.check(True)
    r.close()


# ------------------------------------------------------ the third stress round --

def _column(code, layer=(300.0, 6000.0), cover=0.95, rh_layer=97.0, low_cover=90.0, mid_cover=95.0,
            elevation=0.0, rh2=80.0, t_sfc=18.0, lapse=6.5, lat=48.9):
    """A forecast-like sounding: pressure levels with the model's cloud
    fraction (cover inside `layer`, none outside), a surface row without
    one (as Open-Meteo gives it), and the present weather `code`."""
    from .sounding import Level, Sounding, P0, dewpoint_from_rh
    levels = []
    for z in [elevation + 2.0] + [float(z) for z in range(250, 14001, 250) if z > elevation + 30.0]:
        t = t_sfc - lapse * (z - elevation) / 1000.0
        p = P0 * (1.0 - z / 44330.0) ** (1.0 / 0.1903)
        inside = layer[0] <= z <= layer[1]
        rh = rh2 if z == elevation + 2.0 else (rh_layer if inside else 55.0)
        cc = -1.0 if z == elevation + 2.0 else (cover if inside else 0.0)
        levels.append(Level(p=p, z=z, t=t, rh=rh, td=dewpoint_from_rh(t, rh), u=4.0, v=2.0, cc=cc))
    surface = {"weather_code": float(code), "cloud_cover_low": low_cover, "cloud_cover_mid": mid_cover,
               "cloud_cover_high": 0.0, "precipitation": 1.0 if code >= 51 else 0.0, "visibility": 20000.0,
               "temperature_2m": t_sfc}
    s = Sounding(levels=levels, lat=lat, lon=2.3, when=datetime(2026, 10, 1, 9, tzinfo=timezone.utc),
                 surface=surface, source="test")
    s.elevation = elevation
    return s


def test_round3():
    """X1-X8: what the third stress round found (3 Oct 2026)."""
    r = Rule("X1 a layer of cells is no deeper than its elements allow: Altocumulus, "
             "Stratocumulus and Cirrocumulus half as deep as wide (drawn as deep as their cloudy "
             "band, cells a few hundred metres wide stood 1-2 km tall)")
    kl = sounding.standard_sounding(3.14, 101.69, profile="tropical-fair")
    for g, sp, base, thick in (("Ac", "str", 2796.0, 1600.0), ("Sc", "str", 232.0, 1721.0),
                               ("Ac", "flo", 4000.0, 1500.0), ("Sc", "len", 1900.0, 1800.0),
                               ("Cc", "str", 7600.0, 900.0)):
        d = scene.deck_from_spec(CloudSpec(g, sp), kl, base_m=base, coverage=0.3, thickness_m=thick)
        k = scene.CELL_ASPECT[g]
        r.check(d.thickness <= max(k * d.element_w_m, 50.0) + 1.0,
                f"{g} {sp}: {d.thickness:.0f} m deep with elements {d.element_w_m:.0f} m wide")
        r.check(any("half as deep" in x for x in d.reasons) or d.thickness >= thick - 1.0,
                f"{g} {sp}: the panel does not say why it is drawn {d.thickness:.0f} m deep")
    d = scene.deck_from_spec(CloudSpec("Ac", "cas"), kl, base_m=4000.0, coverage=0.3,
                             thickness_m=1500.0, turret_rise_m=1200.0)
    layer = d.turret * d.thickness
    r.check(layer <= max(0.5 * d.element_w_m, 50.0) + 1.0
            and abs(d.top_m - (4000.0 + 1500.0 + 1200.0)) < 1.0,
            f"castellanus: common layer {layer:.0f} m (elements {d.element_w_m:.0f} m wide), turrets' top "
            f"{d.top_m:.0f} m (the parcel's: 6700)")
    for g in ("St", "As", "Ns", "Cs", "Cb"):
        d = scene.deck_from_spec(CloudSpec(g), kl, base_m=1000.0, coverage=0.5, thickness_m=2000.0)
        r.check(abs(d.thickness - min(2000.0, scene.GENUS_DEFAULTS[g]["max_thick"])) < 1.0,
                f"{g} is not a layer of cells: {d.thickness:.0f} m")
    r.close()

    r = Rule("X2 a deep overcast layer from the low étage is named by its precipitation (Atlas): "
             "rain or snow Nimbostratus, showers or thunder Cumulonimbus, drizzle neither")
    for code, want, not_want in ((63, "Ns", None), (73, "Ns", None), (81, "Cb", "Ns"),
                                 (95, "Cb", "Ns"), (53, None, "Ns"), (3, None, "Ns")):
        decks = scene.diagnose(_column(code))
        gens = [d.spec.genus for d in decks]
        if want:
            r.check(want in gens, f"weather code {code}: {gens}, no {want}")
        if not_want:
            r.check(not_want not in gens, f"weather code {code}: {gens} has {not_want}")
    # rain under a layer whose own peak fraction is short of overcast, while the
    # forecast's low and middle cover say overcast
    decks = scene.diagnose(_column(61, cover=0.7, low_cover=81.0, mid_cover=100.0))
    r.check("Ns" in [d.spec.genus for d in decks], f"overcast by the étage covers: {[d.spec.abbrev() for d in decks]}")
    # a deep layer based in the middle étage stays the middle étage's
    decks = scene.diagnose(_column(95, layer=(3500.0, 9000.0)))
    r.check(all(not (d.spec.genus == "Cb" and d.base_m > 3000.0) for d in decks),
            f"a Cumulonimbus based at {[round(d.base_m) for d in decks if d.spec.genus == 'Cb']} m")
    r.close()

    r = Rule("X3 showers or thunder under an overcast low layer: the convective cloud is still "
             "there (it was dropped whenever a low layer was overcast)")
    from .sounding import Level
    for code, want in ((95, True), (80, True), (3, False)):
        s = sounding.standard_sounding(3.14, 101.69, profile="humid-deep")
        # an overcast low layer over the deep moist column
        s.levels = [Level(p=l.p, z=l.z, t=l.t, rh=max(l.rh, 96.0) if 600 <= l.z <= 1500 else l.rh,
                          td=l.td, u=l.u, v=l.v, cc=0.95 if 600 <= l.z <= 1500 else l.cc) for l in s.levels]
        s.surface["weather_code"] = float(code)
        decks = scene.diagnose(s)
        has_cb = any(d.spec.genus in ("Cb", "Cu") for d in decks)
        overcast = any(d.base_m < 1500 and d.coverage >= 0.85 for d in decks)
        r.check(overcast, f"code {code}: no overcast low layer to test with: {[d.spec.abbrev() for d in decks]}")
        r.check(has_cb == want, f"code {code}: {[d.spec.abbrev() for d in decks]}")
    r.close()

    r = Rule("X4 the screen-level humidity makes no cloud: no layer at or under the ground "
             "(98 % at 2 m drew Stratus based below the station at noon in 24 km visibility)")
    for elev in (0.0, 200.0, 57.0):
        s = _column(3, layer=(9000.0, 12000.0), cover=0.8, elevation=elev, rh2=98.0)
        decks = scene.diagnose(s)
        low = [d.spec.abbrev() for d in decks if d.base_m < elev + 300.0]
        r.check(not low, f"station at {elev:.0f} m, 98 % at 2 m: {low}")
    s = _column(3, layer=(9000.0, 12000.0), cover=0.8, rh2=100.0)
    r.check(True)                               # saturation at 2 m may make fog-level cloud
    r.close()

    r = Rule("X5 the offline Stratus profile has no layer steeper than the dry adiabat, and its "
             "deck no turrets (21 K/km at 1.3-1.6 km made it Stratocumulus castellanus)")
    st = sounding.standard_sounding(3.14, 101.69, profile="stable-stratus")
    worst = max((st.t_at(z) - st.t_at(z + 100.0)) / 0.1 for z in range(0, 6000, 100))
    r.check(worst < 9.8, f"steepest lapse {worst:.1f} K/km")
    decks = scene.diagnose(st)
    r.check(bool(decks) and all(d.spec.species != "cas" for d in decks),
            f"{[d.spec.abbrev() for d in decks]}")
    r.close()

    import re
    from . import regimes
    r = Rule("X6 CIN is given only where a parcel has a level of free convection (with none, the "
             "negative area summed to the top of the profile read -14,000 to -18,500 J/kg in the panel)")
    kinds = set()
    for prof in ("tropical-fair", "deep-moist", "stable-stratus", "dry-cirrus"):
        s = sounding.standard_sounding(3.14, 101.69, profile=prof)
        cape, cin, zl, lfc, el, _ = s.parcel(mixed_depth=400)
        txt = sounding.inhibition_text(cin, lfc)
        line = " ".join(e.line() for e in regimes.choose(s).evidence if "CAPE" in e.line())
        kinds.add(lfc is None)
        if lfc is None:
            r.check(txt == "no LFC" and "no LFC" in line and "CIN" not in line, f"{prof}: {line}")
        else:
            r.check(-1000.0 < cin <= 0.0 and txt == f"CIN {cin:.0f} J/kg" and txt in line,
                    f"{prof}: {line}")
        for d in scene.diagnose(s):
            for x in d.reasons:
                m = re.search(r"CIN (-?\d+)", x)
                r.check(m is None or abs(int(m.group(1))) < 1000, f"{prof} {d.spec.abbrev()}: {x}")
    r.check(kinds == {True, False}, "profiles with and without a level of free convection")
    r.check(sounding.inhibition_text(-45.4, 900.0, unit=False) == "CIN -45"
            and sounding.inhibition_text(-17424.0, None, unit=False) == "no LFC", "the panel's form")
    r.close()

    from types import SimpleNamespace
    r = Rule("X7 lenticularis is separate lenses in a strong wind across a stable layer: not a sheet "
             "of 60 % of the sky or more (by wind alone, overcast opacus Stratocumulus was lenticularis)")
    stable = sounding.standard_sounding(3.14, 101.69, profile="stable-stratus")
    plain = sounding.standard_sounding(3.14, 101.69, profile="tropical-fair")
    ph = {"lapse_above": 0.0, "cape": 0.0, "lapse_in": -3.0, "rh_below": 80.0}
    lay = SimpleNamespace(base=800.0, top=1100.0, cover=1.0)
    for snd, kind, wind, cov, want in ((stable, "stable", 22.0, 1.0, "str"), (stable, "stable", 22.0, 0.3, "len"),
                                       (plain, "standard lapse", 22.0, 0.3, "str"),
                                       (stable, "stable", 8.0, 0.3, "str"), (stable, "stable", 22.0, 0.59, "len"),
                                       (stable, "stable", 22.0, 0.6, "str")):
        spec, _, why = scene._low_species(snd, lay, ph, wind, cov)
        r.check(spec.species == want, f"Sc, {kind} layer, {wind:.0f} m/s, cover {cov}: {spec.abbrev()} "
                                      f"(want {want}) {why[-1]}")
        if want == "str" and wind > 15:
            r.check(why[-1].endswith("not lenticularis"), f"the panel does not say why: {why}")
    for cov, want in ((1.0, "str"), (0.3, "len")):
        spec, _, _ = scene._middle_species(stable, lay, ph, 3.14, 9, 22.0, cov)
        r.check(spec.species == want, f"Ac, stable layer, 22 m/s, cover {cov}: {spec.abbrev()} (want {want})")
    r.close()

    r = Rule("X8 a layer named Nimbostratus because the forecast's low and middle cover say overcast "
             "is drawn overcast, as much as the larger of the two étages it spans (at its own peak "
             "fraction it kept gaps: 70 % under 81/100 %)")
    for frac, low, mid in ((0.7, 81.0, 100.0), (0.5, 90.0, 85.0), (0.95, 81.0, 92.0), (0.97, 81.0, 92.0)):
        ns = [d for d in scene.diagnose(_column(61, cover=frac, low_cover=low, mid_cover=mid))
              if d.spec.genus == "Ns"]
        want = max(frac, max(low, mid) / 100.0)
        r.check(len(ns) == 1 and abs(ns[0].coverage - want) < 1e-6,
                f"peak fraction {frac}, low {low} %, mid {mid} %: "
                f"{[round(d.coverage, 3) for d in ns]} (want {want:.2f})")
    r.close()


def test_round3_gl():
    """X9: the still picture after the sky changes (needs OpenGL)."""
    try:
        import moderngl
        ctx = moderngl.create_standalone_context(backend="egl", require=330)
    except Exception:
        try:
            import moderngl
            ctx = moderngl.create_standalone_context(require=330)
        except Exception as e:                                  # noqa: BLE001
            RESULTS.append(("X9 the still picture after a change", 0,
                            [f"skipped, no OpenGL context ({type(e).__name__})"]))
            return
    from . import gl_sky, timeline
    r = Rule("X9 when the sky changes while the picture stands still, the picture is the new sky: "
             "the old one stayed in it at 94 % a frame (a clear sky with the Sun's disc under a "
             "new Nimbostratus a dozen frames later)")
    snd = sounding.standard_sounding(48.9, 2.5, profile="humid-deep")
    geom = astro.sky_geometry(datetime(2026, 10, 1, 8, tzinfo=timezone.utc), 48.9, 2.5)
    cam = gl_sky.Camera(150.0, 20.0, 80.0)
    ns = scene.deck_from_spec(CloudSpec("Ns"), snd, base_m=400.0, coverage=1.0, thickness_m=6000.0)
    real = timeline.build_realization(timeline.realization_params(ns), ns.seed & 0x3FFFFFFF,
                                      gl_sky.MAP_SIZE)
    st = gl_sky.static_state(ns, 0, real)

    def run(seq):
        rd = gl_sky.SkyRenderer(ctx, 160, 120, quality="low")
        rd.adaptive = False
        img = None
        for states, uploads, n, anim in seq:
            rd.set_deck_states(states, uploads)         # the app's path (timeline states)
            for k in range(n):
                img = rd.read_image(rd.render(cam, geom, 0.0, accumulate=True,
                                              animating=anim and k == 0)).astype(float)
        return img
    fresh = run([([st], [(0, 0, real["pack"])], 12, False)])
    clear = run([([], [], 12, False)])
    after = run([([], [], 30, False), ([st], [(0, 0, real["pack"])], 12, True)])
    share = float(((after - fresh) * (clear - fresh)).sum() / max(((clear - fresh) ** 2).sum(), 1e-9))
    r.check(abs(clear.mean() - fresh.mean()) > 40.0, "the two skies differ")
    r.check(abs(share) < 0.02, f"{share:.0%} of the sky before the change is still in the picture")
    r.check(abs(after.mean() - fresh.mean()) < 2.0,
            f"mean {after.mean():.1f}, the new sky alone {fresh.mean():.1f}")
    r.close()


def test_ground_look():
    """Y1-Y3, Y6, Y7: clouds as they look from the ground (4 Oct 2026)."""
    from . import app, timeline
    r = Rule("Y1 an element of Altocumulus, Cirrocumulus or Stratocumulus seen anywhere above "
             "30 degrees is inside its Atlas band: wide enough 30 degrees up, small enough "
             "overhead, where it is twice as near (sized at 30 degrees alone a Paris "
             "altocumulus was 6.6 degrees wide overhead)")
    n_dec = 0
    for lat, lon, elev in ((3.14, 101.69, 0.0), (49.01, 2.55, 0.0), (64.1, -21.9, 0.0),
                           (-16.5, -68.15, 3640.0)):
        snd = sounding.standard_sounding(lat, lon, profile="tropical-fair")
        snd.elevation = elev
        for g in ("Ac", "Cc", "Sc"):
            lo, hi = atlas.ELEMENT_WIDTH_DEG[g]
            blo, bhi = atlas.etage_range_m(g, lat)
            for sp in ("str", "len", "flo" if g != "Sc" else "cas"):
                for base in np.linspace(max(blo, elev + 300.0), max(bhi, elev + 900.0), 5):
                    for cover in (0.15, 0.5, 0.95):
                        d = scene.deck_from_spec(CloudSpec(g, sp), snd, base_m=float(base),
                                                 coverage=cover, seed=n_dec + 3)
                        n_dec += 1
                        h = max(d.base_m - elev, 150.0)
                        w30 = atlas.metres_to_angular_width(d.element_w_m, h, 30.0)
                        w90 = atlas.metres_to_angular_width(d.element_w_m, h, 90.0)
                        floor = lo if lo > 0 else 0.20
                        r.check(w30 >= floor * 0.999,
                                f"{g} {sp} {base:.0f} m (station {elev:.0f} m) cover {cover}: "
                                f"{w30:.2f}° seen 30° up, under {floor}°")
                        r.check((w90 <= hi * 1.001) if hi < 11.0 else (w30 <= 45.0 * 1.001),
                                f"{g} {sp} {base:.0f} m (station {elev:.0f} m) cover {cover}: "
                                f"{w90:.2f}° overhead, over {hi}°")
                        r.check(abs(d.element_w_m - scene.element_width(d.element_m, d.coverage))
                                < 1e-6, f"{g} {sp}: element width is not the cloudy part of a cell")
    r.check(n_dec >= 500, f"only {n_dec} decks")
    # and the panel says it, in degrees, at 30 degrees up and overhead
    d = scene.deck_from_spec(CloudSpec("Ac", "str"), sounding.standard_sounding(49.01, 2.55),
                             base_m=2796.0, coverage=0.32, thickness_m=4220.0)
    r.check(any("seen 30 deg up" in x and "overhead" in x for x in d.reasons),
            f"the reasons do not give the element's size: {d.reasons}")
    r.close()

    r = Rule("Y2 the elements of a layer are flat: no deeper than half as wide (Altocumulus, "
             "Stratocumulus, Cirrocumulus; castellanus turrets rise above the common layer)")
    snd = sounding.standard_sounding(49.01, 2.55, profile="tropical-fair")
    for g in ("Ac", "Cc", "Sc"):
        blo, bhi = atlas.etage_range_m(g, 49.0)
        for sp in ("str", "len", "und", "cas" if g != "Cc" else "flo"):
            if sp == "und":
                spec = CloudSpec(g, "str", ["un"])
            else:
                spec = CloudSpec(g, sp)
            for base in np.linspace(max(blo, 300.0), bhi, 4):
                for band in (150.0, 800.0, 4000.0):
                    d = scene.deck_from_spec(spec, snd, base_m=float(base), coverage=0.5,
                                             thickness_m=band, turret_rise_m=800.0)
                    depth = d.top_m - d.base_m
                    layer = d.turret * depth if d.turret > 0.0 else depth
                    r.check(layer <= max(0.5 * d.element_w_m, 50.0) + 1.0,
                            f"{spec.abbrev()} {base:.0f} m band {band:.0f}: {layer:.0f} m deep, "
                            f"{d.element_w_m:.0f} m wide")
                    capped = band > max(0.5 * d.element_w_m, 50.0) + 1.0
                    r.check(not capped or any("half as deep" in x for x in d.reasons),
                            f"{spec.abbrev()}: drawn {layer:.0f} of {band:.0f} m, the panel does not say why")
    r.close()

    r = Rule("Y3 a layer's elements are rounded masses, not balls: each stirred within itself, "
             "the cover still exact, blurred edges; only Altocumulus, Stratocumulus and "
             "Cirrocumulus have them (not lenticularis); a layer changing its kind changes its "
             "shape smoothly")
    snd = sounding.standard_sounding(49.01, 2.55, profile="tropical-fair")
    for g in atlas.GENUS_ORDER:
        for sp in (None, "len"):
            if sp == "len" and "len" not in atlas.GENERA[g]["species"]:
                continue
            d = scene.deck_from_spec(CloudSpec(g, sp), snd)
            want = 1.0 if (g in scene.CELLULAR and sp != "len") else 0.0
            r.check(d.cell_dome == want, f"{g} {sp}: cell_dome {d.cell_dome}")
            r.check(timeline.realization_params(d)["lumpy"] == want,
                    f"{g} {sp}: its patterns are built lumpy={timeline.realization_params(d)['lumpy']}")
            soft = (0.12 if g in ("Cu", "Cb") else 0.60 if g in ("Ac", "Cc") else
                    0.45 if g == "Sc" else 0.3)
            r.check(abs(d.edge_softness - soft) < 1e-9, f"{g}: edge softness {d.edge_softness}")
    f0 = noise.deck_fields(384, 12000.0, 600.0, cellularity=0.95, seed=11, lumpy=0.0)
    f1 = noise.deck_fields(384, 12000.0, 600.0, cellularity=0.95, seed=11, lumpy=1.0)
    for cov in (0.15, 0.5, 0.85):
        m = f1["z"] > noise.zstar_for_cover(cov)
        r.check(abs(float(m.mean()) - cov) < 0.02, f"lumpy: cover {m.mean():.3f} for {cov}")
    # each element stirred within itself: the same cells, not the same shapes
    cc = float(np.corrcoef(f0["z"].ravel(), f1["z"].ravel())[0, 1])
    r.check(0.5 < cc < 0.97, f"lumpy against plain pattern: correlation {cc:.3f}")
    a = scene.deck_from_spec(CloudSpec("Ac", "str"), snd, base_m=3500.0, coverage=0.5)
    b = scene.deck_from_spec(CloudSpec("As"), snd, base_m=3500.0, coverage=0.9)
    vals = [timeline.interp_deck(a, b, f).cell_dome for f in np.linspace(0.0, 1.0, 11)]
    r.check(all(abs(v - (1.0 - f)) < 1e-6 for v, f in zip(vals, np.linspace(0.0, 1.0, 11))),
            f"Ac to As: cell_dome {np.round(vals, 3)}")
    r.close()

    r = Rule("Y7 towering convective clouds are drawn as plumes (Cumulus congestus, "
             "Cumulonimbus): Deck.tower set for them alone, packed for the renderer, and a layer "
             "changing its kind changes its shape smoothly")
    snd = sounding.standard_sounding(49.01, 2.55, profile="tropical-fair")
    from . import gl_sky as _gs
    for g in atlas.GENUS_ORDER:
        for sp in [None] + list(atlas.GENERA[g]["species"]):
            d = scene.deck_from_spec(CloudSpec(g, sp), snd)
            want = 1.0 if (g == "Cb" or (g == "Cu" and sp == "con")) else 0.0
            r.check(d.tower == want, f"{g} {sp}: tower {d.tower}")
            r.check(float(_gs._deck_rows(d)[9, 1]) == want, f"{g} {sp}: row 9.y {_gs._deck_rows(d)[9, 1]}")
    a = scene.deck_from_spec(CloudSpec("Cu", "med"), snd, coverage=0.3)
    b = scene.deck_from_spec(CloudSpec("Cu", "con"), snd, coverage=0.3)
    vals = [timeline.interp_deck(a, b, f).tower for f in np.linspace(0.0, 1.0, 6)]
    r.check(all(abs(v - f) < 1e-6 for v, f in zip(vals, np.linspace(0.0, 1.0, 6))),
            f"Cu med to Cu con: tower {np.round(vals, 3)}")
    r.close()

    r = Rule("Y6 the program opens at a 60 degree view: at 80 the frame stretched clouds at its "
             "top and bottom edges 1.7 times, at 60 1.33")
    fov = float(app.DEFAULT_VIEW["fov"])
    r.check(fov == 60.0, f"default view {fov}°")
    r.check(1.0 / math.cos(math.radians(fov / 2.0)) ** 2 < 1.34, "edge stretch")
    r.check(app.DEFAULT_VIEW["az"] == 150.0 and app.DEFAULT_VIEW["alt"] == 18.0,
            "the default direction is as before (az 150, 18° up)")
    r.close()


def test_ground_look_gl():
    """Y4-Y5: the light of a layer of rounded masses; Y8: towers that are not walls
    (4 Oct 2026)."""
    try:
        import moderngl
        ctx = moderngl.create_standalone_context(backend="egl", require=330)
    except Exception:
        try:
            import moderngl
            ctx = moderngl.create_standalone_context(require=330)
        except Exception as e:                                  # noqa: BLE001
            for name in ("Y4 an element of a layer has the deck's optical depth in its middle",
                         "Y5 an element lit from the side has a shaded side",
                         "Y8 a cumulonimbus's tower is not a wall"):
                RESULTS.append((name, 0, [f"skipped, no OpenGL context ({type(e).__name__})"]))
            return
    from . import gl_sky, shaders
    r = Rule("Y4 an element of a layer has the deck's optical depth in its middle: the extinction "
             "of a layer of rounded masses is raised by what its shape leaves out (it was drawn "
             "about three times thinner than the panel said); every other layer as before")
    import re
    m = re.search(r"const float DOME_COLUMN = ([0-9.]+);", shaders.SKY_FRAG)
    r.check(m is not None and abs(float(m.group(1)) - gl_sky.DOME_COLUMN) < 1e-9,
            f"shaders.py DOME_COLUMN {m.group(1) if m else None}, gl_sky {gl_sky.DOME_COLUMN}")
    snd = sounding.standard_sounding(49.01, 2.55, profile="tropical-fair")
    for g in atlas.GENUS_ORDER:
        d = scene.deck_from_spec(CloudSpec(g), snd)
        row = gl_sky._deck_rows(d)
        k = gl_sky.DOME_COLUMN if d.cell_dome == 1.0 else 1.0
        r.check(abs(float(row[0, 2]) - d.sigma_e * k) <= 1e-6 * k * d.sigma_e + 1e-12,
                f"{g}: extinction {row[0, 2]:.5g}, sigma_e {d.sigma_e:.5g}")
        r.check(float(row[6, 0]) == d.cell_dome, f"{g}: row 6.x {row[6, 0]}")
    r.close()

    r = Rule("Y5 an element lit from the side has a shaded side: brighter toward the Sun than "
             "away from it (the two-stream light of a layer without edges drew elements evenly "
             "lit - Monte Carlo: twice as bright)")
    geom = astro.sky_geometry(datetime(2026, 9, 28, 12, tzinfo=timezone.utc), 49.01, 2.55)
    d = scene.deck_from_spec(CloudSpec("Ac", "str"), snd, base_m=2796.0, coverage=0.6,
                             thickness_m=4220.0, seed=1000)
    rend = gl_sky.SkyRenderer(ctx, 300, 420, quality="high")
    rend.ground_elev = 100.0
    rend.set_decks([d])
    from scipy import ndimage
    # looking across the Sun's azimuth (Sun at 186 degrees, to the right)
    cam = gl_sky.Camera(96.0, 40.0, 60.0)
    for _ in range(16):
        img = rend.read_image(rend.render(cam, geom, 0.0, accumulate=True))
    im = img.astype("f8")[:220]
    lum = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]
    cloud = ndimage.binary_erosion((im[..., 2] - im[..., 0]) < 20, iterations=1)
    lab, n = ndimage.label(cloud)
    left, right = [], []
    for i in range(1, n + 1):
        mk = lab == i
        if mk.sum() < 150:
            continue
        xs = np.nonzero(mk)[1]
        x0, x1 = np.percentile(xs, 10), np.percentile(xs, 90)
        cols = np.arange(mk.shape[1])[None, :]
        left.append(lum[mk & (cols < x0 + (x1 - x0) / 3)].mean())
        right.append(lum[mk & (cols > x1 - (x1 - x0) / 3)].mean())
    r.check(len(left) >= 15, f"only {len(left)} elements measured")
    ratio = float(np.mean(left) / max(np.mean(right), 1e-6)) if left else 1.0
    r.check(ratio < 0.95, f"away from the Sun / toward it {ratio:.3f} on screen (evenly lit: 0.98)")
    r.close()

    r = Rule("Y8 a cumulonimbus's tower is not a wall: seen across from mid-height its outline "
             "bulges in and out with height (the cell's outline stood up drew boxes and slabs)")
    geom8 = astro.sky_geometry(datetime(2026, 9, 28, 12, tzinfo=timezone.utc), 49.01, 2.55)

    def persistence(tower):
        d8 = scene.deck_from_spec(CloudSpec("Cb", "cap"), snd, base_m=1000.0, coverage=0.2, seed=37)
        d8.tower = tower
        rd8 = gl_sky.SkyRenderer(ctx, 200, 120, quality="medium")
        rd8.observer_alt = d8.base_m + 0.4 * (d8.top_m - d8.base_m)
        rd8.adaptive = False
        rd8.set_decks([d8])
        for _ in range(6):
            im8 = rd8.read_image(rd8.render(gl_sky.Camera(66.0, 0.0, 60.0), geom8, 0.0,
                                            accumulate=True)).astype("f8")
        cl = ((im8.max(axis=2) - im8.min(axis=2)) < 40) & (im8.mean(axis=2) > 110)
        rows = cl[:int(cl.shape[0] * 0.6)]
        edges = np.abs(np.diff(rows.astype(int), axis=1)) > 0
        same = tot = 0
        for y in range(rows.shape[0] - 1):
            xs = np.nonzero(edges[y])[0]
            nx = np.nonzero(edges[y + 1])[0]
            if len(xs) and len(nx):
                same += int((np.abs(xs[:, None] - nx[None, :]).min(axis=1) == 0).sum())
            tot += len(xs)
        return same / max(tot, 1), tot
    p1, n1 = persistence(1.0)
    p0, n0 = persistence(0.0)
    r.check(n1 > 100 and n0 > 100, f"too few outline points ({n1}, {n0})")
    r.check(p0 > 0.45, f"control: the old shape's walls keep their edge from row to row {p0:.2f}")
    r.check(p1 < 0.42, f"the plumes' outline keeps its edge from row to row {p1:.2f} (walls {p0:.2f})")
    r.close()


def test_air_light_gl():
    """Y9: the light of the air in a cloud's shadow (4 Oct 2026).  The sky
    shader's own atmosphere() is run along single rays with the shadow map
    replaced by a uniform cloud of optical depth tau."""
    name = ("Y9 seen from above, a cloud's shadow darkens the hazy air below it; from the ground "
            "the shafts toward a low Sun stay")
    try:
        import moderngl
        try:
            ctx = moderngl.create_standalone_context(backend="egl", require=330)
        except Exception:                                        # noqa: BLE001
            ctx = moderngl.create_standalone_context(require=330)
    except Exception as e:                                       # noqa: BLE001
        RESULTS.append((name, 0, [f"skipped, no OpenGL context ({type(e).__name__})"]))
        return
    from . import gl_sky, shaders
    r = Rule("Y9 seen from above, a cloud's shadow darkens the hazy air below it, whatever the "
             "cloud's optical depth and wherever the Sun (the diffuse light under a cloud was taken "
             "as scattered evenly every way: from 15 km the shadows came out up to four times "
             "brighter than the sunlit air, white trails radiating from the point opposite the "
             "Sun); from the ground the shafts toward a low Sun stay, and seen level the air under "
             "a cloud is at most three times as bright as in sunlight (a thin veil made it six); "
             "the diffuse field's phase over the upper hemisphere is the haze's, integrated")
    U = timezone.utc
    try:
        rr = gl_sky.SkyRenderer(ctx, 64, 48, quality="medium")
        rr.set_decks([])
        rr.haze = gl_sky.haze_from_visibility(5000.0)
        src = shaders.SKY_FRAG
        k = src.rindex("void main() {")

        def program(main):
            p = ctx.program(vertex_shader=shaders.VERT, fragment_shader=src[:k] + main)
            for nm, unit in rr.units.items():
                if nm in p:
                    p[nm].value = unit
            if "DeckBlock" in p:
                p["DeckBlock"].binding = rr.deck_binding
            return p, ctx.vertex_array(p, [(rr.quad, "2f", "in_pos")])

        out = ctx.texture((1, 1), 4, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[out])

        def pixel(p, vao):
            fbo.use()
            ctx.viewport = (0, 0, 1, 1)
            vao.render(moderngl.TRIANGLES)
            return np.frombuffer(out.read(), "f4")[:3].astype(float)

        # the shader's hemiDown against the integral it stands for
        ph_, va_ = program("uniform float uMu;\nvoid main() { f_colour = vec4(hemiDown(uMu), 0.0, 0.0, 1.0); }\n")
        n = 240
        th = (np.arange(n) + 0.5) / n * math.pi
        az = (np.arange(2 * n) + 0.5) / (2 * n) * 2.0 * math.pi
        TH, AZ = np.meshgrid(th, az, indexing="ij")
        dy, dx = np.cos(TH), np.sin(TH) * np.cos(AZ)
        dw = np.sin(TH) * (math.pi / n) * (math.pi / n) * (dy < 0)
        worst = 0.0
        for mu in np.linspace(-1.0, 1.0, 21):
            c = dx * math.sqrt(max(1.0 - mu * mu, 0.0)) + dy * mu
            den = 0.09 + 1.4 * (1.0 - c)                         # (1-g)^2 + 2g(1-c), g 0.7
            want = float((0.51 / (4.0 * math.pi * den * np.sqrt(den)) * dw).sum())
            ph_["uMu"].value = float(mu)
            got = float(pixel(ph_, va_)[0])
            worst = max(worst, abs(got - want))
        r.check(worst < 0.006, f"hemiDown off the integral by {worst:.4f}")

        pa, vaa = program("uniform vec3 uTestDir;\nuniform float uTestLen;\nvoid main() {\n"
                          "    vec3 L, T;\n    atmosphere(normalize(uTestDir), 0.0, uTestLen, 64, 0.5, L, T);\n"
                          "    f_colour = vec4(L, 1.0);\n}\n")
        sh = ctx.texture((8, 8), 4, dtype="f4")
        sh.filter = (moderngl.NEAREST, moderngl.NEAREST)
        taus = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)

        def air(when, eye, az_, alt_):
            geom = astro.sky_geometry(when, 22.5726, 88.3639)
            rr.observer_alt = eye
            light = rr.light_source(geom)
            rd = np.array(gl_sky.Camera(az_, alt_, 60.0).basis()[2], "f8")
            R = 6360000.0 + rr.ground_elevation()
            o = np.array([0.0, R + eye, 0.0])
            b, c = float(np.dot(o, rd)), float(np.dot(o, o) - R * R)
            hit = -b - math.sqrt(b * b - c) if b * b - c > 0.0 else -1.0
            vals = []
            for tau in taus:
                d = np.zeros((8, 8, 4), "f4")
                d[..., 0], d[..., 1], d[..., 2], d[..., 3] = -1e7, 1.0, tau, 1.0
                sh.write(d.tobytes())
                rr._bind_textures()
                rr._common_uniforms(pa, geom, 0.0, light)
                sh.use(7)
                pa["uTestDir"].value = tuple(float(v) for v in rd)
                pa["uTestLen"].value = float(hit if 0.0 < hit < 60000.0 else 60000.0)
                vals.append(float(pixel(pa, vaa).mean()))
            return np.array(vals), geom

        noon = datetime(2026, 10, 4, 9, 0, tzinfo=U)            # Kolkata, the Sun 37 deg up
        late = datetime(2026, 10, 4, 11, 0, tzinfo=U)           # ... 11 deg up
        sun_az = astro.sky_geometry(noon, 22.5726, 88.3639).sun_az
        for what, rel in (("toward the Sun", 0.0), ("across its light", 100.0), ("away from it", 180.0)):
            v, _ = air(noon, 15000.0, sun_az + rel, -40.0)
            ratio = v[1:] / v[0]
            r.check(v[0] > 0.0 and ratio.max() <= 1.001,
                    f"from 15 km, 40 deg down, {what}: the shadowed air at tau {taus[1:]} is "
                    f"{', '.join(f'{x:.2f}' for x in ratio)} x the sunlit")
            r.check(v[3] >= v[4] >= v[5] and v[5] <= 0.25 * v[0],
                    f"from 15 km {what}: thicker cloud not darker ({', '.join(f'{x:.2f}' for x in ratio)})")
        v, geom = air(late, 2.0, astro.sky_geometry(late, 22.5726, 88.3639).sun_az, 5.0)
        r.check(v[2] <= 0.6 * v[0] and v[3] <= 0.2 * v[0],
                f"from the ground toward the Sun {geom.sun_alt:.0f} deg up: shadowed air "
                f"{v[2] / v[0]:.2f} (tau 3), {v[3] / v[0]:.2f} (tau 10) of the sunlit - no shafts")
        for what, rel in (("across the Sun's light", 90.0), ("away from the Sun", 180.0)):
            v, _ = air(noon, 2.0, sun_az + rel, 2.0)
            r.check((v[1:] / v[0]).max() <= 3.0,
                    f"from the ground, level, {what}: the air under cloud "
                    f"{', '.join(f'{x:.2f}' for x in v[1:] / v[0])} x the sunlit")
    except Exception as e:                                       # noqa: BLE001
        import traceback
        r.check(False, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}")
    finally:
        r.close()
        ctx.release()


def _window_forecast(t0: datetime, n: int):
    """A forecast answer of n hours from t0 (the Kuala Lumpur sounding of
    25 Sept, held), read by the program's own parser."""
    from . import forecast
    hq = {"time": [(t0 + timedelta(hours=k)).strftime("%Y-%m-%dT%H:00") for k in range(n)],
          "temperature_2m": [31.9] * n, "dew_point_2m": [23.2] * n, "relative_humidity_2m": [60] * n,
          "surface_pressure": [1004.5] * n, "wind_speed_10m": [0.9] * n, "wind_direction_10m": [80] * n,
          "weather_code": [3] * n}
    for p, vals in KL_0925_05.items():
        for name, v in zip(("temperature", "relative_humidity", "cloud_cover", "wind_speed",
                            "wind_direction", "geopotential_height"), vals):
            hq[f"{name}_{p}hPa"] = [v] * n
    return forecast.Forecast({"elevation": 62.0, "hourly": hq}, 3.14, 101.69)


def test_dropdowns():
    """G17-G19: the place, date and hour dropdowns in the program itself (a
    hidden window, offline): the keys and the mouse go to the dropdown that
    is open, a pick moves the place or the clock, the forecast is asked for
    when the one there does not reach the time chosen."""
    import time as _time
    try:
        import pyglet
        from pyglet.window import key, mouse
        from . import app, places
    except ModuleNotFoundError:
        raise
    U = timezone.utc
    real_window = app._make_window

    def hidden_window(width, height):
        from pyglet.gl import Config
        last = None
        for major, minor in ((4, 3), (3, 3)):
            try:
                cfg = Config(major_version=major, minor_version=minor, forward_compatible=True,
                             double_buffer=True, depth_size=0, sample_buffers=0)
                return pyglet.window.Window(width=width, height=height, config=cfg, visible=False)
            except Exception as e:                               # noqa: BLE001
                last = e
        raise last

    name17 = ("G17 the dropdowns: while one is open the keys type in it (Esc closes it, not the "
              "program; the camera keys rest), the mouse picks; a city or coordinates picked move "
              "the place, its clock and its sky")
    try:
        app._make_window = hidden_window
        try:
            cs = app.CloudSim(width=1280, height=760, quality="low", offline=True, sim="none",
                              when=datetime(2026, 7, 1, 10, 0, tzinfo=U))
        finally:
            app._make_window = real_window
    except Exception as e:                                       # noqa: BLE001
        RESULTS.append((name17, 0, [f"skipped, no window can be made here ({type(e).__name__})"]))
        return
    pyglet.clock.unschedule(cs.update)
    W = cs.win
    W._enable_event_queue = False            # events to the handlers at once, as in the event loop

    def button(prefix):
        for w in cs.panel.widgets:
            if getattr(w, "text", "").startswith(prefix) and hasattr(w, "callback"):
                return w
        raise KeyError(f"no button {prefix!r}")

    def click(prefix):
        w = button(prefix)
        x, y = int(w.x + w.w / 2), int(w.y + w.h / 2)
        W.dispatch_event("on_mouse_press", x, y, mouse.LEFT, 0)
        W.dispatch_event("on_mouse_release", x, y, mouse.LEFT, 0)

    def typ(text):
        for ch in text:
            W.dispatch_event("on_text", ch)

    def press(sym, mods=0):
        W.dispatch_event("on_key_press", sym, mods)
        W.dispatch_event("on_key_release", sym, mods)

    def settle():
        # until the new place's weather has come and its sky, made ready in
        # the background (the loading card up), has been cut to
        t_end = _time.time() + 60.0
        while (cs._new_place or cs._pending_forecast is not None or cs._fetching
               or cs._sky_job is not None) and _time.time() < t_end:
            _time.sleep(0.03)
            cs.update(0.03)
        cs.update(0.03)

    r = Rule(name17)
    try:
        kl = places.find("Kuala Lumpur")
        click("world cities")
        p = cs.picker
        r.check(p is not None and cs.picker_kind == "city" and len(p.rows) == 1584,
                "the city list opens with every city")
        r.check(p is not None and p.rows[p.sel][1] == ("city", kl), "it opens on the place there is")
        typ("pari")
        r.check(cs.picker.rows[0][0] == "Paris, France", f"typed 'pari': {cs.picker.rows[0][0]}")
        az = cs.cam.az
        cs.keys.data[key.A] = True                 # an "a" held down while typing
        cs.update(0.2)
        cs.keys.data[key.A] = False
        r.check(cs.cam.az == az, "the camera turned while typing in the list")
        press(key.ESCAPE)
        r.check(cs.picker is None and not cs.sky._stop, "Esc closed the program, or not the list")
        click("world cities")
        typ("paris")
        old_sky = cs.sky
        press(key.ENTER)
        settle()
        r.check(cs.place_name == "Paris, France" and cs.zone == "Europe/Paris",
                f"picked Paris: {cs.place_name} {cs.zone}")
        r.check(cs.utc_offset_s() == 7200 and cs.local.utcoffset() == timedelta(hours=2),
                f"Paris in July: {cs.utc_offset_s()} s")
        r.check(old_sky._stop and cs.sky is not old_sky, "the old place's sky goes on")
        texts = [getattr(w, "text", "") for w in cs.panel.widgets]
        r.check("Paris, France   48.869, 2.331" in texts, f"the panel's place line: {texts[1]!r}")
        r.check(any(t.endswith("(UTC+2)") for t in texts), "the panel's clock does not say UTC+2")
        r.check(cs.status.startswith("Paris, France: "), f"status {cs.status!r}")
        g = astro.sky_geometry(cs.when, 48.8686, 2.3314)
        r.check(abs(cs.geom.sun_alt - g.sun_alt) < 1e-6, "the Sun is not Paris's")
        # coordinates: the place's own to start from, replaced by what is typed
        click("latitude, longitude")
        r.check(cs.picker.query == "48.8686, 2.3314" and cs.picker._fresh, f"start text {cs.picker.query!r}")
        typ("-33.87, 151.21")
        row = cs.picker.rows[0]
        r.check(cs.picker.query == "-33.87, 151.21" and row[1] == ("coords", -33.87, 151.21, None)
                and "UTC+10" in row[2], f"typed coordinates: {cs.picker.query!r} -> {row}")
        press(key.ENTER)
        settle()
        r.check(cs.zone == "Australia/Sydney" and cs.place_name == "near Sydney, Australia"
                and (cs.lat, cs.lon) == (-33.87, 151.21), f"{cs.place_name} {cs.zone} {cs.lat} {cs.lon}")
        click("latitude, longitude")
        W.dispatch_event("on_text_motion", key.MOTION_BACKSPACE)
        r.check(cs.picker.query == "", "Backspace clears the start text")
        typ("10, 20 +3")
        press(key.ENTER)
        settle()
        r.check(cs.zone is None and cs.fixed_s == 10800 and cs.utc_offset_s() == 10800,
                f"a fixed clock typed: {cs.zone} {cs.fixed_s}")
        # another dropdown's button switches to it; its own button closes it; the sky closes it
        click("world cities")
        date_label = cs.local.strftime("%a %d %b %Y")
        click(date_label)
        r.check(cs.picker_kind == "date", f"switched to {cs.picker_kind}")
        click(date_label)
        r.check(cs.picker is None, "its own button closes it")
        click("world cities")
        az = cs.cam.az
        W.dispatch_event("on_mouse_press", 700, 60, mouse.LEFT, 0)
        W.dispatch_event("on_mouse_drag", 760, 60, 60, 0, mouse.LEFT, 0)
        W.dispatch_event("on_mouse_release", 760, 60, mouse.LEFT, 0)
        r.check(cs.picker is None, "a click on the sky closes it")
        # h, m, space ... type while a list is open
        click("world cities")
        help0, sim0, play0 = cs.show_help, cs.sim_key, cs.playing
        for sym in (key.H, key.M, key.SPACE, key.T, key.R):
            press(sym)
        r.check((cs.show_help, cs.sim_key, cs.playing) == (help0, sim0, play0) and cs.picker is not None,
                "a key's own action ran while typing in the list")
        try:
            W.set_clipboard_text("  Kathmandu \n")
            pasted = True
        except Exception:                                        # noqa: BLE001
            pasted = False                     # no clipboard here
        if pasted:
            press(key.V, key.MOD_CTRL)
            r.check(cs.picker.query == "h Kathmandu" or cs.picker.query.endswith("Kathmandu"),
                    f"pasted: {cs.picker.query!r}")
        press(key.ESCAPE)
        # a model running goes on at the new place, the clock as it was
        cs.sim_size = "fast"
        cs.start_sim("bomex")
        cs.playing = False
        cs.set_place("Kuala Lumpur", 3.139, 101.6869, zone="Asia/Kuala_Lumpur")
        settle()
        r.check(cs.sim_key == "bomex" and not cs.playing and cs.sim_start_when <= cs.when,
                f"the model at the new place: {cs.sim_key}, playing {cs.playing}")
        cs.stop_sim(now=True)
        # a narrow window: the list stays on the screen
        W.set_size(640, 400)
        cs.on_resize(640, 400)
        click("world cities")
        p = cs.picker
        r.check(W.width == 640 and p.x >= 0 and p.x + p.w <= 640 and p.y >= 0 and p.y + p.h <= 400,
                f"{W.width}x{W.height}: the list at {p.x},{p.y} {p.w}x{p.h}")
        press(key.ESCAPE)
        W.set_size(1280, 760)
        cs.on_resize(1280, 760)
    except Exception as e:                                       # noqa: BLE001
        import traceback
        r.check(False, f"{type(e).__name__}: {e} {traceback.format_exc()[-300:]}")
    r.close()

    r = Rule("G18 a date and an hour chosen move the clock on the place's own clock (summer time "
             "followed); the list tells where the clock skips an hour")
    try:
        cs.set_place("Paris, France", 48.8686, 2.3314, zone="Europe/Paris")
        settle()
        cs._goto(datetime(2026, 10, 24, 10, 0, tzinfo=U))          # 12:00 in Paris, summer time
        click("Sat 24 Oct 2026")
        p = cs.picker
        r.check(p is not None and p.rows[p.sel][1] == ("date", datetime(2026, 10, 24).date()),
                "the date list opens on the clock's date")
        r.check(all(row[2] == "" for row in p.rows), "offline, the list promises forecasts")
        typ("25/10/2026")
        r.check(p.rows[0][1] == ("date", datetime(2026, 10, 25).date()), f"25/10/2026: {p.rows[0]}")
        press(key.ENTER)
        r.check(cs.when == datetime(2026, 10, 25, 11, 0, tzinfo=U),
                f"Sunday 25th at noon in Paris (winter time from 3 am) is 11:00 UTC, not {cs.when}")
        click("12:00")
        p = cs.picker
        r.check(len(p.rows) == 24 and p.rows[p.sel][0] == "12:00", "the hour list opens on the hour")
        u = places.to_utc(datetime(2026, 10, 25, 15, 0), "Europe/Paris")
        r.check(p.rows[15][2] == f"sun {astro.sky_geometry(u, cs.lat, cs.lon).sun_alt:+.0f}°",
                f"15:00's Sun: {p.rows[15][2]}")
        typ("3am")
        press(key.ENTER)
        settle()
        t_end = _time.time() + 10.0
        while cs._anim is not None and _time.time() < t_end:
            _time.sleep(1 / 30)
            cs.update(1 / 30)
        r.check(cs.when == datetime(2026, 10, 25, 2, 0, tzinfo=U),
                f"3 am that Sunday (after the change) is 02:00 UTC, not {cs.when}")
        cs._goto(datetime(2026, 3, 29, 10, 0, tzinfo=U))
        click(cs.local.strftime("%H:%M"))
        note = cs.picker.rows[2][2]
        r.check(note.startswith("not on the clock that day - 03:00"), f"29 March, 02:00: {note!r}")
        press(key.ESCAPE)
        cs.reset_time()
        r.check(abs((cs.when - datetime.now(U)).total_seconds()) < 5.0, "now")
    except Exception as e:                                       # noqa: BLE001
        import traceback
        r.check(False, f"{type(e).__name__}: {e} {traceback.format_exc()[-300:]}")
    r.close()

    r = Rule("G19 the forecast is asked for when the one there does not reach the time chosen (and "
             "one on its way will not); the status says when none reaches it, and stops saying so; "
             "'fetch live weather' fetches it after an offline profile, never with --offline")
    calls = []
    real_get = app.forecast.get_forecast
    try:
        t0 = datetime(2026, 9, 25, 0, tzinfo=U)
        fc = _window_forecast(t0, 48)
        cs.offline = False
        cs.forecast = fc
        cs.fetch_weather = lambda when=None, force=False, status="", live=False, card="": \
            calls.append((when, force))
        cs._fetching = False
        cs.goto_time(t0 + timedelta(hours=20))
        r.check(calls == [], f"inside the forecast: asked for another {calls}")
        cs.goto_time(t0 + timedelta(hours=70))
        r.check(len(calls) == 1 and calls[0] == (t0 + timedelta(hours=70), True),
                f"outside it: {calls}")
        cs._fetching = True
        cs._fetch_span = (t0 + timedelta(hours=60), t0 + timedelta(hours=120))
        calls.clear()
        cs.goto_time(t0 + timedelta(hours=80))
        r.check(calls == [], "a fetch on its way reaches it: asked again")
        cs.goto_time(t0 + timedelta(hours=200))
        r.check(len(calls) == 1, "a fetch on its way does not reach it: not asked")
        cs._fetching = False
        del cs.fetch_weather
        # what came does not reach the clock: said; the clock back inside: no longer said
        cs._goto(t0 + timedelta(hours=60))
        cs._take_forecast(fc, "Open-Meteo, 48 hours")
        r.check("does not reach" in cs.status, f"status {cs.status!r}")
        cs._goto(t0 + timedelta(hours=40))
        r.check(cs.status == "Open-Meteo, 48 hours", f"back inside: {cs.status!r}")
        cs._new_place = True
        cs._goto(t0 + timedelta(hours=60))
        cs._take_forecast(_window_forecast(t0, 48), "Open-Meteo, 48 hours")
        t_end = _time.time() + 60.0                 # its sky made ready, then cut to
        while cs._sky_job is not None and _time.time() < t_end:
            _time.sleep(0.02)
            cs._poll_sky_job()
        r.check(cs.status.startswith(cs.place_name + ": Open-Meteo, 48 hours - it does not reach"),
                f"a new place's forecast that does not reach the clock: {cs.status!r}")
        # the button after an offline profile
        seen = []

        def stub(lat, lon, when=None, allow_network=True, fallback_profile="tropical-fair"):
            seen.append(allow_network)
            return real_get(lat, lon, when, allow_network=False, fallback_profile=fallback_profile)
        app.forecast.get_forecast = stub
        for no_net, want in ((False, True), (True, False)):
            cs.no_network = no_net
            cs.use_profile("stable-stratus")
            seen.clear()
            button("fetch live weather").callback()
            t_end = _time.time() + 10.0
            while cs._fetching and _time.time() < t_end:
                _time.sleep(0.02)
            cs.update(0.02)
            r.check(seen == [want] and cs.offline == (not want),
                    f"--offline {no_net}: the network {'asked' if seen and seen[0] else 'not asked'}")
    except Exception as e:                                       # noqa: BLE001
        import traceback
        r.check(False, f"{type(e).__name__}: {e} {traceback.format_exc()[-300:]}")
    finally:
        app.forecast.get_forecast = real_get
        r.close()
        cs.offline = True
        cs.sim.stop()
        cs.sky.close()
        W.close()


# ------------------------------------------- loading window and compass --

def test_loading():
    """Z1-Z2: the loading window's protocol and bar; a sky made ready in the
    background before it is cut to."""
    import subprocess
    import sys
    import tempfile
    import time as _time
    from . import splash, timeline

    r = Rule("Z1 the loading window: what the program says reaches it as said (in order, any "
             "language), the bar never goes back and never runs past where the program has got")
    p = splash.parse("P 0.2500 0.4000 Compiling the sky's shaders")
    r.check(p == ("P", 0.25, 0.4, "Compiling the sky's shaders"), f"parsed {p}")
    r.check(splash.parse("Q") == ("Q",) and splash.parse("P x y z") is None
            and splash.parse("hello") is None, "a line that is not P or Q is ignored")
    r.check(splash.parse("P 1.7 -2 t")[1:3] == (1.0, 0.0), "fractions kept within 0..1")
    vals = [splash.creep(0.4, 0.6, t) for t in (0.0, 0.5, 1.0, 2.0, 5.0, 20.0, 200.0)]
    r.check(vals[0] == 0.4, f"the bar starts where the program said: {vals[0]}")
    r.check(all(b >= a for a, b in zip(vals, vals[1:])), f"the bar went back: {vals}")
    r.check(max(vals) <= 0.6 + 1e-12 and vals[-1] > 0.599, f"the bar ran past 0.6, or stuck: {vals}")
    L = splash.Loader(None)
    L.step(0.3, "a", 0.5)
    L.step(0.2, "b")
    L.step(0.6, "c", 0.4)
    r.check([e[1] for e in L.log] == [0.3, 0.3, 0.6], f"went back: {[e[1] for e in L.log]}")
    r.check([e[2] for e in L.log] == [0.5, 0.3, 0.6], f"'to' below where it is: {[e[2] for e in L.log]}")
    L.close()
    L.close()
    r.check(L.closed and not L.alive() and not splash.NoLoader().alive(), "a loader with no window")
    # through a real pipe: a stand-in for the window process that writes
    # down what it reads
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    out.close()
    try:
        code = ("import sys; d = sys.stdin.buffer.read(); "
                f"open({out.name!r}, 'wb').write(d)")
        proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE)
        L = splash.Loader(proc)
        t0 = _time.time()
        L.step(0.1, "Fetching the weather for Zürich (Open-Meteo)", 0.2)
        L.step(0.5, "Building the cloud patterns  ·  3 of 8")
        L.step(0.5, "Building the cloud patterns  ·  3 of 8")    # the same again: not sent twice
        L.close()
        r.check(_time.time() - t0 < 0.5, "the program waited for the pipe")
        proc.wait(timeout=10)
        lines = open(out.name, "rb").read().decode("utf-8").splitlines()
        r.check(lines == ["P 0.1000 0.2000 Fetching the weather for Zürich (Open-Meteo)",
                          "P 0.5000 0.5000 Building the cloud patterns · 3 of 8", "Q"],
                f"the window process read {lines}")
    except Exception as e:                                       # noqa: BLE001
        r.check(False, f"{type(e).__name__}: {e}")
    finally:
        try:
            import os as _os
            _os.unlink(out.name)
        except Exception:                                        # noqa: BLE001
            pass
    r.close()

    r = Rule("Z2 a sky made ready in the background (Sky.prepare) asks for exactly what a sync "
             "update builds and touches nothing the renderer holds; cut to, it has nothing left to "
             "build and is the sky a sync update makes")
    sky, t0 = _sky_hands(10, 64)
    when = t0 + timedelta(hours=4, minutes=20)
    have0, need = sky.prepare(when)
    r.check(need > 0 and have0 == 0, f"needed {need}, built before asking {have0}")
    r.check(not sky.group_of and not sky.pending_uploads
            and all(v is None for v in sky.slot_real.values()), "prepare touched the renderer's slots")
    built = sky.build_now()
    have, need2 = sky.prepare(when, request=False)
    r.check(have == need2 == need == built, f"built {built}, then {have} of {need2} (asked {need})")
    calls = []
    real = timeline.build_realization
    timeline.build_realization = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        sky.update(when, 0.0, sync=True)
    finally:
        timeline.build_realization = real
    r.check(not calls, f"the sync update after it built {len(calls)} more")
    ref, _ = _sky_hands(10, 64)
    ref.update(when, 0.0, sync=True)

    def sig(s):
        return sorted((st.deck.spec.abbrev(), round(st.deck.coverage, 9), round(st.theta, 9),
                       tuple(round(w, 9) for w in st.weights)) for st in s.states)
    r.check(sig(sky) == sig(ref) and len(sky.states) > 0,
            f"made ready then cut to: {sig(sky)[:2]} - in one sync step: {sig(ref)[:2]}")
    fr = sky.fresh()
    h, n = fr.prepare(when, request=False)
    r.check(h == n == need and not fr.group_of and fr.tracks is not sky.tracks
            and fr._epoch0 == sky._epoch0 and fr._salt == sky._salt,
            f"a fresh sky with the patterns built: {h} of {n}")
    now_only = sky.prepare(when, request=False, ahead=False)
    r.check(now_only[1] * 4 == need * 3, f"without the next realization: {now_only[1]} of {need}")
    # the manual sky: the layers chosen by hand, held
    snd = sounding.standard_sounding(48.86, 2.35, profile="tropical-fair")
    man = timeline.Sky(manual=[scene.deck_from_spec(CloudSpec("Ac", "str"), snd, base_m=3500.0,
                                                    coverage=0.5, seed=4)],
                       threaded=False, n=64, seed=3)
    h0, n0 = man.prepare(t0)
    man.build_now()
    h1, n1 = man.prepare(t0, request=False)
    r.check(n0 == 4 and h0 == 0 and h1 == n1 == 4, f"a layer chosen by hand: {h0}/{n0} then {h1}/{n1}")
    for s in (sky, ref, fr, man):
        s.close()
    r.close()


def test_compass():
    """Z6: the compass points where the renderer draws those directions."""
    from . import gl_sky, viewinfo
    r = Rule("Z6 the compass points (Stellarium's Q) stand where the renderer draws those "
             "directions on the horizon, for any view and any eye height")
    # the tangent from the eye to the sphere, the other way: tan(dip) =
    # sqrt(2Rh + h^2) / R
    for h_ in (1.7, 300.0, 20000.0):
        want = math.degrees(math.atan(math.sqrt(2 * viewinfo.RG * h_ + h_ * h_) / viewinfo.RG))
        r.check(near(viewinfo.horizon_dip_deg(h_), want, 1e-6),
                f"dip {viewinfo.horizon_dip_deg(h_):.6f} at {h_} m, not {want:.6f}")
    r.check(near(viewinfo.horizon_dip_deg(1.7), 0.0419, 0.0001)
            and near(viewinfo.horizon_dip_deg(20000.0), 4.5379, 0.0001)
            and viewinfo.horizon_dip_deg(0.0) == 0.0,
            f"dip {viewinfo.horizon_dip_deg(1.7):.4f} at 1.7 m, "
            f"{viewinfo.horizon_dip_deg(20000.0):.4f} at 20 km")
    names = [c[0] for c in viewinfo.COMPASS]
    r.check(names == ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            and [c[2] for c in viewinfo.COMPASS] == [True, False] * 4, "eight points, four cardinal")
    rng = np.random.default_rng(11)
    W_, H_ = 1280.0, 760.0
    seen = 0
    for k in range(400):
        az, alt = float(rng.uniform(0, 360)), float(rng.uniform(-60, 60))
        fov, aspect = float(rng.uniform(4, 140)), float(rng.uniform(0.6, 2.4))
        eye = float(10 ** rng.uniform(0.2, 4.3))
        cam = gl_sky.Camera(az, alt, fov)
        marks = viewinfo.horizon_marks(cam, aspect, W_, H_, eye)
        right, up, fwd = (np.asarray(v, "f8") for v in cam.basis())
        t = math.tan(math.radians(fov) * 0.5)
        dip = viewinfo.horizon_dip_deg(eye)
        for name, x, y, cardinal in marks:
            nx, ny = 2.0 * x / W_ - 1.0, 2.0 * y / H_ - 1.0
            rd = fwd + right * nx * t * aspect + up * ny * t      # the shader's ray
            rd /= np.linalg.norm(rd)
            a = math.degrees(math.atan2(rd[0], rd[2])) % 360.0
            want = dict((c[0], c[1]) for c in viewinfo.COMPASS)[name]
            da = abs((a - want + 180.0) % 360.0 - 180.0)
            el = math.degrees(math.asin(max(-1.0, min(1.0, rd[1]))))
            seen += 1
            r.check(da < 0.01 and abs(el + dip) < 0.01 and cardinal == (len(name) == 1),
                    f"view {az:.0f}/{alt:.0f}/{fov:.0f}: {name} drawn at az {a:.3f} el {el:.3f}")
        # nothing on the screen left out
        for name, az_c, _ in viewinfo.COMPASS:
            d = np.array([math.cos(math.radians(dip)) * math.sin(math.radians(az_c)),
                          -math.sin(math.radians(dip)),
                          math.cos(math.radians(dip)) * math.cos(math.radians(az_c))])
            z = float(d @ fwd)
            if z > 0.05:
                nx = float(d @ right) / (z * t * aspect)
                ny = float(d @ up) / (z * t)
                if abs(nx) < 0.95 and abs(ny) < 0.95:
                    r.check(name in [m[0] for m in marks], f"view {az:.0f}/{alt:.0f}/{fov:.0f}: "
                                                           f"{name} on the screen but not marked")
    r.check(seen > 400, f"only {seen} marks checked")
    cam = gl_sky.Camera(0.0, 0.0, 60.0)
    m = {k[0]: k[1:] for k in viewinfo.horizon_marks(cam, 16 / 9, 1600, 900, 1.7)}
    r.check(set(m) == {"N", "NE", "NW"} and near(m["N"][0], 800.0, 0.01)
            and near(m["N"][1], 450.0 - 450.0 * math.tan(math.radians(viewinfo.horizon_dip_deg(1.7))) / math.tan(math.radians(30)), 1e-6),
            f"facing north at the horizon: {m}")
    r.check(viewinfo.horizon_marks(gl_sky.Camera(0.0, -89.0, 90.0), 16 / 9, 1600, 900, 20000.0) == [],
            "looking straight down from 20 km the horizon is not on the screen")
    r.close()


def test_loading_gl():
    """Z3-Z5, Z7-Z9: the loading window's process; the program's start with
    it; the loading card in the program; the compass points drawn; the text
    on whole pixels."""
    import os as _os
    import re
    import subprocess
    import sys
    import threading
    import time as _time
    try:
        import pyglet
        from pyglet.window import key
        from . import app, gl_sky, splash, timeline, viewinfo
        probe = pyglet.window.Window(width=64, height=64, visible=False)
        probe.close()
    except ModuleNotFoundError:
        raise
    except Exception as e:                                       # noqa: BLE001
        for n in ("Z3 the loading window opens, follows and closes",
                  "Z4 the program's start", "Z5 the loading card", "Z7 the horizon drawn",
                  "Z8 the compass points drawn", "Z9 the text on whole pixels"):
            RESULTS.append((n, 0, [f"skipped, no window can be made here ({type(e).__name__})"]))
        return
    U = timezone.utc

    # Z3: the window process itself
    r = Rule("Z3 the loading window opens, takes what it is told, ignores what it cannot read, and "
             "closes when told to or when the program's end of the pipe closes")
    for how in ("Q", "EOF"):
        try:
            proc = subprocess.Popen([sys.executable, _os.path.abspath(splash.__file__), "--line",
                                     "Paris  ·  48.857, 2.352", "--report"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            got = []
            th = threading.Thread(target=lambda: got.extend(
                x.decode().strip() for x in iter(proc.stdout.readline, b"")), daemon=True)
            th.start()
            t_end = _time.time() + 30.0
            while "UP" not in got and proc.poll() is None and _time.time() < t_end:
                _time.sleep(0.05)
            r.check("UP" in got, f"({how}) the window did not come up: {got}, exit {proc.poll()}")
            for line in (b"P 0.3000 0.4000 Compiling the sky's shaders\n", b"garbage\n",
                         "P 0.6000 0.7000 Fetching the weather for Zürich\n".encode()):
                proc.stdin.write(line)
                proc.stdin.flush()
            _time.sleep(0.4)
            r.check(proc.poll() is None, f"({how}) the window closed on a line it cannot read")
            if how == "Q":
                proc.stdin.write(b"Q\n")
                proc.stdin.flush()
            else:
                proc.stdin.close()
            code = proc.wait(timeout=10)
            th.join(2.0)
            r.check(code == 0 and got[-1:] == ["BYE"], f"({how}) closed with {code}, said {got}")
        except Exception as e:                                   # noqa: BLE001
            r.check(False, f"({how}) {type(e).__name__}: {e}")
            try:
                proc.kill()
            except Exception:                                    # noqa: BLE001
                pass
    r.close()

    # Z4: the program's start with a loading window up (a stand-in that
    # records what it is told; the program's window is kept hidden)
    class Stand(splash.Loader):
        def __init__(self):
            super().__init__(None)

        def alive(self):
            return not self.closed

    made = []
    real_window = app._make_window

    def maker(width, height, visible=True):
        made.append(visible)
        from pyglet.gl import Config
        last = None
        for major, minor in ((4, 3), (3, 3)):
            try:
                cfg = Config(major_version=major, minor_version=minor, forward_compatible=True,
                             double_buffer=True, depth_size=0, sample_buffers=0)
                return pyglet.window.Window(width=width, height=height, config=cfg, visible=False)
            except Exception as e:                               # noqa: BLE001
                last = e
        raise last

    r = Rule("Z4 the program's start: its window is made hidden while the loading window counts "
             "the start through, in order and never back; it shows with its first frames drawn, "
             "and the loading window closes once it has drawn on the screen")
    L = Stand()
    cs = None
    try:
        app._make_window = maker
        try:
            cs = app.CloudSim(width=320, height=200, quality="low", offline=True, sim="none",
                              when=datetime(2026, 7, 1, 10, 0, tzinfo=U), loader=L)
        finally:
            app._make_window = real_window
        pyglet.clock.unschedule(cs.update)
        r.check(made == [False], f"the window made visible={made}")
        steps = [e[3] for e in L.log]
        order = ["Opening the OpenGL window", "Compiling the sky's shaders",
                 "Preparing the cloud-resolving model", "Making the tropical-fair sounding",
                 "Building the cloud patterns"]
        idx = [next((i for i, s in enumerate(steps) if s.startswith(o)), -1) for o in order]
        r.check(-1 not in idx and idx == sorted(idx), f"the steps: {steps[:8]}")
        counts = [tuple(map(int, m.groups())) for s in steps
                  for m in [re.search(r"(\d+) of (\d+)", s)] if m]
        r.check(bool(counts) and counts[-1][0] == counts[-1][1] > 0
                and all(b[0] >= a[0] for a, b in zip(counts, counts[1:])),
                f"the patterns counted: {counts[:3]} ... {counts[-1:]}")
        fr = [e[1] for e in L.log]
        r.check(all(b >= a for a, b in zip(fr, fr[1:])) and fr[0] <= 0.1, f"the bar: {fr[:6]}")
        h, n = cs.sky.prepare(cs.when, request=False)
        r.check(h == n > 0 and len(cs.decks) > 0, f"the sky shown at the start: {h} of {n} built")
        drawn = []
        real_draw = cs.on_draw
        cs.on_draw = lambda: (drawn.append(bool(cs.win.visible)), real_draw())[1]
        cs.warm_up()
        r.check(drawn == [False, False, False] and not L.closed and L.log[-1][1:] == (1.0, 1.0, "Ready"),
                f"warm-up frames {drawn}, loading window closed {L.closed}, last {L.log[-1]}")
        shown = []
        cs.win.set_visible = lambda v=True: shown.append(v)
        cs.win.activate = lambda: None
        cs.show()
        r.check(shown == [True] and not L.closed, f"shown {shown}; closed at once {L.closed}")
        cs.on_draw()
        first = L.closed
        cs.on_draw()
        r.check(not first and L.closed, f"closed after the first frame {first}, the second {L.closed}")
    except Exception as e:                                       # noqa: BLE001
        import traceback
        r.check(False, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}")
    r.close()
    if cs is None:
        for n in ("Z5 the loading card", "Z9 the text on whole pixels"):
            RESULTS.append((n, 0, ["skipped, the program did not start"]))
        return

    # Z5: the loading card, with the forecast answered by a stand-in after a while
    r = Rule("Z5 the loading card: up for a load that was asked for (a place, a date, the sky of a "
             "far moment, R) and gone the frame it is done; a new place's sky is made ready while "
             "the sky there is drawn, then cut to, the program's thread building nothing; the "
             "program's own refresh shows no card")
    real_get = app.forecast.get_forecast
    main_builds = []
    real_build = timeline.build_realization
    main_thread = threading.current_thread()

    def counting(*a, **k):
        if threading.current_thread() is main_thread:
            main_builds.append(1)
        return real_build(*a, **k)

    def stub(lat, lon, when=None, allow_network=True, fallback_profile="tropical-fair"):
        _time.sleep(0.4)
        return _window_forecast(datetime(2026, 7, 1, 0, tzinfo=U), 48), "stand-in forecast"

    def run_until(cond, limit=90.0):
        seen = []
        t_end = _time.time() + limit
        while _time.time() < t_end:
            cs.update(0.03)
            st = cs.loading()
            seen.append((st, cs.sky))
            if cond():
                break
            _time.sleep(0.03)
        return seen

    try:
        r.check(cs.loading() is None, f"a card at rest: {cs.loading()}")
        app.forecast.get_forecast = stub
        timeline.build_realization = counting
        cs.offline = False
        old_sky = cs.sky
        cs.set_place("Paris, France", 48.8686, 2.3314, zone="Europe/Paris")
        st = cs.loading()
        r.check(st is not None and st[0] == "Fetching the weather for Paris, France" and st[2] is None,
                f"while the weather comes: {st}")
        seen = run_until(lambda: cs._sky_job is None and not cs._fetching
                         and cs._pending_forecast is None)
        titles = [s[0][0] if s[0] else None for s in seen]
        build = [k for k, s in enumerate(seen) if s[0] and s[0][0].startswith("Building the sky of Paris")]
        r.check(bool(build), f"no 'Building the sky' card: {sorted(set(titles), key=str)}")
        if build:
            r.check(all(seen[k][1] is old_sky for k in build),
                    "the new place's sky was drawn before it was built")
            fr = [seen[k][0][2] for k in build]
            r.check(all(b >= a for a, b in zip(fr, fr[1:])), f"the card's bar went back: {fr}")
        r.check(titles[-1] is None and seen[-1][1] is not old_sky and old_sky._stop,
                f"after the cut: card {titles[-1]}, the new sky {seen[-1][1] is not old_sky}")
        r.check(None not in titles[:-1], "the card went away between the weather and its sky")
        r.check(not main_builds, f"the program's thread built {len(main_builds)} patterns (froze)")
        r.check(cs.status.startswith("Paris, France: stand-in forecast"), f"status {cs.status!r}")
        # the program's own refresh: no card
        cs.fetch_weather()
        r.check(cs.loading() is None, f"a card for the program's own refresh: {cs.loading()}")
        run_until(lambda: not cs._fetching and cs._pending_forecast is None and cs._sky_job is None)
        # a far jump: the new moment's sky forms with the card up
        cs.offline = True
        cs._goto(cs.when + timedelta(hours=7))
        st = cs.loading()
        r.check(st is not None and st[0].startswith("Forming the sky of"), f"a far jump: {st}")
        run_until(lambda: cs._sky_job is None)
        h, n = cs.sky.prepare(cs.when, request=False, ahead=False)
        r.check(cs.loading() is None and h == n, f"after the jump: {cs.loading()}, {h} of {n}")
        # R: made again in the background, then cut to
        old_sky = cs.sky
        cs.rebuild_sky()
        r.check(cs._sky_job is not None and cs._sky_job["then"] is not None, "R: no sky made ready")
        seen = run_until(lambda: cs._sky_job is None)
        r.check(cs.sky is not old_sky and old_sky._stop and cs.loading() is None,
                "R: not cut to the sky made again")
        # a plain watch never drops a cut on its way
        cs._watch_sky("t", sky=cs.sky.fresh(), then=lambda: None, ahead=True)
        cs._watch_sky("Building the sky")
        r.check(cs._sky_job["then"] is not None, "a plain watch dropped the cut")
        cs._sky_job["sky"].close()
        cs._sky_job = None
        r.check(not main_builds, f"the program's thread built {len(main_builds)} patterns")
        # the card itself: in the middle of the sky between the panels, the
        # bar never back within one load
        cs.card.reset()
        cs.card.layout(352, 880, 760, "Building the sky of Paris, France", "cloud patterns 3 of 8",
                       0.5, 10.0)
        c = cs.card.card
        r.check(352 <= c.x and c.x + c.width <= 880 and near(c.x + c.width / 2, 616.0, 0.5)
                and near(c.y + cs.card.H / 2, 380.0, 0.5), f"the card at {c.x},{c.y} {c.width}")
        cs.card.layout(352, 880, 760, "x", "y", 0.3, 10.1)
        r.check(cs.card.pct.text == "50%", f"the bar went back: {cs.card.pct.text}")
    except Exception as e:                                       # noqa: BLE001
        import traceback
        r.check(False, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}")
    finally:
        app.forecast.get_forecast = real_get
        timeline.build_realization = real_build
        cs.offline = True
    r.close()

    # Z9: text on whole pixels
    r = Rule("Z9 the text stands on whole pixels - the loading card in a window an odd number of "
             "pixels wide and tall, the compass points as the view turns, the panels scrolled by a "
             "touchpad's fractions (half a pixel off, pyglet draws each letter rounded on its own "
             "and the letters of a word stand a pixel apart); small scrolls still add up")
    try:
        from . import ui as _ui

        def whole(lab):
            left = getattr(lab, "left", lab.x)
            return all(float(v).is_integer() for v in (lab.x, lab.y, left))

        def panel_labels(p):
            out = []
            for w in p.widgets:
                if getattr(w, "_l", None) is not None:
                    out.append(w._l)
                out += list(getattr(w, "_labels", []))
                out += [q for q in getattr(w, "_parts", []) if isinstance(q, pyglet.text.Label)]
            return out

        r.check([_ui.px(v) for v in (0.5, 1.5, -0.5, 2.49, 7.0)] == [1, 2, 0, 2, 7],
                "px() does not round to the nearest pixel")
        cs.card.reset()
        cs.card.layout(351, 880, 761, "Fetching the weather for Kolkata, India",
                       "Open-Meteo forecast  ·  2 s", None, 10.0)
        c = cs.card
        bad = [n for n, lab in (("title", c.title), ("detail", c.detail), ("pct", c.pct))
               if not whole(lab)]
        r.check(not bad and float(c.card.x).is_integer() and float(c.card.y).is_integer(),
                f"the card off the pixels: {bad}, card at {c.card.x},{c.card.y}")
        r.check(near(c.card.x + c.card.width / 2, 615.5, 0.5) and near(c.card.y + c.H / 2, 380.5, 0.5),
                f"the card not in the middle: {c.card.x},{c.card.y} {c.card.width}")
        cs.win.switch_to()
        fbo = cs.ctx.simple_framebuffer((320, 200))
        fbo.use()
        offs = []
        for az in (13.37, 101.9, 222.25):
            cs.cam = gl_sky.Camera(az, 0.7, 61.3)
            cs.draw_directions()
            marks = viewinfo.horizon_marks(cs.cam, cs.renderer.rw / float(cs.renderer.rh),
                                           cs.win.width, cs.win.height, cs.renderer.observer_alt)
            for m in marks:
                labs = cs._compass_labels.get(m[0])
                offs += [m[0] for lab in labs if not whole(lab)]
        cs.ctx.screen.use()
        fbo.release()
        r.check(not offs, f"compass points off the pixels: {offs[:6]}")
        p = cs.panel
        p.scroll = 0.0
        p.end()
        y0 = p.widgets[0].y
        for _ in range(10):
            p.on_scroll(-0.05)                                   # a touchpad's small steps
            p.end()
        moved = p.widgets[0].y - y0
        offp = [getattr(lab, "text", "?") for lab in panel_labels(p) if not whole(lab)]
        r.check(not offp and bool(panel_labels(p)), f"panel text off the pixels: {offp[:4]}")
        r.check(moved == _ui.px(min(10 * 0.05 * 42, max(0.0, p.content_h - (p.height - 16)))),
                f"ten small scrolls moved the panel {moved} px")
        p.scroll = 0.0
        p.end()
    except Exception as e:                                       # noqa: BLE001
        import traceback
        r.check(False, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}")
    r.close()

    # Z8: the compass points drawn by the program, Q and the panel's toggle
    r = Rule("Z8 the compass points are drawn on the horizon in Stellarium's red, the cardinal "
             "ones bigger; Q and the panel's toggle show and hide them; on from the start")
    try:
        import moderngl as _mgl
        r.check(cs.show_directions, "off at the start")
        r.check(any(getattr(w, "text", "").startswith("directions") for w in cs.panel.widgets),
                "no toggle on the panel")
        r.check(any(s.startswith("q ") for s in app.CloudSim.HELP_LINES), "not in the help")
        cs.on_key_press(key.Q, 0)
        off = cs.show_directions
        cs.on_key_press(key.Q, 0)
        r.check(off is False and cs.show_directions, "Q does not toggle them")
        cs.win.switch_to()
        fbo = cs.ctx.simple_framebuffer((320, 200))
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        cs.cam = gl_sky.Camera(0.0, 0.0, 60.0)
        marks = {m[0]: m for m in viewinfo.horizon_marks(cs.cam, cs.renderer.rw / float(cs.renderer.rh),
                                                         cs.win.width, cs.win.height,
                                                         cs.renderer.observer_alt)}
        cs.draw_directions()
        px = np.frombuffer(fbo.read(components=3), np.uint8).reshape(200, 320, 3).astype(int)
        red = (px[..., 0] > 150) & (px[..., 1] < 90) & (px[..., 2] < 70)
        ys, xs = np.nonzero(red)
        nx, ny = marks["N"][1], marks["N"][2]
        # (the framebuffer's rows from the bottom, as the window's y): the
        # letter stands on the horizon point, centred on it
        r.check(len(xs) > 20 and abs(xs.mean() - nx) < 6 and 0 < ys.mean() - ny < 20,
                f"N drawn at {xs.mean() if len(xs) else -1:.1f},{ys.mean() if len(ys) else -1:.1f} "
                f"for {nx:.0f},{ny:.0f} ({len(xs)} red pixels)")
        sizes = {}
        for nm in ("N", "NE"):
            labs = cs._compass_labels.get(nm)
            sizes[nm] = labs[1].font_size if labs else None
        r.check(sizes["N"] == 17 and sizes["NE"] == 12, f"sizes {sizes}")
        cs.ctx.screen.use()
        fbo.release()
    except Exception as e:                                       # noqa: BLE001
        import traceback
        r.check(False, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}")
    finally:
        r.close()
        cs.sim.stop()
        cs.sky.close()
        cs.win.close()

    # Z7: the horizon in the renderer's picture is where the compass marks it
    r = Rule("Z7 the horizon the renderer draws is where the compass points stand (the ground a "
             "sphere: from 300 m it is 0.56 deg below the level)")
    try:
        import moderngl as _mgl
        try:
            ctx = _mgl.create_standalone_context(backend="egl", require=330)
        except Exception:                                        # noqa: BLE001
            ctx = _mgl.create_standalone_context(require=330)
        when = datetime(2026, 6, 21, 10, 0, tzinfo=U)
        geom = astro.sky_geometry(when, 48.86, 2.35)
        rr = gl_sky.SkyRenderer(ctx, 240, 240, quality="high")
        rr.haze = gl_sky.haze_from_visibility(300000.0)
        rr.set_decks([])
        cam = gl_sky.Camera(0.0, 0.0, 4.0)                      # north, level, 4 deg: 30 px a degree
        for eye in (100.0, 300.0):
            rr.observer_alt = eye
            rr.shadow_dirty = True
            rr.accum_frames = 0
            img = rr.read_image(rr.render(cam, geom, 0.0)).astype(float)
            # the clear sky is the same all along a row; the ground has its
            # fields: the horizon is the first row (from the top, below the
            # middle) whose colours vary along it
            sd = img.std(axis=1).max(axis=1)
            row = next((i for i in range(100, 240) if sd[i] > 2.0), None)
            mark = dict((m[0], m) for m in viewinfo.horizon_marks(cam, 1.0, 240, 240, eye))["N"]
            want = 240 - mark[2]                                # the mark's height, from the top
            r.check(row is not None and abs(row - want) <= 2.0,
                    f"from {eye:.0f} m the horizon is drawn at row {row}, the mark at {want:.1f}")
        ctx.release()
    except Exception as e:                                       # noqa: BLE001
        r.check(False, f"{type(e).__name__}: {e}")
    r.close()


def main(verbose=False):
    RESULTS.clear()
    for fn in (test_taxonomy, test_elements, test_thermo, test_astro,
               test_geometry, test_model, test_layer_physics, test_regimes,
               test_region_and_view, test_fall_and_waves, test_stress_regressions,
               test_zero_low_cloud, test_app_logic, test_places, test_round3,
               test_ground_look, test_loading, test_compass,
               test_transitions, test_twilight, test_species, test_optics,
               test_panels, test_dropdowns, test_loading_gl, test_gl, test_round3_gl,
               test_ground_look_gl, test_air_light_gl, test_model_gpu):
        try:
            fn()
        except ModuleNotFoundError as e:
            if e.name not in ("moderngl", "pyglet"):
                raise
            RESULTS.append((fn.__name__, 0, [f"skipped, {e.name} not installed"]))
        except Exception as e:                                   # noqa: BLE001
            import traceback
            # a crash is a failure, never a skip
            RESULTS.append((fn.__name__ + " CRASHED", 1,
                            [traceback.format_exc() if verbose else repr(e)]))
    width = max(len(n) for n, _, _ in RESULTS) + 2
    total = fails = skipped = 0
    print()
    for name, n, bad in RESULTS:
        total += n
        state = "skip" if n == 0 and bad else ("PASS" if not bad else "FAIL")
        if state == "skip":
            skipped += 1        # a skip is not a failure
        else:
            fails += len(bad)
        print(f"  {state}  {name:<{width}} {n:5d} checks"
              + (f"   {len(bad)} failed" if bad and state == "FAIL" else ""))
        if bad and (verbose or len(bad) <= 6):
            for b in bad[:12]:
                print(f"          - {b}")
        elif bad:
            for b in bad[:4]:
                print(f"          - {b}")
            print(f"          ... and {len(bad) - 4} more (use --verbose)")
    print(f"\n  {total} checks, {fails} failed"
          + (f", {skipped} rules skipped" if skipped else "") + "\n")
    return 1 if fails else 0
