# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
What the camera is looking at, and why it looks like that.

Casts a grid of rays through the current view - the same rays the renderer
casts, through the same cloud field (the model's columns continued over the
region, the Atlas layers, the upper-atmosphere clouds) - and reports, for
each kind of cloud the rays meet first:

  * its Atlas name, and how much of the frame it fills;
  * where it is: its height, how far away, which part of the frame;
  * the physics that gives it this appearance, with the model's own numbers
    (the size of its cells and why they have that size, how perspective
    shrinks them toward the horizon, why its base is dark or its edge
    bright, why the precipitation under it vanishes before the ground);
  * the weather data that made the program put it there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import atlas, crm, region
from .atlas import CloudSpec

RG = 6360000.0

#: The compass points written on the horizon, as Stellarium does (its key
#: Q): the four cardinal points bigger and in its red, the four between
#: them smaller.  (label, azimuth in degrees east of north, cardinal)
COMPASS = [("N", 0.0, True), ("NE", 45.0, False), ("E", 90.0, True), ("SE", 135.0, False),
           ("S", 180.0, True), ("SW", 225.0, False), ("W", 270.0, True), ("NW", 315.0, False)]


def horizon_dip_deg(eye_m: float, ground_r: float = RG) -> float:
    """How far below the level the horizon lies seen from eye_m metres up:
    the renderer's ground is a sphere of radius ground_r, so the horizon is
    where the line of sight grazes it, acos(R / (R + h)) below the level
    (0.04 deg at 1.7 m, 4.5 deg at 20 km)."""
    return math.degrees(math.acos(ground_r / (ground_r + max(float(eye_m), 0.0))))


def horizon_marks(cam, aspect: float, width: float, height: float, eye_m: float = 1.7,
                  ground_r: float = RG, margin: float = 40.0) -> list:
    """Where the compass points stand in the window: [(label, x, y,
    cardinal)] for those in front of the camera and within `margin` pixels
    of the window, (x, y) from the bottom left.  On the horizon as the
    renderer draws it (horizon_dip_deg), through the renderer's own
    rectilinear projection: a direction d is drawn at normalised
    coordinates (d.right / (d.fwd T aspect), d.up / (d.fwd T)), T the
    tangent of half the vertical field of view (shaders: rd = fwd +
    right x T aspect + up y T)."""
    right, up, fwd = (np.asarray(v, "f8") for v in cam.basis())
    t = math.tan(math.radians(cam.fov) * 0.5)
    dip = math.radians(horizon_dip_deg(eye_m, ground_r))
    out = []
    for name, az, cardinal in COMPASS:
        a = math.radians(az)
        d = np.array([math.cos(dip) * math.sin(a), -math.sin(dip), math.cos(dip) * math.cos(a)])
        z = float(d @ fwd)
        if z <= 1e-3:
            continue                                  # behind the camera
        nx = float(d @ right) / (z * t * aspect)
        ny = float(d @ up) / (z * t)
        x, y = (nx + 1.0) * 0.5 * width, (ny + 1.0) * 0.5 * height
        if -margin <= x <= width + margin and -margin <= y <= height + margin:
            out.append((name, x, y, cardinal))
    return out


@dataclass
class Seen:
    key: str
    name: str
    abbr: str
    share: float = 0.0                 # fraction of the frame
    dist: list = field(default_factory=list)
    elev: list = field(default_factory=list)
    az: list = field(default_factory=list)
    heights: tuple = (0.0, 0.0)
    why: list = field(default_factory=list)
    data: list = field(default_factory=list)

    def where(self) -> str:
        if not self.dist:
            return ""
        d0, d1 = float(np.percentile(self.dist, 5)), float(np.percentile(self.dist, 95))
        e0, e1 = min(self.elev), max(self.elev)
        az = math.degrees(math.atan2(np.mean(np.sin(np.radians(self.az))),
                                     np.mean(np.cos(np.radians(self.az))))) % 360.0
        if e0 > 60:
            pos = "overhead"
        elif e1 < 8:
            pos = "low on the horizon"
        elif e0 < 5 and e1 > 45:
            pos = "from overhead down to the horizon"
        else:
            pos = f"{e0:.0f}-{e1:.0f} deg up"
        comp = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((az + 22.5) // 45) % 8]
        return (f"{pos}, toward {comp}; {d0 / 1000:.1f}-{d1 / 1000:.0f} km away, "
                f"{self.heights[0]:.0f}-{self.heights[1]:.0f} m up")


def _shell(ro_r, rd_y, h0, h1, tmax):
    """Vectorised entry/exit distances of rays from an eye at radius ro_r
    (rays with vertical component rd_y) through the shell h0..h1 - the
    first stretch only (enough for the census)."""
    def sphere(rad):
        b = ro_r * rd_y
        c = ro_r * ro_r - rad * rad
        h = b * b - c
        ok = h >= 0
        s = np.sqrt(np.maximum(h, 0.0))
        return np.where(ok, -b - s, -1.0), np.where(ok, -b + s, -1.0), ok
    ox, oy, ok_o = sphere(RG + h1)
    ix, iy, ok_i = sphere(RG + max(h0, 0.0))
    hit_inner = ok_i & (h0 > 0.0) & (iy > 0.0)
    a0 = np.maximum(ox, 0.0)
    a1 = np.where(hit_inner, np.where(ix > 0.0, ix, -1.0), oy)
    # the eye below the shell: the stretch after the inner sphere
    below = hit_inner & (ix <= 0.0)
    a0 = np.where(below, np.maximum(iy, 0.0), a0)
    a1 = np.where(below, oy, a1)
    a1 = np.minimum(a1, tmax)
    ok = ok_o & (oy > 0.0) & (a1 > a0)
    return a0, a1, ok


def _ground_dist(ro_r, rd_y):
    b = ro_r * rd_y
    c = ro_r * ro_r - RG * RG
    h = b * b - c
    t = -b - np.sqrt(np.maximum(h, 0.0))
    return np.where((h >= 0) & (t > 0), t, np.inf)


class Census:
    """Recomputed a few times a second by the app."""

    def __init__(self, nu: int = 36, nv: int = 22, samples: int = 140):
        self.nu, self.nv, self.samples = nu, nv, samples
        self.seen: list[Seen] = []
        self.sky_share = 0.0
        self.ground_share = 0.0
        self.scale_m = None
        self.band_deg = None
        self.anisotropy = 0.0

    # ------------------------------------------------------ model columns --
    @staticmethod
    def columns(driver):
        """Per-column (top, bottom, water path, ice fraction, precipitation
        bottom, rain rate, column optical depth) of the running model."""
        m = driver.model
        if m is None:
            return None
        g = driver.sc.grid
        if driver.gpu_ok:
            cd = m.column_diag().astype("f8")
            top, bot, lwp = cd[..., 5], cd[..., 6], cd[..., 7]
            ice = cd[..., 9] / np.maximum(cd[..., 10], 1e-12)
            pbot, rate = cd[..., 14], cd[..., 11]
        else:
            rho = m.b.rho0.reshape(-1, 1, 1)
            cond = (m.qc + m.qi).astype("f8")
            cl = cond > 1e-5
            zc = g.zc.reshape(-1, 1, 1)
            top = np.where(cl.any(0), (np.where(cl, zc, -1e9).max(0) + 0.5 * g.dz), 0.0)
            bot = np.where(cl.any(0), (np.where(cl, zc, 1e9).min(0) - 0.5 * g.dz), 1e9)
            lwp = (rho * cond).sum(0) * g.dz
            ice = m.qi.sum(0) / np.maximum(cond.sum(0), 1e-12)
            opt = m.optics()
            pr = (opt["rain"] + opt["snow"]) > crm.PRECIP_VISIBLE
            pbot = np.where(pr.any(0), np.where(pr, zc, 1e9).min(0) - 0.5 * g.dz, 1e9)
            rate = m.rain_rate
        tau = lwp * (150.0 * (1.0 - ice) + 65.0 * ice)
        return dict(top=top, bot=bot, lwp=lwp, ice=ice, pbot=pbot, rate=rate, tau=tau)

    def pattern(self, cols, grid):
        """The model's dominant horizontal scale and how banded it is.  The
        cell size is twice the first zero of the radially averaged
        autocorrelation of the column optical depth (for a cellular field
        the correlation turns negative half a cell away); the banding is
        how much of the spectrum's power lies along one direction."""
        t = np.log1p(cols["tau"])
        t = t - t.mean()
        self.scale_m, self.band_deg, self.anisotropy = None, None, 0.0
        if not np.any(t):
            return
        ny, nx = t.shape
        F = np.fft.fft2(t)
        P = np.abs(F) ** 2
        ac = np.real(np.fft.ifft2(P))
        ac /= max(ac[0, 0], 1e-30)
        yy = np.fft.fftfreq(ny) * ny * grid.dy
        xx = np.fft.fftfreq(nx) * nx * grid.dx
        r = np.hypot(yy[:, None], xx[None, :])
        dr = max(grid.dx, grid.dy)
        nb = int(min(nx * grid.dx, ny * grid.dy) / 2 / dr)
        prof = []
        for b in range(1, nb):
            m = (r >= (b - 0.5) * dr) & (r < (b + 0.5) * dr)
            prof.append(ac[m].mean() if m.any() else 0.0)
        zero = next((b for b, v in enumerate(prof, start=1) if v <= 0.0), None)
        if zero is not None:
            self.scale_m = 2.0 * zero * dr
        P[0, 0] = 0.0
        ky = np.fft.fftfreq(ny, grid.dy)[:, None] + 0.0 * np.zeros((1, nx))
        kx = np.fft.fftfreq(nx, grid.dx)[None, :] + 0.0 * np.zeros((ny, 1))
        k = np.hypot(kx, ky)
        w = np.where(k > 1.5 / (nx * grid.dx), P, 0.0)
        ang = np.arctan2(ky, kx)
        c2 = (w * np.cos(2 * ang)).sum() / max(w.sum(), 1e-30)
        s2 = (w * np.sin(2 * ang)).sum() / max(w.sum(), 1e-30)
        self.anisotropy = float(math.hypot(c2, s2))
        kdir = 0.5 * math.degrees(math.atan2(s2, c2))        # direction of k, from east
        # the bands run across k; as a bearing from north
        self.band_deg = (90.0 - (kdir + 90.0)) % 180.0

    # ------------------------------------------------------------ census --
    def update(self, cam, aspect, eye_alt, driver, regime, decks, rmaps, deck_maps,
               world_time, special=None, sim_fade=1.0):
        """deck_maps: per deck, a function (east m, north m) -> cloud fraction
        (timeline.Sky.frac_at), or a (map, extent) pair for a deck that does
        not evolve; sim_fade: how much of the model's field is shown."""
        right, up, fwd = (np.asarray(v, "f8") for v in cam.basis())
        th = math.tan(math.radians(cam.fov) * 0.5)
        u = (np.arange(self.nu) + 0.5) / self.nu * 2.0 - 1.0
        v = (np.arange(self.nv) + 0.5) / self.nv * 2.0 - 1.0
        uu, vv = np.meshgrid(u, v)
        rd = (fwd[None, None, :] + right[None, None, :] * (uu * th * aspect)[..., None]
              + up[None, None, :] * (vv * th)[..., None])
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
        rd = rd.reshape(-1, 3)
        nray = rd.shape[0]
        ro_r = RG + eye_alt
        tg = _ground_dist(ro_r, rd[:, 1])
        best_t = np.full(nray, np.inf)
        best_k = np.full(nray, "", dtype=object)
        info = {}

        # --- the model's field -------------------------------------------
        sc = driver.sc if driver is not None else None
        view = driver.view if driver is not None else None
        cols = self.columns(driver) if (sc is not None and driver.model is not None) else None
        if cols is not None:
            g = sc.grid
            self.pattern(cols, g)
            a0, a1, ok = _shell(ro_r, rd[:, 1], view.zlo, view.zhi, np.minimum(tg, 300000.0))
            s = (np.arange(self.samples) + 0.5) / self.samples
            # geometric spacing: dense near the entry, sparse far out
            tt = a0[:, None] + (a1 - a0)[:, None] * (np.expm1(3.0 * s) / math.expm1(3.0))[None, :]
            p = rd[:, None, :] * tt[..., None]
            cen = np.array([0.0, -ro_r, 0.0])
            alt = np.linalg.norm(p - cen, axis=-1) - RG
            warp, cov = rmaps.sample(p[..., 0], p[..., 2]) if rmaps is not None else (
                np.zeros(p.shape[:2] + (2,)), np.ones(p.shape[:2] + (4,)))
            sx = view.obs[0] + p[..., 0] - warp[..., 0]
            sy = view.obs[1] + p[..., 2] - warp[..., 1]
            inside = ok[:, None] & np.ones_like(sx, bool)
            if not (sc.tiling & 1):
                inside &= (sx >= 0) & (sx < g.lx)
            if not (sc.tiling & 2):
                inside &= (sy >= 0) & (sy < g.ly)
            ii = np.floor(sx / g.dx).astype(int) % g.nx
            jj = np.floor(sy / g.dy).astype(int) % g.ny
            top = cols["top"][jj, ii]
            bot = cols["bot"][jj, ii]
            tau = cols["tau"][jj, ii]
            km = getattr(rmaps, "keep_map", None) if rmaps is not None else None
            tau_k = km[jj, ii] if km is not None and km.shape == cols["tau"].shape else tau
            # the forecast's cover out there (the renderer's simKeep)
            r = np.clip(cov[..., 3], 0.0, 1.0) * float(min(max(sim_fade, 0.0), 1.0))
            tab = np.asarray(rmaps.tau_th if rmaps is not None else [0.0] * 17)
            f = r * 16.0
            k0 = np.floor(f).astype(int)
            thr = tab[k0] * (1 - (f - k0)) + tab[np.minimum(k0 + 1, 16)] * (f - k0)
            keep = ((r >= 0.999) | (tau_k > thr)) & (r > 0.001)
            cloud = inside & keep & (tau > 0.5) & (alt >= bot - g.dz) & (alt <= top + g.dz)
            pb = cols["pbot"][jj, ii]
            rain = inside & keep & (pb < 1e8) & (alt >= pb) & (alt < bot) & ~cloud
            hitc = cloud.any(1)
            hitr = rain.any(1)
            ic = np.argmax(cloud, axis=1)
            ir = np.argmax(rain, axis=1)
            tc = np.where(hitc, tt[np.arange(nray), ic], np.inf)
            trn = np.where(hitr, tt[np.arange(nray), ir], np.inf)
            # name every hit column
            for n in np.nonzero(hitc | hitr)[0]:
                if tc[n] <= trn[n]:
                    k = ic[n]
                    j_, i_ = jj[n, k], ii[n, k]
                    key = self._model_key(regime, sc, cols, j_, i_)
                    t_hit = tc[n]
                else:
                    k = ir[n]
                    j_, i_ = jj[n, k], ii[n, k]
                    reach = cols["pbot"][j_, i_] - g.z0 <= 1.5 * g.dz and g.z0 <= 0.0
                    key = "model:pra" if reach else "model:vir"
                    t_hit = trn[n]
                if t_hit < best_t[n]:
                    best_t[n] = t_hit
                    best_k[n] = key
                    info.setdefault(key, []).append((n, t_hit, j_, i_))

        # --- the Atlas layers --------------------------------------------
        for di, d in enumerate(decks or []):
            a0, a1, ok = _shell(ro_r, rd[:, 1], max(d.base_m, 20.0), d.top_m, tg)
            if not ok.any():
                continue
            tmid = np.where(ok, a0 + 0.3 * (a1 - a0), np.inf)
            p = rd * np.where(ok, tmid, 0.0)[:, None]
            src = deck_maps[di] if deck_maps and di < len(deck_maps) else None
            if callable(src):
                c = np.asarray(src(p[:, 0], p[:, 2]), "f8")
            elif src is not None:
                m, ex = src
                uvx = ((p[:, 0] - d.wind_u * world_time) / ex) % 1.0
                uvy = ((p[:, 2] - d.wind_v * world_time) / ex) % 1.0
                n = m.shape[0]
                c = m[(uvy * n).astype(int) % n, (uvx * n).astype(int) % n, 0].astype("f8")
            else:
                c = np.full(nray, d.coverage)
            if rmaps is not None and rmaps.has_data:
                _, cv = rmaps.sample(p[:, 0], p[:, 2])
                we = region.etage_weights(d.base_m)
                c = c * np.clip(sum(w * cv[:, e] for e, w in enumerate(we) if w > 0.0), 0, 1.5)
            hit = ok & (c > 0.3) & (tmid < best_t)
            key = f"deck:{di}"
            for n in np.nonzero(hit)[0]:
                best_t[n] = tmid[n]
                best_k[n] = key
                info.setdefault(key, []).append((n, tmid[n], 0, 0))

        # --- the upper atmosphere ------------------------------------------
        if special:
            for sp in special:
                mask = sp.visible_mask(rd) & ~np.isfinite(best_t)
                key = "special:" + sp.key
                for n in np.nonzero(mask)[0]:
                    best_k[n] = key
                    info.setdefault(key, []).append((n, sp.distance(rd[n]), 0, 0))

        # --- tally ---------------------------------------------------------
        cen = np.array([0.0, -ro_r, 0.0])
        self.ground_share = float(np.mean(np.isfinite(tg) & (best_k == "")))
        self.sky_share = float(np.mean(~np.isfinite(tg) & (best_k == "")))
        out = []
        for key, hits in info.items():
            idx = np.array([h[0] for h in hits])
            ts = np.array([h[1] for h in hits])
            # count only the rays this kind actually won
            won = best_k[idx] == key
            if not won.any():
                continue
            idx, ts = idx[won], ts[won]
            pts = rd[idx] * ts[:, None]
            alts = np.linalg.norm(pts - cen, axis=-1) - RG
            elev = np.degrees(np.arcsin(np.clip(rd[idx, 1], -1, 1)))
            az = np.degrees(np.arctan2(rd[idx, 0], rd[idx, 2])) % 360.0
            seen = self._describe(key, regime, driver, cols, decks, special)
            if seen is None:
                continue
            seen.share = len(idx) / float(nray)
            seen.dist = list(ts)
            seen.elev = list(elev)
            seen.az = list(az)
            if key.startswith("model") or key.startswith("deck"):
                seen.heights = (float(np.percentile(alts, 5)), float(np.percentile(alts, 95)))
            seen.why += self._perspective(seen, ts, elev)
            out.append(seen)
        out.sort(key=lambda s: -s.share)
        self.seen = out
        return out

    # ------------------------------------------------------------ naming --
    def _model_key(self, regime, sc, cols, j, i):
        key = sc.key
        if key in ("weather", "supercell", "bomex", "cumulus"):
            top, bot = cols["top"][j, i], cols["bot"][j, i]
            depth = top - bot
            ice = cols["ice"][j, i]
            trop = (regime.facts.get("trop") if regime is not None else None) or 12000.0
            if ice > 0.25 and depth > 5000 and top > trop + 600.0:
                return "model:ovt"                       # overshooting top
            if bot > 5500 and ice > 0.5:
                return "model:inc"                       # anvil sheet
            if ice > 0.25 and depth > 5000:
                return "model:Cb cap" if ice > 0.45 else "model:Cb cal"
            if depth < 1000:
                return "model:Cu hum"
            if depth < 2500:
                return "model:Cu med"
            return "model:Cu con"
        return "model:layer"

    def _describe(self, key, regime, driver, cols, decks, special):
        data = []
        if regime is not None:
            data = [regime.decision] + [e.line() for e in regime.evidence[:6]]
        if key.startswith("deck:"):
            d = decks[int(key.split(":")[1])]
            why = list(d.reasons[:5])
            return Seen(key, d.spec.latin(), d.spec.abbrev(), why=why,
                        data=["drawn as an Atlas layer (not simulated): " + "; ".join(d.reasons[:3])])
        if key.startswith("special:"):
            sp = next((s for s in special if "special:" + s.key == key), None)
            return Seen(key, sp.name, sp.abbr, why=list(sp.why), data=list(sp.data)) if sp else None
        sc = driver.sc
        g = sc.grid
        dg = driver.diag or {}
        shows = sc.shows or [(sc.name, sc.description)]
        why = []
        if key == "model:layer":
            name = shows[0][0]
            spec = _layer_spec(name, cols, self, sc)
            why.append(shows[0][1])
            young = driver.model_time < 900.0
            if young:
                why.append(f"the model is {driver.model_time / 60:.0f} min old: its overturning "
                           f"is still organising, the cells are not yet at their size")
            elif self.scale_m and sc.key == "asperitas":
                why.append(f"the pattern printed on the base is about {self.scale_m:.0f} m "
                           f"across; Kelvin-Helmholtz waves on a shear layer 200 m deep grow "
                           f"fastest at about 7 times its depth, some 1.4 km")
            elif self.scale_m:
                depth = getattr(sc, "layer_depth", 0.0) or max(
                    dg.get("cloud_top", 0) - dg.get("cloud_base", 0), g.dz)
                why.append(f"its cells are about {self.scale_m:.0f} m across, "
                           f"{self.scale_m / depth:.1f} times the {depth:.0f} m depth of the "
                           f"overturning layer - the aspect ratio of convection cells")
            if not young and self.band_deg is not None:
                if self.anisotropy > 0.4:
                    along = _along_wind(self.band_deg, sc)
                    why.append(f"the cells line up in bands running {self.band_deg:.0f} deg "
                               f"from north, " + ("along the wind: rolls (radiatus)" if along
                                                  else "across the wind: waves (undulatus)"))
                elif any(v in sc.name for v in ("radiatus", "undulatus")):
                    why.append("the bands this case is built for have not organised on this "
                               "model grid yet (a finer grid resolves them)")
            tau_m = float(np.median(cols["tau"][cols["tau"] > 0.5])) if (cols["tau"] > 0.5).any() else 0.0
            if tau_m:
                t = 1.0 / (1.0 + 0.105 * tau_m)
                why.append(f"typical optical depth {tau_m:.0f}: the base passes about "
                           f"{t * 100:.0f}% of the light falling on the top, which is why "
                           f"the thick cell centres look darker than their thin edges")
            if sc.reasons:
                why += sc.reasons[:2]
            return Seen(key, spec.latin(), spec.abbrev(), why=why, data=data)
        if key in ("model:vir", "model:pra"):
            vir = key == "model:vir"
            name = "virga (the model's precipitation, evaporating)" if vir else \
                "praecipitatio (precipitation reaching the ground)"
            why = ["precipitation falls out of the cloud base at 1-6 m/s",
                   "the air under the cloud is dry, so it evaporates or sublimates before the "
                   "ground" if vir else
                   (f"the air below is moist enough for it to reach the ground "
                    f"(up to {dg.get('rain_rate_max', 0):.0f} mm/h)")]
            lean = _trail_lean(sc, dg)
            if lean:
                why.append(lean)
            return Seen(key, name, "vir" if vir else "pra", why=why, data=data)
        # convective clouds
        k = key.split(":")[1]
        cape = regime.facts.get("cape", 0) if regime else 0
        lcl = dg.get("cloud_base", 0)
        if k == "ovt":
            spec = CloudSpec("Cb", "cap", supplementary=["inc"])
            why = ["overshooting top: the strongest updraught is still rising when it reaches "
                   "the tropopause and carries on above the anvil by its momentum, a dome that "
                   "falls back within minutes",
                   f"tops {dg.get('cloud_top', 0) / 1000:.1f} km, updraught up to "
                   f"{dg.get('w_max', 0):.0f} m/s"]
            lat = abs(regime.facts.get("lat", 0.0)) if regime is not None else 0.0
            if dg.get("cloud_top", 0) > 14000:
                why.append("in the tropics such a tower is a 'hot tower' (Riehl & Malkus 1958): "
                           "it carries the heat of the ocean's evaporation to the tropopause")
            return Seen(key, spec.latin() + " with an overshooting top", spec.abbrev(),
                        why=why, data=data)
        if k == "inc":
            spec = CloudSpec("Cb", "cap", supplementary=["inc"])
            why = ["the updraughts have reached the tropopause, where the air stops being "
                   "buoyant; the ice they carry spreads out sideways into an anvil",
                   f"cloud tops {dg.get('cloud_top', 0) / 1000:.1f} km"]
        elif k.startswith("Cb"):
            spec = CloudSpec("Cb", k.split()[1])
            why = [f"a thermal released CAPE {cape:.0f} J/kg and rose "
                   f"{(dg.get('cloud_top', 0) - lcl) / 1000:.1f} km above its base; "
                   f"its top froze (ice {dg.get('ice_frac', 0) * 100:.0f}% of the cloud)",
                   "capillatus: the frozen top has lost the sharp cauliflower outline"
                   if k.endswith("cap") else "calvus: the top is freezing but still has its bulges"]
        else:
            sp = k.split()[1]
            spec = CloudSpec("Cu", sp)
            if sp == "con":
                return Seen(key, spec.latin() + " (towering cumulus, TCu; also called Cumulus "
                            "castellanus)", spec.abbrev(),
                            why=[f"a thermal from the sun-warmed ground reached its condensation "
                                 f"level ({lcl:.0f} m) and strong buoyancy (CAPE {cape:.0f} J/kg) "
                                 f"carried it several km, taller than wide",
                                 "the next stage is cumulonimbus: when its top freezes"],
                            data=data)
            if sc.key == "bomex":
                return Seen(key, spec.latin() + " (trade-wind cumulus)", spec.abbrev(),
                            why=["trade-wind cumulus: moist air over a warm ocean rises until "
                                 "the trade inversion near 2 km caps it (BOMEX, Siebesma et al. "
                                 "2003)"], data=data)
            why = [f"a thermal from the sun-warmed ground reached its condensation level "
                   f"({lcl:.0f} m) - hence the flat base - and kept rising while warmer than "
                   f"its surroundings",
                   {"hum": "humilis: a stable layer or dry air stops it within 1 km",
                    "med": "mediocris: it grows as tall as it is wide before the buoyancy runs out",
                    "con": "congestus: strong buoyancy carries it several km, taller than wide"}[sp]]
        return Seen(key, spec.latin(), spec.abbrev(), why=why, data=data)

    def _perspective(self, seen, ts, elev):
        """Why the same clouds look smaller and flatter further away."""
        if not seen.key.startswith("model") or self.scale_m is None or len(ts) < 4:
            return []
        if seen.key in ("model:vir", "model:pra"):
            return []
        lam = self.scale_m
        h = max(seen.heights[0], 50.0)
        near = float(np.percentile(ts, 5))
        far = float(np.percentile(ts, 95))
        if far < 2.0 * near:
            return []
        a_near = math.degrees(2.0 * math.atan(lam / (2.0 * max(near, 1.0))))
        a_far = math.degrees(lam / max(far, 1.0))
        e_far = max(float(np.percentile(elev, 5)), 0.5)
        return [f"perspective: a {lam / 1000:.1f} km cell {near / 1000:.1f} km away spans "
                f"{a_near:.0f} deg; {far / 1000:.0f} km away it spans {a_far:.1f} deg across and, "
                f"seen at {e_far:.0f} deg elevation, only {a_far * math.sin(math.radians(e_far)):.2f} "
                f"deg tall - which is why the same pattern shrinks, flattens and crowds into "
                f"rows toward the horizon"]


def _trail_lean(sc, dg) -> str | None:
    """Which way the precipitation trails hang: the cloud drifts with the
    wind at its base, its precipitation falls into the wind below, and the
    difference carries the trail off to one side - from the scenario's
    winds, which in a sky chosen from the data are the forecast's."""
    from .regimes import compass, wind_from
    zt = float(dg.get("cloud_base", 0.0))
    zb = float(dg.get("precip_bottom", -1.0))
    if sc is None or zt <= 0.0 or zb < 0.0 or zt - zb < 100.0:
        return None
    b, zc = sc.base, sc.grid.zc
    zm = 0.5 * (zt + zb)
    ut, vt = float(np.interp(zt, zc, b.u0)), float(np.interp(zt, zc, b.v0))
    ul, vl = float(np.interp(zm, zc, b.u0)), float(np.interp(zm, zc, b.v0))
    du, dv = ul - ut, vl - vt
    src = " (the forecast's winds at those heights)" if "from your sounding" in sc.name else ""
    if math.hypot(du, dv) < 0.7:
        return (f"the wind hardly changes between {zt / 1000:.1f} and {zb / 1000:.1f} km, so the "
                f"trails hang almost straight down{src}")
    toward = compass(math.degrees(math.atan2(du, dv)))
    return (f"the trails lean: the cloud at {zt / 1000:.1f} km drifts with the wind there "
            f"({math.hypot(ut, vt):.0f} m/s from {compass(wind_from(ut, vt))}) while what falls "
            f"from it meets the wind at {zm / 1000:.1f} km ({math.hypot(ul, vl):.0f} m/s from "
            f"{compass(wind_from(ul, vl))}); relative to its cloud it drifts "
            f"{math.hypot(du, dv):.0f} m/s toward the {toward}, so each trail hangs back toward "
            f"the {toward}{src}")


def _along_wind(band_deg: float, sc) -> bool:
    """Do bands running band_deg (from north) lie along the layer's wind?"""
    um, vm = sc.translate
    if math.hypot(um, vm) < 1.0:
        return True
    wdir = math.degrees(math.atan2(um, vm)) % 180.0
    d = abs((band_deg - wdir + 90.0) % 180.0 - 90.0)
    return d < 35.0


def _layer_spec(name: str, cols, census: Census, sc=None) -> CloudSpec:
    """The Atlas name of the model's layer cloud, with the varieties its
    field actually shows (the ones a case is built for are only claimed
    once they have formed)."""
    try:
        spec = atlas.parse(name.split("(")[0].strip())
    except Exception:
        spec = CloudSpec("Sc", "str")
    tau = cols["tau"]
    cloudy = tau > 1.0
    cover = float(cloudy.mean())
    tmed = float(np.median(tau[cloudy])) if cloudy.any() else 0.0
    gv = atlas.GENERA[spec.genus]["varieties"]
    v = [x for x in spec.varieties if x not in ("tr", "op", "pe", "ra", "un")]
    if tmed and tmed < 5 and "tr" in gv:
        v.append("tr")
    elif tmed > 15 and "op" in gv:
        v.append("op")
    if 0.5 <= cover <= 0.92 and "pe" in gv and "op" not in v:
        v.append("pe")
    if census.anisotropy > 0.4 and census.band_deg is not None and sc is not None:
        along = _along_wind(census.band_deg, sc)
        if along and "ra" in gv:
            v.append("ra")
        elif not along and "un" in gv:
            v.append("un")
    spec.varieties = v
    return spec
