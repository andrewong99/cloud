# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
The cloud around the place, seen from above - the satellite's view that the
sky over the observer is a piece of.

The forecast gives cloud cover and wind on a grid of points around the
observer (sounding.fetch_region).  From them this module makes the maps the
renderer continues the simulated cloud field with, beyond the model's own
domain:

  * where the forecast has less of the model's kind of cloud than there is
    overhead, the model's thinnest columns are left out, so the field thins
    and ends where the forecast's cloud thins and ends - it keeps the
    model's cells and rows, it only has fewer of them;
  * the field is displaced by how differently the air there has been carried
    over the last hours (the forecast's wind there minus the wind here, times
    two hours), so what lies 60 km away is not a copy of what is overhead.

Offline (no regional data) the cover is the same everywhere and the
displacement is the smooth stretch described in `synthetic_warp`.
"""

from __future__ import annotations

import math

import numpy as np

ADVECTION_S = 2.0 * 3600.0          # how long the far field has been carried apart
MAP_N = 128                          # texels across the regional maps


def _upsample(grid: np.ndarray, n_out: int) -> np.ndarray:
    """Smooth (cubic where scipy is there) resampling of an n x n grid of
    point values to n_out x n_out texels covering the same square; NaNs are
    filled from their neighbours first."""
    g = np.array(grid, "f8")
    if np.isnan(g).all():
        return np.zeros((n_out, n_out))
    while np.isnan(g).any():
        m = np.isnan(g)
        p = np.pad(g, 1, mode="edge")
        acc = np.zeros_like(g)
        cnt = np.zeros_like(g)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = p[1 + dy:1 + dy + g.shape[0], 1 + dx:1 + dx + g.shape[1]]
                ok = ~np.isnan(v)
                acc[ok] += v[ok]
                cnt[ok] += 1
        fill = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        g[m] = fill[m]
    n = g.shape[0]
    # texel centres in grid coordinates (the grid points sit at 0..n-1)
    x = (np.arange(n_out) + 0.5) / n_out * n - 0.5
    try:
        from scipy.ndimage import map_coordinates
        yy, xx = np.meshgrid(x, x, indexing="ij")
        return map_coordinates(g, [yy, xx], order=3, mode="nearest")
    except Exception:
        x = np.clip(x, 0, n - 1)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        f = x - i0
        rows = g[:, i0] * (1 - f) + g[:, i1] * f
        return rows[i0, :] * (1 - f)[:, None] + rows[i1, :] * f[:, None]


def _wind_uv(speed, direction):
    s = np.nan_to_num(speed, nan=0.0)
    d = np.radians(np.nan_to_num(direction, nan=0.0))
    return -s * np.sin(d), -s * np.cos(d)


def synthetic_warp(n: int, half_m: float, period_m: float) -> np.ndarray:
    """A smooth, slowly varying displacement (three long waves of 55-120 km,
    up to 0.35 of the model's domain), zero within two domain widths of the
    observer: without regional data it is what keeps the far field from
    repeating (this program's choice)."""
    x = ((np.arange(n) + 0.5) / n * 2.0 - 1.0) * half_m
    yy, xx = np.meshgrid(x, x, indexing="ij")
    amp = 0.35 * period_m
    waves = [(55e3, 0.4, 1.1), (80e3, 2.1, 0.3), (120e3, 4.0, 2.5)]
    wx = np.zeros_like(xx)
    wy = np.zeros_like(xx)
    for lam, ang, ph in waves:
        k = 2.0 * math.pi / lam
        c, s = math.cos(ang), math.sin(ang)
        a = k * (c * xx + s * yy) + ph
        wx += amp / 3.0 * np.sin(a)
        wy += amp / 3.0 * np.cos(a + 0.7)
    r = np.hypot(xx, yy)
    ramp = np.clip((r - 2.0 * period_m) / (2.0 * period_m), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    w = np.stack([wx * ramp, wy * ramp], -1)
    return w - _centre(w)


def _centre(a):
    """The bilinear value at the map's exact centre (between four texels
    for an even size)."""
    n = a.shape[0]
    h = n // 2
    return a[h - 1:h + 1, h - 1:h + 1].mean(axis=(0, 1)) if n % 2 == 0 else a[h, h]


def demo_region(cover_fn, etage: int, n: int = 15, spacing_km: float = 8.0,
                speed: float = 0.0, direction: float = 270.0):
    """A made-up forecast grid for a reference case - this program's own
    composition, labelled as such wherever it is used: cover_fn(east_km,
    north_km) gives the cover (%) of the model's étage there, the other
    étages are clear, and the wind is the same everywhere."""
    from .sounding import Region, RegionPoint, region_offsets
    names = ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high")
    pts = []
    for (e, nk) in region_offsets(n, spacing_km):
        vals = {nm: (float(cover_fn(e, nk)) if i == etage else 0.0) for i, nm in enumerate(names)}
        for lvl in ("850hPa", "700hPa"):
            vals[f"wind_speed_{lvl}"] = speed
            vals[f"wind_direction_{lvl}"] = direction
        pts.append(RegionPoint(e, nk, 0.0, 0.0, vals, {}))
    return Region(pts, spacing_km, n, source="demo")


def keep_tau_map(col_tau: np.ndarray, dx: float, radius_m: float = 300.0) -> np.ndarray:
    """The column optical depth the thinning decides on: each column takes
    the largest within radius_m (the model's domain is periodic), so a cell
    - a kilometre or so across - is kept or left out whole instead of being
    cut along its columns."""
    a = np.asarray(col_tau, "f8")
    r = max(1, int(round(radius_m / max(dx, 1.0))))
    try:
        from scipy.ndimage import maximum_filter
        return maximum_filter(a, size=2 * r + 1, mode="wrap")
    except Exception:
        out = a.copy()
        for dy in range(-r, r + 1):
            for dx_ in range(-r, r + 1):
                out = np.maximum(out, np.roll(np.roll(a, dy, 0), dx_, 1))
        return out


def source_for(key, sc, snd):
    """What the regional maps are built from.  The forecast's regional cover
    belongs to the sky chosen from the data ('auto') and to the convection
    grown from the sounding ('weather'); a reference case shows its
    mechanism with its own composition if it has one (Scenario.demo_cover)
    and the same everywhere if not - never with a forecast that did not ask
    for its cloud."""
    if sc is None or key in (None, "auto", "weather"):
        return snd
    if getattr(sc, "demo_cover", None) is not None:
        zb = sc.random_zmin if sc.random_zmin > 0 else sc.grid.z0
        return type("DemoSounding", (), {"region": demo_region(sc.demo_cover,
                                                                etage_of_height(zb))})()
    return None


def etage_of_height(z_m: float) -> int:
    """0 low, 1 middle, 2 high (by base height above the ground)."""
    return 0 if z_m < 2000.0 else (1 if z_m < 7000.0 else 2)


def _smooth(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def etage_weights(base_m: float) -> tuple[float, float, float]:
    """How much of the low, middle and high étage's regional cover a deck
    based at base_m takes: its own étage's, blended across the boundaries
    (2 km over 600 m, 7 km over 1 km), so a layer that rises through one
    changes gradually instead of taking another étage's cover at once (the
    shader's regionDeck)."""
    k1 = _smooth(1700.0, 2300.0, base_m)
    k2 = _smooth(6500.0, 7500.0, base_m)
    return (1.0 - k1, k1 * (1.0 - k2), k2)


class RegionMaps:
    """The regional maps for the renderer and for the in-view census.

    Between two forecast hours the maps are not rebuilt as the clock moves:
    both hours' maps (raw_a, raw_b: the forecast's cover in percent per
    étage; warp_a, warp_b) stay as they are for the whole hour, and the
    moment enters through a few numbers (set_time): how far into the hour
    the clock is (f), how far each étage's field has been carried since the
    first hour and will be until the second (shift_a, shift_b), and the
    cover here now.  The renderer blends the two per sample - the
    advection-corrected interpolation of build_series, continuous at every
    frame, with no work on the CPU as the clock runs (the maps used to be
    rebuilt once a simulated minute, 15 ms on the main thread, and changed
    in steps)."""

    ETAGE_VARS = ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high")

    def __init__(self, n: int = MAP_N):
        self.n = n
        self.half_m = 1.0
        self.raw_a = np.full((n, n, 3), 100.0, "f4")
        self.raw_b = self.raw_a
        self.warp_a = np.zeros((n, n, 2), "f4")
        self.warp_b = self.warp_a
        self.f = 0.0
        self.shift_a = np.zeros((3, 2))
        self.shift_b = np.zeros((3, 2))
        self.winds = np.zeros((3, 2))
        self.local = (0.0, 0.0, 0.0)
        self.local_now = np.array([100.0, 100.0, 100.0])
        self.sim_etage = 0
        self.tau_th = [0.0] * 17
        self.note = "no regional data"
        self.has_data = False
        self.keep_map = None
        #: bumped whenever the maps themselves change (the renderer uploads)
        self.version = 0
        self._pair = None
        self._hours = {}

    # ------------------------------------------------------------- building --
    def _maps_of(self, reg, sim_et: int, half: float, period_m: float):
        """(raw cover %, warp) of one Region: the cover of each étage, and the
        displacement (wind there - wind here) x two hours at the level that
        carries the model's cloud, plus the smooth stretch far out so a
        uniform flow does not leave the field repeating."""
        n = self.n
        raw = np.stack([_upsample(reg.grid(v), n) for v in self.ETAGE_VARS], -1)
        lvl = "850hPa" if sim_et == 0 else "700hPa"
        u, v = _wind_uv(reg.grid(f"wind_speed_{lvl}"), reg.grid(f"wind_direction_{lvl}"))
        c = reg.n // 2
        w = np.stack([(_upsample(u, n) - u[c, c]) * ADVECTION_S,
                      (_upsample(v, n) - v[c, c]) * ADVECTION_S], -1)
        w = w + synthetic_warp(n, half, max(period_m, 1000.0))
        return raw.astype("f4"), (w - _centre(w)).astype("f4")

    def build(self, snd, sim_layer_z: float | None, period_m: float):
        """From the sounding's region (if any), holding in time.  sim_layer_z
        is the height of the model's cloud layer (its base), which picks the
        étage whose cover the model's field follows and the level of the
        wind that carries it; period_m is the model domain's width."""
        reg = getattr(snd, "region", None)
        n = self.n
        self._pair = None
        self.sim_etage = etage_of_height(sim_layer_z if sim_layer_z is not None else 1000.0)
        if reg is None or not reg.points:
            self.half_m = 120000.0
            self.warp_a = self.warp_b = synthetic_warp(n, self.half_m, max(period_m, 1000.0)).astype("f4")
            self.raw_a = self.raw_b = np.full((n, n, 3), 100.0, "f4")
            self.has_data = False
            self.note = ("no regional data: the cloud is taken to be the same all around, and "
                         "the model's field continues with a smooth stretch so it never repeats")
            self.version += 1
            self.set_time(0.0)
            return self
        half = reg.spacing_km * (reg.n // 2 + 0.5) * 1000.0
        self.half_m = half
        self.raw_a, self.warp_a = self._maps_of(reg, self.sim_etage, half, period_m)
        self.raw_b, self.warp_b = self.raw_a, self.warp_a
        self.has_data = True
        lvl = "850hPa" if self.sim_etage == 0 else "700hPa"
        if getattr(reg, "source", "") == "demo":
            self.note = ("the layer's extent around you is this program's composition for the "
                         "demonstration (with live data the forecast grid decides it); the "
                         "cells, gaps and trails are the model's")
        else:
            self.note = (f"forecast cover on a {reg.n}x{reg.n} grid {reg.spacing_km:.0f} km apart; "
                         f"the model's field continues displaced by the difference in the "
                         f"{lvl[:3]} hPa wind over two hours")
        self.version += 1
        self.set_time(0.0)
        return self

    def build_series(self, series, when, sim_layer_z: float | None, period_m: float,
                     high_wind=None):
        """The maps between the forecast's hours (forecast.RegionSeries).
        Each étage's cover is carried along its own wind from the hour before
        and from the hour after and the two are blended by time -
        advection-corrected interpolation (Anagnostou & Krajewski 1999, as in
        pysteps) - so a band of cloud travels across the maps between the
        hours instead of fading out in one place and in at another.  The low
        étage moves with the region's mean 850 hPa wind, the middle one with
        its 700 hPa wind, the high one with high_wind(hour index) (the local
        sounding's, the grid has none up there); each étage's wind is the
        mean of the two hours'.  Only when the clock enters another hour are
        maps made (and those of an hour already seen are kept); within the
        hour this only sets the time (set_time)."""
        x = series.index(when)
        i = int(math.floor(x))
        j = min(i + 1, len(series.times) - 1)
        et = etage_of_height(sim_layer_z if sim_layer_z is not None else 1000.0)
        pair = (id(series), i, j, et, round(float(period_m), 1))
        if pair != self._pair:
            half = series.spacing_km * (series.n // 2 + 0.5) * 1000.0
            if self._pair is None or self._pair[0] != id(series) or self._pair[3:] != pair[3:]:
                self._hours = {}
            for k in (i, j):
                if k not in self._hours:
                    self._hours[k] = self._maps_of(series.region_hour(k), et, half, period_m)
            self._hours = {k: v for k, v in self._hours.items() if abs(k - i) <= 2}
            self.raw_a, self.warp_a = self._hours[i]
            self.raw_b, self.warp_b = self._hours[j]
            self.half_m = half
            self.sim_etage = et

            def mean_wind(reg, lvl):
                u, v = _wind_uv(reg.grid(f"wind_speed_{lvl}"), reg.grid(f"wind_direction_{lvl}"))
                ok = np.isfinite(u) & np.isfinite(v)
                return (float(np.mean(u[ok])), float(np.mean(v[ok]))) if ok.any() else (0.0, 0.0)

            ra, rb = series.region_hour(i), series.region_hour(j)
            ha = tuple(high_wind(i)) if callable(high_wind) else tuple(high_wind or (0.0, 0.0))
            hb = tuple(high_wind(j)) if callable(high_wind) else ha
            wa = [mean_wind(ra, "850hPa"), mean_wind(ra, "700hPa"), ha]
            wb = [mean_wind(rb, "850hPa"), mean_wind(rb, "700hPa"), hb]
            self.winds = np.nan_to_num(0.5 * (np.asarray(wa, "f8") + np.asarray(wb, "f8")))
            self._pair = pair
            self.has_data = True
            lvl = "850hPa" if et == 0 else "700hPa"
            self.note = (f"forecast cover on a {series.n}x{series.n} grid {series.spacing_km:.0f} km "
                         f"apart, carried along the wind between the hours; the model's field "
                         f"continues displaced by the difference in the {lvl[:3]} hPa wind over "
                         f"two hours")
            self.version += 1
        f = (min(max(x, 0.0), float(len(series.times) - 1)) - i) if j != i else 0.0
        self.set_time(f)
        return self

    def set_time(self, f: float):
        """The moment within the pair of hours: f of the way from the first
        to the second.  Each étage's field has been carried u f H since the
        first hour and will be carried u (1 - f) H more until the second."""
        f = float(min(max(np.nan_to_num(f), 0.0), 1.0))
        self.f = f
        w = np.nan_to_num(np.asarray(self.winds, "f8"), nan=0.0, posinf=0.0, neginf=0.0)
        self.shift_a = w * f * 3600.0
        self.shift_b = -w * (1.0 - f) * 3600.0
        loc = []
        for e in range(3):
            a = self._bil(self.raw_a[..., e], -self.shift_a[e, 0], -self.shift_a[e, 1])
            b = self._bil(self.raw_b[..., e], -self.shift_b[e, 0], -self.shift_b[e, 1])
            loc.append(float((1.0 - f) * a + f * b))
        self.local = tuple(loc)
        # a gap in the maps is the cover of here: 100 %, a ratio of one
        self.local_now = np.maximum(np.nan_to_num(np.asarray(loc, "f8"), nan=100.0), 5.0)
        return self

    # --------------------------------------------------------------- lookup --
    def _uv(self, east_m, north_m):
        n = self.n
        u = (np.asarray(east_m, "f8") / (2.0 * self.half_m) + 0.5) * n - 0.5
        v = (np.asarray(north_m, "f8") / (2.0 * self.half_m) + 0.5) * n - 0.5
        u = np.nan_to_num(u, nan=0.5 * (n - 1), posinf=n - 1, neginf=0.0)
        v = np.nan_to_num(v, nan=0.5 * (n - 1), posinf=n - 1, neginf=0.0)
        return np.clip(u, 0, n - 1), np.clip(v, 0, n - 1)

    def _bil(self, a, east_m, north_m):
        """Bilinear lookup (clamped at the edge) of a map at positions
        relative to the observer - the renderer's texture lookup."""
        u, v = self._uv(east_m, north_m)
        n = self.n
        i0 = np.floor(v).astype(int)
        j0 = np.floor(u).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        j1 = np.minimum(j0 + 1, n - 1)
        fi, fj = v - i0, u - j0
        if a.ndim == 3:
            fi, fj = fi[..., None], fj[..., None]
        return ((a[i0, j0] * (1 - fj) + a[i0, j1] * fj) * (1 - fi)
                + (a[i1, j0] * (1 - fj) + a[i1, j1] * fj) * fi)

    def sample(self, east_m, north_m):
        """Warp (east, north) and cover ratios (low, middle, high, the
        model's étage) at positions relative to the observer (arrays): the
        same lookup and blend the shader does (regionWarp, regionEtage)."""
        e_m = np.asarray(east_m, "f8")
        n_m = np.asarray(north_m, "f8")
        f = self.f
        warp = (1.0 - f) * self._bil(self.warp_a, e_m, n_m) + f * self._bil(self.warp_b, e_m, n_m)
        ratios = []
        for e in range(3):
            a = self._bil(self.raw_a[..., e], e_m - self.shift_a[e, 0], n_m - self.shift_a[e, 1])
            b = self._bil(self.raw_b[..., e], e_m - self.shift_b[e, 0], n_m - self.shift_b[e, 1])
            c = ((1.0 - f) * a + f * b) / self.local_now[e]
            ratios.append(np.clip(np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.5))
        ratios.append(ratios[min(max(self.sim_etage, 0), 2)])
        warp = np.nan_to_num(warp, nan=0.0, posinf=0.0, neginf=0.0)
        return warp, np.stack(ratios, -1)

    @property
    def warp(self) -> np.ndarray:
        """The displacement map at the moment (n, n, 2)."""
        return ((1.0 - self.f) * self.warp_a + self.f * self.warp_b).astype("f4")

    @property
    def cover(self) -> np.ndarray:
        """The cover ratios at the moment on the map's texels (n, n, 4)."""
        x = ((np.arange(self.n) + 0.5) / self.n * 2.0 - 1.0) * self.half_m
        yy, xx = np.meshgrid(x, x, indexing="ij")
        return self.sample(xx, yy)[1].astype("f4")

    def gpu_maps(self):
        """(warp of both hours (n, n, 4), cover of the first hour and of the
        second (n, n, 4) each, in percent) for the renderer; a gap in them is
        no displacement and the cover of here."""
        warp = np.concatenate([self.warp_a, self.warp_b], -1)
        warp = np.nan_to_num(warp.astype("f4"), nan=0.0, posinf=0.0, neginf=0.0)
        out = []
        for raw in (self.raw_a, self.raw_b):
            c = np.array(raw, "f4")
            for e in range(3):
                bad = ~np.isfinite(c[..., e])
                c[..., e][bad] = self.local_now[e]
            out.append(np.concatenate([c, np.zeros(c.shape[:2] + (1,), "f4")], -1))
        return warp, out[0], out[1]

    def set_tau_table(self, col_tau: np.ndarray, dx: float | None = None,
                      tau_min: float = 0.3):
        """Column optical depth below which the model's columns are left out
        where the forecast has r = k/16 of the cover here.  Given the grid
        spacing dx and a (ny, nx) map, the decision is made on
        keep_tau_map(col_tau), so whole cells are kept or left out."""
        col_tau = np.asarray(col_tau, "f8")
        if dx is not None and col_tau.ndim == 2:
            col_tau = keep_tau_map(col_tau, dx)
            self.keep_map = col_tau.astype("f4")
        else:
            self.keep_map = None
        t = col_tau.ravel()
        cloudy = t > tau_min
        c_local = float(cloudy.mean()) if t.size else 0.0
        table = []
        for k in range(17):
            r = k / 16.0
            if r >= 1.0 or c_local <= 0.0:
                table.append(0.0)
                continue
            keep = r * c_local
            if keep <= 0.0:
                table.append(float(t.max() * 2.0 + 1.0))
            else:
                table.append(float(np.quantile(t, 1.0 - keep)))
        self.tau_th = table
        return table
