# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
Procedural noise used by the cloud renderer.

Two jobs:

1. Tileable 3D volumes (Perlin-Worley shape noise and Worley detail noise)
   uploaded once as GL 3D textures.  These give a cloud its *internal*
   structure - billows, erosion, wisps.

2. Tileable 2D "deck control maps", one per cloud layer.  These carry the
   Atlas's *macroscopic* structure: element spacing (the apparent-width
   criterion), regular arrangement, undulatus waves, lacunosus holes,
   perlucidus gaps, radiatus/fibratus stretching.  They are generated on the
   CPU so their properties can be measured and asserted in selftest.py.

Everything is tileable so a deck can be advected by the wind indefinitely
without a visible seam.
"""

from __future__ import annotations

import hashlib
import math
import os
import numpy as np

F32 = np.float32


# --------------------------------------------------------------- lattices ---

def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin3(n: int, freq: int, seed: int) -> np.ndarray:
    """Tileable 3D Perlin gradient noise on an n^3 grid, roughly -1..1."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(freq, freq, freq, 3)).astype(F32)
    g /= np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9

    c = (np.arange(n, dtype=F32) + 0.5) / n * freq
    i0 = np.floor(c).astype(np.int32) % freq
    i1 = (i0 + 1) % freq
    f = (c - np.floor(c)).astype(F32)
    u = _fade(f)

    out = np.zeros((n, n, n), F32)
    for dx in (0, 1):
        ix = i0 if dx == 0 else i1
        wx = (1 - u) if dx == 0 else u
        fx = f - dx
        for dy in (0, 1):
            iy = i0 if dy == 0 else i1
            wy = (1 - u) if dy == 0 else u
            fy = f - dy
            for dz in (0, 1):
                iz = i0 if dz == 0 else i1
                wz = (1 - u) if dz == 0 else u
                fz = f - dz
                gv = g[np.ix_(ix, iy, iz)]                     # (n,n,n,3)
                dot = (gv[..., 0] * fx[:, None, None]
                       + gv[..., 1] * fy[None, :, None]
                       + gv[..., 2] * fz[None, None, :])
                out += dot * (wx[:, None, None] * wy[None, :, None] * wz[None, None, :])
    return out


def worley3(n: int, cells: int, seed: int) -> np.ndarray:
    """Tileable 3D Worley (cellular) noise, 0 at feature points to 1 far away."""
    rng = np.random.default_rng(seed)
    jit = rng.random((cells, cells, cells, 3)).astype(F32)

    c = (np.arange(n, dtype=F32) + 0.5) / n * cells
    idx = np.floor(c).astype(np.int32)
    frac = (c - idx).astype(F32)
    idx %= cells

    best = np.full((n, n, n), 9.0, F32)
    for dx in (-1, 0, 1):
        nx = (idx + dx) % cells
        for dy in (-1, 0, 1):
            ny = (idx + dy) % cells
            for dz in (-1, 0, 1):
                nz = (idx + dz) % cells
                j = jit[np.ix_(nx, ny, nz)]
                ex = (dx + j[..., 0]) - frac[:, None, None]
                ey = (dy + j[..., 1]) - frac[None, :, None]
                ez = (dz + j[..., 2]) - frac[None, None, :]
                np.minimum(best, ex * ex + ey * ey + ez * ez, out=best)
    return np.sqrt(best, out=best)


def perlin2(n: int, fx: int, fy: int, seed: int) -> np.ndarray:
    """Tileable 2D Perlin noise, allowing different frequencies per axis
    (that is how anisotropic stretching - radiatus, fibratus - is produced)."""
    rng = np.random.default_rng(seed)
    ang = rng.random((fx, fy)).astype(F32) * (2 * np.pi)
    gx, gy = np.cos(ang), np.sin(ang)

    cx = (np.arange(n, dtype=F32) + 0.5) / n * fx
    cy = (np.arange(n, dtype=F32) + 0.5) / n * fy
    ix0 = np.floor(cx).astype(np.int32) % fx
    iy0 = np.floor(cy).astype(np.int32) % fy
    ix1, iy1 = (ix0 + 1) % fx, (iy0 + 1) % fy
    tx = (cx - np.floor(cx)).astype(F32)
    ty = (cy - np.floor(cy)).astype(F32)
    ux, uy = _fade(tx), _fade(ty)

    out = np.zeros((n, n), F32)
    for dx in (0, 1):
        ax = ix0 if dx == 0 else ix1
        wx = (1 - ux) if dx == 0 else ux
        ddx = tx - dx
        for dy in (0, 1):
            ay = iy0 if dy == 0 else iy1
            wy = (1 - uy) if dy == 0 else uy
            ddy = ty - dy
            dot = (gx[np.ix_(ax, ay)] * ddx[:, None] + gy[np.ix_(ax, ay)] * ddy[None, :])
            out += dot * (wx[:, None] * wy[None, :])
    return out


def worley2(n: int, cx: int, cy: int, seed: int, jitter: float = 1.0):
    """Tileable 2D Worley.  Returns (F1, F2) distances in cell units.

    cx/cy differing gives elongated cells - the mechanism behind radiatus
    bands and the stretched elements of a sheared deck.
    """
    rng = np.random.default_rng(seed)
    jit = (0.5 + (rng.random((cx, cy, 2)).astype(F32) - 0.5) * jitter)

    ax = (np.arange(n, dtype=F32) + 0.5) / n * cx
    ay = (np.arange(n, dtype=F32) + 0.5) / n * cy
    ix = np.floor(ax).astype(np.int32)
    iy = np.floor(ay).astype(np.int32)
    fx = (ax - ix).astype(F32)
    fy = (ay - iy).astype(F32)
    ix %= cx
    iy %= cy

    f1 = np.full((n, n), 9.0, F32)
    f2 = np.full((n, n), 9.0, F32)
    for dx in (-1, 0, 1):
        nx = (ix + dx) % cx
        for dy in (-1, 0, 1):
            ny = (iy + dy) % cy
            j = jit[np.ix_(nx, ny)]
            ex = (dx + j[..., 0]) - fx[:, None]
            ey = (dy + j[..., 1]) - fy[None, :]
            d = ex * ex + ey * ey
            np.minimum(f2, np.maximum(f1, d), out=f2)
            np.minimum(f1, d, out=f1)
    return np.sqrt(f1), np.sqrt(f2)


# ------------------------------------------------------------ 3D textures ---

def build_shape_volume(n: int = 128, seed: int = 20260910) -> np.ndarray:
    """RGBA n^3 uint8.

    R: Perlin-Worley (the classic remap that keeps Perlin's connectedness but
       gains Worley's billowy interiors)
    G,B,A: Worley at increasing frequency, used to erode the shape.
    """
    pf = np.zeros((n, n, n), F32)
    amp, tot = 1.0, 0.0
    for o, f in enumerate((4, 8, 16)):
        pf += amp * perlin3(n, f, seed + o)
        tot += amp
        amp *= 0.5
    pf = (pf / tot) * 0.5 + 0.5
    pf = np.clip(pf, 0.0, 1.0)

    w = [1.0 - worley3(n, c, seed + 100 + i) for i, c in enumerate((4, 8, 16))]
    w = [np.clip(x, 0.0, 1.0) for x in w]
    wfbm = w[0] * 0.625 + w[1] * 0.25 + w[2] * 0.125

    # remap Perlin by the Worley fBm  (Schneider 2015)
    pw = np.clip((pf - (1.0 - wfbm)) / np.maximum(wfbm, 1e-4), 0.0, 1.0)

    out = np.stack([pw, w[0], w[1], w[2]], axis=-1)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def build_detail_volume(n: int = 32, seed: int = 424242) -> np.ndarray:
    """RGB n^3 uint8 of high-frequency Worley, for eroding cloud edges."""
    w = [1.0 - worley3(n, c, seed + i) for i, c in enumerate((4, 8, 16))]
    out = np.stack([np.clip(x, 0, 1) for x in w], axis=-1)
    return (out * 255.0 + 0.5).astype(np.uint8)


def build_blue_noise(n: int = 64, seed: int = 5) -> np.ndarray:
    """Small tileable dither texture used to jitter raymarch start offsets."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n)).astype(F32)
    # a couple of void-and-cluster-ish smoothing passes to decorrelate
    for _ in range(2):
        k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], F32) / 16.0
        b = sum(np.roll(np.roll(a, i - 1, 0), j - 1, 1) * k[i, j]
                for i in range(3) for j in range(3))
        a = np.clip(a + (a - b) * 1.2, 0, 1)
    r = a.argsort(axis=None).argsort().reshape(a.shape).astype(F32) / (n * n - 1)
    return (r * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------- deck coverage ----

def line_mask(n: int, extent_m: float, spacing_m: float, angle_rad: float,
              seed: int) -> np.ndarray:
    """0..1, 1 on parallel lines `spacing_m` apart running along `angle_rad`
    (radians from east towards north).  The map's rows run north and its
    columns east; the lines' normal is rounded to whole wave numbers so the
    tile stays seamless, and a slow wobble keeps them from looking ruled."""
    k = extent_m / max(spacing_m, 1.0)
    ne = int(round(-k * math.sin(angle_rad)))     # east component of the normal
    nn = int(round(k * math.cos(angle_rad)))      # north component
    if ne == 0 and nn == 0:
        nn = 1
    t = (np.arange(n, dtype=F32) + 0.5) / n
    s = nn * t[:, None] + ne * t[None, :] + 0.22 * perlin2(n, 3, 3, seed)
    d = np.abs(s - np.round(s))                   # distance to a line, in spacings
    return np.exp(-(d / 0.17) ** 2).astype(F32)


def castellanus_turrets(n: int, cells: int, frac: np.ndarray, lines: np.ndarray,
                        base_top: float, seed: int):
    """Turrets standing on a common layer, for Altocumulus / Stratocumulus /
    Cirrocumulus castellanus (Atlas: cumuliform turrets standing on a shared
    horizontal base, often lined up in rows).

    One candidate turret per element (a jittered grid of `cells` x `cells`);
    a candidate becomes a turret when it stands on a line and on the layer,
    and three in four of those do.  Each turret is a dome of its own radius
    (0.30-0.46 of an element) and height (45-100% of the room above the
    layer).  Returns (tops, footprint): the top of the cloud at each point as
    a fraction of the deck's depth (base_top on the layer), and 1 where a
    turret stands."""
    dome = castellanus_domes(n, cells, frac, lines, seed)
    tops = base_top + (1.0 - base_top) * dome
    return tops.astype(F32), (dome > 0.0).astype(F32)


def castellanus_domes(n: int, cells: int, frac: np.ndarray, lines: np.ndarray,
                      seed: int) -> np.ndarray:
    """The turrets of castellanus_turrets as one field: each turret's height
    times its dome profile, as a fraction of the room above the common layer
    (0 where no turret stands)."""
    rng = np.random.default_rng(seed)
    jit = (0.5 + (rng.random((cells, cells, 2)) - 0.5) * 0.8).astype(F32)
    draw = rng.random((cells, cells))
    height = rng.uniform(0.45, 1.0, (cells, cells)).astype(F32)
    radius = rng.uniform(0.30, 0.46, (cells, cells)).astype(F32)
    ci = np.arange(cells)
    pr = (((ci[:, None] + jit[..., 0]) / cells) * n).astype(np.int64) % n
    pc = (((ci[None, :] + jit[..., 1]) / cells) * n).astype(np.int64) % n
    keep = (lines[pr, pc] > 0.45) & (frac[pr, pc] > 0.5) & (draw < 0.75)

    a = (np.arange(n, dtype=F32) + 0.5) / n * cells
    ia = np.floor(a).astype(np.int64)
    fa = (a - ia).astype(F32)
    ia %= cells
    best = np.full((n, n), 9.0, F32)
    bi = np.zeros((n, n), np.int64)
    bj = np.zeros((n, n), np.int64)
    for dx in (-1, 0, 1):
        nx = (ia + dx) % cells
        for dy in (-1, 0, 1):
            ny = (ia + dy) % cells
            j = jit[np.ix_(nx, ny)]
            ex = (dx + j[..., 0]) - fa[:, None]
            ey = (dy + j[..., 1]) - fa[None, :]
            d = np.sqrt(ex * ex + ey * ey)
            better = d < best
            best = np.where(better, d, best)
            bi = np.where(better, nx[:, None], bi)
            bj = np.where(better, ny[None, :], bj)
    r = radius[bi, bj]
    inside = keep[bi, bj] & (best < r)
    dome = np.sqrt(np.clip(1.0 - (best / r) ** 2, 0.0, 1.0))
    return np.where(inside, height[bi, bj] * dome, 0.0).astype(F32)


def _threshold_for_coverage(field: np.ndarray, coverage: float) -> float:
    """Pick the threshold that makes exactly `coverage` of the map cloudy.
    This is what makes the rendered sky cover the number of oktas asked for."""
    coverage = float(np.clip(coverage, 0.0, 1.0))
    if coverage <= 0.0:
        return float(field.max()) + 1.0
    if coverage >= 1.0:
        return float(field.min()) - 1.0
    return float(np.quantile(field, 1.0 - coverage))


#: The share of the area that the net of gaps takes when fully developed:
#: perlucidus's clear, sometimes very small gaps between the elements, and
#: lacunosus's round holes spread fairly evenly (as the WMO Atlas defines
#: them; the shares are this program's choice).
GAP_SHARE = {"perlucidus": 0.12, "lacunosus": 0.35}
#: width of the soft edge of the gaps, in units of the gap field's deviation
GAP_SOFT = 0.18
#: width of a deck's soft edge in units of the pattern's deviation, per unit
#: of Deck.edge_softness (the deployed maps used 3 x edge_softness x the raw
#: field's deviation, the same thing for a near-Gaussian field)
SOFT_PER_EDGE = 3.0
#: a cumulative normal ramp of deviation s rises about as steeply at its middle
#: as a linear ramp of width s * sqrt(2 pi)
RAMP_TO_SIGMA = 1.0 / math.sqrt(2.0 * math.pi)


def gaussianize(field: np.ndarray, seed: int = 0) -> np.ndarray:
    """The histogram transformation of Heitz & Neyret (2018): each value
    replaced by the standard-normal quantile of its rank, so the field keeps
    its shapes but has exactly a standard normal distribution.  Ties (flat
    stretches) are broken at random, not in memory order."""
    from scipy.special import ndtri
    f = np.asarray(field, "f8").ravel()
    span = float(np.ptp(f)) or 1.0
    rng = np.random.default_rng(seed + 424243)
    f = f + rng.random(f.size) * (1e-9 * span)
    order = np.argsort(f, kind="stable")
    ranks = np.empty(f.size, np.int64)
    ranks[order] = np.arange(f.size)
    z = ndtri((ranks + 0.5) / f.size)
    return z.reshape(np.shape(field)).astype(F32)


def norm_cdf(x):
    """Standard normal cumulative distribution (the shader's normCdf)."""
    from scipy.special import ndtr
    return ndtr(x)


def norm_ppf(p: float) -> float:
    """Standard normal quantile."""
    from scipy.special import ndtri
    return float(ndtri(min(max(float(p), 1e-12), 1.0 - 1e-12)))


def zstar_for_cover(cover: float) -> float:
    """The level of a standard-normal pattern above which `cover` of it lies:
    the deck is cloud where its pattern exceeds this (a statistical cloud
    scheme's rule: cloud fraction = the part of the sub-grid distribution
    above saturation)."""
    c = float(np.clip(cover, 0.0, 1.0))
    if c <= 0.0:
        return 40.0
    if c >= 1.0:
        return -40.0
    return norm_ppf(1.0 - c)


#: undulatus: the amplitude of a layer's waves in its pattern's units, per
#: unit of Deck.undulatus.  The waves are one field of the layer that moves
#: with it (timeline: DeckState.wave), added to its evolving pattern -
#: z + a sin(k.x + phase) - not a part of each realization: realizations
#: that all carried the same waves were correlated (0.7-0.8), their blend no
#: longer standard normal, and an undulatus layer was drawn with the wrong
#: cover (a 16 % layer as 26 %).
UND_Z = 1.6
_WAVE_PHI = (np.arange(48) + 0.5) / 48.0 * 2.0 * math.pi
_WAVE_SIN = np.sin(_WAVE_PHI)


def zstar_with_wave(cover: float, a: float) -> float:
    """The level T over which `cover` of z + a sin(phi) lies, z standard
    normal and phi uniform - the cover of a pattern with waves of amplitude a
    running through it, over whole wavelengths (a = 0: zstar_for_cover).
    Newton's method kept inside a bracket that it narrows, bisecting when a
    step would leave it: the mixture's tails are flat, and an unguarded step
    from a tiny cover overshot to 'all cloud'."""
    if a <= 1e-9:
        return zstar_for_cover(cover)
    c = float(np.clip(cover, 0.0, 1.0))
    if c <= 0.0:
        return 40.0
    if c >= 1.0:
        return -40.0
    from scipy.special import ndtr
    sw = a * _WAVE_SIN
    lo, hi = -40.0, 40.0                  # f(lo) > 0 > f(hi)
    t = min(max(zstar_for_cover(c) * math.sqrt(1.0 + 0.5 * a * a), lo), hi)
    for _ in range(100):
        u = t - sw
        f = float(np.mean(ndtr(-u))) - c
        if f > 0.0:
            lo = t
        else:
            hi = t
        fp = -float(np.mean(np.exp(-0.5 * u * u))) / math.sqrt(2.0 * math.pi)
        nt = t - f / fp if fp < -1e-300 else 0.5 * (lo + hi)
        if not (lo < nt < hi):
            nt = 0.5 * (lo + hi)
        if abs(nt - t) < 1e-12 or hi - lo < 1e-12:
            t = nt
            break
        t = nt
    return t


def deck_fields(n: int,
                extent_m: float,
                element_m: float,
                cellularity: float = 0.5,
                anisotropy: float = 1.0,
                angle_rad: float = 0.0,
                undulatus: float = 0.0,
                undulatus_ratio: float = 6.0,
                fibrosity: float = 0.0,
                warp: float = 0.045,
                turret: float = 0.0,
                turret_cover: float = 0.5,
                gaps: str = "",
                blur_px: float = 0.0,
                seed: int = 1,
                lumpy: float = 0.0) -> dict:
    """One realization of a deck's pattern - everything about its look that
    does not depend on how much of the sky it covers.

    lumpy (0..1): a layer's elements as lumpy masses - each cell stirred by a
    flow of its own size and roughened by lumps half its size, so the
    elements keep their regular arrangement but lose their round outlines
    (round cells read as coins).

      z     the pattern, histogram-transformed to exactly standard normal:
            the deck is cloud where z exceeds zstar_for_cover(cover), so the
            cover can change continuously over one fixed pattern (the
            elements grow and new ones appear where the pattern is highest
            first; they shrink and vanish lowest first)
      u     the element-top noise (Perlin, raw): an element's top is
            1 - top_variation * clip(1.6 u + 0.5) * (1.15 - 0.5 frac)
      dome  castellanus: each turret's height times its dome profile, as a
            fraction of the room above the common layer (0 elsewhere);
            turrets stand where the layer is cloud at turret_cover
      gap   a standard-normal field that is lowest in the interstices -
            perlucidus: on the walls between the elements; lacunosus: at the
            centres of the net's holes (zeros when gaps is '')

    Two realizations with different seeds are independent, so a pattern
    can be made to evolve by blending them (Pattern / evolve)."""
    element_m = max(element_m, extent_m / (n / 2.0))
    cells = max(2, int(round(extent_m / element_m)))
    cells_long = max(2, int(round(cells / max(anisotropy, 1e-3))))
    cells = max(2, cells)

    # cellular part: regularly arranged elements
    f1, f2 = worley2(n, cells_long, cells, seed, jitter=1.0 - 0.35 * cellularity)
    cellular = np.clip(1.0 - f1 / 1.15, 0.0, 1.0)

    # turbulent part: irregular masses
    fb = np.zeros((n, n), F32)
    amp, tot = 1.0, 0.0
    for o in range(4):
        fx = max(2, int(cells_long * (2 ** o) / 2))
        fy = max(2, int(cells * (2 ** o) / 2))
        fb += amp * perlin2(n, fx, fy, seed + 31 * o + 7)
        tot += amp
        amp *= 0.5
    fb = fb / tot * 0.5 + 0.5

    field = fb * (1.0 - cellularity) + cellular * cellularity

    # fibratus / uncinus: long filaments carved by strong 1-D anisotropy
    if fibrosity > 0.0:
        # Ridged noise stretched along the shear makes filaments.  Two things
        # keep them from reading as a drawn starburst: the ridges are only a
        # few cells long rather than spanning the tile, and the patchy fBm is
        # never fully replaced, so the filaments live inside patches the way
        # real Cirrus does.
        fil = perlin2(n, max(3, cells_long), max(6, cells * 4), seed + 991)
        fil = np.clip(1.0 - np.abs(fil) * 2.2, 0.0, 1.0)
        fil = fil * 0.62 + fb * 0.38
        field = field * (1.0 - 0.70 * fibrosity) + fil * (0.70 * fibrosity)

    # undulatus: a regular wave modulation across the deck
    if undulatus > 0.0:
        lam = element_m * undulatus_ratio
        k = 2.0 * np.pi * extent_m / max(lam, 1.0)
        k = max(1.0, round(k / (2 * np.pi))) * 2 * np.pi     # keep it tileable
        u = (np.arange(n, dtype=F32) + 0.5) / n
        wob = 0.12 * perlin2(n, 3, 3, seed + 55)
        wave = np.sin(k * (u[:, None] * math.cos(angle_rad)
                           + u[None, :] * math.sin(angle_rad)) + wob * 6.0)
        field = field * (1.0 - undulatus) + (field * (0.55 + 0.45 * wave)) * undulatus

    # the interstices of perlucidus: the walls between the cells, where the
    # distances to the two nearest cell centres are equal (F2 - F1 = 0)
    gapraw = None
    if gaps == "perlucidus":
        gapraw = (f2 - f1).astype(F32)
    elif gaps == "lacunosus":
        # a net: round holes centred on a coarser lattice's points
        h1, _ = worley2(n, max(2, cells_long // 2), max(2, cells // 2), seed + 777, 0.35)
        gapraw = h1.astype(F32)

    # Domain warp: displace the whole field by a low-frequency flow.  Real
    # cloud fields are stirred by the wind, so perfectly straight rows of
    # elements or perfectly parallel filaments look drawn rather than seen.
    # The gaps are displaced with the elements they separate.
    if lumpy > 0.0:
        # lumps about half an element across, then a stirring of each
        # element's own size (both tileable: whole periods across the tile)
        lump = perlin2(n, max(2, cells_long * 2), max(2, cells * 2), seed + 607)
        field = field * (1.0 - 0.22 * lumpy) + (lump * 0.5 + 0.5) * (0.22 * lumpy)
        from scipy import ndimage
        a = lumpy * 0.38 * n / max(cells, 2)
        wx = perlin2(n, max(2, cells_long), max(2, cells), seed + 611) * a
        wy = perlin2(n, max(2, cells_long), max(2, cells), seed + 612) * a
        yy, xx = np.meshgrid(np.arange(n, dtype=F32), np.arange(n, dtype=F32), indexing="ij")
        coords = [(yy + wy) % n, (xx + wx) % n]
        field = ndimage.map_coordinates(field, coords, order=1, mode="grid-wrap").astype(F32)
        if gapraw is not None:
            gapraw = ndimage.map_coordinates(gapraw, coords, order=1, mode="grid-wrap").astype(F32)

    if warp > 0.0:
        from scipy import ndimage
        # filaments need more stirring than blobs, or they stay dead straight
        amp = n * warp * (1.0 + 1.6 * fibrosity)
        wx = perlin2(n, 3, 3, seed + 501) * amp
        wy = perlin2(n, 3, 3, seed + 502) * amp
        yy, xx = np.meshgrid(np.arange(n, dtype=F32), np.arange(n, dtype=F32),
                             indexing="ij")
        coords = [(yy + wy) % n, (xx + wx) % n]
        field = ndimage.map_coordinates(field, coords, order=1, mode="grid-wrap").astype(F32)
        if gapraw is not None:
            gapraw = ndimage.map_coordinates(gapraw, coords, order=1,
                                             mode="grid-wrap").astype(F32)

    lines = None
    if turret > 0.0:
        # castellanus: the layer's elements gather on lines four elements
        # apart, along the shear (gather, not fuse: they stay elements)
        lines = line_mask(n, extent_m, 4.0 * element_m, angle_rad, seed + 61)
        field = field * 0.78 + lines * 0.22

    if blur_px > 0.0:
        # Fibres one or two texels wide beat against each other and against
        # the pixel grid into a woven moire; real fibres are smooth.
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(field.astype("f8"), blur_px, mode="wrap").astype(F32)

    z = gaussianize(field, seed)
    # How high each element grows, as a fraction of the deck depth: a field
    # at the scale of the elements, not a per-pixel ramp (one turret has one
    # top).
    u = perlin2(n, max(2, cells_long // 2), max(2, cells // 2), seed + 877).astype(F32)
    dome = np.zeros((n, n), F32)
    if lines is not None:
        frac0 = (z > zstar_for_cover(turret_cover)).astype(F32)
        dome = castellanus_domes(n, cells, frac0, lines, seed + 919)
    gap = gaussianize(gapraw, seed + 5) if gapraw is not None else np.zeros((n, n), F32)
    return dict(z=z, u=u, dome=dome, gap=gap, gaps=gaps, cells=cells)


def compose(z, zstar: float, soft_z: float, var=None, u=None, top_variation: float = 0.45,
            gap=None, gap_amount: float = 0.0, gap_star: float = 0.0,
            dome=None, growth: float = 1.0, turret: float = 0.0):
    """Cloud fraction and element tops from a pattern - the renderer's own
    rule (shaders.deckDensity), for the census and the self-test.

    frac = Phi((z - zstar) / sqrt(var + s^2)), s = soft_z / sqrt(2 pi): the
    fraction of a footprint that is cloud when the pattern varies within it
    by var (0 for a single texel); area with frac > 0.5 is exactly the part
    of the pattern above zstar.  Gaps take gap_amount of the cloud out
    where the gap field is below gap_star.  Castellanus: the turrets
    (dome x growth x the layer's own cloud fraction there) stand on cloud
    and set the tops; elsewhere the tops follow u.  Returns (frac, tops) -
    tops as a fraction of the deck depth (absolute for castellanus, the
    element scale otherwise)."""
    z = np.asarray(z, "f8")
    s = max(float(soft_z), 1e-4) * RAMP_TO_SIGMA
    v = 0.0 if var is None else np.asarray(var, "f8")
    frac = norm_cdf((z - zstar) / np.sqrt(v + s * s))
    base = frac
    if gap is not None and gap_amount > 0.0:
        inside = norm_cdf((np.asarray(gap, "f8") - gap_star) / GAP_SOFT)
        frac = frac * (1.0 - gap_amount * (1.0 - inside))
    if turret > 0.0 and dome is not None:
        # a turret stands on its layer's cloud: where the layer is cloud now
        # (the turrets thin and sink with their layer and go when it goes)
        g = np.clip((base - 0.5) / 0.3, 0.0, 1.0)
        d = np.clip(growth, 0.0, 1.0) * np.asarray(dome, "f8") * (g * g * (3.0 - 2.0 * g))
        frac = np.maximum(frac, np.clip(d * 400.0, 0.0, 1.0))
        tops = turret + (1.0 - turret) * d
    else:
        tv = float(np.clip(top_variation, 0.0, 1.0))
        uu = np.clip((np.zeros_like(z) if u is None else np.asarray(u, "f8")) * 1.6 + 0.5, 0.0, 1.0)
        tops = np.clip(1.0 - tv * uu * (1.15 - 0.5 * frac), 0.22, 1.0)
    return frac.astype(F32), tops.astype(F32)


def gap_star(kind: str) -> float:
    """The level of the gap field below which a fully developed net of gaps
    lies (GAP_SHARE of the area)."""
    return norm_ppf(GAP_SHARE.get(kind, 0.0)) if kind in GAP_SHARE else -40.0


def deck_map(n: int,
             extent_m: float,
             element_m: float,
             coverage: float,
             cellularity: float = 0.5,
             anisotropy: float = 1.0,
             angle_rad: float = 0.0,
             undulatus: float = 0.0,
             undulatus_ratio: float = 6.0,
             lacunosus: float = 0.0,
             perlucidus: float = 0.0,
             edge_softness: float = 0.25,
             fibrosity: float = 0.0,
             top_variation: float = 0.45,
             warp: float = 0.045,
             turret: float = 0.0,
             seed: int = 1,
             blur_px: float = 0.0,
             lumpy: float = 0.0) -> np.ndarray:
    """One deck's control map at one moment: a single realization of its
    pattern (deck_fields) put through the renderer's rule (compose).

    Returns float32 (n, n, 3):
      [...,0]  cloud fraction 0..1        (density multiplier)
      [...,1]  element top scale 0..1     (how tall this element grows); with
               turret > 0 the top itself, as a fraction of the deck's depth
      [...,2]  age / erosion 0..1         (not drawn; kept for the layout)

    turret     0, or castellanus: the top of the common layer as a fraction of
               the deck's depth.  The layer's elements gather on lines along
               `angle_rad` and turrets stand on them (castellanus_turrets).

    extent_m   physical size of the tile in metres
    element_m  spacing of the individual macroscopic elements in metres.
               This is the Atlas's apparent-width criterion, converted to
               metres by atlas.angular_width_to_metres().
    anisotropy element length / element width, >1 stretches along `angle_rad`
               (radiatus, fibratus, wind shear).
    """
    gaps = "lacunosus" if lacunosus > 0.0 else ("perlucidus" if perlucidus > 0.0 else "")
    tb = float(np.clip(turret, 0.1, 0.9)) if turret > 0.0 else 0.0
    f = deck_fields(n, extent_m, element_m, cellularity=cellularity, anisotropy=anisotropy,
                    angle_rad=angle_rad, undulatus=undulatus, undulatus_ratio=undulatus_ratio,
                    fibrosity=fibrosity, warp=warp, turret=tb, turret_cover=coverage,
                    gaps=gaps, blur_px=blur_px, seed=seed, lumpy=lumpy)
    # the cover is exact on the tile: the threshold is the pattern's own
    # quantile (for a standard normal pattern it is zstar_for_cover(coverage))
    zs = _threshold_for_coverage(f["z"], coverage)
    amount = lacunosus if gaps == "lacunosus" else perlucidus
    frac, tops = compose(f["z"], zs, SOFT_PER_EDGE * edge_softness, u=f["u"],
                         top_variation=top_variation, gap=f["gap"], gap_amount=amount,
                         gap_star=gap_star(gaps), dome=f["dome"], growth=1.0, turret=tb)
    cells = f["cells"]
    age = np.clip(0.5 + 0.5 * perlin2(n, max(2, cells // 2), max(2, cells // 2), seed + 313),
                  0.0, 1.0)
    return np.stack([frac, tops, age], axis=-1).astype(F32)


def measured_element_spacing_m(cov: np.ndarray, extent_m: float) -> float:
    """Dominant horizontal spacing of the elements in a control map, in metres.

    Measured from the radially-averaged power spectrum, so it is an
    independent check on what deck_map() was asked to produce.
    """
    a = cov - cov.mean()
    p = np.abs(np.fft.rfft2(a)) ** 2
    n = cov.shape[0]
    fy = np.fft.fftfreq(n) * n
    fx = np.fft.rfftfreq(n) * n
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    rb = r.astype(np.int32)
    nb = rb.max() + 1
    power = np.bincount(rb.ravel(), p.ravel(), minlength=nb)
    count = np.bincount(rb.ravel(), minlength=nb)
    power = power / np.maximum(count, 1)
    power[0] = 0.0
    k = int(np.argmax(power))
    if k == 0:
        return float("inf")
    return extent_m / k


# ------------------------------------------------------------- disk cache ---

def cache_dir() -> str:
    base = (os.environ.get("LOCALAPPDATA")
            or os.environ.get("XDG_CACHE_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache"))
    d = os.path.join(base, "cloudsim")
    os.makedirs(d, exist_ok=True)
    return d


def cached(name: str, builder, *args, **kwargs) -> np.ndarray:
    """Build once, then reuse from the user cache directory.

    Only ever writes inside the OS cache directory - never next to the
    program and never into any of the user's own folders.
    """
    key = hashlib.sha1(f"{name}|{args}|{sorted(kwargs.items())}".encode()).hexdigest()[:16]
    path = os.path.join(cache_dir(), f"{name}-{key}.npy")
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass
    arr = builder(*args, **kwargs)
    try:
        np.save(path, arr)
    except Exception:
        pass
    return arr
