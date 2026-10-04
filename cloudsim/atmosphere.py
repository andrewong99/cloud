# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
Atmospheric scattering look-up tables, built on the CPU with numpy.

Model: Rayleigh + Mie + ozone absorption on a spherical Earth, the standard
set of coefficients used since Bruneton & Neyret (2008) and refined by
Hillaire (2020).  Two tables are produced:

  transmittance  (mu, altitude)  -> RGB fraction of light surviving the path
                                    from that point to the top of the
                                    atmosphere in that direction.
  multi-scatter  (mu_sun, alt)   -> RGB isotropic multiple-scattering term.

The shader then does single scattering by ray marching and adds the
multiple-scattering term, which is what gives a believable twilight and a
correctly bright horizon rather than a flat blue gradient.

All lengths here are kilometres.
"""

from __future__ import annotations

import numpy as np

F32 = np.float32

RG = 6360.0          # km, ground radius
RT = 6460.0          # km, top of atmosphere

# per-kilometre scattering / absorption at the RGB primaries
RAYLEIGH_S = np.array([5.802, 13.558, 33.100], F32) * 1e-3
RAYLEIGH_H = 8.0
MIE_S = np.array([3.996, 3.996, 3.996], F32) * 1e-3
MIE_E = np.array([4.400, 4.400, 4.400], F32) * 1e-3
MIE_H = 1.2
MIE_G = 0.80
OZONE_A = np.array([0.650, 1.881, 0.085], F32) * 1e-3
OZONE_CENTRE = 25.0
OZONE_WIDTH = 30.0
GROUND_ALBEDO = 0.10


def densities(h_km):
    """Rayleigh, Mie and ozone density fractions at altitude h (km)."""
    h = np.maximum(h_km, 0.0)
    r = np.exp(-h / RAYLEIGH_H)
    m = np.exp(-h / MIE_H)
    o = np.maximum(0.0, 1.0 - np.abs(h - OZONE_CENTRE) / (OZONE_WIDTH * 0.5))
    return r, m, o


def extinction_rgb(h_km):
    """Total extinction per km at altitude h; result shape is (..., 3)."""
    r, m, o = densities(h_km)
    return (RAYLEIGH_S * r[..., None] + MIE_E * m[..., None] + OZONE_A * o[..., None])


def _ray_sphere_outer(r, mu):
    """Distance to the top of the atmosphere from radius r, cosine mu."""
    disc = r * r * (mu * mu - 1.0) + RT * RT
    return np.maximum(0.0, -r * mu + np.sqrt(np.maximum(disc, 0.0)))


def _hits_ground(r, mu):
    disc = r * r * (mu * mu - 1.0) + RG * RG
    return (mu < 0.0) & (disc >= 0.0)


def transmittance_lut(width: int = 256, height: int = 64, steps: int = 48) -> np.ndarray:
    """(height, width, 3) float32.  x = 0.5*(mu+1), y = (r-RG)/(RT-RG)."""
    mu = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    alt = (np.arange(height, dtype=np.float64) + 0.5) / height
    r = RG + alt * (RT - RG)

    R, MU = np.meshgrid(r, mu, indexing="ij")
    dmax = _ray_sphere_outer(R, MU)
    ground = _hits_ground(R, MU)

    t = (np.arange(steps, dtype=np.float64) + 0.5) / steps
    ds = dmax[..., None] / steps
    d = dmax[..., None] * t
    rr = np.sqrt(np.maximum(R[..., None] ** 2 + 2.0 * R[..., None] * MU[..., None] * d
                            + d * d, 1.0))
    h = rr - RG
    ext = extinction_rgb(h)                      # (H, W, steps, 3)
    tau = (ext * ds[..., None]).sum(axis=-2)     # (H, W, 3)
    tr = np.exp(-tau)
    tr[ground] = 0.0
    return np.ascontiguousarray(tr.astype(F32))


def _sample_transmittance(lut: np.ndarray, r, mu):
    """Bilinear sample of the transmittance table (numpy side)."""
    h, w = lut.shape[:2]
    x = np.clip((mu + 1.0) * 0.5 * w - 0.5, 0, w - 1)
    y = np.clip((r - RG) / (RT - RG) * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    return ((lut[y0, x0] * (1 - fx) + lut[y0, x1] * fx) * (1 - fy)
            + (lut[y1, x0] * (1 - fx) + lut[y1, x1] * fx) * fy)


def multiscatter_lut(tr_lut: np.ndarray, size: int = 32,
                     dirs: int = 64, steps: int = 24) -> np.ndarray:
    """(size, size, 3) float32.  x = 0.5*(mu_sun+1), y = altitude fraction.

    Hillaire's isotropic approximation: second-order in-scattering divided by
    (1 - f_ms) to account for all higher orders.
    """
    mus = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    alt = (np.arange(size, dtype=np.float64) + 0.5) / size
    rr0 = RG + alt * (RT - RG) * 0.999

    # uniform directions on the sphere (Fibonacci)
    i = np.arange(dirs, dtype=np.float64) + 0.5
    cz = 1.0 - 2.0 * i / dirs
    sz = np.sqrt(np.maximum(0.0, 1.0 - cz * cz))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    dvec = np.stack([sz * np.cos(phi), sz * np.sin(phi), cz], axis=-1)   # z = up

    out = np.zeros((size, size, 3), F32)
    t = (np.arange(steps, dtype=np.float64) + 0.5) / steps
    for iy, r0 in enumerate(rr0):
        for ix, mu_s in enumerate(mus):
            sun = np.array([np.sqrt(max(0.0, 1 - mu_s * mu_s)), 0.0, mu_s])
            # ray start at (0,0,r0); direction dvec
            mu = dvec[:, 2]
            dmax = _ray_sphere_outer(np.full(dirs, r0), mu)
            ground = _hits_ground(np.full(dirs, r0), mu)
            dmax = np.where(ground,
                            np.maximum(0.0, -r0 * mu - np.sqrt(np.maximum(
                                r0 * r0 * (mu * mu - 1.0) + RG * RG, 0.0))),
                            dmax)
            ds = dmax / steps
            d = dmax[:, None] * t[None, :]
            pos = np.stack([dvec[:, 0][:, None] * d,
                            dvec[:, 1][:, None] * d,
                            r0 + dvec[:, 2][:, None] * d], axis=-1)      # (dirs,steps,3)
            rr = np.linalg.norm(pos, axis=-1)
            h = rr - RG
            rd, md, od = densities(h)
            scat = RAYLEIGH_S * rd[..., None] + MIE_S * md[..., None]
            ext = extinction_rgb(h)
            # transmittance from the start point to each sample
            tau = np.cumsum(ext * ds[:, None, None], axis=1) - 0.5 * ext * ds[:, None, None]
            tr_view = np.exp(-tau)
            # sun transmittance and shadow
            mu_sun = (pos @ sun) / np.maximum(rr, 1e-6)
            tr_sun = _sample_transmittance(tr_lut, rr, mu_sun)
            shadow = (~_hits_ground(rr, mu_sun)).astype(np.float64)[..., None]

            iso = 1.0 / (4.0 * np.pi)
            l2 = (tr_view * scat * tr_sun * shadow * iso * ds[:, None, None]).sum(axis=1)
            # sunlight reflected by the ground where the ray ends on it
            # (Lambertian, GROUND_ALBEDO), as in Hillaire's LUT
            pg = pos[:, -1, :] + dvec * (0.5 * ds)[:, None]
            rg = np.linalg.norm(pg, axis=-1)
            mu_g = (pg @ sun) / np.maximum(rg, 1e-6)
            tr_g = _sample_transmittance(tr_lut, np.full(dirs, RG + 1e-3), mu_g)
            gnd = (tr_view[:, -1, :] * tr_g * np.maximum(mu_g, 0.0)[:, None]
                   * GROUND_ALBEDO / np.pi) * ground[:, None]
            l2 = l2 + gnd
            fms = (tr_view * scat * ds[:, None, None]).sum(axis=1)
            # Hillaire (2020) eqs. 5 and 7: both gathered over the sphere with
            # the isotropic phase function, i.e. the plain mean over
            # directions (4 pi / N per direction times 1 / (4 pi))
            l2 = l2.mean(axis=0)
            fms = fms.mean(axis=0)
            out[iy, ix] = (l2 / np.maximum(1.0 - fms, 1e-3)).astype(F32)
    return out


def build_luts(tr_size=(256, 64), ms_size=32):
    tr = transmittance_lut(tr_size[0], tr_size[1])
    ms = multiscatter_lut(tr, ms_size)
    return tr, ms


def sun_colour_at_ground(tr_lut: np.ndarray, sun_alt_deg: float,
                         observer_km: float = 0.0) -> np.ndarray:
    """Direct solar spectrum reaching the observer, as an RGB fraction."""
    mu = np.sin(np.radians(sun_alt_deg))
    r = RG + max(observer_km, 0.0)
    return _sample_transmittance(tr_lut, np.array([r]), np.array([mu]))[0]


def sky_radiance(tr_lut, ms_lut, dirs, sun_dir, alt_m=0.0, steps=20):
    """Sky radiance (relative to a solar irradiance of 1) for a set of
    directions, using the same single-scatter + multiple-scatter model the
    shader uses.  `dirs` is (N,3) with y up; `sun_dir` is (3,)."""
    dirs = np.asarray(dirs, np.float64)
    sun = np.asarray(sun_dir, np.float64)
    r0 = RG + alt_m / 1000.0
    ro = np.array([0.0, r0, 0.0])
    mu = dirs[:, 1]
    dmax = _ray_sphere_outer(np.full(len(dirs), r0), mu)
    hitg = _hits_ground(np.full(len(dirs), r0), mu)
    disc = r0 * r0 * (mu * mu - 1.0) + RG * RG
    dground = np.where(hitg, -r0 * mu - np.sqrt(np.maximum(disc, 0.0)), dmax)
    dmax = np.where(hitg, np.maximum(dground, 0.0), dmax)

    t = (np.arange(steps) + 0.5) / steps
    ds = dmax / steps
    d = dmax[:, None] * t[None, :]
    pos = ro[None, None, :] + dirs[:, None, :] * d[..., None]
    rr = np.linalg.norm(pos, axis=-1)
    h = rr - RG
    rd_, md_, od_ = densities(h)
    ext = extinction_rgb(h)
    tau = np.cumsum(ext * ds[:, None, None], axis=1) - 0.5 * ext * ds[:, None, None]
    tview = np.exp(-tau)
    mus = (pos @ sun) / np.maximum(rr, 1e-9)
    trsun = _sample_transmittance(tr_lut, rr, mus)
    shade = (~_hits_ground(rr, mus))[..., None]
    cost = dirs @ sun
    pr = (3.0 / (16.0 * np.pi) * (1.0 + cost * cost))[:, None, None]
    g = MIE_G
    pm = ((1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * cost) ** 1.5))[:, None, None]
    single = (RAYLEIGH_S * rd_[..., None] * pr + MIE_S * md_[..., None] * pm) * trsun * shade
    scat = RAYLEIGH_S * rd_[..., None] + MIE_S * md_[..., None]
    ms = _sample_transmittance(ms_lut, rr, mus) if ms_lut.shape[:2] == tr_lut.shape[:2] \
        else _sample_ms(ms_lut, rr, mus)
    return ((single + scat * ms) * tview * ds[:, None, None]).sum(axis=1)


def _sample_ms(ms_lut, r, mu):
    h, w = ms_lut.shape[:2]
    x = np.clip((mu + 1.0) * 0.5 * w - 0.5, 0, w - 1)
    y = np.clip((r - RG) / (RT - RG) * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    return ((ms_lut[y0, x0] * (1 - fx) + ms_lut[y0, x1] * fx) * (1 - fy)
            + (ms_lut[y1, x0] * (1 - fx) + ms_lut[y1, x1] * fx) * fy)


def sky_irradiance(tr_lut, ms_lut, sun_alt_deg, alt_m=0.0, n=96):
    """Downwelling sky irradiance on a horizontal surface, relative to a solar
    irradiance of 1.  Cosine-weighted hemisphere sampling, so the integral is
    simply pi times the mean radiance."""
    i = np.arange(n) + 0.5
    # cosine-weighted hemisphere (Fibonacci)
    r = np.sqrt(i / n)
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    x, z = r * np.cos(phi), r * np.sin(phi)
    y = np.sqrt(np.maximum(0.0, 1.0 - x * x - z * z))
    dirs = np.stack([x, y, z], axis=-1)
    sa = np.radians(sun_alt_deg)
    sun = np.array([np.cos(sa), np.sin(sa), 0.0])
    L = sky_radiance(tr_lut, ms_lut, dirs, sun, alt_m)
    return (L.mean(axis=0) * np.pi).astype(F32)
