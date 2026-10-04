# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
Mie scattering by water drops, and the rain phase function that paints
rainbows.

A rainbow is not drawn here; it falls out of the physics.  Rain is a cloud of
spherical water drops.  Mie theory gives how much light a drop of radius r
scatters into each angle at each wavelength; averaging over a Marshall-Palmer
spectrum of drop sizes and over the visible spectrum (weighted by sunlight and
the CIE 1931 colour-matching functions) gives an RGB phase function.  Its peak
near 138 degrees of scattering is the primary bow, 42 degrees from the
antisolar point; the peak near 129 degrees is the secondary bow; the dim gap
between is Alexander's dark band; the bright spike at 180 degrees is the
glory.  The renderer multiplies this function into the single scattering of
the rain shafts, so a bow appears exactly where sunlight falls on rain behind
the observer, with the colours in the right order.

Sources:
  * Bohren & Huffman (1983), appendix A: BHMIE, the series used below.
  * Water refractive index: Daimon & Masumura (2007), 20 C, four-term
    Sellmeier fit.
  * CIE 1931 2-degree colour matching functions: the multi-lobe Gaussian fit
    of Wyman, Sloan & Shirley (2013).
  * Drop sizes: Marshall & Palmer (1948), N(D) = N0 exp(-L D),
    L = 4.1 R^-0.21 mm^-1.
  * Sun: a 5778 K black body; its disc (0.533 degrees) is convolved in, which
    is what smooths the finest interference ripples away.

The same machinery, for nearly equal ice spheres of 0.6-3 um, gives the
iridescence of nacreous clouds (data/psc_phase.npy, one row per radius).

The tables are precomputed and shipped in data/; ``python -m cloudsim.mie``
rebuilds them.
"""

from __future__ import annotations

import math
import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data", "rain_phase.npy")
PSC_DATA = os.path.join(os.path.dirname(__file__), "data", "psc_phase.npy")
LUT_SIZE = 4096
# nacreous-cloud table: ice spheres of these mean radii (um), one row each
PSC_RADII = np.linspace(0.6, 3.0, 16)


def water_index(lam_um):
    """Real refractive index of liquid water at 20 C (Daimon & Masumura 2007)."""
    l2 = np.asarray(lam_um, "f8") ** 2
    b = (5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1)
    c = (5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1)
    n2 = 1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c))
    return np.sqrt(n2)


def bhmie(x: float, m: complex, mu: np.ndarray):
    """Amplitude functions S1, S2 at cos(theta) = mu, and Qext, Qsca, g.

    Bohren & Huffman's BHMIE, evaluated directly at every requested angle
    (no half-range symmetry trick), in double precision."""
    mu = np.asarray(mu, "f8")
    nstop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    y = m * x
    nmx = int(max(nstop, abs(y)) + 15)
    d = np.zeros(nmx + 1, complex)
    for n in range(nmx, 0, -1):
        en = n / y
        d[n - 1] = en - 1.0 / (d[n] + en)
    psi0, psi1 = math.cos(x), math.sin(x)
    chi0, chi1 = -math.sin(x), math.cos(x)
    xi1 = complex(psi1, -chi1)
    pi0 = np.zeros_like(mu)
    pi1 = np.ones_like(mu)
    s1 = np.zeros(mu.shape, complex)
    s2 = np.zeros(mu.shape, complex)
    qsca = 0.0
    gsca = 0.0
    an1 = bn1 = 0.0
    for n in range(1, nstop + 1):
        fn = float(n)
        fn2 = (2.0 * fn + 1.0) / (fn * (fn + 1.0))
        psi = (2.0 * fn - 1.0) * psi1 / x - psi0
        chi = (2.0 * fn - 1.0) * chi1 / x - chi0
        xi = complex(psi, -chi)
        dn = d[n]
        an = ((dn / m + fn / x) * psi - psi1) / ((dn / m + fn / x) * xi - xi1)
        bn = ((m * dn + fn / x) * psi - psi1) / ((m * dn + fn / x) * xi - xi1)
        qsca += (2.0 * fn + 1.0) * (abs(an) ** 2 + abs(bn) ** 2)
        gsca += (2.0 * fn + 1.0) / (fn * (fn + 1.0)) * (an * bn.conjugate()).real
        if n > 1:
            gsca += (fn - 1.0) * (fn + 1.0) / fn * (an1 * an.conjugate() + bn1 * bn.conjugate()).real
        pi = pi1
        tau = fn * mu * pi - (fn + 1.0) * pi0
        s1 += fn2 * (an * pi + bn * tau)
        s2 += fn2 * (an * tau + bn * pi)
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = complex(psi1, -chi1)
        an1, bn1 = an, bn
        pi1 = ((2.0 * fn + 1.0) * mu * pi - (fn + 1.0) * pi0) / fn
        pi0 = pi
    qsca *= 2.0 / x ** 2
    gsca *= 4.0 / (x ** 2 * qsca)
    return s1, s2, qsca, gsca


def phase_from_s(s1, s2, x, qsca):
    """Normalised phase function (1/sr): integrates to one over the sphere."""
    return (np.abs(s1) ** 2 + np.abs(s2) ** 2) / (2.0 * math.pi * x ** 2 * qsca)


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_xyz(lam_nm):
    """CIE 1931 colour matching functions, Wyman-Sloan-Shirley 2013 fit."""
    l = np.asarray(lam_nm, "f8")
    x = 1.056 * _g(l, 599.8, 37.9, 31.0) + 0.362 * _g(l, 442.0, 16.0, 26.7) \
        - 0.065 * _g(l, 501.1, 20.4, 26.2)
    y = 0.821 * _g(l, 568.8, 46.9, 40.5) + 0.286 * _g(l, 530.9, 16.3, 31.1)
    z = 1.217 * _g(l, 437.0, 11.8, 36.0) + 0.681 * _g(l, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])


def planck(lam_nm, t=5778.0):
    lam = np.asarray(lam_nm, "f8") * 1e-9
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return 1.0 / (lam ** 5 * (np.exp(h * c / (lam * k * t)) - 1.0))


def angle_grid():
    """Scattering angles (radians): log-spaced through the diffraction peak,
    0.5 degree steps to 100 degrees, 0.02 degree steps over the bows and
    the glory."""
    fwd = np.geomspace(1e-6, math.radians(1.0), 260)
    mid = np.radians(np.arange(1.25, 100.0, 0.25))
    back = np.radians(np.arange(100.0, 180.0 + 1e-9, 0.02))
    return np.concatenate([[0.0], fwd, mid, back])


def lut_theta(n=LUT_SIZE):
    """The LUT's angle for index i: theta = pi * u^2, u = (i + 0.5)/n.  Dense
    near the forward peak, still 0.09 degrees wide per texel at 180."""
    u = (np.arange(n) + 0.5) / n
    return math.pi * u * u


def build_rain_phase(rain_rate_mm_h=5.0, n_lambda=16, n_radius=18, verbose=False):
    lams = np.linspace(400.0, 700.0, n_lambda)
    th = angle_grid()
    mu = np.cos(th)
    lam_mp = 4.1 * rain_rate_mm_h ** -0.21            # mm^-1, in diameter
    r_mm = np.geomspace(0.05, 2.0, n_radius)
    dr = np.gradient(r_mm)
    # number per radius interval (N(D) dD with D = 2r), weight by scattering later
    nr = np.exp(-lam_mp * 2.0 * r_mm) * 2.0 * dr
    p_lam = np.zeros((n_lambda, th.size))
    g_lam = np.zeros(n_lambda)
    for il, lam in enumerate(lams):
        m = complex(float(water_index(lam / 1000.0)), 0.0)
        acc = np.zeros(th.size)
        wsum = 0.0
        gacc = 0.0
        for ir, r in enumerate(r_mm):
            x = 2.0 * math.pi * (r * 1000.0) / (lam / 1000.0)   # r in um / lambda in um
            s1, s2, qsca, g = bhmie(x, m, mu)
            csca = qsca * math.pi * r ** 2
            w = nr[ir] * csca
            acc += w * phase_from_s(s1, s2, x, qsca)
            gacc += w * g
            wsum += w
        p_lam[il] = acc / wsum
        g_lam[il] = gacc / wsum
        if verbose:
            print(f"  lambda {lam:5.1f} nm  n = {m.real:.5f}  g = {g_lam[il]:.4f}", flush=True)
    lut, rgb = _spectral_lut(th, p_lam, lams)
    return lut, {"theta": th, "rgb": rgb, "g": g_lam, "lambda": lams}


def _spectral_lut(th, p_lam, lams):
    """Phase functions per wavelength on the angle grid -> the renderer's RGB
    table at theta = pi u^2: the Sun's disc convolved in, each wavelength
    normalised, the spectrum weighted by sunlight and the CIE matching
    functions into linear sRGB (a grey scatterer gives RGB = 1, 1, 1)."""
    # the Sun's disc: a 1-D chord-length kernel of radius 0.2665 degrees
    p_lam = _convolve_disc(th, p_lam, math.radians(0.2665))
    p_lam = np.stack([_normalise(th, pl) for pl in p_lam])
    w = planck(lams)[:, None] * cie_xyz(lams)                # (n_lambda, 3)
    xyz = np.einsum("lk,lt->tk", w, p_lam)
    rgb = xyz @ XYZ_TO_SRGB.T
    white = (w.sum(axis=0)) @ XYZ_TO_SRGB.T
    rgb = rgb / white[None, :]
    rgb = np.maximum(rgb, 1e-6)
    lut_th = lut_theta()
    lut = np.stack([np.exp(np.interp(lut_th, th, np.log(rgb[:, c]))) for c in range(3)], -1)
    # the table's own quadrature (the one the renderer effectively uses)
    w = 2.0 * math.pi * np.sin(lut_th) * np.gradient(lut_th)
    lut = lut / (lut * w[:, None]).sum(axis=0)[None, :]
    return lut.astype("f4"), rgb


def ice_index(lam_um):
    """Real refractive index of ice near 0.4-0.7 um (Warren & Brandt 2008),
    fitted linearly in 1/lambda^2."""
    return 1.3040 + 0.00178 / np.asarray(lam_um, "f8") ** 2


def build_psc_phase(n_lambda=16, n_size=9, sigma_g=1.04, verbose=False):
    """Nacreous (type II polar stratospheric) cloud: nearly equal ice
    spheres.  For each mean radius in PSC_RADII, a narrow log-normal spread
    (sigma_g) of sizes, Mie theory at each wavelength: the rows of vivid,
    radius-dependent colours that are the cloud's iridescence."""
    lams = np.linspace(400.0, 700.0, n_lambda)
    th = angle_grid()
    mu = np.cos(th)
    out = []
    for a in PSC_RADII:
        z = np.linspace(-2.0, 2.0, n_size)
        radii = a * sigma_g ** z
        wn = np.exp(-0.5 * z * z)
        p_lam = np.zeros((n_lambda, th.size))
        for il, lam in enumerate(lams):
            m = complex(float(ice_index(lam / 1000.0)), 0.0)
            acc = np.zeros(th.size)
            wsum = 0.0
            for r, wr in zip(radii, wn):
                x = 2.0 * math.pi * r / (lam / 1000.0)
                s1, s2, qsca, g = bhmie(x, m, mu)
                w = wr * qsca * r * r
                acc += w * phase_from_s(s1, s2, x, qsca)
                wsum += w
            p_lam[il] = acc / wsum
        lut, _ = _spectral_lut(th, p_lam, lams)
        out.append(lut)
        if verbose:
            print(f"  radius {a:.2f} um", flush=True)
    return np.stack(out)


def psc_phase_lut() -> np.ndarray:
    """(len(PSC_RADII), LUT_SIZE, 3) float32: RGB phase functions of nacreous
    cloud (1/sr) at theta = pi u^2, one row per mean radius."""
    if os.path.exists(PSC_DATA):
        return np.load(PSC_DATA).astype("f4")
    return build_psc_phase()


def _convolve_disc(th, p, radius):
    """Average each angle over the solar disc (chord-length weights).

    A one-dimensional average in theta is only a good stand-in for the
    two-dimensional convolution away from the poles; near 0 and 180 degrees
    it would smear the diffraction peak over a ring and inflate its energy.
    So it is applied between 3 and 177 degrees and faded out towards the
    poles, and the result is renormalised by the caller."""
    offs = np.linspace(-radius, radius, 21)
    wts = np.sqrt(np.maximum(1.0 - (offs / radius) ** 2, 0.0))
    wts /= wts.sum()
    out = np.zeros_like(p)
    for o, wgt in zip(offs, wts):
        tt = np.clip(np.abs(th + o), 0.0, math.pi)
        tt = np.where(th + o > math.pi, 2 * math.pi - (th + o), tt)
        for i in range(p.shape[0]):
            out[i] += wgt * np.exp(np.interp(tt, th, np.log(np.maximum(p[i], 1e-30))))
    deg = np.degrees(th)
    blend = np.clip((deg - 1.0) / 2.0, 0.0, 1.0) * np.clip((179.0 - deg) / 2.0, 0.0, 1.0)
    return p * (1.0 - blend) + out * blend


def _normalise(th, p):
    """Scale so that the integral over the sphere is exactly one."""
    w = 2.0 * math.pi * np.sin(th)
    return p / np.trapezoid(p * w, th)


def rain_phase_lut() -> np.ndarray:
    """(LUT_SIZE, 3) float32: RGB phase function of rain (1/sr) at
    theta = pi u^2."""
    if os.path.exists(DATA):
        return np.load(DATA)
    lut, _ = build_rain_phase()
    return lut


def lookup(lut, theta):
    """Evaluate the LUT at scattering angles theta (radians) - the same
    interpolation the shader does."""
    u = np.sqrt(np.clip(np.asarray(theta, "f8") / math.pi, 0.0, 1.0))
    f = u * lut.shape[0] - 0.5
    i0 = np.clip(np.floor(f).astype(int), 0, lut.shape[0] - 1)
    i1 = np.clip(i0 + 1, 0, lut.shape[0] - 1)
    t = np.clip(f - i0, 0.0, 1.0)[..., None]
    return lut[i0] * (1 - t) + lut[i1] * t


if __name__ == "__main__":
    import time
    t0 = time.time()
    lut, info = build_rain_phase(verbose=True)
    np.save(DATA, lut)
    print(f"saved {DATA}  ({time.time() - t0:.0f} s)")
    t0 = time.time()
    np.save(PSC_DATA, build_psc_phase(verbose=True).astype("f2"))
    print(f"saved {PSC_DATA}  ({time.time() - t0:.0f} s)")
