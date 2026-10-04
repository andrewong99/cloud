# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
A cloud-resolving model: anelastic dynamics and single-moment bulk
microphysics, built to the recipe of NCAR's CM1 (Bryan & Fritsch 2002).

This module is the specification.  It holds the physical constants, the
thermodynamics, the reference state, the scenarios, the microphysical process
rates, and a complete NumPy implementation of the model.  The GPU model in
``crm_gpu.py`` is a line-by-line port of it and the self-test runs the two
side by side; the NumPy model is also what runs when the graphics card cannot
do compute shaders.

Equations (CM1 psolver=4, eqtset=1, anelastic):

    du/dt   = ADV(u) - d(phi)/dx                    phi = p'/rho0
    dv/dt   = ADV(v) - d(phi)/dy
    dw/dt   = ADV(w) - d(phi)/dz + B
    d(th')/dt = ADV(th) + latent heating / (cp pi0)
    dq/dt   = ADV(q) + microphysics                  q = qv qc qr qi qs
    div(rho0 u) = 0

    B = g [ th'/th0 + (Rv/Rd - 1)(qv - qv0) - (qc + qr + qi + qs) ]   CM1 eq. 8

Numerics (CM1 defaults for its LES and supercell cases):
  * Arakawa C grid, periodic in x and y, rigid lid and ground in z.
  * Wicker-Skamarock three-stage Runge-Kutta; fifth-order upwind flux-form
    advection of every field (third and second order on the two levels next
    to the ground and the lid, where the stencil runs out).
  * Pressure: the anelastic Poisson equation is solved EXACTLY at every stage
    by an FFT in x and y and a tridiagonal (Thomas) solve in z for every
    horizontal wavenumber - no iteration count to argue about.
  * Microphysics once per step, after the dynamics, in CM1's order:
    positivity, fall speeds, sedimentation, conversions, saturation
    adjustment.

Microphysics:
  * warm rain: CM1's Kessler scheme exactly (SI constants of kessler.F);
  * cloud liquid/ice partition: SAM1MOM's temperature ramp, 0 C to -20 C;
  * frozen precipitation: one category that is snow in the cold anvil and
    graupel lower down (SAM1MOM's ramp); Marshall-Palmer spectra with SAM's
    intercepts give fall speeds, accretion, melting and sublimation;
  * rain freezes by Bigg's formula with Lin et al. (1983) constants;
  * saturation adjustment conserves cp T - Lv qc - Ls qi exactly, by Newton
    iteration on temperature.

Where a constant comes from, it says so.  Where a choice is this program's
own, it says that instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------ constants ----
# CM1 constants.F
G = 9.81
RD = 287.04
RV = 461.5
CP = 1005.7
CPV = 1870.0
CPL = 4190.0
CPI = 2106.0
P00 = 1.0e5
EPS = RD / RV                     # 0.62197
REPSM1 = RV / RD - 1.0            # 0.60779
XLV = 2.501e6
XLS = 2.834e6
LV1 = XLV + (CPL - CPV) * 273.15  # 3.1347e6
LV2 = CPL - CPV                   # 2320
LS1 = XLS + (CPI - CPV) * 273.15  # 2.8985e6
LS2 = CPI - CPV                   # 236
T0K = 273.15
RHOW = 1000.0
RHOI = 917.0

# Kessler (CM1 kessler.F, SI)
K_AUTO = 1.0e-3                   # s^-1
Q_AUTO = 1.0e-3                   # kg/kg
K_ACCR = 2.2
VR_COEF = 14.34                   # m/s (rho qr)^0.1346 sqrt(1.15/rho)
VR_EXP = 0.1346

# SAM1MOM (micro_params.f90, precip_init.f90)
N0R, N0S, N0G = 8.0e6, 3.0e6, 4.0e6          # m^-4
RHO_S, RHO_G = 100.0, 400.0                   # kg m^-3
A_S, B_S = 4.84, 0.25                         # snow  V = a D^b
A_G, B_G = 94.5, 0.5                          # graupel (Lin 1983, rhog=400)
E_CLOUD_BY_ICE_PRECIP = 1.0
E_ICE_BY_ICE_PRECIP = 0.1
QI_AUTO = 1.0e-4                              # kg/kg
K_AUTO_ICE = 1.0e-3                           # s^-1, times exp(0.025 (T-T0))
RHO_REF_ICE = 1.29                            # SAM's reference density
# Bigg (1953) freezing of rain, Lin et al. (1983) constants
BIGG_A = 0.66                                 # K^-1
BIGG_B = 100.0                                # m^-3 s^-1
# conduction and diffusion (Rutledge & Hobbs 1983 style process rates)
K_AIR = 2.4e-2                                # W m^-1 K^-1
DV0 = 2.26e-5                                 # m^2/s at 273.15 K, 1000 hPa
F_VENT = 1.0                                  # ventilation factor (this program's choice)

# cloud optics: effective radii (this program's choice, typical values)
R_EFF_LIQ = 10.0e-6
R_EFF_ICE = 25.0e-6

# Long-wave absorption by cloud ice relative to cloud water, per unit mass.
# In the geometric limit a particle's absorption cross-section per unit mass
# is 3/(2 rho r_eff), so ice of 25 um absorbs (1000 x 10)/(917 x 25) = 0.44
# as much as droplets of 10 um (this program's choice; the DYCOMS-II scheme
# itself is liquid only).
LW_ICE_RATIO = (RHOW * R_EFF_LIQ) / (RHOI * R_EFF_ICE)

# Precipitation counts as there to be seen where its extinction exceeds this:
# a ray grazing 2 km of it then crosses 0.02 optical depths (this program's
# choice).  About 0.5 mg/kg of rain or 0.15 mg/kg of snow.
PRECIP_VISIBLE = 1.0e-5                       # m^-1


def _gamma(x):
    return math.gamma(x)


# Marshall-Palmer derived coefficients (exact under the stated spectra).
# fall speed (mass weighted):  V = a Gamma(4+b)/6 (rho q / (pi rho_x N0))^(b/4)
VS_COEF = A_S * _gamma(4.0 + B_S) / 6.0 / (math.pi * RHO_S * N0S) ** (B_S / 4.0)
VG_COEF = A_G * _gamma(4.0 + B_G) / 6.0 / (math.pi * RHO_G * N0G) ** (B_G / 4.0)
# collection: dq_c/dt = -C (rho q)^((3+b)/4) q_c sqrt(rho_ref/rho)
CS_COEF = (math.pi / 4.0) * A_S * N0S * _gamma(3.0 + B_S) / (math.pi * RHO_S * N0S) ** ((3.0 + B_S) / 4.0)
CG_COEF = (math.pi / 4.0) * A_G * N0G * _gamma(3.0 + B_G) / (math.pi * RHO_G * N0G) ** ((3.0 + B_G) / 4.0)


# ---------------------------------------------------------- thermodynamics --
def esat_liq(t):
    """Saturation vapour pressure over liquid, Pa.  Bolton (1980), as CM1."""
    return 611.2 * np.exp(17.67 * (t - T0K) / (t - 29.65))


def esat_ice(t):
    """Over ice, Pa.  Tao et al. (1989), as CM1 cm1libs.F."""
    return 611.2 * np.exp(21.8745584 * (t - T0K) / (t - 7.66))


def qsat_liq(t, p):
    es = np.minimum(esat_liq(t), 0.5 * p)
    return EPS * es / (p - es)


def qsat_ice(t, p):
    es = np.minimum(esat_ice(t), 0.5 * p)
    return EPS * es / (p - es)


def dqsat_liq_dt(t, p, qs):
    es = np.minimum(esat_liq(t), 0.5 * p)
    return qs * (p / (p - es)) * 17.67 * (T0K - 29.65) / (t - 29.65) ** 2


def dqsat_ice_dt(t, p, qs):
    es = np.minimum(esat_ice(t), 0.5 * p)
    return qs * (p / (p - es)) * 21.8745584 * (T0K - 7.66) / (t - 7.66) ** 2


def liquid_fraction(t):
    """Cloud condensate that is liquid.  SAM1MOM: all liquid above 0 C, all
    ice below -20 C, linear between (micro_params.f90)."""
    return np.clip((t - 253.16) / 20.0, 0.0, 1.0)


def graupel_fraction(t):
    """Frozen precipitation that is graupel rather than snow (SAM1MOM)."""
    return np.clip((t - 223.16) / 60.0, 0.0, 1.0)


def latent_vap(t):
    return LV1 - LV2 * t


def latent_sub(t):
    return LS1 - LS2 * t


def saturation_adjust_arrays(t, qv, qc, qi, p, ice: bool, iterations: int = 4):
    """Condense or evaporate cloud so that vapour equals saturation over the
    mixed-phase cloud wherever there is cloud, conserving
    h = cp T - Lv qc - Ls qi at constant pressure.  Newton on T.  Works on
    arrays of any shape; returns (T, qv, qc, qi)."""
    qn = qc + qi
    qt = qv + qn
    lv = latent_vap(t)
    lf = latent_sub(t) - lv
    # conserved moist static energy per unit mass / cp: the clear-air
    # temperature if all the cloud evaporated
    tl = t - (lv * qc + (lv + lf) * qi) / CP

    def qsat_mix(tt):
        if not ice:
            qw = qsat_liq(tt, p)
            return qw, dqsat_liq_dt(tt, p, qw), np.ones_like(tt), np.zeros_like(tt)
        fl = liquid_fraction(tt)
        qw = qsat_liq(tt, p)
        qii = qsat_ice(tt, p)
        dfl = np.where((tt > 253.16) & (tt < 273.16), 1.0 / 20.0, 0.0)
        qsm = fl * qw + (1.0 - fl) * qii
        dq = fl * dqsat_liq_dt(tt, p, qw) + (1.0 - fl) * dqsat_ice_dt(tt, p, qii) + dfl * (qw - qii)
        return qsm, dq, fl, dfl

    qs_l, _, _, _ = qsat_mix(tl)
    cloudy = qt > qs_l
    # Newton on g(T) = T - tl - Leff(T) (qt - qsat(T)) / cp, which is smooth,
    # increasing and convex when the condensate is allowed to go negative
    # inside the iteration: started from tl (left of the root) the first step
    # lands right of it and the rest converge monotonically and
    # quadratically.  Clipping the condensate at zero inside the loop instead
    # gives the wrong slope and crawls.
    tn = np.array(tl, dtype="f8", copy=True)
    for _ in range(iterations):
        qsm, dq, fl, dfl = qsat_mix(tn)
        leff = lv + (1.0 - fl) * lf
        cond = qt - qsm
        f = tn - leff * cond / CP - tl
        fp = 1.0 + (lf * dfl * cond + leff * dq) / CP
        tn = np.where(cloudy, tn - f / fp, tl)
    qsm, _, fl, _ = qsat_mix(tn)
    qn_new = np.where(cloudy, np.maximum(qt - qsm, 0.0), 0.0)
    if not ice:
        fl = np.ones_like(tn)
    return tn, qt - qn_new, fl * qn_new, (1.0 - fl) * qn_new


# ------------------------------------------------------------------ grid ----
@dataclass
class Grid:
    """nx x ny x nz cells of dx x dy x dz.  z0 lifts the whole domain off
    the ground: a layer cloud at 3 km needs its 20 m levels where the cloud
    is, not 150 of them between it and the ground.  Heights (zc, zf, top)
    are above the ground."""
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    z0: float = 0.0

    @property
    def lx(self):
        return self.nx * self.dx

    @property
    def ly(self):
        return self.ny * self.dy

    @property
    def top(self):
        return self.z0 + self.nz * self.dz

    @property
    def zc(self):
        return self.z0 + (np.arange(self.nz) + 0.5) * self.dz

    @property
    def zf(self):
        return self.z0 + np.arange(self.nz + 1) * self.dz

    @property
    def cells(self):
        return self.nx * self.ny * self.nz


# ------------------------------------------------------------ base state ----
@dataclass
class BaseState:
    """Hydrostatic reference state on the model levels, built exactly as
    CM1 base.F does from a theta / qv sounding."""
    grid: Grid
    th0: np.ndarray       # K, cell centres
    qv0: np.ndarray       # kg/kg
    u0: np.ndarray        # m/s, ground relative
    v0: np.ndarray
    pi0: np.ndarray
    p0: np.ndarray        # Pa
    t0: np.ndarray        # K
    rho0: np.ndarray      # dry-air density, centres
    rhof: np.ndarray      # dry-air density, w levels (nz+1)
    psfc: float
    thsfc: float
    qvsfc: float

    @staticmethod
    def build(grid: Grid, theta_fn, qv_fn, u_fn, v_fn, psfc: float,
              rh_fn=None, qv_cap: float | None = None,
              max_rh: float = 0.985) -> "BaseState":
        """theta_fn(z) etc. are callables of height above ground.  If rh_fn is
        given, qv is derived from relative humidity, iterating 20 times
        because it depends on the pressure it helps set (CM1 base.F).
        Vapour is capped at max_rh of saturation so the reference state does
        not condense on the first step - a saturated reference would be a
        cloud already, and cloud layers are the deck renderer's job."""
        # An elevated domain (grid.z0 > 0) is reached by integrating the
        # hydrostatic equation up from the surface through 25 m sub-levels
        # of the same sounding, so its pressure is the sounding's pressure
        # there and not a one-step guess across kilometres.
        zc_model = grid.zc
        zsub = np.arange(12.5, zc_model[0] - 1.0, 25.0) if grid.z0 > 0.0 else np.zeros(0)
        nsub = len(zsub)
        zc = np.concatenate([zsub, zc_model])
        th = np.array([theta_fn(z) for z in zc], "f8")
        th_s = float(theta_fn(0.0))
        u0 = np.array([u_fn(z) for z in zc_model], "f8")
        v0 = np.array([v_fn(z) for z in zc_model], "f8")
        if rh_fn is None:
            qv = np.array([qv_fn(z) for z in zc], "f8")
            qv_s = float(qv_fn(0.0))
        else:
            qv = np.zeros_like(th)
            qv_s = 0.0
        pi_s = (psfc / P00) ** (RD / CP)
        for _ in range(20 if rh_fn is not None else 1):
            thv = th * (1.0 + qv / EPS) / (1.0 + qv)
            thv_s = th_s * (1.0 + qv_s / EPS) / (1.0 + qv_s)
            pi0 = np.zeros_like(th)
            pi0[0] = pi_s - G * zc[0] / (CP * 0.5 * (thv_s + thv[0]))
            for k in range(1, len(zc)):
                pi0[k] = pi0[k - 1] - G * (zc[k] - zc[k - 1]) / (CP * 0.5 * (thv[k] + thv[k - 1]))
            p0 = P00 * pi0 ** (CP / RD)
            t0 = th * pi0
            if rh_fn is not None:
                rh = np.array([rh_fn(z) for z in zc], "f8")
                qv = rh * qsat_liq(t0, p0)
                if qv_cap is not None:
                    qv = np.minimum(qv, qv_cap)
                t_s = th_s * pi_s
                qv_s = float(rh_fn(0.0) * qsat_liq(t_s, psfc))
                if qv_cap is not None:
                    qv_s = min(qv_s, qv_cap)
        qv = np.minimum(qv, max_rh * qsat_liq(t0, p0))
        th, qv, pi0, p0, t0 = (a[nsub:] for a in (th, qv, pi0, p0, t0))
        rho0 = p0 / (RD * t0 * (1.0 + qv / EPS))
        rhof = np.zeros(grid.nz + 1)
        rhof[1:-1] = 0.5 * (rho0[1:] + rho0[:-1])
        rhof[0] = 1.5 * rho0[0] - 0.5 * rho0[1]
        rhof[-1] = 1.5 * rho0[-1] - 0.5 * rho0[-2]
        return BaseState(grid, th, qv, u0, v0, pi0, p0, t0, rho0, rhof, psfc,
                         th_s, qv_s)

    @staticmethod
    def from_liquid(grid: Grid, thl_fn, qt_fn, u_fn, v_fn, psfc: float,
                    ice: bool = False, iterations: int = 4):
        """Reference state from liquid-water potential temperature and total
        water - the variables every layer-cloud case (DYCOMS-II, and the
        cloud layers this program builds from a sounding) is specified in.
        Where qt exceeds saturation the level is brought to equilibrium by
        the model's own saturation adjustment, so the reference state IS the
        initial mean state: theta and vapour at saturation inside the cloud,
        and the cloud water returned separately.  Returns (base, qc, qi) on
        the model levels."""
        zc = grid.zc
        thl = np.array([thl_fn(z) for z in zc], "f8")
        qt = np.array([qt_fn(z) for z in zc], "f8")
        dth = np.zeros_like(thl)            # theta - theta_l
        dqv = np.zeros_like(qt)             # qt - qv (the condensate)
        qc = np.zeros_like(qt)
        qi = np.zeros_like(qt)
        base = None
        for _ in range(iterations):
            f_th = (lambda d: (lambda z: float(thl_fn(z)) + float(np.interp(z, zc, d, left=0.0, right=0.0))))(dth.copy())
            f_qv = (lambda d: (lambda z: float(qt_fn(z)) - float(np.interp(z, zc, d, left=0.0, right=0.0))))(dqv.copy())
            base = BaseState.build(grid, f_th, f_qv, u_fn, v_fn, psfc, max_rh=1.0 + 1e-6)
            # every level starts as clear air at T_l = theta_l pi
            tl = thl * base.pi0
            t, qv, qc, qi = saturation_adjust_arrays(tl, qt.copy(), np.zeros_like(qt),
                                                     np.zeros_like(qt), base.p0, ice)
            dth = t / base.pi0 - thl
            dqv = qc + qi
        return base, qc, qi

    def cape(self, ice: bool = False):
        """CAPE / CIN of a surface parcel (pseudo-adiabatic), J/kg, from the
        reference state - the number the scenario is judged by."""
        return parcel_cape(self.grid.zc, self.t0, self.qv0, self.p0,
                           self.thsfc * (self.psfc / P00) ** (RD / CP),
                           self.qvsfc, self.psfc, ice=ice)


def parcel_cape(z, t_env, qv_env, p_env, t_parcel0, qv_parcel0, p_start, ice=False):
    """Pseudo-adiabatic parcel with virtual-temperature buoyancy, integrated
    in pressure steps of 200 Pa (CM1 getcape.F's pinc); returns (cape, cin,
    z_lcl, z_el)."""
    # environment as functions of pressure (log-linear)
    lp = np.log(p_env[::-1])
    def env(p, arr):
        return np.interp(math.log(p), lp, arr[::-1])
    t, qv, p = float(t_parcel0), float(qv_parcel0), float(p_start)
    th = t * (P00 / p) ** (RD / CP)
    cape = cin = 0.0
    z_lcl = z_el = None
    prev_b = None
    zp = 0.0
    saturated = False
    pinc = 200.0
    while p > p_env[-1] + pinc:
        p_new = p - pinc
        if not saturated:
            t_new = th * (p_new / P00) ** (RD / CP)
            if qv >= float(qsat_liq(np.float64(t_new), p_new)):
                saturated = True
                z_lcl = env(p_new, z)
            t, p = t_new, p_new
        else:
            # pseudo-adiabatic step: all condensate removed
            for _ in range(3):
                qs = float(qsat_liq(np.float64(t), p))
            lv = LV1 - LV2 * t
            fl = 1.0
            if ice:
                fl = float(np.clip((t - 233.15) / 40.0, 0.0, 1.0))
                qs = fl * qs + (1 - fl) * float(qsat_ice(np.float64(t), p))
                lv = fl * lv + (1 - fl) * (LS1 - LS2 * t)
            # moist adiabatic dT/dp
            rv = qs
            num = (RD * t + lv * rv) / p
            den = CP + lv * lv * rv * EPS / (RD * t * t)
            t_new = t - num / den * pinc
            p_new = p - pinc
            t, p = t_new, p_new
            qv = float(qsat_liq(np.float64(t), p)) if not ice else \
                (fl * float(qsat_liq(np.float64(t), p)) + (1 - fl) * float(qsat_ice(np.float64(t), p)))
        te = env(p, t_env)
        qe = env(p, qv_env)
        tvp = t * (1.0 + qv / EPS) / (1.0 + qv)
        tve = te * (1.0 + qe / EPS) / (1.0 + qe)
        b = G * (tvp - tve) / tve
        zn = env(p, z)
        dzs = zn - zp
        if prev_b is not None:
            bb = 0.5 * (b + prev_b)
            if bb > 0:
                cape += bb * dzs
            elif cape == 0.0:
                cin += bb * dzs
            if prev_b > 0 >= b:
                z_el = zn
        prev_b, zp = b, zn
    return cape, cin, z_lcl, z_el


# ------------------------------------------------------------ scenarios ----
@dataclass
class Scenario:
    key: str
    name: str
    description: str
    grid: Grid
    base: BaseState
    micro_ice: bool = True
    # SAM1MOM's dograupel: frozen precipitation is part graupel (by
    # temperature) in deep convection; in layer clouds, whose weak updraughts
    # cannot rime ice into graupel, it is all snow
    graupel: bool = True
    # initial perturbations, between random_zmin and random_depth (heights
    # above the ground)
    random_th: float = 0.25          # K, uniform +-
    random_qv: float = 0.0           # kg/kg
    random_depth: float = 1000.0
    random_zmin: float = 0.0
    bubbles: list = field(default_factory=list)   # (x, y, z, rh, rz, dth)
    line_bubbles: list = field(default_factory=list)  # (x, z, rh, rz, dth): along all of y
    # initial cloud condensate on the model levels (from BaseState.from_liquid)
    init_qc: object = None
    init_qi: object = None
    # surface
    flux_mode: str = "sun"           # sun | fixed | bulk | none
    bowen: float = 0.5
    sw_absorb: float = 0.77          # share of the sunlight the ground absorbs (1 - albedo)
    fixed_wth: float = 0.0           # K m/s
    fixed_wqv: float = 0.0           # m/s
    heterogeneity: float = 0.35
    drag: bool = True
    z0: float = 0.1                  # roughness length, m
    cd: float = 0.0                  # fixed drag coefficient; 0: from z0 (log law)
    sst: float = 0.0                 # K, surface temperature for flux_mode "bulk"
    # rotation: du/dt += f (v - vg), dv/dt -= f (u - ug), with the
    # geostrophic wind given by ug_fn/vg_fn (ground relative) or, if None,
    # the reference wind
    coriolis: float = 0.0            # s^-1
    ug_fn: object = None
    vg_fn: object = None
    # long-wave radiation, the DYCOMS-II parameterisation (Stevens et al.
    # 2005): F = F0 exp(-Q(z,inf)) + F1 exp(-Q(0,z)) [+ rho_i cp D ((z-zi)^4/3 / 4
    # + zi (z-zi)^1/3) above zi], Q = kappa int rho (ql + 0.44 qi) dz
    lw_f0: float = 0.0               # W m^-2, cloud-top cooling
    lw_f1: float = 0.0               # W m^-2, cloud-base warming
    lw_kappa: float = 85.0           # m^2 kg^-1
    lw_div: float = 0.0              # s^-1, D of the above-inversion term (0: off)
    lw_zi_qt: float = 8.0e-3         # zi is where qt falls through this value
    lw_rhoi: float = 1.13            # kg m^-3
    # relaxation of the layer's horizontal-MEAN theta_l and q_t towards the
    # initial profile between nudge_zmin and nudge_zmax, with time scale
    # nudge_tau (s; 0: off) - the standard way an LES of an observed case is
    # held to the observed (here: forecast) mean state while its eddies,
    # cells and bands develop freely
    nudge_tau: float = 0.0
    nudge_zmin: float = 0.0
    nudge_zmax: float = 0.0
    nudge_uv: bool = False           # ... and the mean wind (keeps a shear layer sheared)
    # microphysics
    q_auto: float = Q_AUTO           # Kessler autoconversion threshold, kg/kg
    qi_auto: float = QI_AUTO         # cloud ice -> snow threshold, kg/kg
    # large-scale forcing (BOMEX, DYCOMS-II)
    subsidence: object = None        # callable z -> w_ls
    rad_cool: object = None          # callable z -> K/s
    moist_adv: object = None         # callable z -> kg/kg/s
    # frame
    translate: tuple = (0.0, 0.0)    # domain translation, m/s
    # a reference case may set the extent of its layer around the observer
    # (east_km, north_km) -> cover %, this program's composition
    demo_cover: object = None
    tile: bool = True                # periodic cloud field drawn to the horizon
    tile_axes: int = -1              # 1 x, 2 y, 3 both; -1: 3 if tile else 0
    lateral_sponge: bool = False
    sponge_axes: int = 3             # 1 x edges, 2 y edges, 3 both
    sponge_depth: float = 5000.0
    observer: tuple = (0.5, 0.5)     # fraction of the domain at t = 0
    seed: int = 1
    # what the scenario is meant to show, for the in-view panel: a list of
    # (Atlas name, the physics that makes it) and the weather-data reasons
    # it was chosen
    shows: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    layer_depth: float = 0.0         # depth of the overturning layer, m (0: unknown)

    @property
    def tiling(self) -> int:
        if self.tile_axes >= 0:
            return self.tile_axes
        return 3 if self.tile else 0

    def summary(self) -> str:
        g = self.grid
        s = (f"{g.nx}x{g.ny}x{g.nz} cells, {g.dx:.0f} m x {g.dz:.0f} m, "
             f"{g.lx/1000:.1f} km domain")
        if g.z0 > 0:
            return s + f", {g.z0/1000:.1f}-{g.top/1000:.1f} km up"
        cape, cin, zl, ze = self.base.cape()
        return s + f"; CAPE {cape:.0f} J/kg"


def _interp_fn(zs, vals):
    zs = np.asarray(zs, "f8")
    vals = np.asarray(vals, "f8")
    return lambda z: float(np.interp(z, zs, vals))


def wk82_base(grid: Grid) -> BaseState:
    """Weisman & Klemp (1982) sounding exactly as CM1 isnd=5 codes it."""
    z_trop, th_trop, t_trop, th_sfc, qv_pbl = 12000.0, 343.0, 213.0, 300.0, 0.014

    def theta(z):
        if z <= z_trop:
            return th_sfc + (th_trop - th_sfc) * (z / z_trop) ** 1.25
        return th_trop * math.exp(G * (z - z_trop) / (CP * t_trop))

    def rh(z):
        if z <= z_trop:
            return 1.0 - 0.75 * (z / z_trop) ** 1.25
        return 0.25

    def uq(z):          # CM1 iwnd=2 quarter circle, ground relative
        if z <= 2000.0:
            return 7.0 - 7.0 * math.cos(0.5 * math.pi * z / 2000.0)
        if z <= 6000.0:
            return 7.0 + (z - 2000.0) * (31.0 - 7.0) / 4000.0
        return 31.0

    def vq(z):
        if z <= 2000.0:
            return 7.0 * math.sin(0.5 * math.pi * z / 2000.0)
        return 7.0

    return BaseState.build(grid, theta, None, uq, vq, 100000.0, rh_fn=rh,
                           qv_cap=qv_pbl, max_rh=1.0)


def bomex_base(grid: Grid) -> BaseState:
    """BOMEX (Siebesma et al. 2003) as CM1 isnd=19 / MicroHH tabulate it."""
    zs = [0.0, 520.0, 1480.0, 2000.0, 3000.0]
    thl = [298.7, 298.7, 302.4, 308.2, 311.85]
    qt = [17.0e-3, 16.3e-3, 10.7e-3, 4.2e-3, 3.0e-3]            # specific
    th_f = _interp_fn(zs + [30000.0], thl + [311.85 + 27000.0 * 3.65e-3])
    q_f = _interp_fn(zs + [30000.0], qt + [3.0e-3])

    def qv(z):
        q = q_f(z)
        return q / (1.0 - q)

    def u(z):
        if z <= 700.0:
            return -8.75
        return -8.75 + (z - 700.0) * (-4.61 + 8.75) / (3000.0 - 700.0)

    return BaseState.build(grid, th_f, qv, u, lambda z: 0.0, 101500.0)


def sounding_profiles(snd, top: float):
    """theta(z), mixing ratio(z), u(z), v(z) of a CloudSim sounding as
    functions of height above the ground, extended stably above `top` if
    the sounding stops short, and the surface pressure (Pa)."""
    elev = float(getattr(snd, "elevation", 0.0) or 0.0)
    zs, ths, qvs, us, vs = [], [], [], [], []
    for lv in snd.levels:
        z = lv.z - elev
        if z < -1.0:
            continue
        t = lv.t + T0K
        p = lv.p * 100.0
        th = t * (P00 / p) ** (RD / CP)
        e = float(esat_liq(np.float64(lv.td + T0K)))
        e = min(e, 0.5 * p)
        zs.append(max(z, 0.0))
        ths.append(th)
        qvs.append(EPS * e / (p - e))
        us.append(lv.u)
        vs.append(lv.v)
    order = np.argsort(zs)
    zs, ths, qvs, us, vs = (list(np.asarray(a)[order]) for a in (zs, ths, qvs, us, vs))
    if zs[-1] < top + 1000.0:
        zs.append(top + 5000.0)
        ths.append(ths[-1] + (top + 5000.0 - zs[-2]) * 0.012)
        qvs.append(qvs[-1] * 0.5)
        us.append(us[-1])
        vs.append(vs[-1])
    psfc = float(snd.levels[0].p) * 100.0
    if snd.levels[0].z - elev > 5.0:
        t0 = snd.levels[0].t + T0K
        psfc *= math.exp(G * (snd.levels[0].z - elev) / (RD * t0))
    return (_interp_fn(zs, ths), _interp_fn(zs, qvs), _interp_fn(zs, us),
            _interp_fn(zs, vs), psfc)


def sounding_base(grid: Grid, snd) -> BaseState:
    """Reference state from a CloudSim sounding (Open-Meteo or synthetic)."""
    elev = float(getattr(snd, "elevation", 0.0) or 0.0)
    zs, ths, qvs, us, vs = [], [], [], [], []
    for lv in snd.levels:
        z = lv.z - elev
        if z < -1.0:
            continue
        t = lv.t + T0K
        p = lv.p * 100.0
        th = t * (P00 / p) ** (RD / CP)
        e = float(esat_liq(np.float64(lv.td + T0K)))
        e = min(e, 0.5 * p)
        q = EPS * e / (p - e)
        zs.append(max(z, 0.0))
        ths.append(th)
        qvs.append(q)
        us.append(lv.u)
        vs.append(lv.v)
    order = np.argsort(zs)
    zs = list(np.asarray(zs)[order])
    ths = list(np.asarray(ths)[order])
    qvs = list(np.asarray(qvs)[order])
    us = list(np.asarray(us)[order])
    vs = list(np.asarray(vs)[order])
    # extend upward isothermally-stable if the sounding stops short
    if zs[-1] < grid.top + 1000.0:
        zs.append(grid.top + 5000.0)
        ths.append(ths[-1] + (grid.top + 5000.0 - zs[-2]) * 0.012)
        qvs.append(qvs[-1] * 0.5)
        us.append(us[-1])
        vs.append(vs[-1])
    psfc = float(snd.levels[0].p) * 100.0
    if snd.levels[0].z - elev > 5.0:
        # the lowest level is above the ground: extend hydrostatically
        t0 = snd.levels[0].t + T0K
        psfc *= math.exp(G * (snd.levels[0].z - elev) / (RD * t0))
    return BaseState.build(grid, _interp_fn(zs, ths), _interp_fn(zs, qvs),
                           _interp_fn(zs, us), _interp_fn(zs, vs), psfc)


GRID_PRESETS = {
    # (nx=ny, dx, nz, dz) per scenario family and model size
    "deep":     {"fast": (64, 500.0, 48, 375.0), "standard": (128, 250.0, 72, 250.0),
                 "fine": (256, 125.0, 96, 187.5)},
    "storm":    {"fast": (64, 1000.0, 40, 500.0), "standard": (128, 500.0, 64, 312.5),
                 "fine": (256, 250.0, 80, 250.0)},
    "shallow":  {"fast": (64, 250.0, 40, 75.0), "standard": (128, 125.0, 48, 62.5),
                 "fine": (256, 62.5, 64, 50.0)},
    # a cloud layer and the air just around it (stratocumulus, altocumulus,
    # asperitas): LES spacing, ~1.2 km deep, lifted to the layer by Grid.z0
    "layer":    {"fast": (64, 62.5, 48, 25.0), "standard": (128, 50.0, 64, 20.0),
                 "fine": (256, 35.0, 96, 12.5)},
    # a convective boundary layer with room to deepen (cloud streets)
    "bl":       {"fast": (64, 250.0, 40, 50.0), "standard": (128, 125.0, 48, 40.0),
                 "fine": (256, 62.5, 64, 30.0)},
    # a precipitating layer and the dry air its precipitation falls through
    # (virga, fall streaks)
    "fall":     {"fast": (64, 125.0, 48, 75.0), "standard": (128, 100.0, 64, 55.0),
                 "fine": (256, 62.5, 96, 40.0)},
    # a thick cloud layer and the sheared stable layer under it (asperitas),
    # ~1.75 km deep
    "wavebase": {"fast": (64, 62.5, 56, 31.25), "standard": (128, 50.0, 72, 25.0),
                 "fine": (256, 35.0, 108, 16.67)},
}


def make_grid(family: str, size: str, z0: float = 0.0) -> Grid:
    n, dx, nz, dz = GRID_PRESETS[family][size]
    return Grid(n, n, nz, dx, dx, dz, z0)


def dry_pi(z: float, theta_fn, psfc: float, step: float = 10.0) -> float:
    """Exner function at height z of a dry hydrostatic atmosphere with the
    given theta(z), integrated up from the surface."""
    pi = (psfc / P00) ** (RD / CP)
    zz = 0.0
    while zz < z:
        h = min(step, z - zz)
        pi -= G * h / (CP * theta_fn(zz + 0.5 * h))
        zz += h
    return pi


def _smooth(x, a, b):
    t = min(1.0, max(0.0, (x - a) / max(b - a, 1e-9)))
    return t * t * (3.0 - 2.0 * t)


# The Noah land surface's loam (Chen & Dudhia 2001; SOILPARM.TBL, STAS row 6:
# WLTSMC 0.066, REFSMC 0.329): wilting point and reference (field-capacity)
# soil water, m3/m3
SOIL_WILT, SOIL_REF = 0.066, 0.329
EF_WET, EF_DRY = 0.69, 0.03       # evaporative fraction of moist and of bone-dry ground


def surface_state(snd, default_bowen: float = 0.45):
    """(Bowen ratio, share of the sunlight absorbed, note) of the ground under
    a sounding.  The Sun's energy is split between heating the air and
    evaporating water by how much water the root zone holds: the
    evaporative fraction falls from 0.69 (Bowen ratio 0.45, moist ground)
    at field capacity to 0.03 at the wilting point - a desert heats its air
    and does not moisten it.  Bare dry ground reflects more (sand: albedo
    0.35, absorbed 0.65).  Lying snow reflects the sunlight: the absorbed
    share falls to 0.36 (albedo 0.64, snow-covered fields and woods).  Without the forecast's ground: default_bowen and no
    snow (the note is then empty)."""
    sf = getattr(snd, "surface", None) or {}

    def num(k):
        try:
            v = float(sf.get(k))
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    notes = []
    bowen = float(default_bowen)
    th = num("soil_moisture_9_to_27cm")
    if th is not None and th >= 0.0:
        f = min(max((th - SOIL_WILT) / (SOIL_REF - SOIL_WILT), 0.0), 1.0)
        ef = EF_DRY + (EF_WET - EF_DRY) * f
        bowen = (1.0 - ef) / ef
        notes.append(f"root-zone soil water {th:.2f} m3/m3: Bowen ratio {bowen:.2f}")
    depth = num("snow_depth")
    snow = min(max(depth / 0.10, 0.0), 1.0) if depth is not None else 0.0
    # bare dry ground reflects more: albedo 0.23 -> 0.35 (desert sand)
    x = min(max((0.20 - th) / (0.20 - SOIL_WILT), 0.0), 1.0) if th is not None and th >= 0.0 else 0.0
    dry = x * x * (3.0 - 2.0 * x)
    absorb = 0.77 - dry * 0.12
    absorb = absorb - snow * (absorb - 0.36)
    if snow > 0.0:
        notes.append(f"{100.0 * depth:.0f} cm of snow: the ground absorbs {100.0 * absorb:.0f}% "
                     "of the sunlight")
    return bowen, absorb, "; ".join(notes)


def scenario_weather(snd, size="standard", deep=True, sun_up=True) -> Scenario:
    grid = make_grid("deep", size)
    base = sounding_base(grid, snd)
    lx = grid.lx
    rng = np.random.default_rng(7)
    bubbles = []
    if sun_up:
        # first-generation thermals, so the sky does not wait an hour for the
        # boundary layer to organise itself (this program's choice)
        for _ in range(14):
            bubbles.append((rng.uniform(0, lx), rng.uniform(0, lx),
                            600.0, rng.uniform(1200, 2200), 600.0,
                            rng.uniform(0.6, 1.3)))
    src = getattr(snd, "source", "sounding")
    bowen, absorb, ground = surface_state(snd)
    return Scenario("weather", "The weather",
                    f"Your sounding ({src}) heated by the Sun: thermals, "
                    "cumulus, and whatever they grow into."
                    + (f" The ground: {ground}." if ground else ""),
                    grid, base, micro_ice=True, random_th=0.3, random_depth=900.0,
                    bubbles=bubbles, flux_mode="sun", bowen=bowen, sw_absorb=absorb,
                    heterogeneity=0.35, tile=True)


def scenario_supercell(size="standard") -> Scenario:
    grid = make_grid("storm", size)
    base = wk82_base(grid)
    lx, ly = grid.lx, grid.ly
    return Scenario("supercell", "Supercell (Weisman & Klemp 1982)",
                    "The textbook supercell sounding and quarter-circle "
                    "hodograph, triggered by a 3 K warm bubble.",
                    grid, base, micro_ice=True, random_th=0.0,
                    bubbles=[(0.5 * lx, 0.5 * ly, 1500.0, 10000.0, 1500.0, 3.0)],
                    flux_mode="none", drag=False, translate=(12.5, 3.0),
                    tile=False, lateral_sponge=True, observer=(0.78, 0.40))


def scenario_squall(size="standard") -> Scenario:
    """A squall line (CM1's squall-line set-up after Rotunno, Klemp & Weisman
    1988): the Weisman-Klemp sounding with 15 m/s of westerly shear in the
    lowest 2.5 km, triggered by a line thermal.  Its rain-cooled outflow
    spreads as a cold pool whose leading edge, the gust front, lifts the
    warm, moist inflow: the cloud that forms on that lift is the shelf cloud
    (arcus).  Periodic along the line, sponged at its ends across it."""
    grid = make_grid("storm", size)
    wk = wk82_base(grid)
    us = 15.0

    def u(z):
        return us * min(z / 2500.0, 1.0)

    th_f = _interp_fn(grid.zc, wk.th0)
    base = BaseState.build(grid, lambda z: float(np.interp(z, grid.zc, wk.th0)),
                           lambda z: float(np.interp(z, grid.zc, wk.qv0)), u, lambda z: 0.0,
                           wk.psfc, max_rh=1.0)
    lx = grid.lx
    return Scenario("squall", "Squall line with a shelf cloud",
                    "A line of thunderstorms driven by its own cold pool; the gust front "
                    "lifts the inflow into a shelf cloud (arcus) along the line.",
                    grid, base, micro_ice=True, random_th=0.2, random_depth=1500.0,
                    line_bubbles=[(0.4 * lx, 1500.0, 10000.0, 1500.0, 2.0)],
                    flux_mode="none", drag=False, translate=(12.0, 0.0),
                    tile=False, tile_axes=2, lateral_sponge=True, sponge_axes=1,
                    observer=(0.72, 0.5),
                    shows=[("Cumulonimbus capillatus praecipitatio arcus",
                            "rain-cooled outflow spreads ahead of the storms as a cold pool; its "
                            "leading edge lifts the warm inflow, which condenses into a low, "
                            "layered shelf cloud along the gust front")])


def scenario_bomex(size="standard") -> Scenario:
    grid = make_grid("shallow", size)
    base = bomex_base(grid)

    def w_ls(z):
        if z <= 1500.0:
            return -0.0065 * z / 1500.0
        if z <= 2100.0:
            return -0.0065 * (1.0 - (z - 1500.0) / 600.0)
        return 0.0

    def rad(z):                     # DALES taper (see research notes)
        if z <= 1500.0:
            return -2.315e-5
        if z <= 2500.0:
            return -2.315e-5 * (1.0 - (z - 1500.0) / 1000.0)
        return 0.0

    def madv(z):
        if z <= 300.0:
            return -1.2e-8
        if z <= 500.0:
            return -1.2e-8 * (1.0 - (z - 300.0) / 200.0)
        return 0.0

    return Scenario("bomex", "Trade cumulus (BOMEX)",
                    "Siebesma et al. (2003): shallow cumulus over a tropical "
                    "ocean, with the large-scale forcing of the case.",
                    grid, base, micro_ice=False, random_th=0.1, random_qv=2.5e-5,
                    random_depth=1600.0, flux_mode="fixed", fixed_wth=8.0e-3,
                    fixed_wqv=5.2e-5, heterogeneity=0.0, z0=2e-4,
                    subsidence=w_ls, rad_cool=rad, moist_adv=madv, tile=True)


class _DryColumn:
    """A dry hydrostatic column of theta(z), tabulated every 10 m, for
    turning relative humidity into mixing ratio when a case is built."""

    def __init__(self, theta_fn, psfc: float, ztop: float):
        self.z = np.arange(0.0, ztop + 20.0, 10.0)
        pi = np.empty_like(self.z)
        pi[0] = (psfc / P00) ** (RD / CP)
        for k in range(1, len(self.z)):
            pi[k] = pi[k - 1] - G * 10.0 / (CP * theta_fn(self.z[k] - 5.0))
        self.pi = pi
        self.theta_fn = theta_fn

    def p(self, z):
        return P00 * float(np.interp(z, self.z, self.pi)) ** (CP / RD)

    def t(self, z):
        return self.theta_fn(z) * float(np.interp(z, self.z, self.pi))

    def qsat(self, z):
        return float(qsat_liq(np.float64(self.t(z)), self.p(z)))


def scenario_stratocumulus(size: str = "standard") -> Scenario:
    """DYCOMS-II RF01 (Stevens et al. 2005, Mon. Wea. Rev. 133, 1443), as
    specified for the GCSS intercomparison and coded in MicroHH: nocturnal
    marine stratocumulus under an 8.5 K inversion at 840 m, kept alive by
    the long-wave cooling of its own top."""
    grid = make_grid("layer", size)
    zi = 840.0

    def thl(z):
        return 289.0 if z <= zi else 297.5 + (z - zi) ** (1.0 / 3.0)

    def qt(z):
        q = 9.0e-3 if z <= zi else 1.5e-3          # specific humidity
        return q / (1.0 - q)

    base, qc, _ = BaseState.from_liquid(grid, thl, qt, lambda z: 7.0, lambda z: -5.5, 101780.0)
    rho_s = float(base.rho0[0])
    d_ls = 3.75e-6
    return Scenario(
        "stratocumulus", "Stratocumulus (DYCOMS-II RF01)",
        "Marine stratocumulus under an 840 m inversion: the cloud's top radiates "
        "to space, the chilled air sinks in sheets and the layer overturns into "
        "cells about 1.5 times as wide as it is deep.",
        grid, base, micro_ice=False, random_th=0.1, random_qv=2.5e-5,
        random_depth=795.0, init_qc=qc,
        flux_mode="fixed", fixed_wth=15.0 / (rho_s * CP),
        fixed_wqv=115.0 / (rho_s * latent_vap(292.5)), heterogeneity=0.0,
        cd=0.0011, z0=2e-4, coriolis=7.62e-5,
        lw_f0=70.0, lw_f1=22.0, lw_kappa=85.0, lw_div=d_ls, lw_zi_qt=8.0e-3, lw_rhoi=1.13,
        subsidence=lambda z: -d_ls * z, translate=(7.0, -5.5), sponge_depth=250.0,
        tile=True, layer_depth=840.0,
        shows=[("Stratocumulus stratiformis",
                "cloud-top long-wave cooling (70 W/m2) drives cellular overturning "
                "of the 840 m boundary layer")])


def _layer_case(grid: Grid, psfc: float, th_below, z_b: float, z_i: float, z_cb: float,
                jump: float, lapse_above: float, rh_below: float, rh_above: float,
                wind_u, wind_v, ice: bool):
    """A well-mixed, cloud-topped layer between z_b and z_i (theta_l and q_t
    constant, q_t set so the cloud base is at z_cb) with a theta jump at its
    top - the structure of every layer cloud, stratocumulus or
    altocumulus.  Returns (base, qc, qi, theta_l(z), q_t(z))."""
    th_layer = th_below(z_b)

    def thl(z):
        if z < z_b:
            return th_below(z)
        if z <= z_i:
            return th_layer
        return th_layer + jump + lapse_above * (z - z_i)

    col = _DryColumn(thl, psfc, grid.top + 200.0)
    q_layer = float(qsat_liq(np.float64(th_layer * float(np.interp(z_cb, col.z, col.pi))),
                             col.p(z_cb)))

    def qt(z):
        if z < z_b:
            return rh_below * col.qsat(z)
        if z <= z_i:
            return q_layer
        return rh_above * col.qsat(z)

    base, qc, qi = BaseState.from_liquid(grid, thl, qt, wind_u, wind_v, psfc, ice=ice)
    return base, qc, qi, thl, qt


def scenario_altocumulus(size: str = "standard", shear: float = 0.0,
                         kh: bool = False) -> Scenario:
    """Altocumulus: a 400 m moist layer at 3.0-3.4 km, cloudy from 3.15 km,
    under a 4 K inversion with 45% humidity above it, radiating from its top (this program's
    construction on the pattern of DYCOMS-II; long-wave fluxes estimated for
    a +3 C cloud top under a dry mid-troposphere: F0 90, F1 30 W/m2).
    shear (1/s) across the layer lines the cells up into rows along it -
    the radiatus variety."""
    grid = make_grid("layer", size, z0=2700.0)

    def th_below(z):
        return 300.0 + 3.5e-3 * z

    def u_fn(z):
        if kh:
            # a shear layer 100 m thick at the cloud top: Ri ~ 0.2 across the
            # 4 K inversion, below the 0.25 of Kelvin-Helmholtz instability
            return 10.0 + 4.0 * math.tanh((z - 3400.0) / 50.0)
        return 10.0 + shear * min(max(z - 3200.0, -250.0), 250.0)

    base, qc, qi, _, _ = _layer_case(grid, 101300.0, th_below, 3000.0, 3400.0, 3080.0,
                                     4.0, 3.5e-3, 0.55, 0.45, u_fn, lambda z: 0.0, ice=True)
    rad = shear >= 0.005 and not kh
    name = "Altocumulus stratiformis" + (" radiatus" if rad else "") + (" undulatus" if kh else "")
    why = ("cloud-top cooling makes the layer overturn in cells ~2 layer depths "
           "across" + ("; the shear through the layer rolls the cells into rows along "
                       "the wind, which perspective draws converging on the horizon"
                       if rad else "")
           + ("; Kelvin-Helmholtz waves on the sheared inversion ripple its top into "
              "bands across the wind, ~7 shear-layer depths apart" if kh else ""))
    return Scenario(
        "altocumulus" + ("_ra" if rad else "") + ("_un" if kh else ""), name,
        "A thin cloud layer at 3 km overturning under its own radiative cooling"
        + (", rolled into bands by the shear" if rad else "")
        + (", its top rippled by Kelvin-Helmholtz waves" if kh else "") + ".",
        grid, base, micro_ice=True, graupel=False, random_th=0.1, random_qv=2.5e-5,
        random_zmin=3000.0, random_depth=3400.0, init_qc=qc, init_qi=qi,
        flux_mode="none", drag=False, coriolis=1.0e-4,
        lw_f0=80.0, lw_f1=20.0, lw_kappa=85.0, translate=(10.0, 0.0),
        nudge_tau=900.0, nudge_zmin=2900.0, nudge_zmax=3600.0, nudge_uv=kh,
        sponge_depth=250.0, tile=True, layer_depth=400.0, shows=[(name, why)])


def scenario_streets(size: str = "standard") -> Scenario:
    """Cloud streets: a cold-air outbreak over a warmer sea (this program's
    construction; the setting of the Black Sea and Great Lakes streets).
    Air at 0 C flows at 14 m/s over water at 8 C; the heated boundary layer
    organises into horizontal roll vortices along the wind, 2-3 boundary-
    layer depths apart, and cloud forms over their rising branches."""
    grid = make_grid("bl", size)
    zi0 = 700.0

    def thl(z):
        return 272.0 if z <= zi0 else 275.0 + 6.0e-3 * (z - zi0)

    def qt(z):
        return 2.35e-3 if z <= zi0 else 1.4e-3 * math.exp(-(z - zi0) / 3000.0)

    base, qc, qi = BaseState.from_liquid(grid, thl, qt, lambda z: 14.0, lambda z: 0.0,
                                         101500.0, ice=True)
    return Scenario(
        "streets", "Cloud streets (cold-air outbreak)",
        "Cold air pouring over warm water: roll vortices along the wind lift "
        "cloud in parallel lines 2-3 boundary-layer depths apart.",
        grid, base, micro_ice=True, graupel=False, random_th=0.2, random_qv=2.0e-5, random_depth=500.0,
        init_qc=qc, init_qi=qi, flux_mode="bulk", sst=281.5, z0=2e-4, heterogeneity=0.0,
        coriolis=1.0e-4, lw_f0=70.0, lw_f1=22.0, lw_kappa=85.0, translate=(12.0, 0.0),
        sponge_depth=400.0, tile=True,
        shows=[("Cumulus mediocris radiatus (cloud streets)",
                "surface heating in a strong, sheared wind organises the boundary "
                "layer into roll vortices along the wind; cloud forms over their rising "
                "branches, in lines 2-3 boundary-layer depths apart")])


def scenario_asperitas(size: str = "standard", du: float = 3.0, veer: float = 60.0) -> Scenario:
    """Asperitas: a thick stratocumulus layer (1.6-2.3 km, capped by a 5 K
    inversion) resting on a moist, stable, sheared layer.  Across the 200 m
    under the cloud base the air warms by 1 K (N ~ 0.009 1/s) while the wind
    changes by 2 du (6 m/s) and turns through `veer` degrees: the Richardson
    number is ~0.2, below the 0.25 at which the layer breaks into
    Kelvin-Helmholtz waves.  Its air is nearly saturated (85% relative
    humidity rising to 99% at the base), so wherever a wave lifts it by
    some tens of metres it becomes cloud: the waves are drawn into the
    underside of the layer.  Because the wind turns with height the waves
    run in several directions at once - the chaotic, 'rough sea seen from
    below' look that distinguishes asperitas from undulatus (Harrison et
    al. 2017: asperitas with shear and gravity waves at the base).  Drizzle
    (autoconversion from 0.5 g/kg) evaporating below the base chills the
    air there and makes the troughs sag (the settling-evaporation
    instability of Ravichandran & Govindarajan 2020, 2022).  The layer's
    mean temperature, moisture and wind are held to these profiles (15-min
    relaxation), as the large-scale flow that made them would.  This
    program's construction; the waves are the model's own."""
    grid = make_grid("wavebase", size, z0=900.0)
    z_if, d_if, dth_if = 1500.0, 100.0, 1.0
    z_cb, z_i, jump = 1600.0, 2300.0, 5.0
    th_sub = 290.0
    th_top = th_sub + dth_if

    def thl(z):
        if z < z_if - d_if:
            return th_sub
        if z < z_if + d_if:
            return th_sub + dth_if * (z - (z_if - d_if)) / (2.0 * d_if)
        if z <= z_i:
            return th_top
        return th_top + jump + 5.0e-3 * (z - z_i)

    col = _DryColumn(thl, 101300.0, grid.top + 200.0)
    q_layer = float(qsat_liq(np.float64(th_top * float(np.interp(z_cb, col.z, col.pi))),
                             col.p(z_cb)))

    def qt(z):
        if z < z_if - d_if:
            return 0.85 * col.qsat(z)
        if z < z_if + d_if:
            f = (z - (z_if - d_if)) / (2.0 * d_if)
            return min((0.85 + 0.14 * f) * col.qsat(z), q_layer)
        if z <= z_i:
            return q_layer
        return 0.30 * col.qsat(z)

    def _wind(z):
        t = math.tanh((z - z_if) / d_if)
        a = math.radians(veer) * 0.5 * t
        sp = 8.0 + du * t
        return sp * math.cos(a), sp * math.sin(a)

    base, qc, qi = BaseState.from_liquid(grid, thl, qt, lambda z: _wind(z)[0],
                                         lambda z: _wind(z)[1], 101300.0, ice=False)
    name = "Stratocumulus stratiformis opacus asperitas"
    return Scenario(
        "asperitas", name,
        "Kelvin-Helmholtz waves in the moist, sheared stable layer under a thick "
        "cloud layer, running in several directions as the wind turns, are drawn "
        "into its base; evaporating drizzle makes the troughs sag.",
        grid, base, micro_ice=False, graupel=False, random_th=0.1, random_qv=2.5e-5,
        random_zmin=z_if - d_if, random_depth=z_i, init_qc=qc, flux_mode="none",
        drag=False, coriolis=1.0e-4, lw_f0=50.0, lw_f1=10.0, lw_kappa=85.0, q_auto=0.5e-3,
        nudge_tau=900.0, nudge_zmin=z_if - 2.0 * d_if, nudge_zmax=z_i + 150.0, nudge_uv=True,
        translate=_wind(z_cb), sponge_depth=150.0, tile=True, layer_depth=z_i - z_cb,
        shows=[(name, "Kelvin-Helmholtz waves in the sheared, nearly saturated stable layer "
                      "under the base, crossing as the wind turns, drawn into the cloud; "
                      "drizzle evaporating below makes the troughs sag")],
        reasons=[f"under the base the air warms {dth_if:.0f} K and the wind changes "
                 f"{2 * du:.0f} m/s and turns {veer:.0f} deg across 200 m: Richardson number "
                 f"~0.2, below the 0.25 at which a sheared layer overturns in waves",
                 "that layer is 85-99% saturated, so a wave crest a few tens of metres high "
                 "becomes cloud: the waves are printed on the base",
                 "the layer's mean temperature, moisture and wind are held to these profiles "
                 "(15-min relaxation); its waves are the model's own"])


def _moist_bands(east_km: float, north_km: float) -> float:
    """The extent of a reference case's layer around the observer (cover %):
    mid-level moisture drawn out along the wind into bands 44 km apart and
    about 30 km wide, the observer under the northern edge of one - this
    program's composition, so that the layer has edges to see its trails
    against clear sky."""
    return 80.0 * min(max(0.5 + 0.9 * math.cos(2.0 * math.pi * (north_km + 11.0) / 44.0),
                          0.0), 1.0)


QI_AUTO_LAYER = 1.0e-5     # kg/kg: cloud ice -> snow in thin mixed-phase layers


def scenario_virga(size: str = "standard", shear: float = 0.006, rh_below: float = 0.65,
                   qi_auto: float = QI_AUTO_LAYER, z_b: float = 5400.0, z_i: float = 6100.0,
                   z_cb: float = 5700.0, jump: float = 2.0) -> Scenario:
    """Virga: a mixed-phase altocumulus layer (z_b-z_i, cloudy from z_cb,
    -12 to -20 C) whose ice grows into snow that falls into the drier air
    below (65% relative humidity) and sublimates before it can reach the
    ground; the wind, slowing by `shear` per metre downward, bends the
    trails back.  The cloud-ice threshold for snow is lowered from SAM's
    0.1 g/kg to 0.01 g/kg: in a thin mixed-phase layer the crystals are few
    and grow to falling size within minutes (this program's choice, as the
    drizzle threshold of the asperitas case is), and the frozen
    precipitation is all snow (no graupel in updraughts of 1-3 m/s).  This
    program's construction."""
    probe = make_grid("fall", size)
    grid = make_grid("fall", size, z0=z_i + 500.0 - probe.nz * probe.dz)

    def th_below(z):
        return 300.0 + 3.3e-3 * z

    zmid = 0.5 * (z_b + z_i)

    def u_fn(z):
        return max(4.0, 12.0 + shear * (z - zmid))

    base, qc, qi, _, _ = _layer_case(grid, 101300.0, th_below, z_b, z_i, z_cb,
                                     jump, 3.3e-3, rh_below, 0.25, u_fn, lambda z: 0.0, ice=True)
    return Scenario(
        "virga", "Altocumulus with virga",
        "Snow from a mixed-phase altocumulus layer sublimating in the dry air "
        "below; the wind shear bends the trails.",
        grid, base, micro_ice=True, graupel=False, random_th=0.1, random_qv=2.5e-5,
        random_zmin=z_b, random_depth=z_i, init_qc=qc, init_qi=qi,
        flux_mode="none", drag=False, coriolis=1.0e-4, lw_f0=90.0, lw_f1=30.0,
        lw_kappa=85.0, translate=(u_fn(zmid), 0.0), sponge_depth=300.0, tile=True,
        nudge_tau=900.0, nudge_zmin=z_b - 100.0, nudge_zmax=z_i + 200.0,
        layer_depth=z_i - z_b, qi_auto=qi_auto, demo_cover=_moist_bands,
        shows=[("Altocumulus virga", "ice grows at the expense of the droplets, "
                "falls as snow and sublimates in the dry air below")])


SCENARIOS = ("weather", "supercell", "squall", "bomex", "stratocumulus", "altocumulus",
             "altocumulus_ra", "altocumulus_un", "streets", "asperitas", "virga")


def scenario_by_key(key: str, size: str = "standard", snd=None) -> Scenario:
    """The named reference case."""
    if key == "weather":
        return scenario_weather(snd, size)
    if key == "supercell":
        return scenario_supercell(size)
    if key == "bomex":
        return scenario_bomex(size)
    if key == "stratocumulus":
        return scenario_stratocumulus(size)
    if key == "altocumulus":
        return scenario_altocumulus(size, shear=0.0)
    if key == "altocumulus_ra":
        return scenario_altocumulus(size, shear=0.008)
    if key == "altocumulus_un":
        return scenario_altocumulus(size, kh=True)
    if key == "streets":
        return scenario_streets(size)
    if key == "asperitas":
        return scenario_asperitas(size)
    if key == "virga":
        return scenario_virga(size)
    if key == "squall":
        return scenario_squall(size)
    raise KeyError(key)


# ------------------------------------------------------- advection helpers --
def _flux5(am3, am2, am1, a0, ap1, ap2, vel):
    """Wicker & Skamarock fifth-order upwind flux at the interface between
    a[m-1] and a[m] (a0 = a[m])."""
    return (vel * (37.0 * (a0 + am1) - 8.0 * (ap1 + am2) + (ap2 + am3))
            - np.abs(vel) * ((ap2 - am3) - 5.0 * (ap1 - am2) + 10.0 * (a0 - am1))) / 60.0


def _flux3(am2, am1, a0, ap1, vel):
    return (vel * (7.0 * (a0 + am1) - (ap1 + am2))
            + np.abs(vel) * ((ap1 - am2) - 3.0 * (a0 - am1))) / 12.0


def flux_periodic(a, vel, axis):
    """Flux at interface m (between a[m-1] and a[m]), stored at index m."""
    r = np.roll
    return _flux5(r(a, 3, axis), r(a, 2, axis), r(a, 1, axis), a,
                  r(a, -1, axis), r(a, -2, axis), vel)


def flux_vertical(a, vel):
    """a has N entries along axis 0; vel has N+1 (interfaces 0..N).  The
    flux through interfaces 0 and N is zero; order drops to 3 and then 2
    where the stencil would leave the column."""
    n = a.shape[0]
    f = np.zeros_like(vel)
    if n >= 6:
        f[3:n - 2] = _flux5(a[0:n - 5], a[1:n - 4], a[2:n - 3], a[3:n - 2],
                            a[4:n - 1], a[5:n], vel[3:n - 2])
    for m in (2, n - 2):
        if 2 <= m <= n - 2:
            f[m] = _flux3(a[m - 2], a[m - 1], a[m], a[m + 1], vel[m])
    for m in (1, n - 1):
        if 1 <= m <= n - 1:
            f[m] = vel[m] * 0.5 * (a[m] + a[m - 1])
    if n < 6:
        for m in range(1, n):
            f[m] = vel[m] * 0.5 * (a[m] + a[m - 1])
    return f


# ----------------------------------------------------- microphysics rates --
def rain_fall_speed(qr, rho):
    return np.where(qr > 1e-12, VR_COEF * np.maximum(rho * qr, 0.0) ** VR_EXP
                    * np.sqrt(1.15 / rho), 0.0)


def ice_precip_fall_speed(qs, rho, t, graupel=1.0):
    fg = graupel_fraction(t) * graupel
    rq = np.maximum(rho * qs, 0.0)
    vs = VS_COEF * rq ** (B_S / 4.0)
    vg = VG_COEF * rq ** (B_G / 4.0)
    return np.where(qs > 1e-12, (fg * vg + (1.0 - fg) * vs) * np.sqrt(RHO_REF_ICE / rho), 0.0)


def lambda_mp(rho_q, rho_x, n0):
    return (math.pi * rho_x * n0 / np.maximum(rho_q, 1e-14)) ** 0.25


def extinction_mp(rho_q, rho_x, n0):
    """Geometric-optics extinction (Q = 2) of a Marshall-Palmer spectrum of
    spheres: sigma = pi N0 / lambda^3, m^-1."""
    lam = lambda_mp(rho_q, rho_x, n0)
    return np.where(rho_q > 1e-12, math.pi * n0 / lam ** 3, 0.0)


# ------------------------------------------------------------- the model ----
class CPUModel:
    """The reference implementation.  Fields are NumPy arrays indexed
    [k, j, i] (z, y, x); w has nz+1 levels."""

    FIELDS = ("u", "v", "w", "th", "qv", "qc", "qr", "qi", "qs")
    SCALARS = ("th", "qv", "qc", "qr", "qi", "qs")

    def __init__(self, sc: Scenario, dtype="f8"):
        self.sc = sc
        self.g = sc.grid
        self.b = sc.base
        self.dtype = np.dtype(dtype)
        self._setup_constants()
        self.reset()

    # ---------------------------------------------------------- setup ----
    def _setup_constants(self):
        g, b = self.g, self.b
        dt = self.dtype
        sh = (g.nz, 1, 1)
        self.rho0 = b.rho0.reshape(sh).astype(dt)
        self.rhof = b.rhof.reshape(g.nz + 1, 1, 1).astype(dt)
        self.th0 = b.th0.reshape(sh).astype(dt)
        self.qv0 = b.qv0.reshape(sh).astype(dt)
        self.pi0 = b.pi0.reshape(sh).astype(dt)
        self.p0 = b.p0.reshape(sh).astype(dt)
        um, vm = self.sc.translate
        self.u0 = (b.u0 - um).reshape(sh).astype(dt)          # model-frame base wind
        self.v0 = (b.v0 - vm).reshape(sh).astype(dt)
        # geostrophic wind in the model frame (the Coriolis force acts on the
        # ground-relative wind; the frame only translates, it does not turn)
        sc_ = self.sc
        ugp = np.array([sc_.ug_fn(z) for z in g.zc]) if sc_.ug_fn is not None else b.u0
        vgp = np.array([sc_.vg_fn(z) for z in g.zc]) if sc_.vg_fn is not None else b.v0
        self.ug0 = (np.asarray(ugp, "f8") - um).reshape(sh).astype(dt)
        self.vg0 = (np.asarray(vgp, "f8") - vm).reshape(sh).astype(dt)
        # surface drag: only a domain that touches the ground has any
        if sc_.drag and g.z0 <= 0.0:
            zc0 = 0.5 * g.dz
            self.cd = sc_.cd if sc_.cd > 0.0 else (0.4 / math.log((zc0 + sc_.z0) / sc_.z0)) ** 2
        else:
            self.cd = 0.0
        # Rayleigh damping (CM1): tau(z) = 1/2 [1 - cos(pi (z-zd)/(top-zd))]
        zc, zf = g.zc, g.zf
        zd = g.top - self.sc.sponge_depth
        rd = 1.0 / 300.0

        def prof(z):
            if self.sc.sponge_depth <= 0.0:
                return np.zeros_like(z)
            return np.where(z > zd, 0.5 * (1.0 - np.cos(np.pi * (z - zd) / (g.top - zd))), 0.0) * rd
        self.damp_c = prof(zc).reshape(sh).astype(dt)
        self.damp_f = prof(zf).reshape(g.nz + 1, 1, 1).astype(dt)
        # lateral sponge: relax to the reference within 6 points of the edges
        if self.sc.lateral_sponge:
            ii = np.arange(g.nx)
            jj = np.arange(g.ny)
            big = 1 << 20
            dx_edge = np.minimum(ii, g.nx - 1 - ii) if self.sc.sponge_axes & 1 else np.full(g.nx, big)
            dy_edge = np.minimum(jj, g.ny - 1 - jj) if self.sc.sponge_axes & 2 else np.full(g.ny, big)
            d = np.minimum(dx_edge[None, :], dy_edge[:, None]).astype("f8")
            lat = np.where(d < 6, (1.0 - d / 6.0) ** 2, 0.0) / 120.0
            self.lat_sponge = lat[None].astype(dt)
        else:
            self.lat_sponge = None
        # horizontal eigenvalues of the discrete Laplacian (rfft layout)
        kx = np.arange(g.nx // 2 + 1)
        ky = np.arange(g.ny)
        lam = (2.0 * (np.cos(2 * np.pi * kx[None, :] / g.nx) - 1.0) / g.dx ** 2
               + 2.0 * (np.cos(2 * np.pi * ky[:, None] / g.ny) - 1.0) / g.dy ** 2)
        self.lam = lam
        r0 = b.rho0
        rf = b.rhof
        dz2 = g.dz ** 2
        nz = g.nz
        self.pa = np.zeros(nz)
        self.pc = np.zeros(nz)
        self.pb_z = np.zeros(nz)
        for k in range(nz):
            if k > 0:
                self.pa[k] = rf[k] / (dz2 * r0[k - 1])
                self.pb_z[k] -= rf[k] / (dz2 * r0[k])
            if k < nz - 1:
                self.pc[k] = rf[k + 1] / (dz2 * r0[k + 1])
                self.pb_z[k] -= rf[k + 1] / (dz2 * r0[k])
        # Advection of the reference potential temperature.  The model
        # carries th' = th - th0(z), so rising air must lose th' at the rate
        # w dth0/dz.  It is done with the SAME fifth-order flux operator as
        # everything else, applied to th0 alone: because the operator is
        # linear in the advected field this is exactly the advection of
        # the full th, but written as fluxes of (th0 - th0[k]) - numbers of
        # a kelvin, not hundreds - so single precision on the GPU loses
        # nothing to cancellation.  Per interface m (between levels m-1
        # and m):  F = M (c[m] - th0[ref]) - |M| d[m], with c the centred
        # interpolation and d the upwind correction of the stencil used
        # there.
        th0v = b.th0
        cz = np.zeros(nz + 1)
        dzc = np.zeros(nz + 1)
        for mm in range(1, nz):
            if 3 <= mm <= nz - 3:
                a = th0v[mm - 3:mm + 3]
                cz[mm] = (37 * (a[3] + a[2]) - 8 * (a[4] + a[1]) + (a[5] + a[0])) / 60.0
                dzc[mm] = ((a[5] - a[0]) - 5 * (a[4] - a[1]) + 10 * (a[3] - a[2])) / 60.0
            elif 2 <= mm <= nz - 2:
                a = th0v[mm - 2:mm + 2]
                cz[mm] = (7 * (a[2] + a[1]) - (a[3] + a[0])) / 12.0
                dzc[mm] = -((a[3] - a[0]) - 3 * (a[2] - a[1])) / 12.0
            else:
                cz[mm] = 0.5 * (th0v[mm] + th0v[mm - 1])
                dzc[mm] = 0.0
        self.th0_alo = np.zeros(nz + 1)      # relative to the level below
        self.th0_ahi = np.zeros(nz + 1)      # relative to the level above
        self.th0_alo[1:nz] = cz[1:nz] - th0v[0:nz - 1]
        self.th0_ahi[1:nz] = cz[1:nz] - th0v[1:nz]
        self.th0_d = dzc
        # surface heterogeneity map (fixed to the ground)
        rng = np.random.default_rng(self.sc.seed + 17)
        # nudging targets: the initial theta_l and q_t profiles.  theta_l
        # uses the latent heats at the reference temperature of each level,
        # the same on the CPU and the GPU.
        sc_ = self.sc
        self.lv_ref = latent_vap(b.t0)
        self.ls_ref = latent_sub(b.t0)
        qc0 = np.asarray(sc_.init_qc, "f8") if sc_.init_qc is not None else np.zeros(g.nz)
        qi0 = np.asarray(sc_.init_qi, "f8") if sc_.init_qi is not None else np.zeros(g.nz)
        self.thl_target = b.th0 - (self.lv_ref * qc0 + self.ls_ref * qi0) / (CP * b.pi0)
        self.qt_target = b.qv0 + qc0 + qi0
        self.nudge_mask = ((g.zc >= sc_.nudge_zmin) & (g.zc <= sc_.nudge_zmax)).astype("f8")
        self.het = self._smooth_noise(rng, g.ny, g.nx, 3000.0 / g.dx) \
            if self.sc.heterogeneity > 0 else np.zeros((g.ny, g.nx))

    @staticmethod
    def _smooth_noise(rng, ny, nx, scale_cells):
        f = rng.standard_normal((ny, nx))
        fk = np.fft.rfft2(f)
        ky = np.fft.fftfreq(ny)[:, None]
        kx = np.fft.rfftfreq(nx)[None, :]
        k2 = kx ** 2 + ky ** 2
        filt = np.exp(-k2 * (np.pi * scale_cells) ** 2)
        out = np.fft.irfft2(fk * filt, s=(ny, nx))
        out -= out.mean()
        out /= max(out.std(), 1e-9)
        return np.clip(out, -2.5, 2.5)

    def reset(self):
        g, sc = self.g, self.sc
        dt = self.dtype
        shp = (g.nz, g.ny, g.nx)
        self.u = np.broadcast_to(self.u0, shp).astype(dt).copy()
        self.v = np.broadcast_to(self.v0, shp).astype(dt).copy()
        self.w = np.zeros((g.nz + 1, g.ny, g.nx), dt)
        self.th = np.zeros(shp, dt)
        self.qv = np.broadcast_to(self.qv0, shp).astype(dt).copy()
        self.qc = np.zeros(shp, dt)
        self.qr = np.zeros(shp, dt)
        self.qi = np.zeros(shp, dt)
        self.qs = np.zeros(shp, dt)
        if sc.init_qc is not None:
            self.qc += np.asarray(sc.init_qc, "f8").reshape(-1, 1, 1).astype(dt)
        if sc.init_qi is not None:
            self.qi += np.asarray(sc.init_qi, "f8").reshape(-1, 1, 1).astype(dt)
        self.rain = np.zeros((g.ny, g.nx), "f8")       # accumulated, kg m^-2
        self.rain_rate = np.zeros((g.ny, g.nx), "f8")  # kg m^-2 s^-1
        self.time = 0.0
        self.steps = 0
        rng = np.random.default_rng(sc.seed)
        zc = g.zc
        x = (np.arange(g.nx) + 0.5) * g.dx
        y = (np.arange(g.ny) + 0.5) * g.dy
        for (bx, by, bz, rh, rz, dth) in sc.bubbles:
            ddx = np.minimum(np.abs(x - bx), g.lx - np.abs(x - bx))
            ddy = np.minimum(np.abs(y - by), g.ly - np.abs(y - by))
            beta = np.sqrt(ddx[None, None, :] ** 2 / rh ** 2 + ddy[None, :, None] ** 2 / rh ** 2
                           + (zc[:, None, None] - bz) ** 2 / rz ** 2)
            self.th += np.where(beta < 1.0, dth * np.cos(0.5 * np.pi * beta) ** 2, 0.0).astype(dt)
        for (bx, bz, rh, rz, dth) in sc.line_bubbles:
            ddx = np.minimum(np.abs(x - bx), g.lx - np.abs(x - bx))
            beta = np.sqrt(ddx[None, None, :] ** 2 / rh ** 2
                           + (zc[:, None, None] - bz) ** 2 / rz ** 2) * np.ones((1, g.ny, 1))
            self.th += np.where(beta < 1.0, dth * np.cos(0.5 * np.pi * beta) ** 2, 0.0).astype(dt)
        if sc.random_th > 0 or sc.random_qv > 0:
            k0 = int(np.searchsorted(zc, sc.random_zmin))
            k1 = int(np.searchsorted(zc, sc.random_depth))
            nlev = k1 - k0
            if nlev > 0:
                self.th[k0:k1] += (sc.random_th * (2 * rng.random((nlev, g.ny, g.nx)) - 1)).astype(dt)
                self.qv[k0:k1] += (sc.random_qv * (2 * rng.random((nlev, g.ny, g.nx)) - 1)).astype(dt)
        # the initial velocity field must satisfy continuity (it does: uniform
        # base wind and w = 0), and the saturation adjustment runs once so a
        # bubble that starts supersaturated starts as cloud
        self._saturation_adjust()

    # ------------------------------------------------------ dynamics ----
    def _divergence(self, u, v, w):
        """div(rho0 u) / rho0 at cell centres."""
        g = self.g
        rf = self.rhof
        return ((np.roll(u, -1, 2) - u) / g.dx + (np.roll(v, -1, 1) - v) / g.dy
                + (rf[1:] * w[1:] - rf[:-1] * w[:-1]) / (self.rho0 * g.dz))

    def _adv_scalar(self, s, u, v, mw, dv):
        g = self.g
        fx = flux_periodic(s, u, 2)
        fy = flux_periodic(s, v, 1)
        fz = flux_vertical(s, mw)
        return (-(np.roll(fx, -1, 2) - fx) / g.dx - (np.roll(fy, -1, 1) - fy) / g.dy
                - (fz[1:] - fz[:-1]) / (self.rho0 * g.dz) + s * dv)

    def tendencies(self, st):
        """Right-hand side of everything but the pressure gradient."""
        g = self.g
        u, v, w = st["u"], st["v"], st["w"]
        rho0, rhof = self.rho0, self.rhof
        mw = rhof * w                                    # mass flux at w levels
        dv = self._divergence(u, v, w)
        out = {}
        # scalars
        for name in self.SCALARS:
            out[name] = self._adv_scalar(st[name], u, v, mw, dv)
        # ... plus the advection of the reference state th0(z)
        alo = self.th0_alo.reshape(-1, 1, 1)
        ahi = self.th0_ahi.reshape(-1, 1, 1)
        dd = self.th0_d.reshape(-1, 1, 1)
        flo = mw * alo - np.abs(mw) * dd          # interface fluxes of th0 - th0[m-1]
        fhi = mw * ahi - np.abs(mw) * dd          # ... of th0 - th0[m]
        out["th"] -= (flo[1:] - fhi[:-1]) / (rho0 * g.dz)
        # u momentum (x faces)
        uc = 0.5 * (u + np.roll(u, 1, 2))                 # at interface m = centre m-1
        fx = flux_periodic(u, uc, 2)
        vy = 0.5 * (v + np.roll(v, 1, 2))                 # v at (x face i, y face j)
        fy = flux_periodic(u, vy, 1)
        mz = 0.5 * (mw + np.roll(mw, 1, 2))
        fz = flux_vertical(u, mz)
        du = (-(np.roll(fx, -1, 2) - fx) / g.dx - (np.roll(fy, -1, 1) - fy) / g.dy
              - (fz[1:] - fz[:-1]) / (rho0 * g.dz) + u * 0.5 * (dv + np.roll(dv, 1, 2)))
        # v momentum (y faces)
        ux = 0.5 * (u + np.roll(u, 1, 1))
        fx = flux_periodic(v, ux, 2)
        vc = 0.5 * (v + np.roll(v, 1, 1))
        fy = flux_periodic(v, vc, 1)
        mz = 0.5 * (mw + np.roll(mw, 1, 1))
        fz = flux_vertical(v, mz)
        dvv = (-(np.roll(fx, -1, 2) - fx) / g.dx - (np.roll(fy, -1, 1) - fy) / g.dy
               - (fz[1:] - fz[:-1]) / (rho0 * g.dz) + v * 0.5 * (dv + np.roll(dv, 1, 1)))
        # w momentum (z faces 1..nz-1)
        ru = rho0 * u
        rv_ = rho0 * v
        mxu = np.zeros_like(w)
        mxu[1:-1] = 0.5 * (ru[1:] + ru[:-1])
        myv = np.zeros_like(w)
        myv[1:-1] = 0.5 * (rv_[1:] + rv_[:-1])
        fx = flux_periodic(w, mxu, 2)
        fy = flux_periodic(w, myv, 1)
        wc = 0.5 * (mw[1:] + mw[:-1])                     # mass flux at centres
        vint = np.zeros((g.nz + 2,) + w.shape[1:], w.dtype)
        vint[1:-1] = wc
        fz = flux_vertical(w, vint)                       # shape nz+2
        dw = np.zeros_like(w)
        rfi = rhof[1:-1]
        dvf = 0.5 * (dv[1:] + dv[:-1])
        dw[1:-1] = (-(np.roll(fx, -1, 2) - fx)[1:-1] / (g.dx * rfi)
                    - (np.roll(fy, -1, 1) - fy)[1:-1] / (g.dy * rfi)
                    - (fz[2:-1] - fz[1:-2]) / (g.dz * rfi) + w[1:-1] * dvf)
        # buoyancy (CM1 eq. 8), at scalar points then averaged to w levels
        cond = st["qc"] + st["qr"] + st["qi"] + st["qs"]
        bc = G * (st["th"] / self.th0 + REPSM1 * (st["qv"] - self.qv0) - cond)
        dw[1:-1] += 0.5 * (bc[1:] + bc[:-1])
        # Rayleigh damping near the lid
        du -= self.damp_c * (u - self.u0)
        dvv -= self.damp_c * (v - self.v0)
        dw -= self.damp_f * w
        out["th"] -= self.damp_c * st["th"]
        if self.lat_sponge is not None:
            ls = self.lat_sponge
            du -= ls * (u - self.u0)
            dvv -= ls * (v - self.v0)
            dw[1:-1] -= ls * w[1:-1]
            out["th"] -= ls * st["th"]
            out["qv"] -= ls * (st["qv"] - self.qv0)
            for q in ("qc", "qr", "qi", "qs"):
                out[q] -= ls * st[q]
        # Coriolis force on the departure from the geostrophic wind, with
        # each velocity averaged to the other's points on the C grid
        f = self.sc.coriolis
        if f != 0.0:
            v_at_u = 0.25 * (v + np.roll(v, 1, 2) + np.roll(v, -1, 1)
                             + np.roll(np.roll(v, -1, 1), 1, 2))
            u_at_v = 0.25 * (u + np.roll(u, -1, 2) + np.roll(u, 1, 1)
                             + np.roll(np.roll(u, -1, 2), 1, 1))
            du += f * (v_at_u - self.vg0)
            dvv -= f * (u_at_v - self.ug0)
        # surface drag on the lowest level (ground-relative wind)
        if self.cd > 0.0:
            um, vm = self.sc.translate
            ug = u[0] + um
            vg = v[0] + vm
            spd = np.sqrt(ug ** 2 + vg ** 2 + 1.0)
            du[0] -= self.cd * spd * ug / g.dz
            dvv[0] -= self.cd * spd * vg / g.dz
        out["u"] = du
        out["v"] = dvv
        out["w"] = dw
        return out

    def project(self, u, v, w, dts):
        """Remove the divergent part of (u, v, w) so that div(rho0 u) = 0,
        by solving the anelastic Poisson equation exactly (FFT + Thomas)."""
        g = self.g
        rf = self.rhof
        rhs = (self.rho0 * ((np.roll(u, -1, 2) - u) / g.dx + (np.roll(v, -1, 1) - v) / g.dy)
               + (rf[1:] * w[1:] - rf[:-1] * w[:-1]) / g.dz) / dts
        rh = np.fft.rfft2(rhs, axes=(1, 2))
        nz = g.nz
        pa, pc = self.pa, self.pc
        b = self.pb_z[:, None, None] + self.lam[None]
        # (0,0) mode is singular: pin its bottom value (CM1 does the same)
        b00 = b[:, 0, 0].copy()
        cprime = np.zeros(b.shape)
        dprime = np.zeros(rh.shape, complex)
        bb = b.copy()
        rr = rh.copy()
        bb[0, 0, 0] = 1.0
        rr[0, 0, 0] = 0.0
        c0 = np.full(b.shape[1:], pc[0])
        c0[0, 0] = 0.0
        cprime[0] = c0 / bb[0]
        dprime[0] = rr[0] / bb[0]
        for k in range(1, nz):
            ak = pa[k]
            m = bb[k] - ak * cprime[k - 1]
            ck = pc[k]
            cprime[k] = ck / m
            dprime[k] = (rr[k] - ak * dprime[k - 1]) / m
        x = np.zeros_like(dprime)
        x[-1] = dprime[-1]
        for k in range(nz - 2, -1, -1):
            x[k] = dprime[k] - cprime[k] * x[k + 1]
        pp = np.fft.irfft2(x, s=(g.ny, g.nx), axes=(1, 2))
        phi = pp / self.rho0
        u -= dts * (phi - np.roll(phi, 1, 2)) / g.dx
        v -= dts * (phi - np.roll(phi, 1, 1)) / g.dy
        w[1:-1] -= dts * (phi[1:] - phi[:-1]) / g.dz
        w[0] = 0.0
        w[-1] = 0.0
        self.pprime = pp
        return u, v, w

    def state(self):
        return {n: getattr(self, n) for n in self.FIELDS}

    def step(self, dt: float):
        """One Wicker-Skamarock RK3 step of dynamics, then microphysics and
        surface fluxes."""
        s0 = {n: getattr(self, n) for n in self.FIELDS}
        cur = s0
        for frac in (1.0 / 3.0, 0.5, 1.0):
            tend = self.tendencies(cur)
            dts = dt * frac
            new = {n: s0[n] + dts * tend[n] for n in self.FIELDS}
            new["w"][0] = 0.0
            new["w"][-1] = 0.0
            self.project(new["u"], new["v"], new["w"], dts)
            cur = new
        for n in self.FIELDS:
            setattr(self, n, cur[n].astype(self.dtype, copy=False))
        self._forcing(dt)
        self._microphysics(dt)
        self._surface(dt)
        self.time += dt
        self.steps += 1

    # ---------------------------------------------------- time step -------
    def max_dt(self, courant: float = 0.9, dt_max: float | None = None) -> float:
        """The advective Courant limit, and never more than 10 s, so that
        the once-per-step microphysics stays accurate (this program's
        choice; the sedimentation sub-steps itself)."""
        g = self.g
        cu = np.abs(self.u).max() / g.dx + np.abs(self.v).max() / g.dy \
            + np.abs(self.w).max() / g.dz
        dtl = 10.0
        if dt_max is not None:
            dtl = min(dtl, dt_max)
        return float(min(courant / max(cu, 1e-6), dtl))

    # ---------------------------------------------------- forcing ---------
    def lw_heating(self):
        """Long-wave radiative heating, K/s, of the DYCOMS-II
        parameterisation (Stevens et al. 2005) - the cooling that drives a
        stratocumulus or altocumulus layer's convection from its top."""
        sc, g = self.sc, self.g
        rho = self.b.rho0.reshape(-1, 1, 1)
        ql = np.maximum(self.qc.astype("f8"), 0.0) + LW_ICE_RATIO * np.maximum(self.qi.astype("f8"), 0.0)
        dq = sc.lw_kappa * rho * ql * g.dz                  # optical depth of each cell
        qa = np.zeros((g.nz + 1, g.ny, g.nx))
        qa[:-1] = np.cumsum(dq[::-1], axis=0)[::-1]          # Q(z, inf) at interfaces
        qb = np.zeros_like(qa)
        qb[1:] = np.cumsum(dq, axis=0)                       # Q(0, z)
        flux = sc.lw_f0 * np.exp(-qa) + sc.lw_f1 * np.exp(-qb)
        if sc.lw_div > 0.0:
            zi = self.inversion_height()
            zf = g.zf.reshape(-1, 1, 1)
            d = np.maximum(zf - zi[None], 0.0)
            flux = flux + np.where(zf > zi[None],
                                   sc.lw_rhoi * CP * sc.lw_div
                                   * (d ** (4.0 / 3.0) / 4.0 + zi[None] * d ** (1.0 / 3.0)), 0.0)
        return -(flux[1:] - flux[:-1]) / (rho * CP * g.dz)

    def mean_thl_qt(self):
        """Horizontal means of theta_l and q_t on every level."""
        pi0 = self.b.pi0
        thl = (self.b.th0.reshape(-1, 1, 1) + self.th.astype("f8")
               - (self.lv_ref.reshape(-1, 1, 1) * self.qc.astype("f8")
                  + self.ls_ref.reshape(-1, 1, 1) * self.qi.astype("f8"))
               / (CP * pi0.reshape(-1, 1, 1)))
        qt = (self.qv + self.qc + self.qi).astype("f8")
        return thl.mean(axis=(1, 2)), qt.mean(axis=(1, 2))

    def inversion_height(self):
        """zi of each column: where total water first falls through
        lw_zi_qt going up (DYCOMS-II's 8 g/kg isoline), interpolated
        between levels; +inf where it never does."""
        sc, g = self.sc, self.g
        qt = (self.qv + self.qc + self.qi).astype("f8")
        below = qt < sc.lw_zi_qt
        k = np.argmax(below, axis=0)                          # first level below
        found = below.max(axis=0) & (k > 0)
        kk = np.clip(k, 1, g.nz - 1)
        zc = g.zc
        q_lo = np.take_along_axis(qt, (kk - 1)[None], 0)[0]
        q_hi = np.take_along_axis(qt, kk[None], 0)[0]
        f = np.clip((q_lo - sc.lw_zi_qt) / np.maximum(q_lo - q_hi, 1e-12), 0.0, 1.0)
        zi = zc[kk - 1] + f * g.dz
        return np.where(found, zi, np.inf)

    def _forcing(self, dt):
        sc, g = self.sc, self.g
        zc = g.zc
        # the means the nudging relaxes are those of the state the dynamics
        # handed over, before any forcing (as on the GPU)
        if sc.nudge_tau > 0.0:
            thl_m, qt_m = self.mean_thl_qt()
            if sc.nudge_uv:
                um_ = self.u.astype("f8").mean(axis=(1, 2))
                vm_ = self.v.astype("f8").mean(axis=(1, 2))
        if sc.rad_cool is not None:
            r = np.array([sc.rad_cool(z) for z in zc]).reshape(-1, 1, 1)
            self.th += (dt * r / self.pi0).astype(self.dtype)
        if sc.lw_f0 > 0.0 or sc.lw_f1 > 0.0:
            self.th += (dt * self.lw_heating() / self.b.pi0.reshape(-1, 1, 1)).astype(self.dtype)
        if sc.nudge_tau > 0.0:
            w = self.nudge_mask / sc.nudge_tau
            self.th += (dt * w * (self.thl_target - thl_m)).reshape(-1, 1, 1).astype(self.dtype)
            self.qv += (dt * w * (self.qt_target - qt_m)).reshape(-1, 1, 1).astype(self.dtype)
            if sc.nudge_uv:
                self.u += (dt * w * (self.u0[:, 0, 0] - um_)).reshape(-1, 1, 1).astype(self.dtype)
                self.v += (dt * w * (self.v0[:, 0, 0] - vm_)).reshape(-1, 1, 1).astype(self.dtype)
        if sc.moist_adv is not None:
            m = np.array([sc.moist_adv(z) for z in zc]).reshape(-1, 1, 1)
            self.qv += (dt * m).astype(self.dtype)
        if sc.subsidence is not None:
            # large-scale subsidence (w_ls <= 0), upwind from above, applied
            # to every column's own profile as CM1 does - to the cloud as
            # well as the vapour, so that the conserved theta_l and q_t the
            # cases are specified in are what subsides
            wls = np.array([sc.subsidence(z) for z in zc]).reshape(-1, 1, 1)
            for name in ("th", "qv", "qc", "qi"):
                f = getattr(self, name).astype("f8")
                full = f + (self.b.th0.reshape(-1, 1, 1) if name == "th" else 0.0)
                grad = np.zeros_like(full)
                grad[:-1] = (full[1:] - full[:-1]) / g.dz
                setattr(self, name, (f - dt * wls * grad).astype(self.dtype))

    # ------------------------------------------------ microphysics -------
    def _pdefq(self):
        """CM1 pdefq: zero negative water and rescale each column so its
        mass is unchanged.  A hydrometeor column whose total is itself
        negative (undershoots with no cloud to absorb them) is zeroed and
        the deficit is taken from that column's vapour instead, so total
        water is conserved to round-off in every column - CM1 would
        otherwise gain the deficit."""
        r0 = self.rho0
        deficit = 0.0
        for name in ("qc", "qr", "qi", "qs", "qv"):
            q = getattr(self, name)
            before = (r0 * q).sum(axis=0)
            if name == "qv":
                before = before - deficit
            qp = np.maximum(q, 0.0)
            after = (r0 * qp).sum(axis=0)
            scale = np.where(after > 0, np.maximum(before, 0.0) / np.maximum(after, 1e-30), 0.0)
            if name != "qv":
                deficit = deficit + np.maximum(-before, 0.0)
            setattr(self, name, (qp * scale[None]).astype(self.dtype))

    def _sediment(self, name, vfun, dt):
        """CM1 k_fallout + fallout: fall speeds once per step (copied down
        from the level above where there is no precipitation yet, so the
        leading edge keeps falling), then upstream flux-form sedimentation
        in as many sub-steps as each column needs to keep its fall Courant
        number below 0.8."""
        g = self.g
        q = getattr(self, name).astype("f8")
        r0 = self.b.rho0.reshape(-1, 1, 1)
        total_fall = np.zeros((g.ny, g.nx))
        t = self._temperature() if name == "qs" else None
        v = vfun(q, r0, t) if t is not None else vfun(q, r0)
        for k in range(g.nz - 2, -1, -1):
            v[k] = np.where(v[k] > 0.0, v[k], v[k + 1])
        cr = (v * dt / g.dz).max(axis=0)
        ncol = np.floor(cr / 0.8).astype(int) + 1
        nmax = int(ncol.max())
        dtf = dt / ncol
        for it in range(nmax):
            active = it < ncol
            flux = r0 * v * q                                # kg m^-2 s^-1, downward
            div = np.zeros_like(q)
            div[:-1] = flux[1:] - flux[:-1]
            div[-1] = -flux[-1]
            q = np.where(active[None], q + dtf[None] * div / (r0 * g.dz), q)
            total_fall += np.where(active, dtf * flux[0], 0.0)
        setattr(self, name, q.astype(self.dtype))
        return total_fall

    def _temperature(self):
        return (self.th0 + self.th) * self.pi0

    def _microphysics(self, dt):
        self._pdefq()
        fall_r = self._sediment("qr", rain_fall_speed, dt)
        gr = float(self.sc.graupel)
        fall_s = self._sediment("qs", lambda q, r, t: ice_precip_fall_speed(q, r, t, gr), dt) \
            if self.sc.micro_ice \
            else np.zeros_like(fall_r)
        self.rain += fall_r + fall_s
        self.rain_rate = (fall_r + fall_s) / dt
        self._conversions(dt)
        self._saturation_adjust()

    def _conversions(self, dt):
        th, qv, qc, qr, qi, qs = (getattr(self, n).astype("f8") for n in self.SCALARS)
        pi0 = self.b.pi0.reshape(-1, 1, 1)
        p = self.b.p0.reshape(-1, 1, 1)
        rho = self.b.rho0.reshape(-1, 1, 1)
        t = (self.b.th0.reshape(-1, 1, 1) + th) * pi0
        lv = latent_vap(t)
        lf = latent_sub(t) - lv
        # --- Kessler warm rain (CM1 kessler.F) ---------------------------
        ar = np.maximum(K_AUTO * (qc - self.sc.q_auto), 0.0) * dt
        cr = K_ACCR * qc * np.maximum(qr, 0.0) ** 0.875 * dt
        tot = ar + cr
        lim = np.where(tot > qc, qc / np.maximum(tot, 1e-30), 1.0)
        ar *= lim
        cr *= lim
        qvs = qsat_liq(t, p)
        rq = np.maximum(rho * qr, 0.0)
        er = np.where((qr > 0) & (qv < qvs),
                      (1.6 + 30.3922 * rq ** 0.2046) * (1.0 - qv / qvs) * rq ** 0.525
                      / ((2.03e4 + 9.584e6 / (qvs * p)) * rho), 0.0)
        er = np.minimum(er * dt, qr)
        er = np.where(qv + er > qvs, np.maximum(qvs - qv, 0.0), er)
        qv = qv + er
        qc = qc - ar - cr
        qr = qr + ar + cr - er
        th = th - er * lv / (CP * pi0)
        if self.sc.micro_ice:
            # --- ice processes ----------------------------------------------
            tc = t - T0K
            # autoconversion of cloud ice to snow (SAM1MOM)
            ai = np.maximum(K_AUTO_ICE * np.exp(0.025 * tc) * (qi - self.sc.qi_auto), 0.0) * dt
            ai = np.minimum(ai, qi)
            qi -= ai
            qs += ai
            # collection of cloud water and ice by frozen precipitation
            fg = graupel_fraction(t) * float(self.sc.graupel)
            rqs = np.maximum(rho * qs, 0.0)
            coll = (fg * CG_COEF * rqs ** ((3.0 + B_G) / 4.0)
                    + (1.0 - fg) * CS_COEF * rqs ** ((3.0 + B_S) / 4.0)) \
                * np.sqrt(RHO_REF_ICE / rho)
            dqc = np.minimum(qc, coll * E_CLOUD_BY_ICE_PRECIP * qc * dt)
            dqi = np.minimum(qi, coll * E_ICE_BY_ICE_PRECIP * qi * dt)
            cold = t < T0K
            # riming: frozen onto the particle below 0 C (releases fusion
            # heat); shed as rain above it
            qc -= dqc
            qi -= dqi
            qs += dqi + np.where(cold, dqc, 0.0)
            qr += np.where(cold, 0.0, dqc)
            th += np.where(cold, dqc * lf / (CP * pi0), 0.0)
            # melting of frozen precipitation above 0 C (heat conduction)
            n0 = fg * N0G + (1.0 - fg) * N0S
            rx = fg * RHO_G + (1.0 - fg) * RHO_S
            lam = lambda_mp(rqs, rx, n0)
            melt = np.where((~cold) & (qs > 1e-12),
                            2.0 * math.pi * n0 * F_VENT * K_AIR * (t - T0K)
                            / (lam ** 2 * lf * rho), 0.0) * dt
            melt = np.minimum(melt, qs)
            qs -= melt
            qr += melt
            th -= melt * lf / (CP * pi0)
            # freezing of rain below 0 C: Bigg (1953), Lin et al. (1983)
            rqr = np.maximum(rho * qr, 0.0)
            lamr = lambda_mp(rqr, RHOW, N0R)
            frz = np.where(cold & (qr > 1e-12),
                           20.0 * math.pi ** 2 * BIGG_B * N0R * (RHOW / rho)
                           * (np.exp(BIGG_A * np.minimum(T0K - t, 60.0)) - 1.0)
                           / lamr ** 7, 0.0) * dt
            frz = np.minimum(frz, qr)
            qr -= frz
            qs += frz
            th += frz * lf / (CP * pi0)
            # sublimation of frozen precipitation in air below ice saturation
            qvi = qsat_ice(t, p)
            ls = latent_sub(t)
            esi = np.minimum(esat_ice(t), 0.5 * p)
            dv = DV0 * (t / T0K) ** 1.81 * (1.0e5 / p)
            aa = (ls / (RV * t) - 1.0) * ls / (K_AIR * t)
            bb = RV * t / (dv * esi)
            sub = np.where((qs > 1e-12) & (qv < qvi),
                           2.0 * math.pi * (1.0 - qv / qvi) * n0 * F_VENT
                           / (lam ** 2 * (aa + bb) * rho), 0.0) * dt
            sub = np.minimum(sub, qs)
            sub = np.minimum(sub, np.maximum(qvi - qv, 0.0))
            qs -= sub
            qv += sub
            th -= sub * ls / (CP * pi0)
        for n, arr in zip(self.SCALARS, (th, qv, qc, qr, qi, qs)):
            setattr(self, n, arr.astype(self.dtype))

    def _saturation_adjust(self, iterations: int = 4):
        """Condense or evaporate cloud so that vapour equals saturation over
        the mixed-phase cloud wherever there is cloud, conserving
        h = cp T - Lv qc - Ls qi at constant pressure.  Newton on T."""
        pi0 = self.b.pi0.reshape(-1, 1, 1)
        p = self.b.p0.reshape(-1, 1, 1)
        th0 = self.b.th0.reshape(-1, 1, 1)
        t = (th0 + self.th.astype("f8")) * pi0
        tn, qv, qc, qi = saturation_adjust_arrays(
            t, self.qv.astype("f8"), self.qc.astype("f8"), self.qi.astype("f8"),
            p, self.sc.micro_ice, iterations)
        self.qc = qc.astype(self.dtype)
        self.qi = qi.astype(self.dtype)
        self.qv = qv.astype(self.dtype)
        self.th = (tn / pi0 - th0).astype(self.dtype)

    # ------------------------------------------------------- surface -----
    def surface_fluxes(self):
        """Kinematic surface fluxes (w'th' in K m/s, w'qv' in m/s), per
        column, for the current scenario and sun."""
        sc = self.sc
        g = self.g
        if g.z0 > 0.0:
            return None, None
        if sc.flux_mode == "fixed":
            wth = np.full((g.ny, g.nx), sc.fixed_wth)
            wqv = np.full((g.ny, g.nx), sc.fixed_wqv)
        elif sc.flux_mode == "bulk":
            # bulk aerodynamic formulae over a surface at sc.sst, with the
            # neutral transfer coefficient of the drag law (this program's
            # choice: no stability correction), e.g. the sea under a
            # cold-air outbreak
            um, vm = sc.translate
            uc = 0.5 * (self.u[0] + np.roll(self.u[0], -1, 1)) + um
            vc = 0.5 * (self.v[0] + np.roll(self.v[0], -1, 0)) + vm
            spd = np.sqrt(uc ** 2 + vc ** 2 + 1.0)
            zc0 = 0.5 * g.dz
            ch = (0.4 / math.log((zc0 + sc.z0) / sc.z0)) ** 2
            th_s = sc.sst * (P00 / self.b.psfc) ** (RD / CP)
            pi_1 = self.b.pi0[0]
            th_1 = self.b.th0[0] + self.th[0]
            wth = ch * spd * (th_s - th_1) * pi_1              # temperature flux
            qs_s = float(qsat_liq(np.float64(sc.sst), self.b.psfc))
            wqv = ch * spd * (qs_s - self.qv[0])
        elif sc.flux_mode == "sun":
            rn = self.net_radiation()
            rho_s = self.b.rho0[0]
            day = rn > 0
            avail = np.where(day, 0.9 * rn, 0.8 * rn)
            bw = sc.bowen
            hflux = np.where(day, avail * bw / (1.0 + bw), avail)
            le = np.where(day, avail / (1.0 + bw), 0.0)
            het = 1.0 + sc.heterogeneity * self.het
            wth = hflux * het / (rho_s * CP)
            wqv = le * het / (rho_s * latent_vap(self.b.t0[0]))
        else:
            return None, None
        return wth, wqv

    # set by the app: sun elevation (rad) and azimuth (rad, from north
    # towards east) at the current model time
    sun_elev = 0.9
    sun_az = 3.14

    def net_radiation(self):
        """Surface net radiation, W m^-2: clear-sky shortwave (1098 sin(e)
        exp(-0.057/sin(e))), thinned by the cloud overhead along the Sun's
        direction with the conservative two-stream transmission
        1/(1 + 0.75 (1-g) tau), less 70 W m^-2 of net long-wave
        (this program's simple surface energy budget)."""
        g = self.g
        se = math.sin(max(self.sun_elev, 0.0))
        if se <= 0.01:
            return np.full((g.ny, g.nx), -70.0)
        sw = 1098.0 * se * math.exp(-0.057 / se)
        tau = self.sun_optical_depth()
        trans = 1.0 / (1.0 + 0.75 * 0.14 * tau)
        return self.sc.sw_absorb * sw * trans - 70.0

    def sun_optical_depth(self, steps: int = 48):
        """Cloud optical depth between each ground column and the Sun."""
        g = self.g
        sig = self.optics()["cloud"]
        el = max(self.sun_elev, 0.05)
        dirx = math.sin(self.sun_az) * math.cos(el)
        diry = math.cos(self.sun_az) * math.cos(el)
        dirz = math.sin(el)
        # march through the levels: at height z the ray has moved z/tan(e)
        tau = np.zeros((g.ny, g.nx))
        for k in range(g.nz):
            z = g.z0 + (k + 0.5) * g.dz
            shx = z * dirx / dirz / g.dx
            shy = z * diry / dirz / g.dy
            lay = sig[k]
            # bilinear periodic shift
            ix = int(math.floor(shx))
            fx = shx - ix
            iy = int(math.floor(shy))
            fy = shy - iy
            a = np.roll(np.roll(lay, -ix, 1), -iy, 0)
            b = np.roll(a, -1, 1)
            c = np.roll(a, -1, 0)
            d = np.roll(b, -1, 0)
            tau += ((1 - fx) * (1 - fy) * a + fx * (1 - fy) * b + (1 - fx) * fy * c
                    + fx * fy * d) * g.dz / dirz
        return tau

    def _surface(self, dt):
        wth, wqv = self.surface_fluxes()
        if wth is None:
            return
        g = self.g
        self.th[0] += (dt * wth / (g.dz * self.b.pi0[0])).astype(self.dtype)
        self.qv[0] += (dt * wqv / g.dz).astype(self.dtype)
        self.sfc_wth = wth
        self.sfc_wqv = wqv

    # ------------------------------------------------------- output ------
    def optics(self):
        """Extinction coefficients (m^-1) for the renderer: cloud liquid,
        cloud ice, rain and frozen precipitation."""
        rho = self.b.rho0.reshape(-1, 1, 1)
        t = self._temperature()
        liq = 3.0 * rho * np.maximum(self.qc, 0) / (2.0 * RHOW * R_EFF_LIQ)
        ice = 3.0 * rho * np.maximum(self.qi, 0) / (2.0 * RHOI * R_EFF_ICE)
        rain = extinction_mp(rho * np.maximum(self.qr, 0), RHOW, N0R)
        fg = graupel_fraction(t) * float(self.sc.graupel)
        rqs = rho * np.maximum(self.qs, 0)
        snow = fg * extinction_mp(rqs, RHO_G, N0G) + (1 - fg) * extinction_mp(rqs, RHO_S, N0S)
        return {"liq": liq, "ice": ice, "rain": rain, "snow": snow, "cloud": liq + ice}

    def diagnostics(self) -> dict:
        g = self.g
        rho = self.b.rho0.reshape(-1, 1, 1)
        cond = self.qc + self.qi
        zc = g.zc
        cloudy = cond > 1e-5
        levels = np.where(cloudy.any(axis=(1, 2)))[0]
        top = float(zc[levels[-1]] + 0.5 * g.dz) if levels.size else 0.0
        base = float(zc[levels[0]] - 0.5 * g.dz) if levels.size else 0.0
        lwp = (rho * cond).sum(axis=0) * g.dz
        water = float(((rho * (self.qv + self.qc + self.qr + self.qi + self.qs)).sum()
                       * g.dx * g.dy * g.dz))
        opt = self.optics()
        pr = (opt["rain"] + opt["snow"]) > PRECIP_VISIBLE
        plev = np.where(pr.any(axis=(1, 2)))[0]
        return {
            "precip_bottom": float(g.z0 + plev[0] * g.dz) if plev.size else -1.0,
            "time": self.time,
            "w_max": float(self.w.max()),
            "w_min": float(self.w.min()),
            "cloud_top": top,
            "cloud_base": base,
            "cloud_cover": float((lwp > 0.02).mean()),
            "rain_rate_max": float(self.rain_rate.max() * 3600.0),     # mm/h
            "rain_total": float(self.rain.mean()),                      # mm
            "water": water,
            "ice_frac": float(self.qi.sum() / max(cond.sum(), 1e-20)),
        }
