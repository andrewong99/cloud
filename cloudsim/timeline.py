# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
The sky between the forecast's hours.

A forecast gives the atmosphere on the hour.  The program diagnoses the Atlas
layers of each hour (scene.diagnose), and this module makes one continuous
sky of them:

  tracks      A layer is followed from hour to hour: the layers of two
              neighbouring hours that occupy the same part of the column are
              the same layer (link).  A layer the next hour does not have
              thins to nothing over the hour; one that appears grows from
              nothing.  Between the hours every number of a layer - base,
              top, cover, wind, optical depth, how cellular or fibrous it is
              - is interpolated linearly in time; its name changes at the
              half hour.

  cover       A layer's pattern is a field z with exactly a standard normal
              distribution (noise.deck_fields); the layer is cloud where z
              exceeds zstar = Phi^-1(1 - cover).  This is the rule of a
              statistical cloud scheme (Sommeria & Deardorff 1977; Mellor
              1977): the cloud fraction is the part of the sub-grid
              distribution above saturation.  So when the cover changes, the
              elements grow and new ones appear where the pattern is highest;
              they shrink and vanish from where it is lowest - clouds form and
              evaporate, they do not fade in and out.

  evolution   The elements themselves are born and die.  The pattern is a
              blend of realizations - independent fields, one begun every
              hop, each living three hops:
                  z = sum_j w_j Z_j,  w_j = sqrt(8/9) sin^2(pi (theta - j) / 3)
              The weights of the three living realizations always satisfy
              sum w_j^2 = 1 (sin^4 at three phases a third of a period apart
              sums to 9/8), so z stays exactly standard normal - the cover
              stays exact and nothing ghosts, as it would if two patterns were
              cross-faded (Heitz & Neyret 2018: histogram-preserving
              blending).  That is for independent realizations; on a tile of
              a few large elements two of them are correlated by chance, so
              the blend's spread over the tile is sum_ij w_i w_j rho_ij, and
              zstar is set for that spread (Sky.blend_spread) - the share of
              the tile that is cloud stays the cover.  Each
              weight rises from zero and returns to zero
              with zero slope, and sum (dw_j/dtheta)^2 is constant, so the
              pattern changes at a steady rate, without jerks.  theta
              advances at a rate set by the kind of cloud (RENEWAL_MIN): a
              field of fair-weather cumulus renews itself in minutes, a
              cirrostratus veil in about an hour.

  wind        Each layer is carried by its own wind, integrated over time
              (the wind interpolated between the hours), instead of wind
              times clock time - so a wind that changes does not make the
              whole layer jump.

The state at a moment is a function of that moment, not of how the clock
got there - except while a pattern is still being built, when the layer's
pattern lags (it catches up at CATCHUP_HOPS_PER_S) rather than jumps.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
import threading
import time as _time
from datetime import datetime

import numpy as np

from . import noise, scene
from .scene import Deck

HOUR = 3600.0
SLOTS = 3
MAP_N = 512

# ---------------------------------------------------------------------------
# How fast a layer's pattern renews itself: the time after which half of it
# has been replaced (the lag at which the pattern's correlation falls to 1/2).
#
#   Cu humilis/mediocris  7.5 min: shallow cumulus live 15 +- 2 min (median
#                         of 158 clouds tracked by stereo cameras, Romps et
#                         al. 2021); a snapshot's clouds are on average half
#                         way through their lives, so half are gone after
#                         about half a lifetime.
#   Cb                    15 min: a single thunderstorm cell lasts about 30
#                         minutes (NOAA JetStream); the same half-life rule.
#   the others            this program's estimates from the time scales of
#                         the motions that make each pattern: the overturning
#                         of a mixed layer, depth / velocity (Sc 25 min for a
#                         kilometre-deep layer stirred at ~1 m/s; Ac and Cc,
#                         thin layers overturned by their own radiative
#                         cooling, 15 and 8 min); for ice cloud, the time the
#                         crystals take to fall through the layer (Ci 30 min
#                         for 1.5 km at ~1 m/s; Cs, As, Ns deeper and more
#                         uniform, 45-60 min).
RENEWAL_MIN = {"Cu": 7.5, "Cb": 15.0, "Sc": 25.0, "St": 30.0, "Ac": 15.0, "Cc": 8.0,
               "Ci": 30.0, "Cs": 60.0, "As": 60.0, "Ns": 45.0}
RENEWAL_SPECIES_MIN = {("Cu", "con"): 12.0, ("Cu", "fra"): 6.0}


def _w(x):
    return np.where((x >= 0) & (x < 3), math.sqrt(8.0 / 9.0) * np.sin(math.pi * x / 3.0) ** 2, 0.0)


def pattern_correlation(lag_hops: float) -> float:
    """Correlation of the blended pattern with itself lag_hops later,
    averaged over the phase within a hop (the Z_j are independent, so it is
    sum_j w_j(theta) w_j(theta + lag))."""
    th = (np.arange(600) + 0.5) / 600.0
    acc = np.zeros_like(th)
    for j in range(-5, 6):
        acc += _w(th - j) * _w(th + lag_hops - j)
    return float(np.mean(acc))


def _half_life_hops() -> float:
    lo, hi = 0.0, 3.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if pattern_correlation(mid) > 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


HALF_LIFE_HOPS = _half_life_hops()


def renewal_s(deck: Deck) -> float:
    g, sp = deck.spec.genus, deck.spec.species
    return 60.0 * RENEWAL_SPECIES_MIN.get((g, sp), RENEWAL_MIN.get(g, 30.0))


def hop_rate(deck: Deck) -> float:
    """Realizations begun per second."""
    return HALF_LIFE_HOPS / renewal_s(deck)


def weights(theta: float) -> dict:
    """{realization index: weight} of the realizations alive at theta."""
    J = math.floor(theta)
    out = {}
    for j in (J - 2, J - 1, J):
        out[j] = math.sqrt(8.0 / 9.0) * math.sin(math.pi * (theta - j) / 3.0) ** 2
    return out


def realization_seed(track_seed: int, j: int) -> int:
    h = hashlib.blake2b(f"{track_seed}:{j}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") & 0x3FFFFFFF


# ---------------------------------------------------------------------------
# Interpolating a layer between two hours

_NUMERIC = [f.name for f in dataclasses.fields(Deck)
            if f.type in ("float", float) and f.name not in ("aniso_angle", "ice_frac", "turret")]


def pchip_slope(y0, y1, y2) -> float:
    """The slope at y1 (per hour) of the monotone cubic through three hourly
    values: the harmonic mean of the two secants where they agree in sign,
    zero where they do not (Fritsch & Carlson 1980; Fritsch & Butland 1984 -
    scipy's PchipInterpolator).  An hour at which a quantity turns is a
    turning point of the curve, and between two hours the curve stays
    between their values: no overshoot, so a cover stays within 0..1."""
    if y0 is None or y2 is None:
        return 0.0
    d0, d1 = y1 - y0, y2 - y1
    if d0 * d1 <= 0.0:
        return 0.0
    return 2.0 * d0 * d1 / (d0 + d1)


def hermite(p0: float, p1: float, m0: float, m1: float, f: float) -> float:
    """The cubic between p0 (f = 0) and p1 (f = 1) with slopes m0, m1."""
    f2 = f * f
    f3 = f2 * f
    return ((2.0 * f3 - 3.0 * f2 + 1.0) * p0 + (f3 - 2.0 * f2 + f) * m0
            + (3.0 * f2 - 2.0 * f3) * p1 + (f3 - f2) * m1)


def interp_deck(a: Deck, b: Deck, f: float, before: Deck | None = None,
                after: Deck | None = None, smooth: bool = False) -> Deck:
    """The layer a fraction f of the way from a to b.

    smooth=False: every number linear in time.  smooth=True (a layer
    followed through the hours, `before` and `after` the hours either side
    or None): every number on the monotone cubic through the hourly values
    (pchip_slope, hermite), so each changes smoothly through the hours as
    well as between them - a layer that sinks and then holds eases into
    holding instead of stopping dead on the hour; an hour missing either
    side (a layer's first or last) counts as level.  The stretching
    direction goes as an axis, the shorter way round; the name, reasons and
    odds are the nearer hour's."""
    f = min(max(float(f), 0.0), 1.0)
    near = a if f < 0.5 else b
    d = dataclasses.replace(near)
    if smooth:
        for k in _NUMERIC:
            x, y = getattr(a, k), getattr(b, k)
            m0 = pchip_slope(getattr(before, k) if before is not None else None, x, y)
            m1 = pchip_slope(x, y, getattr(after, k) if after is not None else None)
            setattr(d, k, hermite(x, y, m0, m1, f))
        ia, ib = a.ice_amount, b.ice_amount
        m0 = pchip_slope(before.ice_amount if before is not None else None, ia, ib)
        m1 = pchip_slope(ia, ib, after.ice_amount if after is not None else None)
        d.ice_frac = hermite(ia, ib, m0, m1, f)
        g = hermite(0.0, 1.0, 0.0, 0.0, f)          # the axis: eased, level at the hours
    else:
        for k in _NUMERIC:
            x, y = getattr(a, k), getattr(b, k)
            setattr(d, k, x + (y - x) * f)
        d.ice_frac = (1 - f) * a.ice_amount + f * b.ice_amount
        g = f
    c = (1 - g) * math.cos(2 * a.aniso_angle) + g * math.cos(2 * b.aniso_angle)
    s = (1 - g) * math.sin(2 * a.aniso_angle) + g * math.sin(2 * b.aniso_angle)
    d.aniso_angle = 0.5 * math.atan2(s, c) if (c * c + s * s) > 1e-12 else near.aniso_angle
    d.ice = d.ice_frac >= 0.5
    # castellanus: the common layer's top matters only where there are
    # turrets, and the realizations decide how many turrets there are (the
    # renderer takes the top from them: DeckState.turret_tb)
    ta, tb_ = a.turret, b.turret
    d.turret = (ta + (tb_ - ta) * f) if (ta > 0 and tb_ > 0) else max(ta, tb_)
    d.seed = a.seed
    return d


def ghost(d: Deck) -> Deck:
    """The same layer with nothing of it there: where a layer is born or dies."""
    g = dataclasses.replace(d)
    g.coverage = 0.0
    g.virga = g.precip = 0.0
    return g


# ---------------------------------------------------------------------------
# Following a layer from hour to hour

def _klass(d: Deck) -> str:
    return "convective" if d.spec.genus in ("Cu", "Cb") else "layer"


def link_score(a: Deck, b: Deck) -> float:
    """How surely b (next hour) is the layer a (this hour) became: the
    overlap of their depths, each widened by what a layer can plausibly move
    in an hour (a quarter of its depth, 150-500 m), over their union.  A
    convective cloud and a layer cloud score half."""
    ma = min(max(0.25 * (a.top_m - a.base_m), 150.0), 500.0)
    mb = min(max(0.25 * (b.top_m - b.base_m), 150.0), 500.0)
    a0, a1 = a.base_m - ma, a.top_m + ma
    b0, b1 = b.base_m - mb, b.top_m + mb
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0.0:
        return 0.0
    s = inter / (max(a1, b1) - min(a0, b0))
    if _klass(a) != _klass(b):
        s *= 0.5
    return s


def link(prev: list, nxt: list, threshold: float = 0.12) -> list:
    """Pairs (i, j): prev[i] became nxt[j] - greedily, the surest first."""
    cand = []
    for i, a in enumerate(prev):
        for j, b in enumerate(nxt):
            s = link_score(a, b)
            if s >= threshold:
                cand.append((s, i, j))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_i, used_j, out = set(), set(), []
    for s, i, j in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return out


class Track:
    """One layer through the hours: its diagnosed decks by hour index, and
    the time integrals that make its pattern evolve and drift.  lo and hi
    are the forecast's first and last hours: beyond them the edge hour
    holds (the sky does not empty past the end of the forecast)."""

    def __init__(self, tid: int, seed: int, lo: int = -10 ** 9, hi: int = 10 ** 9):
        self.id = tid
        self.seed = seed
        self.lo, self.hi = lo, hi
        self.keys: dict[int, Deck] = {}
        self._theta: dict[int, float] = {}
        self._off: dict[int, tuple] = {}
        #: (hour index, theta, off x, off y): the integrals' constants
        self.anchor = None

    @property
    def first(self) -> int:
        return min(self.keys)

    @property
    def last(self) -> int:
        return max(self.keys)

    def key(self, i: int) -> Deck | None:
        """The deck at hour i: diagnosed, or a ghost next to the layer's
        life, or None further away; outside the forecast, its edge hour's."""
        i = min(max(i, self.lo), self.hi)
        d = self.keys.get(i)
        if d is not None:
            return d
        if (i + 1) in self.keys:
            return ghost(self.keys[i + 1])
        if (i - 1) in self.keys:
            return ghost(self.keys[i - 1])
        return None

    def alive(self, x: float) -> bool:
        return ((x > self.first - 1 or self.first <= self.lo)
                and (x < self.last + 1 or self.last >= self.hi))

    def deck_at(self, x: float) -> Deck | None:
        """The interpolated deck at fractional hour x (smooth through the
        hours: interp_deck)."""
        i = math.floor(x)
        a, b = self.key(i), self.key(i + 1)
        if a is None and b is None:
            return None
        if a is None:
            return ghost(b)
        if b is None:
            return ghost(a)
        return interp_deck(a, b, x - i, self.key(i - 1), self.key(i + 2), smooth=True)

    def _near(self, i: int) -> Deck:
        if i in self.keys:
            return self.keys[i]
        return self.keys[self.first] if i < self.first else self.keys[self.last]

    def invalidate(self):
        self._theta.clear()
        self._off.clear()

    def _rate(self, i: int) -> float:
        return hop_rate(self._near(i))

    def _wind(self, i: int) -> tuple:
        d = self._near(i)
        still = 1.0 - min(max(max(d.lens, d.roll), 0.0), 1.0)   # lenticularis, volutus stand
        return d.wind_u * still, d.wind_v * still

    def _cum(self, i: int):
        """(theta, off x, off y) at hour i: the rate and the wind integrated
        from the anchor, exactly for quantities linear within each hour."""
        if i in self._theta:
            return (self._theta[i],) + self._off[i]
        a_i, a_th, a_x, a_y = self.anchor
        step = 1 if i > a_i else -1
        k, th, x, y = a_i, a_th, a_x, a_y
        self._theta.setdefault(a_i, a_th)
        self._off.setdefault(a_i, (a_x, a_y))
        while k != i:
            k2 = k + step
            if k2 in self._theta:
                th, (x, y) = self._theta[k2], self._off[k2]
            else:
                lo, hi = min(k, k2), max(k, k2)
                r = 0.5 * (self._rate(lo) + self._rate(hi)) * HOUR
                w0, w1 = self._wind(lo), self._wind(hi)
                th += step * r
                x += step * 0.5 * (w0[0] + w1[0]) * HOUR
                y += step * 0.5 * (w0[1] + w1[1]) * HOUR
                self._theta[k2], self._off[k2] = th, (x, y)
            k = k2
        return th, x, y

    def theta_off(self, x: float) -> tuple:
        """(theta, off x, off y) at fractional hour x."""
        i = math.floor(x)
        tau = (x - i) * HOUR
        th, ox, oy = self._cum(i)
        r0, r1 = self._rate(i), self._rate(i + 1)
        w0, w1 = self._wind(i), self._wind(i + 1)
        q = tau * tau / (2.0 * HOUR)
        return (th + r0 * tau + (r1 - r0) * q,
                ox + w0[0] * tau + (w1[0] - w0[0]) * q,
                oy + w0[1] * tau + (w1[1] - w0[1]) * q)

    def time_of_theta(self, theta: float, x_lo: float, x_hi: float) -> float:
        """The fractional hour at which theta is reached (bisection; theta
        rises with time)."""
        lo, hi = x_lo, x_hi
        if self.theta_off(lo)[0] >= theta:
            return lo
        if self.theta_off(hi)[0] <= theta:
            return hi
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if self.theta_off(mid)[0] < theta:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# What the renderer draws of one layer at one moment

@dataclasses.dataclass
class DeckState:
    track: int
    group: int
    deck: Deck                      # interpolated; coverage includes the multipliers
    weights: tuple                  # per slot
    extents: tuple                  # per slot, m
    offsets: tuple                  # per slot, (x, y) m, reduced modulo the slot's extent
    has_turret: tuple               # per slot
    has_gap: tuple                  # per slot (the realization carries a gap net)
    gap_kind: str
    zstar: float
    soft_z: float
    theta: float
    cavum_offset: tuple = (0.0, 0.0)
    cavum_extent: float = 6000.0
    trend: str = ""
    later: str = ""                 # the name the layer takes next hour, if different
    base_cover: float = 0.0         # the forecast's cover before the multipliers
    #: castellanus: the common layer's top (fraction of the depth) the
    #: turrets of the showing realizations stand on (0: none of them has
    #: turrets) - from the realizations themselves, weighted as they show,
    #: so the turrets' layer does not jump when the forecast's hour changes
    turret_tb: float = 0.0
    #: undulatus: the layer's waves (amplitude in pattern units, wave vector
    #: east and north in rad/m, phase at the observer) - one field that moves
    #: with the layer, added to its evolving pattern (noise.UND_Z)
    wave: tuple = (0.0, 0.0, 0.0, 0.0)
    #: perlucidus / lacunosus: how much of the cloud the gap net takes out,
    #: and the level of the gap field below which it lies - each showing
    #: realization's own kind, weighted as it shows
    gap_amount: float = 0.0
    gap_star: float = 0.0
    #: the standard deviation over the tile of the blend of the showing
    #: realizations (1 when they are uncorrelated; zstar is set for it -
    #: Sky.blend_spread)
    spread: float = 1.0


def turret_top(turret: float) -> float:
    """The renderer's castellanus layer top from Deck.turret (0: none)."""
    return float(np.clip(turret, 0.1, 0.9)) if turret > 0.0 else 0.0


def wave_vector(d: Deck) -> tuple:
    """(kx, ky) rad/m of undulatus waves on layer d: six elements apart
    (noise.deck_fields' undulatus_ratio), across the stretching direction."""
    lam = max(float(d.element_m) * 6.0, 600.0)
    k = 2.0 * math.pi / lam
    return k * math.sin(d.aniso_angle), k * math.cos(d.aniso_angle)


def gap_amount(d: Deck, kind: str) -> float:
    """How much of the cloud a gap net of this kind takes out of layer d."""
    if kind == "lacunosus":
        return float(d.lacunosus)
    if kind == "perlucidus":
        return float(d.perlucidus)
    return 0.0


def map_extent(element_m: float) -> float:
    """Physical size of one pattern tile, metres: about two dozen elements
    across, bounded so the map keeps a useful resolution."""
    return float(min(max(element_m * 24.0, 6000.0), 120000.0))


def realization_params(d: Deck) -> dict:
    gaps = ("lacunosus" if d.lacunosus > 0.05 else
            ("perlucidus" if d.perlucidus > 0.05 else ""))
    tb = float(np.clip(d.turret, 0.1, 0.9)) if d.turret > 0.0 else 0.0
    # (no undulatus: the waves are the layer's, not each realization's -
    # DeckState.wave)
    return dict(extent_m=map_extent(d.element_m), element_m=d.element_m,
                cellularity=d.cellularity, anisotropy=d.anisotropy, angle_rad=d.aniso_angle,
                undulatus=0.0, fibrosity=d.map_fibrosity, turret=tb,
                turret_cover=max(d.coverage, 0.15), gaps=gaps,
                blur_px=1.3 if d.map_fibrosity > 0.3 else 0.0,
                lumpy=float(round(min(max(getattr(d, "cell_dome", 0.0), 0.0), 1.0), 3)))


def build_realization(params: dict, seed: int, n: int = MAP_N) -> dict:
    """One realization, packed for the GPU: (n, n, 4) float16 of z, z^2 (so a
    coarser mip level knows how much the pattern varies within it), the
    element-top noise or (castellanus) the turret domes, and the gap net."""
    p = dict(params)
    extent = p.pop("extent_m")
    element = p.pop("element_m")
    f = noise.deck_fields(n, extent, element, seed=seed, **p)
    third = f["dome"] if params.get("turret", 0.0) > 0.0 else f["u"]
    pack = np.stack([f["z"], f["z"] * f["z"], third, f["gap"]], -1).astype(np.float16)
    return dict(pack=pack, extent=extent, turret=params.get("turret", 0.0) > 0.0,
                tb=float(params.get("turret", 0.0)), gaps=f["gaps"], seed=seed)


# ---------------------------------------------------------------------------

class Sky:
    """The continuous sky: tracks through the forecast's hours (or through
    layers chosen by hand, which hold), their evolving patterns, and which of
    them the renderer holds.

    update(when, real_dt) moves it to a moment; states are then the layers
    to draw, and take_uploads() the patterns to put in the renderer's
    texture slots (gl_sky.SkyRenderer.set_deck_states)."""

    #: the renderer's layer groups (gl_sky.MAXDECKS)
    GROUPS = 12
    #: at most this many realization hops per second of real time while a
    #: pattern catches up with the clock
    CATCHUP_HOPS_PER_S = 3.0
    #: further behind than this and the layer dissolves and forms again
    #: instead of running through every pattern in between
    DISSOLVE_LAG_HOPS = 9.0
    #: real seconds over which a layer fades out or in when it has to (a big
    #: jump, a layer the renderer could not hold, a forecast refresh)
    PRESENCE_S = 0.9
    #: real seconds over which a refreshed forecast's numbers take over
    REFRESH_S = 4.0
    #: diagnose this many hours around the clock
    BEHIND_H, AHEAD_H = 2, 3
    #: realizations kept in memory
    CACHE_MAX = 90

    def __init__(self, forecast=None, lat: float = 0.0, manual: list | None = None,
                 seed: int | None = None, n: int = MAP_N, threaded: bool = True,
                 diagnose=None, groups: int | None = None):
        self.n = n
        self.lat = lat
        self.GROUPS = groups or self.GROUPS
        self.diagnose = diagnose or scene.diagnose
        self._salt = seed if seed is not None else 20260926
        self._rng = random.Random(self._salt)
        self._next_id = 1
        self.forecast = None
        self.manual = None
        self._reset_tracks()
        self.x = 0.0
        self.when = None
        self._epoch0 = math.floor(_time.time() / HOUR) * HOUR
        self.theta_disp: dict[int, float] = {}
        self.presence: dict[int, float] = {}
        self.dissolving: set = set()
        self.group_of: dict[int, int] = {}
        self.slot_real: dict[tuple, int | None] = {}
        self.slot_track: dict[tuple, int | None] = {}
        self.slot_meta: dict[tuple, dict] = {}
        self.pending_uploads: list = []
        self.cache: dict[tuple, dict] = {}
        self._rho: dict[tuple, float] = {}        # (seed, i, j) -> realizations' correlation
        self._used: dict[tuple, float] = {}
        self._wanted: dict[tuple, tuple] = {}
        self._building: set = set()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._thread = None
        self.threaded = threaded
        self.states: list[DeckState] = []
        self.overflow: list[str] = []
        self.build_times: list = []
        self.last_error = ""
        self.model_share: dict[int, float] = {}
        self.fading: dict[int, dict] = {}         # old tracks fading after a refresh
        self.blend: dict[int, dict] = {}          # new track -> the old track it eases from
        if manual is not None:
            self.manual = [dataclasses.replace(d) for d in manual]
        else:
            self.forecast = forecast
        if threaded:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _reset_tracks(self):
        self.tracks: dict[int, Track] = {}
        self.keyframes: dict[int, list] = {}
        self._diag_next = None          # the next hour to diagnose (they go in order)

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # ------------------------------------------------------------ sources --
    def _new_track(self, seed: int | None = None) -> Track:
        lo, hi = self.hour_limits()
        t = Track(self._next_id, seed if seed is not None else self._rng.randrange(1 << 30), lo, hi)
        self._next_id += 1
        self.tracks[t.id] = t
        return t

    @property
    def _static(self) -> bool:
        return self.manual is not None or self.forecast is None or self.forecast.is_static

    def _decks_for_hour(self, i: int) -> list:
        if self.manual is not None:
            return [dataclasses.replace(d) for d in self.manual]
        if self.forecast is None:
            return []
        return list(self.diagnose(self.forecast.sounding_hour(i)))

    def hour_x(self, when: datetime) -> float:
        """The clock as a fractional hour index of the forecast (for a sky
        that holds, of an arbitrary epoch)."""
        if self._static:
            return (when.timestamp() - self._epoch0) / HOUR
        return (when - self.forecast.start).total_seconds() / HOUR

    def hour_limits(self):
        """The forecast's first and last hour indices; a sky that holds (the
        layers chosen by hand, the offline profiles) has the one hour 0,
        which holds at every moment."""
        if self._static:
            return 0, 0
        return 0, len(self.forecast.times) - 1

    def _track_seed(self, i: int, k: int, d: Deck) -> int:
        """A new track's seed, from where it begins: its first hour (as a
        time, so a refreshed forecast starting at another hour gives the
        same layer the same seed), its place among that hour's layers and
        its genus - not from the order the program happened to meet it in."""
        if self._static:
            key = f"held:{k}:{d.spec.genus}:{d.seed}:{self._salt}"
        else:
            key = f"{self.forecast.times[i].isoformat()}:{k}:{d.spec.genus}:{self._salt}"
        h = hashlib.blake2b(key.encode(), digest_size=4).digest()
        return int.from_bytes(h, "little") & 0x3FFFFFFF

    def ensure_hours(self, lo: int, hi: int):
        """Diagnose and link the forecast's hours up to hi (lo is where the
        clock needs them from; they are all diagnosed anyway).  The hours go
        in order from the forecast's first: each hour's layers continue the
        hour before's (link) or begin there, so a layer is only ever begun
        at its first hour, whatever the clock did - the tracks, their seeds
        and so the sky at a moment are the same however the clock got to
        it.  About 3 ms an hour."""
        a, b = self.hour_limits()
        hi = min(max(hi, a), b)          # before the first hour, the first holds
        i = a if self._diag_next is None else self._diag_next
        while i <= hi:
            decks = self._decks_for_hour(i)
            prev = self.keyframes.get(i - 1)
            owner = [None] * len(decks)
            if prev is not None:
                for pi, dj in link([d for d, _ in prev], decks):
                    owner[dj] = prev[pi][1]
            entry = []
            for d in decks:
                # the kind of optics this hour's name makes, as numbers that
                # blend between the hours (Deck.optic_kind)
                d.halo_w, d.corona_w, d.icecorona_w = d.optic_kind()
            for k, (d, tid) in enumerate(zip(decks, owner)):
                if tid is None:
                    seed = self._track_seed(i, k, d)
                    tr = self._new_track(seed)
                    # where in its renewal the pattern starts (from the seed)
                    tr.anchor = (i, (seed % 100003) / 100003.0 * 3.0, 0.0, 0.0)
                    tid = tr.id
                tr = self.tracks[tid]
                d.seed = tr.seed
                tr.keys[i] = d
                tr.invalidate()
                entry.append((d, tid))
            self.keyframes[i] = entry
            i += 1
        self._diag_next = max(i, a if self._diag_next is None else self._diag_next)

    def active_tracks(self, x: float) -> list:
        return [t for t in self.tracks.values() if t.keys and t.alive(x)]

    # ------------------------------------------------------ taking up data --
    def set_source(self, forecast=None, manual: list | None = None, merge: bool = True):
        """Take up a new forecast (or a new set of layers chosen by hand).
        The new layers are matched to the ones showing now: a matched layer
        keeps its pattern and drift and eases to its new numbers over
        REFRESH_S; an unmatched old layer thins away, an unmatched new one
        forms."""
        old_tracks = self.tracks
        old_live = [t for t in old_tracks.values() if t.id in self.group_of and t.keys]
        old_x = self.x
        self._reset_tracks()
        if manual is not None:
            self.manual, self.forecast = [dataclasses.replace(d) for d in manual], None
        else:
            self.manual, self.forecast = None, forecast
        self.blend.clear()
        if self.when is None or not merge:
            self.theta_disp.clear()
            self.presence.clear()
            self.dissolving.clear()
            self.fading.clear()
            for key in list(self.slot_track):
                self.slot_track[key] = None
                self.slot_real[key] = None
                self.slot_meta[key] = {}
            self.group_of.clear()
            return
        x = self.hour_x(self.when)
        self.x = x
        shift = old_x - x                   # an old track's hours at the same moment
        base = math.floor(x)
        self.ensure_hours(base - self.BEHIND_H, base + self.AHEAD_H)
        new = [t for t in self.tracks.values() if t.keys and (t.alive(x) or t.alive(x + 0.5))]
        pairs = []
        for nt in new:
            dn = nt.deck_at(x)
            for ot in old_live:
                do = ot.deck_at(old_x)
                if dn is not None and do is not None:
                    pairs.append((link_score(do, dn), ot.id, nt.id, ot, nt))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        used_o, used_n = set(), set()
        now = _time.time()
        for s, _, _, ot, nt in pairs:
            if s < 0.12 or ot.id in used_o or nt.id in used_n:
                continue
            used_o.add(ot.id)
            used_n.add(nt.id)
            self._adopt(nt, ot, x, shift, now)
        for ot in old_live:
            if ot.id not in used_o:
                self.fading[ot.id] = dict(track=ot, t0=now, shift=shift)
        # fading tracks from an earlier refresh keep their own clocks
        for f in self.fading.values():
            if f["track"].id not in [o.id for o in old_live]:
                f["shift"] = f["shift"] + shift

    def jump(self, when: datetime):
        """The clock jumps too far to play through: the layers showing that
        the new moment does not have dissolve over REFRESH_S, while the new
        moment's layers form."""
        if self.when is None:
            return
        old_x = self.x
        new_x = self.hour_x(when)
        now = _time.time()
        for tid, g in list(self.group_of.items()):
            tr = self.tracks.get(tid)
            if tr is None or tid in self.fading:
                continue
            if tr.alive(new_x) or tr.alive(new_x + 0.5):
                continue                    # it goes on (dissolving and forming if far behind)
            self.fading[tid] = dict(track=tr, t0=now, shift=old_x - new_x)

    def _adopt(self, nt: Track, ot: Track, x: float, shift: float, now: float):
        """Make new track nt continue old track ot: the same seed (so the same
        realizations), theta and drift matched now, the same renderer group;
        its numbers ease from the old track's over REFRESH_S."""
        th_o, ox_o, oy_o = ot.theta_off(x + shift)
        nt.seed = ot.seed
        for d in nt.keys.values():
            d.seed = ot.seed
        i = math.floor(x)
        nt.anchor = (i, 0.0, 0.0, 0.0)
        nt.invalidate()
        th_n, ox_n, oy_n = nt.theta_off(x)
        nt.anchor = (i, th_o - th_n, ox_o - ox_n, oy_o - oy_n)
        nt.invalidate()
        self.blend[nt.id] = dict(track=ot, t0=now, shift=shift)
        g = self.group_of.pop(ot.id, None)
        if g is not None:
            self.group_of[nt.id] = g
            for s in range(SLOTS):
                self.slot_track[(g, s)] = nt.id
                m = self.slot_meta.get((g, s))
                if m:
                    m["track"] = nt.id
        for d_ in (self.theta_disp, self.presence, self.model_share):
            if ot.id in d_:
                d_[nt.id] = d_.pop(ot.id)
        if ot.id in self.dissolving:
            self.dissolving.discard(ot.id)
            self.dissolving.add(nt.id)

    # ---------------------------------------------------------- the worker --
    def _worker(self):
        while True:
            with self._cv:
                while not self._stop and not self._wanted:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                key = min(self._wanted, key=lambda k: self._wanted[k][0])
                pri, params, seed = self._wanted.pop(key)
                self._building.add(key)
            t0 = _time.time()
            try:
                r = build_realization(params, seed, self.n)
            except Exception as e:                              # noqa: BLE001
                r = None
                self.last_error = f"{type(e).__name__}: {e}"
            with self._cv:
                self._building.discard(key)
                if r is not None:
                    self.cache[key] = r
                    self._used[key] = _time.time()
                    self.build_times.append(_time.time() - t0)
                    self.build_times = self.build_times[-50:]

    def _want(self, tr: Track, j: int, priority: float):
        key = (tr.seed, j)
        with self._cv:
            if key in self.cache:
                self._used[key] = _time.time()
                return
            if key in self._building:
                return
            if key in self._wanted and self._wanted[key][0] <= priority:
                return
        params = realization_params(self._structure_deck(tr, j))
        with self._cv:
            self._wanted[key] = (priority, params, realization_seed(tr.seed, j))
            self._cv.notify()

    def _structure_deck(self, tr: Track, j: int) -> Deck:
        """The deck whose look realization j of the track takes: the layer as
        it is when that realization is at its strongest."""
        lo, hi = tr.first - 1.0, tr.last + 1.0
        x = tr.time_of_theta(j + 1.5, lo, hi)
        d = tr.deck_at(min(max(x, float(tr.first)), float(tr.last)))
        return d if d is not None else tr.keys[tr.first]

    def build_now(self, limit: int | None = None) -> int:
        """Build every wanted realization in this thread (headless use), or
        at most `limit` of them; returns how many were built."""
        done = 0
        while limit is None or done < limit:
            with self._cv:
                if not self._wanted:
                    return done
                key = min(self._wanted, key=lambda k: self._wanted[k][0])
                pri, params, seed = self._wanted.pop(key)
            r = build_realization(params, seed, self.n)
            with self._cv:
                self.cache[key] = r
                self._used[key] = _time.time()
            done += 1
        return done

    def pending(self) -> int:
        with self._cv:
            return len(self._wanted) + len(self._building)

    def prepare(self, when: datetime, request: bool = True, ahead: bool = True) -> tuple:
        """The patterns the sky at `when` needs, and how many of them are
        built: (built, needed).  request: ask the worker for the missing
        ones.  The layers are the ones update() draws - those alive at the
        moment and those forming in the next half hour, as many as the
        renderer holds, the cloudiest first - and each needs the
        realizations its pattern blends at that moment (weights) and, with
        ahead, the next one: exactly what update(sync=True) builds, so a
        sync update once everything is built builds nothing and is a cut.
        Touches nothing the renderer holds (no group, slot or upload), so a
        sky can be made ready in the background while another is drawn -
        the loading card follows this count."""
        x = self.hour_x(when)
        base = math.floor(x)
        self.ensure_hours(base - self.BEHIND_H, base + self.AHEAD_H)
        live = [t for t in self.tracks.values() if t.keys and (t.alive(x) or t.alive(x + 0.5))]
        live.sort(key=lambda t: -self._priority(t, x))
        built = needed = 0
        for tr in live[:self.GROUPS]:
            d = tr.deck_at(x)
            if d is None:
                continue
            th = tr.theta_off(x)[0]
            J = math.floor(th)
            for j in ((J - 2, J - 1, J, J + 1) if ahead else (J - 2, J - 1, J)):
                needed += 1
                with self._cv:
                    have = (tr.seed, j) in self.cache
                if have:
                    built += 1
                elif request:
                    self._want(tr, j, priority=(1.0 - min(d.coverage, 1.0)) + 0.25 * abs(j - th))
        return built, needed

    def fresh(self) -> "Sky":
        """A new sky with this one's source, salt and epoch and the patterns
        it has built (its cache), but none of its tracks, fades or renderer
        slots: what set_source(merge=False) makes of this sky, to be made
        ready (prepare) in the background while this one is still drawn."""
        s = Sky(self.forecast, lat=self.lat, manual=self.manual, seed=self._salt, n=self.n,
                threaded=self.threaded, diagnose=self.diagnose, groups=self.GROUPS)
        s._epoch0 = self._epoch0
        with self._cv:
            s.cache = dict(self.cache)
            s._used = dict(self._used)
        return s

    def _evict(self):
        with self._cv:
            if len(self.cache) <= self.CACHE_MAX:
                return
            keep = set()
            for (g, s), tid in self.slot_track.items():
                j = self.slot_real.get((g, s))
                tr = self.tracks.get(tid) if tid is not None else None
                if tr is not None and j is not None:
                    keep.add((tr.seed, j))
            order = sorted(self.cache, key=lambda k: self._used.get(k, 0.0))
            for k in order:
                if len(self.cache) <= int(self.CACHE_MAX * 0.8):
                    break
                if k in keep:
                    continue
                self.cache.pop(k, None)
                self._used.pop(k, None)

    # --------------------------------------------------------------- update --
    def update(self, when: datetime, real_dt: float, sync: bool = False):
        """Move the sky to `when`.  real_dt: real seconds since the last call
        (how fast lagging patterns may catch up and fades run).  sync: build
        what is needed now, in this thread, and show it at once (headless)."""
        self.when = when
        x = self.hour_x(when)
        self.x = x
        base = math.floor(x)
        self.ensure_hours(base - self.BEHIND_H, base + self.AHEAD_H)
        rounds = 3 if sync else 1
        for _ in range(rounds):
            self._step(x, real_dt, sync)
            if sync:
                self.build_now()
        self._evict()

    def _step(self, x: float, real_dt: float, sync: bool):
        tracks = self.active_tracks(x)
        soon = [t for t in self.tracks.values()
                if t.keys and t.alive(x + 0.5) and not t.alive(x)]
        fading = [f["track"] for f in self.fading.values()]
        self._assign_groups(tracks + fading, soon, x)
        states = []
        now = _time.time()
        for tr in tracks + soon:
            g = self.group_of.get(tr.id)
            if g is None:
                continue
            st = self._track_state(tr, g, x, real_dt, sync, now)
            if st is not None:
                states.append(st)
        for tid, f in list(self.fading.items()):
            g = self.group_of.get(tid)
            k = (now - f["t0"]) / self.REFRESH_S if not sync else 1.0
            if g is None or k >= 1.0:
                self.fading.pop(tid, None)
                continue
            k = k * k * (3.0 - 2.0 * k)
            st = self._track_state(f["track"], g, x + f["shift"], real_dt, sync, now, fade=1.0 - k)
            if st is not None:
                states.append(st)
        for tid in list(self.blend):
            if (now - self.blend[tid]["t0"]) >= self.REFRESH_S or sync:
                self.blend.pop(tid, None)
        self.states = sorted(states, key=lambda s: s.deck.base_m)
        self._release_groups(x)

    def _deck_now(self, tr: Track, x: float, now: float) -> Deck | None:
        d = tr.deck_at(x)
        b = self.blend.get(tr.id)
        if d is not None and b is not None:
            k = min(max((now - b["t0"]) / self.REFRESH_S, 0.0), 1.0)
            k = k * k * (3.0 - 2.0 * k)
            od = b["track"].deck_at(x + b["shift"])
            if od is not None:
                d = interp_deck(od, d, k)
        return d

    def _priority(self, tr: Track, x: float) -> float:
        c = 0.0
        for dx in (0.0, 0.25, 0.5):
            e = tr.deck_at(x + dx)
            if e is not None:
                c = max(c, e.coverage * (1.0 if dx == 0.0 else 0.6))
        return c

    def _assign_groups(self, tracks: list, soon: list, x: float):
        used = set(self.group_of.values())
        free = [g for g in range(self.GROUPS) if g not in used]
        want = sorted((t for t in tracks + soon if t.id not in self.group_of),
                      key=lambda t: -self._priority(t, x))
        self.overflow = []
        for t in want:
            if not free:
                d = t.deck_at(x)
                if d is not None and d.coverage > 0.01:
                    self.overflow.append(d.spec.abbrev())
                continue
            g = free.pop(0)
            self.group_of[t.id] = g
            self.presence.setdefault(t.id, 0.0)
            for s in range(SLOTS):
                self.slot_track[(g, s)] = t.id
                self.slot_real[(g, s)] = None
                self.slot_meta[(g, s)] = {}

    def _release_groups(self, x: float):
        fading = set(self.fading)
        for tid, g in list(self.group_of.items()):
            tr = self.tracks.get(tid)
            if tid in fading:
                continue
            gone = tr is None or not tr.keys or not (tr.alive(x) or tr.alive(x + 0.5))
            if gone:
                del self.group_of[tid]
                self.theta_disp.pop(tid, None)
                self.presence.pop(tid, None)
                for s in range(SLOTS):
                    self.slot_track[(g, s)] = None
                    self.slot_real[(g, s)] = None
                    self.slot_meta[(g, s)] = {}

    def realization_corr(self, seed: int, i: int, j: int) -> float:
        """The correlation over the tile of realizations i and j of a track
        (the mean of Z_i Z_j; each is exactly standard normal on its tile),
        as far as it is chance.  Independent fields have none on average,
        but a tile has only so many elements: a veil of a few elements to
        the tile (Cs nebulosus, 40 km elements, three to the tile) holds few
        independent values, and two of its realizations are correlated by
        chance, +-0.2 - the blend's spread over the tile is then not 1 and
        its cover is off by up to 5 %.  Tiles of different sizes do not line
        up: 0.  Castellanus realizations share, by design, the rows their
        turrets stand on (noise.line_mask: the rows persist while the
        turrets come and go), +0.33; that dependence is not a bivariate
        normal one - the rows raise the middle of the distribution more
        than its tails - and a Gaussian spread made of it overcorrects (the
        cover of Ac cas at 35 % went from +0.7 % to -1.2 %), so it is left
        out: 0."""
        if i == j:
            return 1.0
        k = (seed, min(i, j), max(i, j))
        r = self._rho.get(k)
        if r is not None:
            return r
        with self._cv:
            a, b = self.cache.get((seed, i)), self.cache.get((seed, j))
        if a is None or b is None:
            return 0.0                      # not both built (then one of them does not show)
        if abs(a["extent"] - b["extent"]) > 1e-6 * max(a["extent"], 1.0) or \
                (a["turret"] and b["turret"]):
            r = 0.0
        else:
            za = a["pack"][..., 0].reshape(-1).astype("f8")
            zb = b["pack"][..., 0].reshape(-1).astype("f8")
            r = float(np.dot(za, zb)) / za.size
        self._rho[k] = r
        if len(self._rho) > 4 * self.CACHE_MAX:
            for kk in list(self._rho)[: len(self._rho) // 2]:
                self._rho.pop(kk, None)
        return r

    def blend_spread(self, seed: int, w: dict) -> float:
        """The standard deviation over the tile of the blend sum_j w_j Z_j:
        sum_ij w_i w_j rho_ij (sum_j w_j^2 = 1 by construction, so it is 1
        for uncorrelated realizations).  The cover threshold is set for this
        spread, so the share of the tile that is cloud is the cover even
        when the showing realizations happen to be correlated."""
        items = [(j, wj) for j, wj in w.items() if wj > 1e-7]
        v = sum(wj * wj for _, wj in items)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (i, wi), (j, wj) = items[a], items[b]
                v += 2.0 * wi * wj * self.realization_corr(seed, i, j)
        return math.sqrt(max(v, 0.05))

    def _loaded(self, g: int, j: int) -> bool:
        return self.slot_real.get((g, j % SLOTS)) == j

    def _feasible(self, g: int, theta: float) -> bool:
        for j, w in weights(theta).items():
            if w > 1e-7 and not self._loaded(g, j):
                return False
        return True

    def _advance_theta(self, tr: Track, g: int, goal: float, real_dt: float):
        cur = self.theta_disp.get(tr.id)
        if cur is None:
            if not self._feasible(g, goal):
                return None
            self.theta_disp[tr.id] = goal
            return goal
        step = self.CATCHUP_HOPS_PER_S * max(real_dt, 0.0) + 1e-9
        cand = goal if abs(goal - cur) <= step else cur + math.copysign(step, goal - cur)
        if not self._feasible(g, cand):
            # stop where the missing realization's weight is still zero
            if cand > cur:
                b = math.floor(cand)
                while b > cur and not self._feasible(g, b):
                    b -= 1
                cand = float(b) if b > cur else cur
            else:
                b = math.ceil(cand)
                while b < cur and not self._feasible(g, b):
                    b += 1
                cand = float(b) if b < cur else cur
            if not self._feasible(g, cand):
                cand = cur
        self.theta_disp[tr.id] = cand
        return cand

    def _track_state(self, tr: Track, g: int, x: float, real_dt: float, sync: bool,
                     now: float, fade: float = 1.0):
        d = self._deck_now(tr, x, now)
        if d is None:
            return None
        base_cover = d.coverage
        th_goal, ox, oy = tr.theta_off(x)
        cur = self.theta_disp.get(tr.id)
        # far behind (a jump): dissolve, then form again with the new pattern
        if cur is not None and abs(th_goal - cur) > self.DISSOLVE_LAG_HOPS and not sync:
            self.dissolving.add(tr.id)
        if sync and cur is not None and abs(th_goal - cur) > 1e-9:
            self.theta_disp.pop(tr.id, None)
            cur = None
        pres = self.presence.get(tr.id, 0.0)
        if tr.id in self.dissolving:
            pres = max(pres - real_dt / self.PRESENCE_S, 0.0)
            if pres <= 0.0:
                self.theta_disp.pop(tr.id, None)
                for s in range(SLOTS):
                    self.slot_real[(g, s)] = None
                    self.slot_meta[(g, s)] = {}
                self.dissolving.discard(tr.id)
                cur = None
        # ask for the realizations the goal needs, and the next ones
        J = math.floor(th_goal)
        js = {J - 2, J - 1, J, J + 1}
        if cur is not None:
            Jc = math.floor(cur)
            js |= {Jc - 2, Jc - 1, Jc, Jc + 1}
        for j in sorted(js):
            lag = abs(j - (cur if cur is not None else th_goal))
            self._want(tr, j, priority=(1.0 - min(base_cover, 1.0)) + 0.25 * lag)
        self._upload_ready(tr, g, cur if cur is not None else th_goal)
        th = self._advance_theta(tr, g, th_goal, 1e9 if sync else real_dt)
        if th is None:
            self.presence[tr.id] = 0.0
            return None
        if tr.id not in self.dissolving:
            pres = 1.0 if sync else min(pres + real_dt / self.PRESENCE_S, 1.0)
        self.presence[tr.id] = pres
        w = weights(th)
        ws, exts, offs = [0.0] * SLOTS, [0.0] * SLOTS, [(0.0, 0.0)] * SLOTS
        tur, gap = [False] * SLOTS, [False] * SLOTS
        kind, best = "", 0.0
        tw2 = tsum = gw2 = gsum_a = gsum_s = 0.0
        for j, wj in w.items():
            s = j % SLOTS
            m = self.slot_meta.get((g, s), {})
            ex = m.get("extent", map_extent(d.element_m))
            ws[s], exts[s] = wj, ex
            offs[s] = (ox % ex, oy % ex)
            tur[s] = bool(m.get("turret", False))
            gap[s] = bool(m.get("gaps", ""))
            if tur[s]:
                tw2 += wj * wj
                tsum += wj * wj * m.get("tb", turret_top(d.turret))
            if gap[s]:
                gw2 += wj * wj
                gsum_a += wj * wj * gap_amount(d, m["gaps"])
                gsum_s += wj * wj * noise.gap_star(m["gaps"])
            if m.get("gaps") and wj > best:
                kind, best = m.get("gaps"), wj
        cov = base_cover * pres * fade * (1.0 - self.model_share.get(tr.id, 0.0))
        d.coverage = min(max(cov, 0.0), 1.0)
        # undulatus: the waves move with the layer (their phase follows its
        # drift, reduced here in double precision); their wavelength and
        # direction are the layer's at its first hour, so they do not slide
        # through the layer as its elements change
        amp = noise.UND_Z * max(float(d.undulatus), 0.0)
        wave = (0.0, 0.0, 0.0, 0.0)
        if amp > 1e-6:
            kx, ky = wave_vector(tr.keys[tr.first])
            ph = ((tr.seed % 997) / 997.0 * 2.0 * math.pi - (kx * ox + ky * oy)) % (2.0 * math.pi)
            wave = (amp, kx, ky, ph)
        exc = map_extent(tr.keys[tr.first].element_m)
        later = ""
        nxt = tr.keys.get(math.floor(x) + 1)
        if nxt is not None and nxt.spec.abbrev() != d.spec.abbrev():
            later = nxt.spec.abbrev()
        # the threshold for the blend as it is on the tile: z = spread * Z',
        # so z + a sin > T where Z' + (a / spread) sin > T / spread
        spread = self.blend_spread(tr.seed, w)
        zstar = spread * noise.zstar_with_wave(d.coverage, wave[0] / spread)
        return DeckState(track=tr.id, group=g, deck=d, weights=tuple(ws), extents=tuple(exts),
                         offsets=tuple(offs), has_turret=tuple(tur), has_gap=tuple(gap),
                         gap_kind=kind, zstar=zstar, spread=spread,
                         wave=wave,
                         soft_z=noise.SOFT_PER_EDGE * d.edge_softness, theta=th,
                         cavum_offset=(ox % (0.55 * exc), oy % (0.55 * exc)),
                         cavum_extent=exc, trend=self._trend(tr, x), later=later,
                         base_cover=base_cover,
                         turret_tb=(tsum / tw2 if tw2 > 1e-12 else 0.0),
                         gap_amount=(gsum_a / gw2 if gw2 > 1e-12 else 0.0),
                         gap_star=(gsum_s / gw2 if gw2 > 1e-12 else 0.0))

    def _upload_ready(self, tr: Track, g: int, theta: float):
        """Put built realizations in the slots they belong in, once the
        slot's present occupant no longer shows."""
        live = weights(theta)
        J = math.floor(theta)
        for j in sorted(set(live) | {J + 1, J - 3}):
            s = j % SLOTS
            if self.slot_real.get((g, s)) == j:
                continue
            with self._cv:
                r = self.cache.get((tr.seed, j))
            if r is None:
                continue
            occ = self.slot_real.get((g, s))
            if occ is not None and live.get(occ, 0.0) > 1e-7:
                continue
            self.slot_real[(g, s)] = j
            self.slot_meta[(g, s)] = dict(extent=r["extent"], turret=r["turret"], gaps=r["gaps"],
                                          tb=r.get("tb", 0.0), j=j, track=tr.id)
            self.pending_uploads.append((g, s, r["pack"]))

    def take_uploads(self) -> list:
        out, self.pending_uploads = self.pending_uploads, []
        return out

    def _trend(self, tr: Track, x: float) -> str:
        a, b = tr.deck_at(x), tr.deck_at(x + 0.5)
        if a is None or b is None or self._static:
            return ""
        i = math.floor(x)
        if tr.first > i:
            return "forming"
        if tr.last < i + 1:
            return "dissolving"
        parts = []
        dc = b.coverage - a.coverage
        if dc > 0.03:
            parts.append("spreading")
        elif dc < -0.03:
            parts.append("thinning out")
        dz = b.base_m - a.base_m
        if dz > 100:
            parts.append("rising")
        elif dz < -100:
            parts.append("lowering")
        return ", ".join(parts)

    # ------------------------------------------------------------ queries --
    def frac_at(self, st: DeckState, x_east, y_north):
        """Cloud fraction of a layer at points relative to the observer (at
        the layer's own height): the renderer's rule on the CPU (the census,
        the self-test) - nearest texel of each slot, weighted, through
        noise.compose."""
        x = np.asarray(x_east, "f8")
        y = np.asarray(y_north, "f8")
        zsum = np.zeros_like(x)
        usum = np.zeros_like(x)
        gsum = np.zeros_like(x)
        dmax = np.zeros_like(x)
        gw2 = 0.0
        tr = self.tracks.get(st.track) or (self.fading.get(st.track, {}).get("track"))
        for s in range(SLOTS):
            w = st.weights[s]
            if w <= 1e-7 or tr is None:
                continue
            j = self.slot_real.get((st.group, s))
            r = self.cache.get((tr.seed, j)) if j is not None else None
            if r is None:
                continue
            pk = r["pack"]
            n = pk.shape[0]
            ex = st.extents[s]
            ox, oy = st.offsets[s]
            ii = (((y - oy) / ex) % 1.0 * n).astype(int) % n
            jj = (((x - ox) / ex) % 1.0 * n).astype(int) % n
            px = pk[ii, jj].astype("f8")
            zsum += w * px[..., 0]
            if st.has_turret[s]:
                dmax = np.maximum(dmax, min(w * w * 9.0 / 8.0, 1.0) * px[..., 2])
            else:
                usum += w * px[..., 2]
            if st.has_gap[s]:
                gsum += w * px[..., 3]
                gw2 += w * w
        d = st.deck
        a, kx, ky, ph = st.wave
        if a > 0.0:
            zsum = zsum + a * np.sin(kx * x + ky * y + ph)
        gap = gsum / math.sqrt(gw2) if gw2 > 1e-9 else None
        frac, _ = noise.compose(zsum, st.zstar, st.soft_z, u=usum, top_variation=d.top_variation,
                                gap=gap, gap_amount=(st.gap_amount * gw2 if gap is not None else 0.0),
                                gap_star=st.gap_star, dome=dmax, turret=st.turret_tb)
        return frac

    def decks(self) -> list:
        return [s.deck for s in self.states]
