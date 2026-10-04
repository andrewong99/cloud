# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
From atmosphere to picture.

Two directions, both ending in a list of `Deck` objects that the renderer can
draw:

  diagnose(sounding)          the sounding decides which Atlas types are
                              present, where their bases sit, and how much sky
                              they cover.  Every decision records the number it
                              was made on.

  deck_from_spec(spec, ...)   you name an Atlas cloud and it is built to
                              specification, placed at a height the sounding
                              supports (or, failing that, in the middle of the
                              genus's étage for the latitude).

The optical thickness of every deck comes from a water content and an
effective droplet/crystal radius, through the geometric-optics extinction
    sigma_e = 3 * W / (2 * rho * r_eff)
rather than from a hand-tuned "density" slider, so a Stratocumulus is opaque
and a Cirrus is not for the reason they really are.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import atlas, sounding
from .atlas import CloudSpec
from .sounding import Sounding

RHO_W = 1000.0      # kg/m3
RHO_I = 917.0       # kg/m3


def extinction(water_g_m3: float, r_eff_um: float, ice: bool) -> float:
    """Volume extinction coefficient in m^-1 (geometric optics limit)."""
    rho = RHO_I if ice else RHO_W
    return 3.0 * (water_g_m3 * 1e-3) / (2.0 * rho * (r_eff_um * 1e-6))


# ----------------------------------------------------------------------------
# Per-genus physical defaults.
#
#   w      typical condensed water content, g/m3
#   r      effective radius, micrometres
#   thick  typical geometrical thickness, m
#   elem_m element spacing, m.  For Cc, Ac and Sc it is pulled into the
#          Atlas's apparent-width band at the deck's height (deck_from_spec,
#          atlas.ELEMENT_WIDTH_DEG); for the others the Atlas gives no
#          numeric rule and the value is this program's own choice of a
#          plausible structure scale.
#   cum    cumuliformity 0 = flat sheet, 1 = towering
#   cell   how regularly arranged the elements are, 0 = formless, 1 = a grid
#   fib    fibrousness (ice streaks)
# ----------------------------------------------------------------------------
GENUS_DEFAULTS = {
    "Ci": dict(w=0.015, r=40.0, thick=1500.0, elem_m=3000.0, max_thick=4000.0,
               cum=0.15, cell=0.05, fib=0.85, ice=True),
    "Cc": dict(w=0.050, r=25.0, thick=500.0, elem_m=220.0, max_thick=900.0,
               cum=0.35, cell=0.85, fib=0.15, ice=True),
    "Cs": dict(w=0.016, r=35.0, thick=2000.0, elem_m=20000.0, max_thick=4000.0,
               cum=0.05, cell=0.02, fib=0.45, ice=True),
    "Ac": dict(w=0.300, r=8.0, thick=500.0, elem_m=900.0, max_thick=1600.0,
               cum=0.35, cell=0.85, fib=0.05, ice=False),
    "As": dict(w=0.050, r=15.0, thick=2500.0, elem_m=15000.0, max_thick=6000.0,
               cum=0.05, cell=0.05, fib=0.30, ice=False),
    "Ns": dict(w=0.400, r=10.0, thick=4000.0, elem_m=20000.0, max_thick=9000.0,
               cum=0.15, cell=0.02, fib=0.10, ice=False),
    "Sc": dict(w=0.300, r=9.0, thick=600.0, elem_m=1600.0, max_thick=1800.0,
               cum=0.45, cell=0.80, fib=0.00, ice=False),
    "St": dict(w=0.250, r=8.0, thick=300.0, elem_m=8000.0, max_thick=900.0,
               cum=0.05, cell=0.05, fib=0.00, ice=False),
    "Cu": dict(w=0.350, r=9.0, thick=900.0, elem_m=2500.0, max_thick=6000.0,
               cum=0.92, cell=0.35, fib=0.00, ice=False),
    "Cb": dict(w=1.800, r=12.0, thick=9000.0, elem_m=16000.0, max_thick=18000.0,
               cum=1.00, cell=0.15, fib=0.25, ice=False),
}

#: The genera whose elements are the cells of a layer (Atlas: the apparent
#: width of their elements is what tells them apart).
CELLULAR = ("Cc", "Ac", "Sc")
#: How deep a layer's elements may be, as a fraction of their width.  The
#: Atlas describes them as plates (laminae), rounded masses and rolls in Ac,
#: rounded masses and rolls in Sc, grains and ripples in a thin layer in Cc:
#: seen from below they are flat patches, wider than deep.  The forecast's pressure
#: levels, 0.5-1.5 km apart, cannot resolve a layer's depth: drawn as deep as
#: their cloudy band, cells a few hundred metres wide stood 1-2 km tall
#: (columns and curtains hanging from the sky); drawn half as deep as their
#: spacing, they were about as deep as wide - balls.
CELL_ASPECT = {"Cc": 0.5, "Ac": 0.5, "Sc": 0.5}
#: The cover a deck is drawn at when none is given.
DEFAULT_COVER = {"Ci": 0.35, "Cc": 0.45, "Cs": 0.9, "Ac": 0.6, "As": 0.95, "Ns": 1.0,
                 "Sc": 0.75, "St": 0.95, "Cu": 0.25, "Cb": 0.55}


def element_width(spacing_m: float, coverage: float) -> float:
    """The width of a layer's elements: the equivalent diameter of the
    cloudy part of a cell (coverage x spacing^2 of it), at most the spacing
    (merged into a sheet, an element is its cell).  On the drawn patterns
    the median element measures 0.51, 0.67 and 0.74 of the spacing at 20, 32
    and 38 % cover; this gives 0.50, 0.64 and 0.70."""
    return spacing_m * min(1.0, 1.1284 * math.sqrt(max(float(coverage), 0.0)))

#: How much the tops of individual elements vary within a deck.  A single
#: Cumulonimbus is one tower that fills its whole depth; a field of small
#: Cumulus humilis is very uneven.
TOP_VARIATION = {"Ci": 0.35, "Cc": 0.40, "Cs": 0.20, "Ac": 0.40, "As": 0.20,
                 "Ns": 0.20, "Sc": 0.35, "St": 0.20, "Cu": 0.55, "Cb": 0.12}

#: species -> multipliers and overrides
SPECIES_EFFECT = {
    "fib": dict(fib=+0.35, cell=-0.05, cum=-0.10, aniso=2.2),          # straight filaments
    "unc": dict(fib=+0.30, hook=1.0, cum=-0.05, aniso=2.0),            # hooks: fallstreaks
    "spi": dict(w_mul=3.0, thick_mul=1.4, fib=-0.25, cum=+0.15),
    "cas": dict(cum=+0.45, cell=+0.05, thick_mul=1.8, turret=1.0, top_var=0.55),
    "flo": dict(cum=+0.35, cell=-0.25, ragged=0.6, thick_mul=1.2, top_var=0.65),
    "str": dict(cum=-0.15, cell=+0.10),
    "neb": dict(cum=-0.10, cell=-0.30, elem_mul=2.0, smooth=1.0),
    "len": dict(cum=-0.05, cell=-0.40, lens=1.0, elem_mul=3.0, aniso=3.0),
    "vol": dict(roll=1.0, aniso=12.0, cum=+0.20, cell=-0.40),
    "fra": dict(ragged=1.0, cell=-0.30, w_mul=0.5, elem_mul=0.5),
    "hum": dict(thick_abs=700.0, cum=+0.00, w_mul=0.75, top_var=0.62),
    "med": dict(thick_abs=1600.0, cum=+0.03, w_mul=0.9),
    "con": dict(thick_abs=4500.0, cum=+0.06, w_mul=1.2, elem_mul=1.6, top_var=0.30),
    "cal": dict(thick_abs=9000.0, anvil=0.25, glaciated=0.35, top_var=0.14),
    "cap": dict(thick_abs=12000.0, anvil=1.0, glaciated=1.0, fib=+0.35, top_var=0.10),
}

#: variety -> map / optical modifiers
VARIETY_EFFECT = {
    "in": dict(fib=+0.15, tangle=1.0),
    "ve": dict(fib=+0.20, ribs=1.0, aniso=3.0),
    "un": dict(undulatus=0.75),
    "ra": dict(aniso=5.0, radiatus=1.0),
    "la": dict(lacunosus=0.8),
    "du": dict(duplicatus=1.0),
    "tr": dict(w_mul=0.45),
    "pe": dict(perlucidus=0.8, cover_mul=0.85),
    "op": dict(w_mul=1.8),
}


@dataclass
class Deck:
    """One cloud layer, fully parameterised for the renderer."""
    spec: CloudSpec
    base_m: float
    top_m: float
    coverage: float                 # 0..1 fraction of sky
    element_m: float                # horizontal spacing of the elements
    element_deg: float              # an element's apparent width seen 30 deg up
    sigma_e: float                  # extinction, m^-1
    ice: bool
    r_eff_um: float = 10.0          # effective radius of the droplets/crystals
    cumuliformity: float = 0.3
    cellularity: float = 0.5
    fibrosity: float = 0.0
    map_fibrosity: float = 0.0
    anisotropy: float = 1.0
    aniso_angle: float = 0.0        # radians, direction of stretching
    undulatus: float = 0.0
    lacunosus: float = 0.0
    perlucidus: float = 0.0
    edge_softness: float = 0.25
    top_variation: float = 0.45
    detail: float = 0.5
    wind_u: float = 0.0
    wind_v: float = 0.0
    shear_u: float = 0.0
    shear_v: float = 0.0
    fall_speed: float = 0.0         # m/s, for virga and uncinus fallstreaks
    virga: float = 0.0
    precip: float = 0.0
    mamma: float = 0.0
    asperitas: float = 0.0
    anvil: float = 0.0
    cavum: float = 0.0
    fluctus: float = 0.0
    arcus: float = 0.0
    pannus: float = 0.0
    pileus: float = 0.0
    #: the width of one element (element_width: the cloudy part of a cell)
    element_w_m: float = 0.0
    #: a layer's elements as rounded masses: thin at their edges, thickest
    #: and highest in the middle, flat-based (1), or slabs of one depth (0)
    cell_dome: float = 0.0
    # a towering convective cloud (Cu congestus, Cb): drawn as plumes - a dome
    # on a flat base, a cumulonimbus's anvil spreading at its top
    # (shaders.deckDensity).  Cumulus humilis and mediocris keep their look.
    tower: float = 0.0
    #: castellanus: the top of the common layer the turrets stand on, as a
    #: fraction of the deck's depth (0: no turrets)
    turret: float = 0.0
    lens: float = 0.0
    roll: float = 0.0
    ragged: float = 0.0
    hook: float = 0.0
    tilt: float = 0.0
    seed: int = 1
    score: float = 1.0
    reasons: list[str] = field(default_factory=list)
    map_index: int = 0
    #: how likely each species of the genus is for this layer, one line
    #: ("" when the deck was not diagnosed from a forecast)
    odds: str = ""
    #: between two hours, how much of the layer is ice (0 water .. 1 ice);
    #: negative: as `ice` says
    ice_frac: float = -1.0
    #: how much this kind of cloud makes a 22 degree halo, a corona of water
    #: droplets and an ice corona (gl_sky.optic_kind); negative: from the
    #: name.  A layer between two hours carries the blend of the two hours'
    #: kinds, so an optical phenomenon fades with a change of name instead of
    #: switching on or off at the half hour.
    halo_w: float = -1.0
    corona_w: float = -1.0
    icecorona_w: float = -1.0

    @property
    def ice_amount(self) -> float:
        return float(self.ice_frac) if self.ice_frac >= 0.0 else (1.0 if self.ice else 0.0)

    def optic_kind(self) -> tuple[float, float, float]:
        """How much this kind of cloud makes (a 22 degree halo, a corona of
        water droplets, an ice corona): the Atlas associates the halo with
        Cirrostratus (and, less, Cirrus) and coronae and irisation with thin
        Altocumulus and Cirrocumulus.  Between two forecast hours, the blend
        of the two hours' kinds (halo_w, corona_w, icecorona_w)."""
        if self.halo_w >= 0.0:
            return self.halo_w, max(self.corona_w, 0.0), max(self.icecorona_w, 0.0)
        g = self.spec.genus
        return ({"Cs": 1.0, "Ci": 0.35}.get(g, 0.0),
                1.0 if g in ("Ac", "Cc") else 0.0,
                1.0 if g == "Cc" else 0.0)

    @property
    def thickness(self) -> float:
        return max(self.top_m - self.base_m, 50.0)

    @property
    def optical_depth(self) -> float:
        return self.sigma_e * self.thickness

    @property
    def label(self) -> str:
        return self.spec.latin()

    def why(self) -> str:
        return "; ".join(self.reasons)


# ------------------------------------------------------------- build a deck --

def _apply(effects: dict, acc: dict) -> None:
    for k, v in effects.items():
        if k.endswith("_mul"):
            acc[k[:-4] + "_mul"] = acc.get(k[:-4] + "_mul", 1.0) * v
        elif k.endswith("_abs"):
            acc[k[:-4] + "_abs"] = v
        elif k in ("cum", "cell", "fib"):
            acc[k] = acc.get(k, 0.0) + v
        else:
            acc[k] = max(acc.get(k, 0.0), v) if isinstance(v, (int, float)) else v


def deck_from_spec(spec: CloudSpec,
                   snd: Sounding,
                   base_m: float | None = None,
                   coverage: float | None = None,
                   thickness_m: float | None = None,
                   seed: int = 1,
                   reasons: list[str] | None = None,
                   turret_rise_m: float | None = None) -> Deck:
    """Turn an Atlas name into a renderable deck.

    turret_rise_m: for castellanus, how far the turrets rise above the
    common layer (a diagnosed layer passes its parcel's buoyant depth);
    without it the turrets rise 0.8 of the layer's depth."""
    g = spec.genus
    d = dict(GENUS_DEFAULTS[g])
    acc: dict = {}
    if spec.species:
        _apply(SPECIES_EFFECT.get(spec.species, {}), acc)
    for v in spec.varieties:
        _apply(VARIETY_EFFECT.get(v, {}), acc)
    turreted = acc.get("turret", 0.0) > 0.0

    lo, hi = atlas.etage_range_m(g, snd.lat)
    if base_m is None:
        base_m = lo + 0.45 * (hi - lo)
        if g in ("Cu", "Cb", "St"):
            cape, cin, z_lcl, lfc, el, _ = snd.parcel(mixed_depth=400)
            base_m = max(min(z_lcl, hi), 60.0) if g != "St" else max(z_lcl * 0.35, 40.0)
        elif g in ("Sc",):
            inv = snd.inversions()
            base_m = max(inv[0][0] - 400.0, 400.0) if inv else 1000.0
    if coverage is None:
        coverage = DEFAULT_COVER.get(g, 0.5)
    coverage = min(1.0, coverage * acc.get("cover_mul", 1.0))
    # Element size.  For Cc, Ac and Sc the Atlas states an apparent width, so
    # the physical size is derived from it and the deck's height above the
    # ground.  For the other seven genera the Atlas gives no numeric rule, so
    # a physical spacing is used directly and the apparent width is merely
    # reported (element_deg: an element's width seen 30 degrees up).
    elem_m = d["elem_m"] * acc.get("elem_mul", 1.0)
    h_ag = max(base_m - sounding.station_elevation(snd), 150.0)
    band = atlas.ELEMENT_WIDTH_DEG.get(g)
    size_note = ""
    if band:
        # The Atlas: most elements seen more than 30 degrees above the
        # horizon have an apparent width inside the band.  From 30 degrees up
        # to overhead the distance to the layer halves (h / sin e), so the
        # same element looks twice as wide overhead: it must be wide enough
        # at 30 degrees and small enough overhead.  The physical spacing above
        # is a realistic one; outside the band it is pulled back.  (Checked
        # at 30 degrees alone, an altocumulus 3.3 degrees wide there was 6.6
        # overhead.)
        lo_d = band[0] * 1.05 if band[0] > 0 else 0.20
        w = element_width(elem_m, coverage)
        w_lo = atlas.angular_width_to_metres(lo_d, h_ag, 30.0)
        w_hi = (atlas.angular_width_to_metres(band[1] * 0.95, h_ag, 90.0) if band[1] < 11.0
                else atlas.angular_width_to_metres(45.0, h_ag, 30.0))     # this program's cap
        w2 = min(max(w, w_lo), w_hi)
        if w > 1e-9 and abs(w2 - w) > 1e-6:
            elem_m *= w2 / w
        w = element_width(elem_m, coverage)
        size_note = (f"elements {w:.0f} m wide, {h_ag:.0f} m up: "
                     f"{atlas.metres_to_angular_width(w, h_ag, 30.0):.1f} deg seen 30 deg up, "
                     f"{atlas.metres_to_angular_width(w, h_ag, 90.0):.1f} overhead (Atlas: "
                     + (f"{band[0]:g}-{band[1]:g}" if band[1] < 11.0 else f"over {band[0]:g}")
                     + " deg above 30 deg)")
    elem_w = element_width(elem_m, coverage)
    elem_deg = atlas.metres_to_angular_width(elem_w, h_ag, 30.0)

    thick = thickness_m if thickness_m is not None else d["thick"]
    # a layer of cells: its elements no deeper than half as wide (CELL_ASPECT),
    # and no thinner than 50 m, the thinnest layer the renderer draws
    cell_depth = max(CELL_ASPECT[g] * elem_w, 50.0) if g in CELLULAR else None
    asked = thick
    turret_frac = 0.0
    if turreted:
        # castellanus: the layer is the common base, and the turrets rise
        # above it - the deck is the two together.  The turrets keep their
        # tops (they may be taller than wide: Atlas); the common layer is a
        # layer of cells.
        layer = min(thick, d.get("max_thick", 20000.0))
        layer0 = layer
        top_turrets = (base_m + layer + turret_rise_m) if turret_rise_m is not None else None
        if cell_depth is not None:
            layer = min(layer, cell_depth)
        rise = (top_turrets - (base_m + layer)) if top_turrets is not None else 0.8 * layer
        # at most 2.5 km above the cloudy band asked (a common layer drawn
        # thinner than the band does not lower the turrets' tops)
        rise = min(max(rise, 0.3 * layer, 150.0), 2500.0 + (layer0 - layer))
        thick = layer + rise
        turret_frac = layer / thick
    else:
        thick = acc.get("thick_abs", thick) * acc.get("thick_mul", 1.0)

    # Cumulonimbus and Cumulus congestus grow up to their equilibrium level
    if g in ("Cb", "Cu") and thickness_m is None:
        cape, cin, z_lcl, lfc, el, _ = snd.parcel(mixed_depth=400)
        if g == "Cb" and el:
            thick = min(max(el - base_m, 4000.0), 17000.0)
        elif g == "Cu" and spec.species == "con" and el:
            thick = min(max(el - base_m, 2000.0) * 0.55, 6000.0)
    if not turreted:
        thick = min(thick, d.get("max_thick", 20000.0))
        if cell_depth is not None:
            thick = min(thick, cell_depth)
    top_m = base_m + thick
    capped = (cell_depth is not None and thickness_m is not None
              and min(asked, d.get("max_thick", 20000.0)) > cell_depth + 1.0)

    ice = d["ice"]
    if g in ("Cu", "Cb") and acc.get("glaciated", 0.0) > 0.5:
        ice = False                       # base still water; the anvil is ice
    w = d["w"] * acc.get("w_mul", 1.0)
    sigma = extinction(w, d["r"], ice)

    u, v = snd.wind_at((base_m + top_m) * 0.5)
    su, sv = snd.shear_at((base_m + top_m) * 0.5, max(thick, 400.0))
    shear_mag = math.hypot(su, sv) * max(thick, 400.0)
    wind_angle = math.atan2(v, u)

    # Shear stretches the elements along the wind - but only a layer cloud.
    # A convective tower is built by its updraught, not by the shear it sits
    # in, so a deep Cumulonimbus is not drawn out into ribbons.
    aniso = acc.get("aniso", 1.0)
    layerness = 1.0 - min(1.0, max(0.0, d["cum"] + acc.get("cum", 0.0)))
    if aniso <= 1.0:
        aniso = 1.0 + layerness * min(3.0, shear_mag / 8.0)
    aniso = min(aniso, 8.0)
    ang = math.atan2(sv, su) if math.hypot(su, sv) > 1e-6 else wind_angle

    deck = Deck(
        spec=spec, base_m=base_m, top_m=top_m, coverage=coverage,
        element_m=elem_m, element_deg=elem_deg, sigma_e=sigma, ice=ice,
        r_eff_um=float(d["r"]),
        cumuliformity=min(1.0, max(0.0, d["cum"] + acc.get("cum", 0.0))),
        cellularity=min(1.0, max(0.0, d["cell"] + acc.get("cell", 0.0))),
        fibrosity=min(1.0, max(0.0, d["fib"] + acc.get("fib", 0.0))),
        map_fibrosity=min(1.0, max(0.0, d["fib"] + acc.get("fib", 0.0))) * layerness,
        anisotropy=aniso, aniso_angle=ang,
        undulatus=acc.get("undulatus", 0.0),
        lacunosus=acc.get("lacunosus", 0.0),
        perlucidus=acc.get("perlucidus", 0.0),
        # a layer's elements end in blurred, ragged edges (Ac, Cc, and Sc a
        # little less); a cumulus's are sharper
        edge_softness=(0.12 if g in ("Cu", "Cb") else
                       (0.60 if g in ("Ac", "Cc") else (0.45 if g == "Sc" else 0.3))),
        top_variation=acc.get("top_var", TOP_VARIATION.get(g, 0.45)),
        detail=0.6 if g in ("Cu", "Cb", "Sc") else 0.35,
        element_w_m=elem_w,
        cell_dome=1.0 if (g in CELLULAR and spec.species != "len") else 0.0,
        tower=1.0 if (g == "Cb" or (g == "Cu" and spec.species == "con")) else 0.0,
        wind_u=u, wind_v=v, shear_u=su, shear_v=sv,
        turret=turret_frac, lens=acc.get("lens", 0.0),
        roll=acc.get("roll", 0.0), ragged=acc.get("ragged", 0.0),
        hook=acc.get("hook", 0.0), anvil=acc.get("anvil", 0.0),
        seed=seed, reasons=list(reasons or []))
    if size_note:
        deck.reasons.append(size_note)
    if capped:
        deck.reasons.append(f"its elements, {elem_w:.0f} m wide, are at most half as deep: "
                            f"{cell_depth:.0f} m of the {asked:.0f} m cloudy band is drawn"
                            + (" (50 m: the thinnest layer drawn)"
                               if cell_depth > CELL_ASPECT[g] * elem_w + 0.5 else ""))

    # supplementary features and accessory clouds
    sup = set(spec.supplementary)
    accs = set(spec.accessory)
    deck.virga = 1.0 if "vir" in sup else 0.0
    deck.precip = 1.0 if "pra" in sup else 0.0
    deck.mamma = 1.0 if "mam" in sup else 0.0
    deck.asperitas = 1.0 if "asp" in sup else 0.0
    deck.cavum = 1.0 if "cav" in sup else 0.0
    deck.fluctus = 1.0 if "flu" in sup else 0.0
    deck.arcus = 1.0 if "arc" in sup else 0.0
    if "inc" in sup:
        deck.anvil = max(deck.anvil, 1.0)
    deck.pannus = 1.0 if "pan" in accs else 0.0
    deck.pileus = 1.0 if ("pil" in accs or "vel" in accs) else 0.0

    # fall speed of the precipitating particles, for fallstreak geometry
    if deck.virga or deck.precip or deck.hook:
        deck.fall_speed = 1.0 if ice else (5.0 if g in ("Cb", "Ns") else 3.0)
    if deck.hook:
        deck.fall_speed = 0.6                  # ice crystals in cirrus
    return deck


# ------------------------------------------------------------- diagnosis -----

def _stability(snd: Sounding, z0: float, z1: float) -> float:
    """Environmental minus moist-adiabatic lapse rate, K/km.
    Positive means the layer is unstable if saturated - it will grow turrets."""
    return snd.lapse_rate(z0, z1) - snd.moist_adiabatic_lapse((z0 + z1) * 0.5)


def present_weather(snd: Sounding) -> tuple:
    """(kind, description) of the forecast's present weather (WMO code table
    4677 as Open-Meteo gives it): kind is "thunder" (95, 96, 99), "showers"
    (80-82, 85, 86), "rain" (rain or snow: 61-67, 71-75), "drizzle" (51-57,
    and snow grains 77: what Stratus gives) or None."""
    sf = getattr(snd, "surface", None) or {}
    try:
        wc = int(round(float(sf.get("weather_code"))))
    except (TypeError, ValueError):
        return None, ""
    if wc in (95, 96, 99):
        return "thunder", f"thunderstorm{' with hail' if wc != 95 else ''} (weather code {wc})"
    if wc in (80, 81, 82, 85, 86):
        return "showers", f"{'snow' if wc >= 85 else 'rain'} showers (weather code {wc})"
    if 51 <= wc <= 57 or wc == 77:
        return "drizzle", f"{'drizzle' if wc <= 57 else 'snow grains'} (weather code {wc})"
    if 61 <= wc <= 67 or 71 <= wc <= 75:
        return "rain", f"{'rain' if wc <= 67 else 'snow'} (weather code {wc})"
    return None, ""


def _subcloud_rh(snd: Sounding, base: float) -> float:
    zs = [snd.levels[0].z + f * (base - snd.levels[0].z) for f in (0.2, 0.5, 0.8)]
    return sum(snd.rh_at(z) for z in zs) / 3.0


def layer_physics(snd: Sounding, layer) -> dict:
    """The numbers a cloudy layer's species is decided on (layer: a
    sounding.CloudLayer).  All computed on the cloudy part of the layer.

      lapse_in     lapse rate across the layer minus the saturated adiabat
                   there, K/km (over 300 m about the middle when thinner)
      lapse_above  the same over the 1.5 km above the layer's top
      cape         CAPE of a saturated parcel from the layer's peak, over the
                   next 4 km (Sounding.layer_parcel), J/kg
      el           the top of that parcel's buoyant stretch, m, or None
      dthetae      d(theta_e)/dz from 500 m below the layer to 500 m above,
                   K/km (negative: potentially unstable)
      rh_below     mean RH 300, 600 and 900 m under the base, %
      t_base       temperature at the base, C"""
    z0 = snd.levels[0].z
    base, top = layer.base, layer.top
    mid = 0.5 * (base + top)
    gm = snd.moist_adiabatic_lapse
    if top - base >= 300.0:
        lapse_in = snd.lapse_rate(base, top) - gm(mid)
    else:
        lapse_in = snd.lapse_rate(mid - 150.0, mid + 150.0) - gm(mid)
    lapse_above = snd.lapse_rate(top, top + 1500.0) - gm(top + 750.0)
    cape, el = snd.layer_parcel(layer.z_peak)
    zb, zt = max(base - 500.0, z0), top + 500.0
    dthe = (sounding.theta_e(snd.t_at(zt), snd.td_at(zt), snd.p_at(zt))
            - sounding.theta_e(snd.t_at(zb), snd.td_at(zb), snd.p_at(zb))) / ((zt - zb) / 1000.0)
    rh_below = sum(snd.rh_at(max(base - dz, z0)) for dz in (300.0, 600.0, 900.0)) / 3.0
    return dict(lapse_in=lapse_in, lapse_above=lapse_above, cape=cape, el=el,
                dthetae=dthe, rh_below=rh_below, t_base=snd.t_at(base))


# ------------------------------------------------------------ species odds --
#
# How likely each species is, from what observers report.  A SYNOP carries
# three cloud codes (CL, CM, CH).  They name only a few species: CM 8 is
# "Altocumulus castellanus or floccus", CH 9 "Cirrocumulus", CL 6 and 7 are
# Stratus.  For these the probability is fitted to real reports, each paired
# with the forecast sounding of its hour and place (the same Open-Meteo
# pressure levels the program uses).  Every other species is chosen by a
# physical rule, and the panel says so - no probability is shown for it,
# because no observation gives one.  SPECIES_DATA.md records the data, the
# fits and their limits.

#: From SYNOP reports of 2025 (OGIMET archive) paired with the Open-Meteo
#: historical forecast of the same hour and place - the pressure levels this
#: program reads.  SPECIES_DATA.md has the data, the fits and their limits.
#:
#: Ac cas/flo  CM 8 among reports that saw middle cloud (CM 1-9).  Fitted on
#:   8,457 reports with a forecast middle layer at 33 stations in Czechia,
#:   Poland, Slovakia and central Russia, June-August (396 CM 8):
#:   logit P = -3.299 + 0.852 x, x = lapse_in in K/km (deviance -187.5 for
#:   one parameter; area under the ROC curve 0.70 in sample, 0.64-0.82 for
#:   each country left out of the fit; calibrated within 1 % per decile).
#:   rates: the CM 8 share at stations whose observers code honestly (they
#:   report "no middle cloud" at least 2 % of the time): 44 temperate
#:   stations; 164 tropical ones in Malaysia's neighbours, India and Sri
#:   Lanka (8 CM 8 in 43,640 reports - the rarest number here).
#: Cc  CH 9 among reports that saw high cloud (CH 1-9), with a forecast high
#:   layer: 6,819 reports at the same stations and months (82 CH 9):
#:   logit P = -2.3824 + 0.0763 t, t = the layer base's temperature in C
#:   (a warm, low ice layer is likelier to be Cirrocumulus: 3.5% at -12 C,
#:   0.2-0.4% below -39 C); the instability measures do not carry over from
#:   one country to the next and are left out.
#: St  CL 6 or 7 among reports that saw low cloud, by the forecast low
#:   layer's base above the ground (19,760 reports; below 150 m 20% in the
#:   temperate zone and 10% in the tropics, alike above).
SPECIES_DATA = {
    "Ac cas/flo": dict(x="lapse_in", a=-3.2994, b=0.8519, train_rate=0.04461,
                       rates={"tropical": 0.000183, "temperate summer": 0.04078,
                              "temperate winter": 0.004659},
                       source_short="SYNOP CM 8, fitted on 8,457 reports paired with the forecast"),
    "Cc": dict(x="t_base", a=-2.3824, b=0.0763, train_rate=0.01415,
               rates={"tropical": 0.002646, "temperate summer": 0.01188,
                      "temperate winter": 0.009834},
               source_short="SYNOP CH 9, fitted on 6,819 reports paired with the forecast"),
    "St": dict(by_base=[(150.0, 0.134), (300.0, 0.057), (600.0, 0.012), (1000.0, 0.009),
                        (1500.0, 0.006), (1.0e9, 0.011)],
               max_base_m=300.0, source_short="SYNOP CL 6/7 share by forecast base height"),
}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0)))


def climate_of(lat: float, month: int) -> str:
    """'tropical' within 23.5 deg of the equator; elsewhere 'temperate
    summer' in the warm half-year (April-September north of the equator,
    October-March south of it) and 'temperate winter' in the other."""
    if abs(lat) < 23.5:
        return "tropical"
    warm = 4 <= month <= 9
    if lat < 0:
        warm = not warm
    return "temperate summer" if warm else "temperate winter"


def _pct(p: float) -> str:
    q = 100.0 * p
    if q < 0.001:
        return "<0.001%"
    if q < 0.1:
        return f"{q:.2g}%"                       # 0.034%, two significant figures
    if q > 99.995:
        return ">99.99%"
    if q > 99.9:
        return f"{q:.2f}%"
    return f"{q:.1f}%" if (q < 10.0 or q > 90.0) else f"{q:.0f}%"


def _odds_line(items) -> str:
    """'odds: A 96% · B 4%', the most likely first."""
    items = sorted(items, key=lambda t: -t[0])
    return "odds: " + " · ".join(f"{label} {_pct(q)}" for q, label in items)


def reported_odds(model: dict, x: float | None, clim: str) -> float:
    """P(observers report it | the layer), from a model in SPECIES_DATA.

    logit P = a + b x  was fitted where the reports are many (the training
    climate); another climate differs by its own base rate, which shifts the
    log-odds by logit(rate) - logit(training rate) (Bayes' rule with the
    class-conditional distribution of x assumed the same)."""
    z = model["a"] + (model["b"] * x if (x is not None and model.get("b")) else 0.0)
    z += _logit(model["rates"][clim]) - _logit(model["train_rate"])
    return _sigmoid(z)


def _middle_species(snd: Sounding, layer, ph: dict, lat: float, month: int, wind: float,
                    cover: float | None = None):
    """Altocumulus: castellanus/floccus with the probability observers give
    it (SPECIES_DATA['Ac cas/flo']); otherwise lenticularis when a strong
    wind crosses a stable layer in separate elements (rule: under 60 % of
    the sky, the extent opacus asks for a sheet), else stratiformis.  The
    species drawn is the most likely one."""
    m = SPECIES_DATA["Ac cas/flo"]
    clim = climate_of(lat, month)
    p = reported_odds(m, ph[m["x"]], clim)
    mid = 0.5 * (layer.base + layer.top)
    thick = layer.top - layer.base
    n_bv = snd.brunt_vaisala(mid)
    lens = wind > 14.0 and n_bv > 0.012 and thick < 1500.0 and (cover is None or cover < 0.6)
    other = "len" if lens else "str"
    if p > 0.5:
        # castellanus has a common base; floccus is tufts whose ragged bases
        # evaporate into drier air below (rule: the SYNOP code does not
        # separate them)
        sp = "flo" if ph["rh_below"] < 60.0 else "cas"
    else:
        sp = other
    odds = _odds_line([(p, "castellanus or floccus"),
                       (1.0 - p, f"{atlas.SPECIES[other]['name']} or another species")])
    why = [f"castellanus/floccus: {m['source_short']} ({clim} base rate "
           f"{_pct(m['rates'][clim])} of middle-cloud reports); saturated lapse in the layer "
           f"{ph['lapse_in']:+.1f} K/km"]
    if lens:
        why.append(f"{wind:.0f} m/s through a stable layer (N={n_bv:.3f} 1/s) -> lenticularis "
                   f"(a rule: SYNOP has no lenticularis code)")
    why.append(f"drawn: {atlas.SPECIES[sp]['name']}, the most likely")
    return CloudSpec("Ac", sp), odds, why


def _high_species(snd: Sounding, layer, ph: dict, lat: float, month: int, shear: float):
    """Cirrocumulus with the probability observers give it (CH 9); else
    Cirrostratus for an extensive deep veil and Cirrus otherwise (rules)."""
    m = SPECIES_DATA["Cc"]
    clim = climate_of(lat, month)
    x = ph[m["x"]] if m.get("x") else None
    p = reported_odds(m, x, clim)
    cover, thick = layer.cover, layer.top - layer.base
    if p > 0.5:
        spec = CloudSpec("Cc", "str")
        why = ["drawn: Cirrocumulus, the most likely"]
    elif cover >= 0.65 and thick >= 1200:
        spec = CloudSpec("Cs", "neb" if shear < 8 else "fib")
        why = ["extensive uniform ice veil -> Cirrostratus (a rule)"]
    else:
        sp = "unc" if shear > 12 else ("spi" if cover > 0.55 else "fib")
        spec = CloudSpec("Ci", sp)
        why = [f"patchy ice cloud with {shear:.0f} m/s shear -> Cirrus {sp} (a rule)"]
    odds = _odds_line([(p, "Cirrocumulus"), (1.0 - p, "Cirrus or Cirrostratus")])
    why.insert(0, f"Cirrocumulus: {m['source_short']} ({clim} base rate "
                  f"{_pct(m['rates'][clim])} of high-cloud reports); base at {ph['t_base']:.0f} C")
    return spec, odds, why


def stratus_share(height_agl: float) -> float:
    """Share of low-cloud reports that are Stratus (CL 6 or 7), for a
    forecast low layer whose base is this high above the ground."""
    for top, share in SPECIES_DATA["St"]["by_base"]:
        if height_agl < top:
            return share
    return SPECIES_DATA["St"]["by_base"][-1][1]


def _low_species(snd: Sounding, layer, ph: dict, wind: float, cover: float):
    """Stratus only where observers see it - the layer's base within a few
    hundred metres of the ground (CL 6/7 by base height, SPECIES_DATA['St']);
    otherwise Stratocumulus, its species by rule."""
    z0 = snd.levels[0].z
    agl = layer.base - z0
    p_st = stratus_share(agl)
    inv = [i for i in snd.inversions() if layer.base - 200 <= i[0] <= layer.top + 900]
    low_base = agl < SPECIES_DATA["St"]["max_base_m"]
    if low_base and agl <= 250.0 and wind < 6.0 and ph["lapse_in"] < 0.0:
        spec = CloudSpec("St", "fra" if wind > 4 else "neb")
        why = ["saturation at the surface under a stable layer -> Stratus"]
    elif low_base and not inv and cover < 0.5:
        spec = CloudSpec("St", "fra")
        why = ["shallow ragged layer near the ground -> Stratus fractus"]
    else:
        # lenticularis: "lenses or almonds ... with well-defined outlines"
        # (Atlas) - separate elements in a strong wind across a stable layer,
        # as for Altocumulus.  A sheet of 60 % of the sky or more (the extent
        # opacus asks for) has no lenses' outlines: by wind alone, eight
        # overcast opacus sheets in four recorded forecasts were lenticularis
        n_bv = snd.brunt_vaisala(0.5 * (layer.base + layer.top))
        lens = wind > 15 and n_bv > 0.012 and cover < 0.6
        sp = ("cas" if ph["lapse_above"] > 0.4 and ph["cape"] > 0.0 and cover >= 0.5
              else ("len" if lens else "str"))
        spec = CloudSpec("Sc", sp)
        if inv:
            why = [f"boundary layer capped by a {inv[0][2]:.1f} K inversion at {inv[0][0]:.0f} m "
                   f"-> Stratocumulus"]
        elif not low_base:
            why = [f"base {agl:.0f} m above the ground: observers report Stratus for "
                   f"{_pct(p_st)} of such layers -> Stratocumulus"]
        else:
            why = ["extensive low layer -> Stratocumulus"]
        why.append(f"species {atlas.SPECIES[sp]['name']} by a rule (SYNOP has no Sc species)")
        if lens:
            why.append(f"{wind:.0f} m/s through a stable layer (N={n_bv:.3f} 1/s), separate elements "
                       f"({cover*100:.0f}% of the sky) -> lenticularis")
        elif wind > 15 and sp == "str":
            why.append(f"{wind:.0f} m/s, but " + (f"a sheet ({cover*100:.0f}% of the sky) has no "
                       f"lenses' outlines" if cover >= 0.6 else f"no stable layer (N={n_bv:.3f} 1/s) "
                       f"for standing waves") + " -> not lenticularis")
    odds = _odds_line([(p_st, "Stratus"), (1.0 - p_st, "another low genus")]) + " at this base height"
    return spec, odds, why


def _attach_features(spec: CloudSpec, snd: Sounding, ctx: dict,
                     reasons: list[str]) -> None:
    """Add the varieties, supplementary features and accessory clouds that the
    sounding justifies.  Every append records the number behind it."""
    g = spec.genus
    gv = atlas.GENERA[g]
    tau = ctx["tau"]
    ri = ctx["ri"]
    wind = ctx["wind"]
    shear = ctx["shear"]
    cover = ctx["cover"]
    sub_rh = ctx["sub_rh"]
    thick = ctx["thick"]
    t_base = ctx["t_base"]

    # ---- varieties ---------------------------------------------------------
    if "tr" in gv["varieties"] and tau < 3.0:
        spec.varieties.append("tr")
        reasons.append(f"optical depth ~{tau:.1f} -> translucidus")
    elif "op" in gv["varieties"] and tau > 12.0 and cover >= 0.6:
        # Atlas: opacus is a wide sheet, layer or patch dense enough over most
        # of its area to hide the Sun - a sheet, not scattered elements
        # however thick each one is
        spec.varieties.append("op")
        reasons.append(f"an extensive sheet ({cover*100:.0f}% of the sky) of optical depth "
                       f"~{tau:.0f} -> opacus")
    if "un" in gv["varieties"] and 0.25 < ri < 4.0:
        spec.varieties.append("un")
        reasons.append(f"Ri {ri:.2f} in the wave-permitting range -> undulatus")
    if "ra" in gv["varieties"] and wind >= 12 and shear >= 6:
        spec.varieties.append("ra")
        reasons.append(f"{wind:.0f} m/s wind with {shear:.0f} m/s shear -> radiatus")
    if "pe" in gv["varieties"] and 0.45 <= cover <= 0.85 and "op" not in spec.varieties:
        spec.varieties.append("pe")
        reasons.append(f"broken cover ({cover*100:.0f}%) with distinct elements -> perlucidus")
    if "in" in gv["varieties"] and ctx["dir_shear"] > 45.0:
        spec.varieties.append("in")
        reasons.append(f"wind direction turns {ctx['dir_shear']:.0f} deg through the "
                       f"layer -> intortus")

    # ---- supplementary features -------------------------------------------
    precipitating = thick > 900 and (cover > 0.55 or g in ("Cb", "Ns", "Cu"))
    if precipitating and "vir" in gv["supplementary"] and sub_rh < 72:
        spec.supplementary.append("vir")
        reasons.append(f"dry sub-cloud layer ({sub_rh:.0f}% RH) -> virga")
    elif precipitating and "pra" in gv["supplementary"] and sub_rh >= 72:
        spec.supplementary.append("pra")
        reasons.append(f"moist sub-cloud layer ({sub_rh:.0f}% RH) -> praecipitatio")
    if "flu" in gv["supplementary"] and 0.0 < ri < 0.25:
        spec.supplementary.append("flu")
        reasons.append(f"Ri {ri:.2f} below the 0.25 Kelvin-Helmholtz threshold -> fluctus")
    if "cav" in gv["supplementary"] and -30.0 < t_base < -8.0 and thick < 900:
        spec.supplementary.append("cav")
        reasons.append(f"thin supercooled layer at {t_base:.0f} C -> cavum possible")
    if "asp" in gv["supplementary"] and ri < 1.0 and sub_rh > 80 and shear > 10:
        spec.supplementary.append("asp")
        reasons.append(f"strong shear (Ri {ri:.2f}) under a moist stable layer -> asperitas")

    if g == "Cb":
        el, trop, z0, cape = ctx["el"], ctx["trop"], ctx["z0"], ctx["cape"]
        if el and el >= trop - 1800:
            spec.supplementary.append("inc")
            reasons.append(f"equilibrium level {el:.0f} m within 1.8 km of the "
                           f"tropopause ({trop:.0f} m) -> incus")
        s3 = snd.bulk_shear(z0, z0 + 3000)
        s6 = snd.bulk_shear(z0, z0 + 6000)
        if s3 >= 10:
            spec.supplementary.append("arc")
            reasons.append(f"0-3 km shear {s3:.0f} m/s -> arcus")
        if s6 >= 20 and cape >= 1500:
            spec.supplementary += ["mur", "cau"]
            spec.accessory.append("flm")
            reasons.append(f"0-6 km shear {s6:.0f} m/s with CAPE {cape:.0f} J/kg -> "
                           f"supercell features (murus, cauda, flumen)")
        if "inc" in spec.supplementary:
            spec.supplementary.append("mam")
            reasons.append("spreading anvil -> mamma on its underside")

    # ---- accessory clouds --------------------------------------------------
    if "pan" in gv["accessory"] and sub_rh >= 85:
        spec.accessory.append("pan")
        reasons.append(f"saturated sub-cloud air ({sub_rh:.0f}% RH) -> pannus")
    if g == "Cu" and spec.species == "con":
        zt = ctx["base"] + 3000
        if snd.rh_at(zt) > 60 and snd.brunt_vaisala(zt) > 0.013:
            spec.accessory.append("pil")
            reasons.append("moist stable layer just above a growing congestus -> pileus")

    err, _ = spec.validate()
    if err:
        if "tr" in spec.varieties and "op" in spec.varieties:
            spec.varieties.remove("tr")
        reasons.append("combination corrected: " + "; ".join(err))


def _direction_shear(snd: Sounding, z0: float, z1: float) -> float:
    a = math.degrees(math.atan2(*reversed(snd.wind_at(z0))))
    b = math.degrees(math.atan2(*reversed(snd.wind_at(z1))))
    return abs((b - a + 180.0) % 360.0 - 180.0)


def diagnose(snd: Sounding, seed: int = 20260910, max_decks: int = 6) -> list[Deck]:
    """Which Atlas clouds does this atmosphere actually support?

    Layer clouds come from the moist layers; convective clouds come from a
    lifted parcel, because that is what they physically are.
    """
    rng = random.Random(seed)
    lat = snd.lat
    cape, cin, z_lcl, lfc, el, _ = snd.parcel(mixed_depth=400)
    trop = snd.tropopause()
    z0 = snd.levels[0].z
    decks: list[Deck] = []
    seen_etage: dict[str, int] = {}
    low_hi = atlas.etage_range_m("Sc", lat)[1]

    def etage_of_base(b: float) -> str:
        for g in ("Ci", "Ac", "Sc"):
            lo, hi = atlas.etage_range_m(g, lat)
            if lo <= b <= hi:
                return atlas.etage_of(g)
        return "high" if b > atlas.etage_range_m("Ac", lat)[1] else "low"

    def context(base, top, cover, genus) -> dict:
        thick = max(top - base, 100.0)
        mid = (base + top) * 0.5
        gd = GENUS_DEFAULTS[genus]
        return dict(base=base, top=top, cover=cover, thick=thick,
                    tau=extinction(gd["w"], gd["r"], gd["ice"]) * thick,
                    ri=snd.richardson(min(top, base + 400.0)),
                    wind=math.hypot(*snd.wind_at(mid)),
                    shear=snd.bulk_shear(base, top),
                    dir_shear=_direction_shear(snd, base, top),
                    sub_rh=_subcloud_rh(snd, base),
                    t_base=snd.t_at(base), t_top=snd.t_at(top),
                    el=el, trop=trop, z0=z0, cape=cape)

    # ---------------------------------------------------------- layer clouds
    month = snd.when.month if getattr(snd, "when", None) else 7
    # The Atlas: a precipitating layer is Cumulonimbus when its precipitation
    # comes in showers or with lightning, thunder or hail (else
    # Nimbostratus); thunder can be the only sign that a Cumulonimbus is there
    # (CL = 9, by convention when calvus and capillatus cannot be told apart).
    wx, wx_text = present_weather(snd)

    def pct(k):
        try:
            return float(snd.surface.get(k))
        except (TypeError, ValueError, AttributeError):
            return None
    cc_low, cc_mid = pct("cloud_cover_low"), pct("cloud_cover_mid")
    overcast_lm = cc_low is not None and cc_mid is not None and min(cc_low, cc_mid) >= 80.0
    for layer in snd.cloud_layers():
        base, top, cover = layer.base, layer.top, layer.cover
        if cover < 0.06:
            continue
        thick = top - base
        mid = (base + top) * 0.5
        et = etage_of_base(base)
        ph = layer_physics(snd, layer)
        t_base, t_top = snd.t_at(base), snd.t_at(top)
        ri = snd.richardson(min(top, base + 400.0))
        wind = math.hypot(*snd.wind_at(mid))
        shear = snd.bulk_shear(base, top)
        sub_rh = _subcloud_rh(snd, base)
        reasons = [f"cloudy layer {base:.0f}-{top:.0f} m (where the cloud fraction is over "
                   f"half its peak; faint to {layer.edge_base:.0f}-{layer.edge_top:.0f} m), "
                   f"cover {cover*100:.0f}%",
                   f"temperature {t_base:.0f} to {t_top:.0f} C",
                   f"lapse minus saturated adiabat: {ph['lapse_in']:+.1f} K/km in the layer, "
                   f"{ph['lapse_above']:+.1f} above it; a parcel from it: CAPE "
                   f"{ph['cape']:.0f} J/kg",
                   f"wind {wind:.0f} m/s, shear across layer {shear:.0f} m/s, Ri {ri:.2f}"]
        odds = ""
        rise = None

        # a deep, nearly overcast layer from the low or middle étage that
        # precipitates: Cumulonimbus or Nimbostratus, by its precipitation.
        # With thunder the cloud is as deep as it reaches at all (its faint
        # top): a thundercloud's upper part is thin in the forecast's
        # fraction, but it is there.
        # (Cumulonimbus and Nimbostratus stand on the low étage: a deep layer
        # based higher is the middle étage's - Altostratus, Altocumulus - and
        # the thunder's Cumulonimbus is the convective one below)
        deep = thick >= 2500 and top > low_hi and et == "low"
        thunder_layer = (wx == "thunder" and cover >= 0.5 and et == "low"
                         and layer.edge_top - base >= 2500 and layer.edge_top > low_hi)
        shower_layer = wx == "showers" and cover >= 0.5 and deep
        if thunder_layer:
            top = min(max(top, layer.edge_top), trop + 1200.0)
            thick = top - base
            t_top = snd.t_at(top)
            spec = CloudSpec("Cb", "cap")
            reasons.append(f"deep ({thick:.0f} m) layer, {cover*100:.0f}% cover, with {wx_text}: "
                           f"thunder makes it Cumulonimbus (Atlas); capillatus by the CL = 9 "
                           f"convention")
        elif shower_layer:
            # showers come from convective cloud: Cumulonimbus once its top is
            # deep and cold enough to glaciate, else Cumulus congestus
            cb = thick >= 4000 and t_top < -20.0
            spec = CloudSpec("Cb", "cap" if t_top < -35.0 else "cal") if cb else CloudSpec("Cu", "con")
            reasons.append(f"deep ({thick:.0f} m) layer, {cover*100:.0f}% cover, with {wx_text}: "
                           f"showery precipitation comes from convective cloud -> "
                           + ("Cumulonimbus (Atlas: showery, not Nimbostratus)" if cb else
                              f"Cumulus congestus (top {t_top:.0f} C, not glaciated)"))
        elif wx == "rain" and deep and (cover >= 0.8 or overcast_lm):
            # Nimbostratus: its look is blurred by rain or snow that falls
            # almost without pause (Atlas) - drizzle is Stratus's.  Overcast:
            # the layer's own fraction, or the forecast's low and middle cover
            # (a deep layer's levels overlap; its peak fraction understates it)
            spec = CloudSpec("Ns")
            reasons.append(f"deep ({thick:.0f} m) overcast layer from the low étage into the "
                           f"middle, with {wx_text} -> Nimbostratus")
            if overcast_lm and cover < max(cc_low, cc_mid) / 100.0:
                # a layer is Nimbostratus because it is overcast: it is drawn
                # so (at its own peak fraction it kept gaps - 70 % in Paris,
                # 1 Oct 10:00, under 81 % low and 100 % middle cover).  It
                # spans both étages, one continuous layer: their covers
                # overlap fully, so it covers as much as the larger
                cover = max(cc_low, cc_mid) / 100.0
                reasons.append(f"drawn at the forecast's cover of the étages it spans, {cover*100:.0f}%")
        elif et == "high":
            spec, odds, why = _high_species(snd, layer, ph, lat, month, shear)
            reasons += why
        elif et == "middle":
            deep = thick >= 2500 and cover >= 0.8
            if deep and base <= low_hi + 500 and sub_rh >= 78:
                spec = CloudSpec("Ns")
                reasons.append(f"deep ({thick:.0f} m) saturated layer over moist "
                               f"sub-cloud air ({sub_rh:.0f}% RH) -> Nimbostratus")
            elif deep:
                spec = CloudSpec("As")
                reasons.append(f"thick ({thick:.0f} m) uniform middle sheet -> Altostratus")
            else:
                spec, odds, why = _middle_species(snd, layer, ph, lat, month, wind, cover)
                reasons += why
        else:
            spec, odds, why = _low_species(snd, layer, ph, wind, cover)
            reasons += why
        if spec.species == "cas" and ph["el"] is not None:
            rise = ph["el"] - top              # the turrets rise as far as the parcel does

        ctx = context(base, top, cover, spec.genus)
        _attach_features(spec, snd, ctx, reasons)
        if seen_etage.get(et) and "du" in atlas.GENERA[spec.genus]["varieties"]:
            spec.varieties.append("du")
            reasons.append("a second layer in the same étage -> duplicatus")
        seen_etage[et] = seen_etage.get(et, 0) + 1
        if odds:
            reasons.insert(0, odds)
        deck = deck_from_spec(spec, snd, base_m=base, coverage=cover,
                              thickness_m=thick, seed=rng.randrange(1 << 30),
                              reasons=reasons, turret_rise_m=rise)
        deck.score = cover
        deck.odds = odds
        decks.append(deck)

    # ------------------------------------------------------ convective cloud
    # an overcast low layer keeps the sun off the ground and the forecast's
    # surface CAPE from being released - unless showers or thunder say the
    # convection is there
    overcast_low = any(d.base_m < 1500 and d.coverage >= 0.85 for d in decks)
    convective_wx = wx in ("thunder", "showers")
    if (cape >= 60 and z_lcl <= 3500 and (not overcast_low or convective_wx)
            and not any(d.spec.genus in ("Cu", "Cb") for d in decks)):
        base = max(z_lcl, z0 + 60.0)
        t_el = snd.t_at(min(el, trop)) if el else 0.0
        deep = bool(el) and (el - base) >= 4000 and t_el < -20
        reasons = [f"CAPE {cape:.0f} J/kg, {sounding.inhibition_text(cin, lfc)}, LCL {base:.0f} m",
                   f"equilibrium level {'-' if not el else f'{el:.0f} m'} "
                   f"at {t_el:.0f} C, tropopause {trop:.0f} m"]
        low_cover = snd.surface.get("cloud_cover_low")
        if deep and cape >= 500:
            spec = CloudSpec("Cb", "cap" if t_el < -35 else "cal")
            top = min(el, trop + 1200.0)
            # 0 % low cloud is a value, not a gap: no cover (it read as missing: 4/8)
            cover = min(0.7, (low_cover / 100.0) if low_cover is not None else 0.5)
            reasons.append(f"deep buoyancy with a glaciating top -> Cumulonimbus {spec.species}")
        elif wx == "thunder" and el:
            # thunder: Cumulonimbus, whatever its depth says (Atlas: when the
            # cloud's look does not decide, lightning, thunder or hail make it
            # Cumulonimbus)
            spec = CloudSpec("Cb", "cap")
            top = min(max(el, base + 4000.0), trop + 1200.0)
            cover = min(0.7, (low_cover / 100.0) if low_cover is not None else 0.5)
            reasons.append(f"{wx_text}: thunder makes it Cumulonimbus (Atlas); capillatus by "
                           f"the CL = 9 convention")
        else:
            depth = (el - base) if el else 900.0
            sp = "hum" if depth < 1100 else ("med" if depth < 2400 else "con")
            spec = CloudSpec("Cu", sp)
            top = base + {"hum": 800.0, "med": 1800.0, "con": 4200.0}[sp]
            cover = min(0.6, (low_cover / 100.0) if low_cover is not None else 0.22)
            reasons.append(f"convective depth {depth:.0f} m -> Cumulus {sp}")
        ctx = context(base, top, cover, spec.genus)
        _attach_features(spec, snd, ctx, reasons)
        deck = deck_from_spec(spec, snd, base_m=base, coverage=cover,
                              thickness_m=top - base, seed=rng.randrange(1 << 30),
                              reasons=reasons)
        deck.score = cover + 0.2
        decks.append(deck)

    decks.sort(key=lambda d: d.base_m)
    for i, d in enumerate(decks):
        d.map_index = i
    return decks[:max_decks]


def summarise(decks: list[Deck]) -> str:
    if not decks:
        return "clear sky"
    return " / ".join(f"{d.spec.abbrev()} {d.coverage*8:.0f}/8 @ {d.base_m:.0f} m"
                      for d in decks)
