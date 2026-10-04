# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
The OpenGL side of the sky: textures, uniforms, the passes of a frame.

A frame is
  1. the Beer shadow map - one ray per texel from the Sun (or the Moon)
     through every deck and the simulated cloud field;
  2. the sky - atmosphere with crepuscular rays, the Atlas decks, the
     simulated clouds with rain, rainbows and lightning, the ground with the
     clouds' shadows - in linear HDR;
  3. temporal accumulation with exact rotation reprojection;
  4. bloom (Jimenez 2014) and tone mapping (AgX, or Khronos PBR Neutral).

Needs an OpenGL 3.3 core context; the simulation itself (crm_gpu.py) is the
only part that needs 4.3.
"""

from __future__ import annotations

import math

import numpy as np
try:
    import moderngl
except ImportError:          # the model, the census and the self-test run without it
    moderngl = None

from . import atmosphere, mie, noise, shaders, timeline
from .scene import Deck

MAXDECKS = 12          # layer groups the renderer holds (timeline.Sky.GROUPS)
SLOTS = timeline.SLOTS # realizations of each layer's pattern blended at once
DECKTEX = 18
MAP_SIZE = timeline.MAP_N
#: A layer of rounded masses (Deck.cell_dome) fills its depth only in part:
#: flat-based elements with domed roofs, lumpy, thinnest at their edges.  The
#: column through an element's middle (95th percentile of the columns through
#: the elements in view, 16 steps each) came out at 1/2.7 to 1/3.5 of
#: extinction x depth: Paris altocumulus 2.19 of 6.52, stratocumulus 13.4 of
#: 39.1 and 8.3 of 28.6, cirrocumulus 0.08 of 0.21 - the panel's optical depth
#: was about three times what was drawn.  The extinction is raised by this, so
#: an element's middle has the deck's optical depth.  (shaders.py holds the
#: same number.)
DOME_COLUMN = 3.0

QUALITY = {
    "low":    dict(steps=72, light=4, atm=16, scale=0.5, sim=200, shadow=256, shsteps=64),
    "medium": dict(steps=144, light=5, atm=24, scale=0.7, sim=320, shadow=384, shsteps=96),
    "high":   dict(steps=256, light=6, atm=32, scale=1.0, sim=380, shadow=512, shsteps=128),
    "photo":  dict(steps=480, light=8, atm=48, scale=1.0, sim=600, shadow=768, shsteps=192),
}

TONES = ("AgX", "AgX punchy", "PBR Neutral")


class Camera:
    def __init__(self, az=180.0, alt=25.0, fov=70.0):
        self.az = az
        self.alt = alt
        self.fov = fov

    def clamp(self):
        self.alt = max(-89.0, min(89.0, self.alt))
        self.az %= 360.0
        self.fov = max(4.0, min(140.0, self.fov))

    def basis(self):
        self.clamp()
        a, e = math.radians(self.az), math.radians(self.alt)
        fwd = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)], "f4")
        world_up = np.array([0.0, 1.0, 0.0], "f4")
        right = np.cross(world_up, fwd)
        n = np.linalg.norm(right)
        if n < 1e-5:
            right = np.array([1.0, 0.0, 0.0], "f4")
        else:
            right = right / n
        up = np.cross(fwd, right)
        return right, up / max(np.linalg.norm(up), 1e-6), fwd


def jde_params(r_eff_um: float) -> tuple:
    """Jendersie & d'Eon (2023) fit of the Mie phase function of cloud
    droplets, for droplet diameter d = 2 r_eff in micrometres (valid 5-50).
    Returns (gHG, gD, alpha, wD)."""
    d = min(max(2.0 * r_eff_um, 5.01), 50.0)
    g_hg = math.exp(-0.0990567 / (d - 1.67154))
    g_d = math.exp(-2.20679 / (d + 3.91029) - 0.428934)
    alpha = math.exp(3.62489 - 8.29288 / (d + 5.52825))
    w_d = math.exp(-0.599085 / (d - 0.641583) - 0.665888)
    return (g_hg, g_d, alpha, w_d)


def deck_map_extent(deck: Deck) -> float:
    """Physical size of one deck-map tile, metres.  About two dozen elements
    across, so the pattern does not repeat visibly, and bounded so the map
    keeps a useful resolution."""
    return timeline.map_extent(deck.element_m)


def build_deck_map(deck: Deck, size: int = MAP_SIZE) -> np.ndarray:
    """One deck's control map at one moment (cover, tops, age): a single
    realization of its pattern through the renderer's rule.  Fibres one or
    two texels wide would beat into a woven moire, so a fibrous pattern is
    smoothed before its cover is set (the cover stays exact)."""
    ex = deck_map_extent(deck)
    return noise.deck_map(
        size, ex, deck.element_m, deck.coverage,
        cellularity=deck.cellularity,
        anisotropy=deck.anisotropy,
        angle_rad=deck.aniso_angle,
        undulatus=deck.undulatus,
        lacunosus=deck.lacunosus,
        perlucidus=deck.perlucidus,
        edge_softness=deck.edge_softness,
        fibrosity=deck.map_fibrosity,
        top_variation=deck.top_variation,
        turret=deck.turret,
        seed=deck.seed & 0x7FFFFFFF,
        blur_px=1.3 if deck.map_fibrosity > 0.3 else 0.0,
        lumpy=float(getattr(deck, "cell_dome", 0.0)))


def _deck_rows(d: Deck) -> np.ndarray:
    """Rows 0-9 of a deck's parameters: the layer itself."""
    p = np.zeros((10, 4), "f4")
    ex = deck_map_extent(d)
    # cumuliform decks want noise finer than an element; stratiform decks
    # want it coarser, so it shades the elements instead of shredding them
    k = 2.6 - 2.3 * d.cumuliformity
    shape_scale = float(min(max(d.element_m * k, 220.0), 14000.0))
    dome = float(min(max(getattr(d, "cell_dome", 0.0), 0.0), 1.0))
    if dome > 0.0:
        # a layer's elements are shaped, not only shaded, by lumps about
        # half an element across: lobed outlines, brighter and greyer parts
        ew = getattr(d, "element_w_m", 0.0) or d.element_m
        shape_scale = (1.0 - dome) * shape_scale + dome * float(min(max(0.5 * ew, 40.0), 14000.0))
    ice = d.ice_amount
    p[0] = (d.base_m, d.top_m, d.sigma_e * (1.0 + dome * (DOME_COLUMN - 1.0)), d.cumuliformity)
    p[1] = (d.cellularity, d.fibrosity, d.detail, d.edge_softness)
    p[2] = (ex, d.wind_u, d.wind_v, ice)
    p[3] = (d.shear_u, d.shear_v, max(d.fall_speed, 0.2), shape_scale)
    p[4] = (d.virga, d.precip, d.mamma, d.asperitas)
    p[5] = (d.anvil, d.cavum, d.fluctus, d.arcus)
    # (castellanus takes its layer top from the realizations showing, row
    # 14.w - Deck.turret only says which realizations are built with turrets;
    # row 6.x carries how much the elements are rounded masses, Deck.cell_dome)
    p[6] = (dome, d.lens, d.roll, d.ragged)
    p[7] = (d.hook, d.pannus, d.pileus, (d.seed & 0xFFFF) / 65535.0)
    p[8] = jde_params(getattr(d, "r_eff_um", 10.0))
    # the deck's own halo and corona strengths: the shader weights each by
    # how much of a ray's light the deck gives (uHaloStrength, uCoronaStrength
    # only switch them on), so a deck's phenomenon fades as the deck changes
    halo, corona = deck_optics(d)
    # 9.y: a convective cloud's plume shape (Deck.tower; the ice is 2.w)
    p[9] = (getattr(d, "r_eff_um", 10.0), float(min(max(getattr(d, "tower", 0.0), 0.0), 1.0)),
            halo, corona)
    return p


def _state_rows(st) -> np.ndarray:
    """Rows 10-17: the layer's evolving pattern (timeline.DeckState)."""
    p = np.zeros((8, 4), "f4")
    d = st.deck
    w, ex, off = st.weights, st.extents, st.offsets
    p[0] = (w[0], w[1], w[2], st.zstar)
    p[1] = (max(ex[0], 1.0), max(ex[1], 1.0), max(ex[2], 1.0), st.soft_z)
    p[2] = (off[0][0], off[0][1], off[1][0], off[1][1])
    p[3] = (off[2][0], off[2][1], st.gap_amount, st.gap_star)
    p[4] = (float(st.has_turret[0]), float(st.has_turret[1]), float(st.has_turret[2]),
            st.turret_tb)
    p[5] = (float(st.has_gap[0]), float(st.has_gap[1]), float(st.has_gap[2]),
            float(np.clip(d.top_variation, 0.0, 1.0)))
    # 16.w: the layer's cover - its elements' light leaks between them
    # (shaders.deckRadiance)
    p[6] = (st.cavum_offset[0], st.cavum_offset[1], st.cavum_extent,
            float(min(max(d.coverage, 0.0), 1.0)))
    p[7] = tuple(float(v) for v in st.wave)
    return p


def static_state(d: Deck, group: int, pack_meta: dict | None = None):
    """A deck that does not evolve: one realization (slot 0), no drift."""
    ex = deck_map_extent(d)
    m = pack_meta or {}
    gaps = m.get("gaps", "")
    turret = bool(m.get("turret", d.turret > 0.0))
    amp = noise.UND_Z * max(float(d.undulatus), 0.0)
    wave = (amp,) + timeline.wave_vector(d) + (0.0,) if amp > 1e-6 else (0.0, 0.0, 0.0, 0.0)
    return timeline.DeckState(
        track=-1, group=group, deck=d, weights=(1.0, 0.0, 0.0), extents=(ex, ex, ex),
        offsets=((0.0, 0.0),) * 3, has_turret=(turret, False, False),
        has_gap=(bool(gaps), False, False), gap_kind=gaps,
        zstar=noise.zstar_with_wave(d.coverage, wave[0]), wave=wave,
        soft_z=noise.SOFT_PER_EDGE * d.edge_softness, theta=0.0,
        cavum_offset=(0.0, 0.0), cavum_extent=ex, base_cover=d.coverage,
        turret_tb=timeline.turret_top(d.turret) if turret else 0.0,
        gap_amount=timeline.gap_amount(d, gaps), gap_star=noise.gap_star(gaps) if gaps else 0.0)


def pack_states(states: list) -> np.ndarray:
    """DECKTEX RGBA texels per layer group, matching the layout the shader
    reads.  An empty group has zero extinction and is skipped."""
    p = np.zeros((MAXDECKS, DECKTEX, 4), "f4")
    p[:, 10, 3] = 40.0                         # no cover
    for st in states:
        g = st.group
        if not (0 <= g < MAXDECKS):
            continue
        p[g, :10] = _deck_rows(st.deck)
        p[g, 10:] = _state_rows(st)
    return np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)


def pack_deck_params(decks: list[Deck], observer_alt: float = 0.0) -> np.ndarray:
    """DECKTEX RGBA texels per deck (decks that do not evolve, one per
    group), matching the layout the shader reads."""
    return pack_states([static_state(d, i) for i, d in enumerate(decks[:MAXDECKS])])


def optic_kind(d: Deck) -> tuple[float, float, float]:
    """(halo, corona, ice corona) propensities of a deck's kind of cloud
    (scene.Deck.optic_kind)."""
    return d.optic_kind()


def _smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def deck_optics(d: Deck) -> tuple[float, float]:
    """(halo strength, corona strength) of one deck.

    The halo needs ice crystals, a corona droplets (or, weaker, small ice
    crystals in a very thin Cirrocumulus), and both a thin cloud: a halo
    does not survive an optical depth of about four, a corona of six.  Every
    step is continuous - in the amount of ice, the optical depth and the
    kind of cloud - so a layer that thickens, freezes or changes its name
    between two hours fades its halo out instead of switching it off."""
    tau = d.optical_depth
    hw, cw, iw = optic_kind(d)
    ice = _smoothstep(0.35, 0.65, d.ice_amount)
    halo = hw * ice * d.coverage * math.exp(-tau / 2.5) * (1.0 - _smoothstep(3.5, 4.5, tau))
    corona = cw * (1.0 - ice) * d.coverage * math.exp(-tau / 3.0) * 0.6 * \
        (1.0 - _smoothstep(5.5, 6.5, tau))
    corona = max(corona, iw * ice * d.coverage * 0.3 * (1.0 - _smoothstep(1.5, 2.5, tau)))
    return halo, corona


def eye_frame(p, observer_alt: float) -> tuple:
    """A lightning point from simdriver.world_point (made with the eye on the
    ground) in the frame of an eye observer_alt above the ground: the same
    point, that much lower.  Without it a flash seen from an aircraft or a
    satellite was drawn as high above the storm as the eye was."""
    return (float(p[0]), float(p[1]) - float(observer_alt), float(p[2]))


def _halton(i: int, b: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def pixel_jitter(frame: int) -> tuple[float, float]:
    """Sub-pixel offset of frame `frame`, in pixels, each in [-0.5, 0.5):
    the Halton (2, 3) points, which cover the pixel evenly in any run of
    frames, so that a still picture averages over the whole pixel."""
    i = frame % 64 + 1
    return _halton(i, 2) - 0.5, _halton(i, 3) - 0.5


def optical_phenomena(decks: list[Deck]) -> tuple[float, float]:
    """(halo strength, corona strength) of a set of decks: the strongest.
    Where each is seen is decided per ray in the shader, by the decks that
    make it and by what lies in front of them."""
    halo = corona = 0.0
    for d in decks:
        h, c = deck_optics(d)
        halo, corona = max(halo, h), max(corona, c)
    return halo, corona


def haze_from_visibility(vis_m: float) -> float:
    """Boundary-layer aerosol extinction at the ground, m^-1, from the
    meteorological visibility by Koschmieder's relation beta = 3.912 / V,
    less what the clean-air tables already hold (Rayleigh ~1.3e-5 and
    aerosol 4.4e-6 m^-1 at 550 nm)."""
    v = min(max(float(vis_m), 1000.0), 120000.0)
    return max(3.912 / v - 1.8e-5, 0.0)


def visibility_guess(rh_pct: float) -> float:
    """Visibility when the weather source gives none: 45 km in dry air,
    falling as aerosol swells with humidity (this program's rough rule)."""
    x = min(max((rh_pct - 40.0) / 60.0, 0.0), 1.0)
    return 45000.0 * (1.0 - 0.8 * x ** 1.5)


class SimView:
    """What the renderer needs from a running simulation.  Filled in by
    simdriver.SimDriver each frame."""

    def __init__(self):
        self.textures = None          # (optA, optB, auxA, auxB)
        self.mix = 1.0
        self.size = (1.0, 1.0, 1.0)   # lx, ly, depth of the domain
        self.cell = (1.0, 1.0, 1.0)
        self.z0 = 0.0                 # height of the domain's floor
        self.zlo = 0.0                # heights between which there is cloud or precipitation
        self.zhi = 1.0
        self.zbase = 0.0              # height of the lowest cloud
        self.move = (0.0, 0.0)        # the domain's drift, m/s
        self.obs = (0.0, 0.0)
        self.tile = 3                 # bit 1 east-west, bit 2 north-south
        self.time = 0.0
        self.jde = jde_params(10.0)
        self.flash_pos = (0.0, 0.0, 0.0)
        self.flash_power = 0.0
        self.bolt = []                # list of (x, y, z, brightness), relative to the eye


class SkyRenderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int,
                 quality: str = "medium"):
        self.ctx = ctx
        V = shaders.VERT
        self.prog = ctx.program(vertex_shader=V, fragment_shader=shaders.SKY_FRAG)
        self.shadow_prog = ctx.program(vertex_shader=V, fragment_shader=shaders.SHADOW_FRAG)
        self.accum_prog = ctx.program(vertex_shader=V, fragment_shader=shaders.ACCUM_FRAG)
        self.down_prog = ctx.program(vertex_shader=V, fragment_shader=shaders.BLOOM_DOWN)
        self.up_prog = ctx.program(vertex_shader=V, fragment_shader=shaders.BLOOM_UP)
        self.tone_prog = ctx.program(vertex_shader=V, fragment_shader=shaders.TONEMAP)
        self.present = ctx.program(vertex_shader=V, fragment_shader=shaders.PRESENT)
        self.quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())

        def vao(p):
            return ctx.vertex_array(p, [(self.quad, "2f", "in_pos")])
        self.vao = vao(self.prog)
        self.svao = vao(self.shadow_prog)
        self.avao = vao(self.accum_prog)
        self.dvao = vao(self.down_prog)
        self.uvao = vao(self.up_prog)
        self.tvao = vao(self.tone_prog)
        self.pvao = vao(self.present)

        tr, ms = atmosphere.build_luts()
        self.t_trans = ctx.texture(tr.shape[1::-1], 3, tr.tobytes(), dtype="f4")
        self.t_ms = ctx.texture(ms.shape[1::-1], 3, ms.tobytes(), dtype="f4")
        for t in (self.t_trans, self.t_ms):
            t.repeat_x = t.repeat_y = False
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)

        shape = noise.cached("shape128", noise.build_shape_volume, 128)
        detail = noise.cached("detail32", noise.build_detail_volume, 32)
        blue = noise.build_blue_noise(64)
        self.t_shape = ctx.texture3d((128, 128, 128), 4, shape.tobytes())
        self.t_detail = ctx.texture3d((32, 32, 32), 3, detail.tobytes())
        self.t_blue = ctx.texture((64, 64), 1, blue.tobytes())
        for t in (self.t_shape, self.t_detail):
            t.repeat_x = t.repeat_y = t.repeat_z = True
        self.t_shape.build_mipmaps()
        self.t_shape.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_detail.build_mipmaps()
        self.t_detail.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_blue.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # every layer group's three pattern realizations (z, z^2, tops or
        # turrets, gaps), mip-mapped: a coarse level holds the mean and the
        # mean square of the pattern, so a footprint's cloud fraction can be
        # taken from both
        self.t_maps = ctx.texture_array((MAP_SIZE, MAP_SIZE, MAXDECKS * SLOTS), 4, dtype="f2")
        self.t_maps.build_mipmaps()
        self.t_maps.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_maps.repeat_x = self.t_maps.repeat_y = True
        # the layers' parameters: a uniform block (DeckBlock), DECKTEX vec4 each,
        # on the context's last uniform-block binding point.  The binding
        # points are shared by everything drawing in the window: pyglet keeps
        # 0 for its WindowBlock (the projection its shapes and text are drawn
        # with) and hands out 1, 2, ... to any other block of its own.  Bound
        # to 0, the layers' numbers became the panels' projection, and the
        # panels vanished (28-30 Sep 2026).
        try:
            n_bind = int(ctx.info.get("GL_MAX_UNIFORM_BUFFER_BINDINGS", 36))
        except Exception:                                          # noqa: BLE001
            n_bind = 36
        self.deck_binding = max(n_bind - 1, 1)
        self.deck_ubo = ctx.buffer(reserve=MAXDECKS * DECKTEX * 16)
        self.deck_ubo.write(pack_states([]).tobytes())
        self._params_prev = None
        self.deck_changed = False
        self.deck_states = []

        # row 0: rain (the rainbows); rows 1-16: nacreous-cloud ice spheres
        # of mie.PSC_RADII (their iridescence)
        rain = mie.rain_phase_lut().astype("f4")
        psc = mie.psc_phase_lut().astype("f4")
        if psc.shape[1] != rain.shape[0]:
            u_r = (np.arange(rain.shape[0]) + 0.5) / rain.shape[0]
            u_p = (np.arange(psc.shape[1]) + 0.5) / psc.shape[1]
            psc = np.stack([np.stack([np.interp(u_r, u_p, row[:, c]) for c in range(3)], -1)
                            for row in psc]).astype("f4")
        table = np.concatenate([rain[None], psc], 0)                  # (17, n, 3)
        self.t_rain = ctx.texture((table.shape[1], table.shape[0]), 3,
                                  np.ascontiguousarray(table).tobytes(), dtype="f4")
        self.t_rain.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_rain.repeat_x = self.t_rain.repeat_y = False

        # the weather around the place (regional maps), off until set_region
        self.t_rwarp = self.t_rcover = None
        self.region_on = False
        self.region_half = 1.0
        self.region_u = np.zeros((8, 4), "f4")
        self.region_u[0, 1:] = self.region_u[4, 1:] = 100.0
        self.region_new = 1.0
        self.region_sim_et = 0
        self._region_key = None
        self._region_arrays(1)
        self.tau_th = [0.0] * 17
        # the column optical depth the regional thinning decides on
        self.t_keep = ctx.texture((1, 1), 1, np.zeros(1, "f4").tobytes(), dtype="f4")
        self.keep_on = False
        # empty stand-ins for the simulation's textures
        self.t_sim_empty = []
        for _ in range(4):
            t = ctx.texture3d((1, 1, 1), 4, np.zeros(4, "f2").tobytes(), dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.t_sim_empty.append(t)
        self.sim: SimView | None = None
        #: how much of the model's field is shown, 0..1 (handovers; shaders.simKeep)
        self.sim_fade = 1.0

        self.tr_lut = tr
        self.ms_lut = ms
        self.num_decks = 0
        self.decks = []
        self.deck_maps_cpu = []
        self.halo = 0.0
        self.corona = 0.0
        self.quality = quality
        self.exposure_bias = 0.0
        self.tone = 1
        self.bloom_strength = 0.04
        self.haze = haze_from_visibility(25000.0)
        self.show_stars = True
        self.ground_albedo = 0.13    # snow-free ground
        self.snow = 0.0              # fraction of the ground under lying snow (sounding.snow_cover)
        self.dryness = 0.0           # 0 green .. 1 bare sand (sounding.ground_dryness)
        self.ground_elev = 0.0       # the ground's height above sea level, m (the station's)
        self.observer_alt = 1.7      # eye height above the ground, metres
        self.upper = {}              # uniforms of the clouds above the weather (upper.py)
        self.adaptive = True         # meter the frame (auto_exposure)
        self.adapt_rate = 0.08       # per metered frame
        self._fbos = {}
        self._bloom = []
        self._prev_basis = None
        self.min_blend = 0.06
        self.anim_blend = 0.35
        self._was_animating = False  # the last accumulated frame was of a change
        self.shadow_size = 0
        self.shadow_dirty = True
        self.frame = 0
        self.resize(width, height)
        self.accum_frames = 0

        self.units = {"tTransmittance": 0, "tMultiScatter": 1, "tShape": 2, "tDetail": 3,
                      "tBlue": 4, "tDeckMaps": 5, "tShadow": 7,
                      "tRainPhase": 8, "tSimOptA": 9, "tSimOptB": 10, "tSimAuxA": 11,
                      "tSimAuxB": 12, "tRegionWarp": 13, "tRegionCover": 14, "tSimKeep": 15}
        for prog in (self.prog, self.shadow_prog):
            for name, unit in self.units.items():
                if name in prog:
                    prog[name].value = unit
            if "DeckBlock" in prog:
                prog["DeckBlock"].binding = self.deck_binding

    # ---------------------------------------------------------------- state --
    def resize(self, width: int, height: int):
        self.width, self.height = max(width, 8), max(height, 8)
        s = QUALITY[self.quality]["scale"]
        rw, rh = max(int(self.width * s), 8), max(int(self.height * s), 8)
        for f in self._fbos.values():
            f[0].release()
            f[1].release()
        for f in self._bloom:
            f[0].release()
            f[1].release()
        self._fbos = {}
        for k in ("scene", "accA", "accB"):
            tex = self.ctx.texture((rw, rh), 4, dtype="f4" if k != "scene" else "f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            self._fbos[k] = (tex, fbo)
        tex = self.ctx.texture((rw, rh), 4, dtype="f1")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._fbos["final"] = (tex, self.ctx.framebuffer(color_attachments=[tex]))
        self._bloom = []
        w, h = rw, rh
        for _ in range(6):
            w, h = max(w // 2, 1), max(h // 2, 1)
            t = self.ctx.texture((w, h), 4, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self._bloom.append((t, self.ctx.framebuffer(color_attachments=[t])))
            if min(w, h) <= 8:
                break
        self.rw, self.rh = rw, rh
        self.accum_frames = 0
        self._ensure_shadow()

    def _ensure_shadow(self):
        n = QUALITY[self.quality]["shadow"]
        if n == self.shadow_size:
            return
        if self.shadow_size:
            self.t_shadow.release()
            self.shadow_fbo.release()
        self.t_shadow = self.ctx.texture((n, n), 4, dtype="f4")
        self.t_shadow.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_shadow.repeat_x = self.t_shadow.repeat_y = False
        self.shadow_fbo = self.ctx.framebuffer(color_attachments=[self.t_shadow])
        self.shadow_size = n
        self.shadow_dirty = True

    def set_quality(self, q: str):
        if q in QUALITY and q != self.quality:
            self.quality = q
            self.resize(self.width, self.height)

    def upload_patterns(self, uploads):
        """[(group, slot, (n, n, 4) float16 realization)] into the pattern
        texture; the mip levels are rebuilt once for all of them."""
        n = 0
        for g, s, pack in uploads:
            if not (0 <= g < MAXDECKS and 0 <= s < SLOTS):
                continue
            a = np.ascontiguousarray(pack, dtype="f2")
            if a.shape[:2] != (MAP_SIZE, MAP_SIZE):
                from scipy.ndimage import zoom
                f = MAP_SIZE / a.shape[0]
                a = np.ascontiguousarray(zoom(a.astype("f4"), (f, f, 1), order=1,
                                              mode="grid-wrap", grid_mode=True)).astype("f2")
            self.t_maps.write(a.tobytes(), viewport=(0, 0, g * SLOTS + s, MAP_SIZE, MAP_SIZE, 1))
            n += 1
        if n:
            self.t_maps.build_mipmaps()
        return n

    def set_deck_states(self, states: list, uploads=()):
        """The layers to draw this frame (timeline.DeckState, each in its
        group) and any new pattern realizations for their slots.  Marks the
        frame as changing (deck_changed) when anything the shader reads did,
        so a still picture does not blend two skies."""
        up = self.upload_patterns(uploads)
        states = [s for s in states if 0 <= s.group < MAXDECKS]
        p = pack_states(states)
        self.deck_changed = bool(up) or self._params_prev is None or \
            not np.array_equal(p, self._params_prev)
        if self.deck_changed:
            self.deck_ubo.write(p.tobytes())
            self._params_prev = p
        self.deck_states = states
        self.decks = [s.deck for s in states]
        self.num_decks = (max(s.group for s in states) + 1) if states else 0
        # the strongest (for the panel); the shader takes each deck's own
        self.halo, self.corona = optical_phenomena([s.deck for s in states if s.deck.coverage > 0.0])

    def set_decks(self, decks: list[Deck]):
        """Decks that do not evolve (headless renders, the self-test, the
        census's own check): one realization each, group i, slot 0."""
        decks = decks[:MAXDECKS]
        uploads, states = [], []
        self.deck_maps_cpu = []
        for i, d in enumerate(decks):
            d.map_index = i
            r = timeline.build_realization(timeline.realization_params(d), d.seed & 0x3FFFFFFF,
                                           MAP_SIZE)
            uploads.append((i, 0, r["pack"]))
            st = static_state(d, i, r)
            states.append(st)
            pk = r["pack"].astype("f4")
            zmap = pk[..., 0]
            if st.wave[0] > 0.0:
                # the waves on the map's own texels (rows north, columns east)
                xs = (np.arange(zmap.shape[0]) + 0.5) / zmap.shape[0] * r["extent"]
                yy, xx = np.meshgrid(xs, xs, indexing="ij")
                a_, kx, ky, ph = st.wave
                zmap = zmap + a_ * np.sin(kx * xx + ky * yy + ph)
            frac, _ = noise.compose(zmap, st.zstar, st.soft_z, u=pk[..., 2],
                                    top_variation=d.top_variation, gap=pk[..., 3] if r["gaps"] else None,
                                    gap_amount=st.gap_amount, gap_star=st.gap_star,
                                    dome=pk[..., 2] if r["turret"] else None,
                                    turret=st.turret_tb)
            self.deck_maps_cpu.append((np.stack([frac, frac, frac], -1), r["extent"]))
        self.set_deck_states(states, uploads)
        self.accum_frames = 0
        self.shadow_dirty = True

    def set_sim(self, view: SimView | None):
        self.sim = view
        self.shadow_dirty = True
        self.accum_frames = 0

    def _region_arrays(self, n: int):
        for t in (self.t_rwarp, self.t_rcover):
            if t is not None:
                t.release()
        self.t_rwarp = self.ctx.texture_array((n, n, 2), 4, np.zeros((2, n, n, 4), "f4").tobytes(),
                                              dtype="f4")
        self.t_rcover = self.ctx.texture_array((n, n, 4), 4,
                                               np.full((4, n, n, 4), 100.0, "f4").tobytes(), dtype="f4")
        for t in (self.t_rwarp, self.t_rcover):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False

    def set_region(self, rm, old=None, tau_th=None, new_weight: float = 1.0):
        """The regional maps (region.RegionMaps; None switches them off)
        and, while a refreshed forecast takes over, the maps of the forecast
        before it (old; new_weight: how far the refresh has gone).  Both
        hours' maps of each go to the GPU, which blends them per sample for
        the moment (region_frame); they are written again only when the
        clock enters another hour (RegionMaps.version)."""
        if tau_th is not None:
            self.tau_th = list(tau_th)
        if rm is None:
            if self.region_on:
                self._region_arrays(1)
                self.shadow_dirty = True
                self.accum_frames = 0
            self.region_on = False
            self.region_new = 1.0
            self._region_key = None
            return
        n = rm.n
        old = old if (old is not None and old.n == n) else None
        key = (id(rm), rm.version, id(old), old.version if old is not None else None)
        if key != self._region_key:
            warp, ca, cb = rm.gpu_maps()
            wo, oa, ob = old.gpu_maps() if old is not None else (warp, ca, cb)
            if tuple(self.t_rwarp.size[:2]) != (n, n):
                self._region_arrays(n)
            self.t_rwarp.write(np.ascontiguousarray(np.stack([warp, wo], 0), "f4").tobytes())
            self.t_rcover.write(np.ascontiguousarray(np.stack([ca, cb, oa, ob], 0), "f4").tobytes())
            if not self.region_on or abs(float(rm.half_m) - self.region_half) > 1e-6:
                self.shadow_dirty = True
                self.accum_frames = 0
            self._region_key = key
        self.region_on = True
        self.region_half = float(rm.half_m)
        self.region_frame(rm, old, new_weight)

    def region_frame(self, rm, old=None, new_weight: float = 1.0):
        """The moment within the regional maps' hours (per frame, no upload):
        how far between the two hours, how far each étage's field has been
        carried, the cover here now - of the forecast and, during a refresh,
        of the one before it."""
        u = np.zeros((8, 4), "f8")
        for k, m in ((0, rm), (4, old if old is not None else rm)):
            sa, sb = np.asarray(m.shift_a, "f8"), np.asarray(m.shift_b, "f8")
            u[k] = (m.f, *np.asarray(m.local_now, "f8"))
            u[k + 1] = (sa[0, 0], sa[0, 1], sa[1, 0], sa[1, 1])
            u[k + 2] = (sa[2, 0], sa[2, 1], sb[0, 0], sb[0, 1])
            u[k + 3] = (sb[1, 0], sb[1, 1], sb[2, 0], sb[2, 1])
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        u[0, 1:] = np.maximum(u[0, 1:], 5.0)
        u[4, 1:] = np.maximum(u[4, 1:], 5.0)
        self.region_u = u.astype("f4")
        self.region_new = float(min(max(new_weight, 0.0), 1.0)) if old is not None else 1.0
        self.region_sim_et = int(min(max(rm.sim_etage, 0), 2))

    def set_tau_table(self, tau_th):
        self.tau_th = list(tau_th)

    def set_keep_map(self, tau_map):
        """(ny, nx) column optical depth of the model on which the regional
        thinning decides (region.keep_tau_map); None: the columns' own."""
        self.t_keep.release()
        if tau_map is None:
            self.t_keep = self.ctx.texture((1, 1), 1, np.zeros(1, "f4").tobytes(), dtype="f4")
            self.keep_on = False
            return
        a = np.ascontiguousarray(tau_map, "f4")
        self.t_keep = self.ctx.texture((a.shape[1], a.shape[0]), 1, a.tobytes(), dtype="f4")
        self.t_keep.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_keep.repeat_x = self.t_keep.repeat_y = True
        self.keep_on = True

    def effective_albedo(self) -> float:
        """Broadband albedo of the ground as the clouds' bases see it: snow
        lifts it from ~0.13 to ~0.64 (open fields 0.75, woods 0.3, the
        patchwork's quarter of woods) - the bright undersides of clouds
        over snow."""
        s = min(max(float(self.snow), 0.0), 1.0) if math.isfinite(self.snow) else 0.0
        d = getattr(self, "dryness", 0.0)
        d = min(max(float(d), 0.0), 1.0) if math.isfinite(d) else 0.0
        bare = self.ground_albedo + d * (0.35 - self.ground_albedo)      # sand 0.35
        return bare + s * (0.64 - bare)

    def ground_elevation(self) -> float:
        """The station's height above sea level, within what the tables cover."""
        e = float(self.ground_elev)
        return min(max(e, 0.0), 9000.0) if math.isfinite(e) else 0.0

    #: the sky irradiance is tabled every 1/IRR_PER_DEG of the Sun's elevation
    IRR_PER_DEG = 4

    def _irr_node(self, i: int, elev: float):
        key = (i, round(elev))
        cache = self.__dict__.setdefault("_irr_nodes", {})
        v = cache.get(key)
        if v is None:
            s = i / self.IRR_PER_DEG
            v = (np.maximum(atmosphere.sky_irradiance(self.tr_lut, self.ms_lut, s, elev), 1e-30),
                 np.maximum(atmosphere.sky_irradiance(self.tr_lut, self.ms_lut, s, elev + 8000.0), 1e-30))
            if len(cache) > 24:
                cache.clear()
            cache[key] = v
        return v

    def sky_irradiance(self, sun_alt_deg: float):
        """Downwelling sky irradiance at the ground and 8 km above it: tabled
        every quarter degree of the Sun's elevation and interpolated between
        (log-linear, it falls by orders of magnitude through twilight).  A
        value held until the Sun had moved on a quarter degree stepped the
        light on the clouds by 15-70 % at a time in twilight - at 60 times
        speed once a second."""
        elev = self.ground_elevation()
        x = float(sun_alt_deg) * self.IRR_PER_DEG
        if not math.isfinite(x):
            x = 0.0
        i0 = math.floor(x)
        f = x - i0
        a, b = self._irr_node(i0, elev), self._irr_node(i0 + 1, elev)
        lo = np.exp((1.0 - f) * np.log(a[0]) + f * np.log(b[0])).astype("f4")
        hi = np.exp((1.0 - f) * np.log(a[1]) + f * np.log(b[1])).astype("f4")
        return lo, hi

    @staticmethod
    def moon_share(sun_alt_deg: float, moon_alt_deg: float) -> float:
        """How far the Moon has taken over from the Sun (the light on the
        clouds, the exposure): none while the Sun is above -6 degrees, all
        of it below -10, and as the Moon clears the horizon.  No cloud is
        sunlit below -6 (the Earth's shadow is 20 km deep by -4.5), so the
        Sun gives the clouds nothing where the Moon's share begins.  It was
        a switch: at -10 degrees the moonlit clouds appeared in one frame,
        at -6 the whole picture darkened 3.5 times (full moon)."""
        return (_smoothstep(-6.0, -10.0, sun_alt_deg)
                * _smoothstep(-0.5, 1.5, moon_alt_deg))

    @staticmethod
    def exposure_limits(sun_alt_deg: float) -> tuple:
        """How far the metering may push the exposure, as factors of the
        base: most in twilight (0.35-60), less by day (0.35-3), little in the
        deep night (0.5-1.5) - moving with the Sun's elevation over four
        degrees about +4 and -14 (log-linear).  They were steps there, and
        an exposure metered against the wider range jumped to the narrower
        one in a frame: the picture 7 times darker at once at sunrise, 13
        times at the end of dusk."""
        day = _smoothstep(2.0, 6.0, sun_alt_deg)
        deep = _smoothstep(-12.0, -16.0, sun_alt_deg)

        def lmix(p, q, t):
            return math.exp((1.0 - t) * math.log(p) + t * math.log(q))
        return lmix(0.35, 0.5, deep), lmix(lmix(60.0, 3.0, day), 1.5, deep)

    def auto_exposure(self, sun_alt_deg: float, moon_alt_deg: float = -90.0,
                      moon_illum: float = 0.0) -> float:
        """Eyes and cameras both adapt.  The base exposure follows the Sun's
        elevation; on top of it the exposure adapts to the frame's own
        log-average luminance, as a camera meters: an overcast sky becomes
        mid-grey instead of dim, a twilight sky bright enough to show what
        is lit high above it - within limits set by the Sun's elevation, so
        that night stays night.  Every part is continuous in the Sun's and
        the Moon's elevation."""
        night = max(0.0, min(1.0, (-sun_alt_deg) / 18.0))
        ev = 8.0 * (10.0 ** (night * 2.7))
        ev /= 1.0 + 2.5 * moon_illum * self.moon_share(sun_alt_deg, moon_alt_deg)
        lo, hi = self.exposure_limits(sun_alt_deg)
        self._ev_limits = (ev * lo, ev * hi)
        base = ev
        adapted = getattr(self, "_ev_adapted", None)
        if adapted is not None and self.adaptive:
            ev = min(max(adapted, ev * lo), ev * hi)
        self._ev_base = base
        return ev * (2.0 ** self.exposure_bias)

    def meter(self):
        """Read the frame's log-average luminance from the smallest bloom
        level and move the adapted exposure toward 'mid-grey' (key 0.16)."""
        if not self._bloom:
            return
        t, _ = self._bloom[-1]
        try:
            data = np.frombuffer(t.read(), "f2").astype("f4").reshape(t.height, t.width, 4)
        except Exception:
            return
        lum = 0.2126 * data[..., 0] + 0.7152 * data[..., 1] + 0.0722 * data[..., 2]
        lum = lum[np.isfinite(lum)]
        if lum.size == 0 or float(lum.max()) <= 1e-12:
            return                        # nothing to meter: keep the exposure
        avg = float(np.exp(np.mean(np.log(np.maximum(lum, 1e-7)))))
        used = getattr(self, "_ev_used", None)
        if used is None or not math.isfinite(avg) or avg <= 0.0 or not math.isfinite(used):
            return
        target = used * 0.16 / avg / (2.0 ** self.exposure_bias)
        lo, hi = getattr(self, "_ev_limits", (target, target))
        target = min(max(target, lo), hi)
        self._ev_target = target
        if getattr(self, "_ev_adapted", None) is None:
            self._ev_adapted = target

    def adapt_step(self):
        """Every frame, the exposure moves toward what the last metering
        asked for - a third of adapt_rate's step, so over the three frames
        between meterings it goes as far as adapt_rate says, in three even
        steps instead of one (a step every third frame was a flicker in a
        fast twilight)."""
        tgt, cur = getattr(self, "_ev_target", None), getattr(self, "_ev_adapted", None)
        if tgt is None or cur is None or not (tgt > 0.0 and cur > 0.0):
            return
        k = 1.0 - (1.0 - min(max(self.adapt_rate, 0.0), 1.0)) ** (1.0 / 3.0)
        self._ev_adapted = math.exp((1.0 - k) * math.log(cur) + k * math.log(tgt))

    @staticmethod
    def light_source(geom):
        """(direction, colour, is_sun).  By day the Sun lights the clouds; once
        it is down and the Moon is up, the Moon does - with its irradiance
        relative to the Sun the same as its disc's radiance is drawn relative
        to the Sun's (0.06/120), times its illuminated fraction, and times
        its share (moon_share: it rises from nothing where the Sun no longer
        lights any cloud, so the change of source is not seen)."""
        m = SkyRenderer.moon_share(geom.sun_alt, geom.moon_alt)
        if m <= 0.0:
            return tuple(float(x) for x in geom.sun_dir), (1.0, 1.0, 1.0), True
        k = 5.0e-4 * max(geom.moon_illum, 0.02) * m
        return tuple(float(x) for x in geom.moon_dir), (k * 0.86, k * 0.92, k * 1.0), False

    # --------------------------------------------------------------- drawing --
    def _bind_textures(self):
        self.t_trans.use(0)
        self.t_ms.use(1)
        self.t_shape.use(2)
        self.t_detail.use(3)
        self.t_blue.use(4)
        self.t_maps.use(5)
        self.deck_ubo.bind_to_uniform_block(self.deck_binding)
        self.t_shadow.use(7)
        self.t_rain.use(8)
        texs = self.sim.textures if (self.sim and self.sim.textures) else self.t_sim_empty
        for i, t in enumerate(texs):
            t.use(9 + i)
        self.t_rwarp.use(13)
        self.t_rcover.use(14)
        self.t_keep.use(15)

    def _common_uniforms(self, prog, geom, sim_time, light):
        def setu(name, value):
            if name in prog:
                prog[name].value = value
        ldir, lcol, is_sun = light
        setu("uSunDir", tuple(float(x) for x in geom.sun_dir))
        setu("uLightDir", ldir)
        setu("uLightColor", lcol)
        setu("uLightIsSun", 1 if is_sun else 0)
        setu("uObserverAlt", float(self.observer_alt))
        setu("uTime", float(sim_time))
        setu("uNumDecks", int(self.num_decks))
        setu("uGroundAlbedo", float(self.effective_albedo()))
        setu("uSnow", float(min(max(self.snow, 0.0), 1.0)))
        setu("uDry", float(min(max(getattr(self, "dryness", 0.0), 0.0), 1.0)))
        setu("uGroundElev", float(self.ground_elevation()))
        setu("uHaze", float(self.haze))
        lo, hi = self.sky_irradiance(geom.sun_alt)
        setu("uSkyIrrLow", tuple(float(x) for x in lo))
        setu("uSkyIrrHigh", tuple(float(x) for x in hi))
        sv = self.sim
        # a model handed over entirely to the Atlas layers is not marched at all
        on = bool(sv and sv.textures) and self.sim_fade > 0.001
        setu("uSimOn", 1 if on else 0)
        setu("uSimFade", float(min(max(self.sim_fade, 0.0), 1.0)))
        if on:
            setu("uSimMix", float(sv.mix))
            setu("uSimSize", tuple(float(x) for x in sv.size))
            setu("uSimCell", tuple(float(x) for x in sv.cell))
            setu("uSimObs", tuple(float(x) for x in sv.obs))
            setu("uSimTile", int(sv.tile) if not isinstance(sv.tile, bool) else (3 if sv.tile else 0))
            setu("uSimTime", float(sv.time))
            setu("uJdELiq", tuple(float(x) for x in sv.jde))
            setu("uSimZ0", float(sv.z0))
            setu("uSimZLo", float(sv.zlo))
            setu("uSimZHi", float(max(sv.zhi, sv.zlo + 1.0)))
            setu("uSimZBase", float(getattr(sv, "zbase", sv.zlo)))
            setu("uSimMove", tuple(float(x) for x in sv.move))
        else:
            setu("uSimSize", (1.0, 1.0, 1.0))
            setu("uSimZ0", 0.0)
        setu("uRegionOn", 1 if (self.region_on and on) or (self.region_on and self.num_decks) else 0)
        setu("uRegionHalf", float(self.region_half))
        if "uReg" in prog:
            prog["uReg"].write(np.ascontiguousarray(self.region_u, "f4").tobytes())
        setu("uRegNew", float(self.region_new))
        setu("uRegSimEt", int(self.region_sim_et))
        setu("uKeepMapOn", 1 if self.keep_on else 0)
        if "uTauTh" in prog:
            prog["uTauTh"].write(np.asarray(self.tau_th, "f4").tobytes())
        # shadow map frame
        s = np.array(ldir, "f8")
        up = np.array([0.0, 1.0, 0.0]) if abs(s[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(up, s)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(s, e1)
        top = self._cloud_top()
        # the map covers what can be seen: wider for an eye high above the ground
        ext = 40000.0 + 3.0 * float(self.observer_alt)
        # the map plane sits beyond the highest cloud as seen along the light,
        # which for a low Sun is a long way off
        d0 = math.sqrt(ext * ext + top * top) + 5000.0
        d0 = max(d0, top / max(float(s[1]), 0.08) + 5000.0)
        setu("uShadowOn", 1)
        setu("uShE1", tuple(float(x) for x in e1))
        setu("uShE2", tuple(float(x) for x in e2))
        setu("uShExtent", ext)
        setu("uShD0", d0)
        setu("uShTop", top)

    def _cloud_top(self) -> float:
        top = 2000.0
        for d in self.decks:
            top = max(top, d.top_m)
        if self.sim and self.sim.textures:
            top = max(top, self.sim.z0 + self.sim.size[2])
        return top + 500.0

    def render_shadow(self, geom, sim_time, light):
        self._ensure_shadow()
        p = self.shadow_prog
        self._bind_textures()
        self._common_uniforms(p, geom, sim_time, light)
        if "uShSteps" in p:
            p["uShSteps"].value = int(QUALITY[self.quality]["shsteps"])
        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_size, self.shadow_size)
        self.svao.render(moderngl.TRIANGLES)
        self.shadow_dirty = False

    def render(self, cam: Camera, geom, sim_time: float, accumulate: bool = False,
               animating: bool = False):
        right, up, fwd = cam.basis()
        light = self.light_source(geom)
        if self.shadow_dirty or animating:
            self.render_shadow(geom, sim_time, light)
        p = self.prog
        self._bind_textures()
        self._common_uniforms(p, geom, sim_time, light)

        def setu(name, value):
            if name in p:
                p[name].value = value

        setu("uResolution", (float(self.rw), float(self.rh)))
        setu("uCamRight", tuple(float(x) for x in right))
        setu("uCamUp", tuple(float(x) for x in up))
        setu("uCamFwd", tuple(float(x) for x in fwd))
        setu("uTanHalfFov", math.tan(math.radians(cam.fov) * 0.5))
        setu("uAspect", self.rw / float(self.rh))
        setu("uMoonDir", tuple(float(x) for x in geom.moon_dir))
        setu("uMoonBrightDir", tuple(float(x) for x in geom.sun_dir))
        setu("uSunAngRad", math.radians(geom.sun_ang_radius_deg))
        setu("uMoonAngRad", math.radians(geom.moon_ang_radius_deg))
        setu("uMoonIllum", float(geom.moon_illum))
        q = QUALITY[self.quality]
        setu("uSteps", int(q["steps"]))
        setu("uLightSteps", int(q["light"]))
        setu("uAtmSteps", int(q["atm"]))
        setu("uSimSteps", int(q["sim"]))
        setu("uShowStars", 1 if self.show_stars else 0)
        # each deck carries its own strength (_deck_rows); these switch them on
        setu("uHaloStrength", 1.0 if self.halo > 0.0 else 0.0)
        setu("uCoronaStrength", 1.0 if self.corona > 0.0 else 0.0)
        ev = self.auto_exposure(geom.sun_alt, geom.moon_alt, geom.moon_illum)
        self._ev_used = ev
        setu("uExposure", ev)
        setu("uFrameJitter", float((self.frame * 0.6180339887) % 1.0))
        jx, jy = pixel_jitter(self.frame) if (accumulate and not animating) else (0.0, 0.0)
        setu("uPixJitter", (2.0 * jx / max(self.rw, 1), 2.0 * jy / max(self.rh, 1)))
        if "uHorToEqu" in p:
            p["uHorToEqu"].write(self._hor_to_equ(geom).astype("f4").tobytes())
        for name in ("uNlc", "uNacreous", "uNat", "uPscLens"):
            setu(name, float(self.upper.get(name, 0.0)))
        setu("uNlcTypes", int(self.upper.get("uNlcTypes", 0)))
        setu("uUpperWind", tuple(self.upper.get("uUpperWind", (-40.0, 0.0))))
        setu("uPscWind", tuple(self.upper.get("uPscWind", (20.0, 0.0))))
        setu("uUpperTime", float(self.upper.get("uUpperTime", 0.0)))
        setu("uLatSign", 1.0 if getattr(geom, "lat", 1.0) >= 0 else -1.0)
        sv = self.sim
        fade = float(min(max(self.sim_fade, 0.0), 1.0))
        if sv is not None and sv.flash_power > 0.0 and fade > 0.0:
            setu("uFlashPos", eye_frame(sv.flash_pos, self.observer_alt))
            setu("uFlashPower", float(sv.flash_power) * fade)
        else:
            setu("uFlashPower", 0.0)
        bolt = ((sv.bolt if sv is not None else []) or []) if fade > 0.5 else []
        n = min(len(bolt), 48)
        setu("uBoltN", int(n))
        if n and "uBolt" in p:
            arr = np.zeros((48, 4), "f4")
            arr[:n] = np.asarray(bolt[:n], "f4")
            arr[:n, :3] = [eye_frame(q[:3], self.observer_alt) for q in arr[:n]]
            p["uBolt"].write(arr.tobytes())
        self.frame += 1

        scene_tex, scene_fbo = self._fbos["scene"]
        scene_fbo.use()
        self.ctx.viewport = (0, 0, self.rw, self.rh)
        self.vao.render(moderngl.TRIANGLES)

        hdr = scene_tex
        if accumulate:
            prev_tex, _ = self._fbos["accA"]
            _, dst_fbo = self._fbos["accB"]
            dst_fbo.use()
            prev_tex.use(0)
            scene_tex.use(1)
            ap = self.accum_prog
            ap["tPrev"].value = 0
            ap["tNew"].value = 1
            if animating:
                self._was_animating = True
            elif getattr(self, "_was_animating", False):
                # a change has just ended (the layers replaced or a pattern
                # uploaded through set_deck_states, the clock stopped): the
                # still picture starts again from the sky as it is now.  The
                # frames before stayed in it at 94 % a frame - two thirds of a
                # clear sky after a dozen frames under a new Nimbostratus,
                # with the Sun's disc in it
                self.accum_frames = 0
                self._was_animating = False
            floor = self.anim_blend if animating else self.min_blend
            ap["uMix"].value = max(1.0 / (self.accum_frames + 1.0), floor)
            ap["uAspect"].value = self.rw / float(self.rh)
            prev = self._prev_basis or (right, up, fwd, cam.fov)
            for name, v in (("cR", right), ("cU", up), ("cF", fwd),
                            ("pR", prev[0]), ("pU", prev[1]), ("pF", prev[2])):
                ap[name].value = tuple(float(x) for x in v)
            ap["cT"].value = math.tan(math.radians(cam.fov) * 0.5)
            ap["pT"].value = math.tan(math.radians(prev[3]) * 0.5)
            self.avao.render(moderngl.TRIANGLES)
            self._prev_basis = (right, up, fwd, cam.fov)
            self._fbos["accA"], self._fbos["accB"] = self._fbos["accB"], self._fbos["accA"]
            self.accum_frames += 1
            hdr = self._fbos["accA"][0]
        else:
            self.accum_frames = 0
        return self._post(hdr)

    def _post(self, hdr):
        # bloom: down the chain, then back up with a tent filter
        src = hdr
        sw, sh = self.rw, self.rh
        dp = self.down_prog
        for i, (t, f) in enumerate(self._bloom):
            f.use()
            self.ctx.viewport = (0, 0, t.width, t.height)
            src.use(0)
            dp["tSrc"].value = 0
            dp["uTexel"].value = (1.0 / sw, 1.0 / sh)
            dp["uKaris"].value = 1 if i == 0 else 0
            self.dvao.render(moderngl.TRIANGLES)
            src, sw, sh = t, t.width, t.height
        up = self.up_prog
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self._bloom) - 1, 0, -1):
            small, _ = self._bloom[i]
            big, fbig = self._bloom[i - 1]
            fbig.use()
            self.ctx.viewport = (0, 0, big.width, big.height)
            small.use(0)
            up["tSrc"].value = 0
            up["uTexel"].value = (1.0 / small.width, 1.0 / small.height)
            up["uRadius"].value = 1.0
            self.uvao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)
        # the bloom chain was summed; normalise by the number of levels
        final_tex, final_fbo = self._fbos["final"]
        final_fbo.use()
        self.ctx.viewport = (0, 0, self.rw, self.rh)
        tp = self.tone_prog
        hdr.use(0)
        self._bloom[0][0].use(1)
        tp["tHdr"].value = 0
        tp["tBloom"].value = 1
        tp["uBloom"].value = float(self.bloom_strength)
        tp["uTone"].value = int(self.tone)
        if "uDither" in tp:
            tp["uDither"].value = float(self.frame % 64)
        self.tvao.render(moderngl.TRIANGLES)
        if self.adaptive:
            if self.frame % 3 == 0:
                self.meter()
            self.adapt_step()
        return final_tex

    def present_to_screen(self, tex, target=None):
        (target or self.ctx.screen).use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        tex.use(0)
        self.present["tSrc"].value = 0
        self.pvao.render(moderngl.TRIANGLES)

    @staticmethod
    def _hor_to_equ(geom) -> np.ndarray:
        """Columns are the horizon basis vectors written in equatorial
        coordinates, so that (East, Up, North) maps onto the sky."""
        th = math.radians(geom.lst)
        ph = math.radians(geom.lat)
        east = np.array([-math.sin(th), math.cos(th), 0.0])
        zen = np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th),
                        math.sin(ph)])
        north = np.array([-math.sin(ph) * math.cos(th), -math.sin(ph) * math.sin(th),
                          math.cos(ph)])
        return np.stack([east, zen, north], axis=1).T.copy()

    def read_image(self, tex=None) -> np.ndarray:
        final_tex, fbo = self._fbos["final"]
        data = fbo.read(components=3, dtype="f1", alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(self.rh, self.rw, 3)
        return img[::-1].copy()
