# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
CloudSim - the application: window, input, control panel.

Look around with the mouse, exactly as in Stellarium: drag to turn, wheel to
zoom.  The panel on the left decides what is in the sky: what the weather
says should be there, an Atlas cloud you name yourself, or clouds grown by a
cloud-resolving model from the sounding.

Time runs continuously.  The forecast's every hour is kept (forecast.py) and
the sky between them is one continuous thing (timeline.py): layers are
followed from hour to hour, grow and thin, rise and sink, and their patterns
live and drift on their own winds.  The time buttons move the clock through
the hours in between rather than jumping over them; a far jump dissolves the
sky and forms the new one.  The cloud model hands over to the Atlas layers
and back (starting, restarting, a change of weather) by dissolving its
thinnest columns first, never by switching.
"""

from __future__ import annotations

import math
import os
import threading
import time
import traceback
from datetime import datetime, time as dtime, timedelta, timezone

os.environ.setdefault("PYGLET_SHADOW_WINDOW", "0")

import pyglet
# pyglet checks glGetError after every single GL call when this is on, which
# both costs a lot of time and turns any stray error from another GL user in
# the same context into an exception in an unrelated call.
pyglet.options["debug_gl"] = False
from pyglet.gl import Config
from pyglet.window import key, mouse
import moderngl
import numpy as np

from . import (astro, atlas, crm, forecast, gl_sky, places, region, regimes, scene, simdriver,
               sounding, splash, timeline, ui, upper, viewinfo)
from .atlas import CloudSpec

APP_NAME = "CloudSim"
#: (name, latitude, longitude, clock): the clock an IANA zone name, or hours from UTC
DEFAULT_PLACE = ("Kuala Lumpur", 3.1390, 101.6869, "Asia/Kuala_Lumpur")

OFFLINE_PROFILES = [("Trade cumulus", "tropical-fair"),
                    ("Deep moist", "humid-deep"),
                    ("Stratus", "stable-stratus"),
                    ("Dry / cirrus", "dry")]

# "from the weather" is the regime the data call for (regimes.choose); the
# rest are the reference cases, each the physics of one cloud form
SCENARIO_BUTTONS = [("off", None), ("from the weather", "auto")]
REFERENCE_CASES = [("Sc DYCOMS", "stratocumulus"), ("Ac cells", "altocumulus"),
                   ("Ac radiatus", "altocumulus_ra"), ("Ac undulatus", "altocumulus_un"),
                   ("streets", "streets"), ("asperitas", "asperitas"),
                   ("virga", "virga"), ("storms", "weather"), ("supercell", "supercell"),
                   ("squall + arcus", "squall"), ("trade Cu", "bomex")]


def _make_window(width, height, visible=True):
    """An OpenGL 4.3 window if the driver has one (the simulation runs on the
    GPU), else 3.3 (the simulation runs on the CPU).  visible=False: made
    hidden, to show once its first frames are drawn (CloudSim.show)."""
    last = None
    for major, minor in ((4, 3), (3, 3)):
        try:
            cfg = Config(major_version=major, minor_version=minor, forward_compatible=True,
                         double_buffer=True, depth_size=0, sample_buffers=0)
            return pyglet.window.Window(width=width, height=height, config=cfg,
                                        resizable=True, caption=APP_NAME, visible=visible)
        except Exception as e:                                   # noqa: BLE001
            last = e
    raise last


def splash_line(place) -> str:
    """The place as the loading window shows it under the title."""
    name, lat, lon = place[0], float(place[1]), float(place[2])
    coords = f"{lat:.3f}, {lon:.3f}"
    return coords if name == coords else f"{name}  ·  {coords}"


#: The view the program opens with.  60 degrees (vertical): at 80 a
#: rectilinear frame stretched what is at its top and bottom edges 1.7 times
#: along the radius (sec^2 of the half angle), at 60 only 1.33 times - clouds
#: near the edges keep nearer their size and shape.  The slider still goes
#: from 4 to 140 degrees.
DEFAULT_VIEW = dict(az=150.0, alt=18.0, fov=60.0)


class CloudSim:
    def __init__(self, width=1280, height=760, quality="medium",
                 place=DEFAULT_PLACE, when=None, offline=False, sim=None,
                 sim_size=None, loader=None):
        # the loading window (splash.Loader) counts the start through; with
        # one up, this window is made hidden and shows with its first frame
        # drawn (warm_up, show)
        self.loader = loader if loader is not None else splash.NoLoader()
        hidden = self.loader.alive()
        self.place_name, self.lat, self.lon, clock = place
        self._set_clock(clock)
        self.picker = None                # the dropdown open over the sky, if any
        self._new_place = False           # a place was chosen: its weather is on the way
        self.when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self.playing = False
        # simulated seconds per real second: real time, so the clouds drift at
        # their wind's speed and renew at their lifetimes; x10, x60 and x300
        # are time-lapses (it was x60, playing, from the start)
        self.time_speed = 1.0
        self.offline = offline
        self.no_network = offline         # --offline: the network is never touched
        self.profile = "tropical-fair"
        self.mode = "auto"                # or "manual"
        self.status = "starting"
        self.busy = False
        self.selected = 0
        self.sim_time = 0.0
        self.show_help = False
        self.fps = 0.0
        self._last = time.time()

        self.manual = CloudSpec("Cu", "med")
        self.manual_cover = 0.35
        self.manual_base = None
        self.manual_decks: list = []
        self.all_decks: list = []
        # the compass points on the horizon (Stellarium's Q), on from the start
        self.show_directions = True
        self._compass_labels: dict = {}
        # what is loading that was asked for: the loading card over the sky
        # (loading()), the sky being built that it follows (_watch_sky)
        self.card = None
        self._card_t0 = None
        self._sky_job = None
        self._fetch_card = False
        self._fetch_title = ""

        self._boot(0.08, "Opening the OpenGL window", 0.10)
        self.win = _make_window(width, height, visible=False) if hidden else \
            _make_window(width, height)
        self.ctx = moderngl.create_context()
        self.card = ui.LoadingCard()
        self._boot(0.10, "Compiling the sky's shaders", 0.38)
        self.renderer = gl_sky.SkyRenderer(self.ctx, width, height, quality)
        self.cam = gl_sky.Camera(**DEFAULT_VIEW)
        self.panel = ui.Panel()
        self.decks: list[scene.Deck] = []

        self._boot(0.38, "Preparing the cloud-resolving model", 0.45)
        self.sim = simdriver.SimDriver(self.ctx, prefer_gpu=True)
        self.sim_key = None
        self.sim_size = sim_size or ("standard" if self.sim.gpu_ok else "fast")
        self.sim_start_when = self.when
        self.regime = None
        self.upper = upper.UpperClouds()
        self.rmaps = region.RegionMaps()
        self.census = viewinfo.Census()
        self.info = ui.Panel(width=400)
        self._census_t = 0.0
        self._tau_t = 0.0
        self._cam_key = None

        # every hour of the forecast, and the sky as a function of time
        self._boot(0.45, (f"Fetching the weather for {self.place_name} (Open-Meteo)"
                          if not offline else f"Making the {self.profile} sounding"), 0.57)
        self.forecast, self.note = forecast.get_forecast(
            self.lat, self.lon, self.when, allow_network=not offline,
            fallback_profile=self.profile)
        self.snd = self.forecast.sounding_at(self.when)
        self.status = self.note           # (a model started below says what it grows)
        self.sky = timeline.Sky(self.forecast, lat=self.lat)
        self._anim = None                 # a time shift being played through
        self.sim_fade = 0.0               # how much of the model's field is shown
        self._restart_pending = None      # ("key" or None): the model restarts once hidden
        self._env_t = 0.0
        self._env_when = None
        self._region_old = None           # the regional maps of the forecast before a refresh
        self._region_static_key = None
        self._regime_hour = None
        self._fetching = False
        self._last_fetch = 0.0
        self._pending_forecast = None
        self._sky_t = time.time()
        self._build_with_progress(0.58, 0.86)
        self.rebuild_scene(sync=True)
        # by default the sky is the one the weather data call for
        sim = "auto" if sim is None else sim
        if sim and sim != "none":
            self._boot(0.86, "Starting the cloud model", 0.90)
            self.start_sim(sim)
        # the sky opens still: play (or space) starts the clock; a model
        # started now spins up to the clock meanwhile and forms in
        self.playing = False
        self.build_panel()

        self.win.push_handlers(
            on_draw=self.on_draw, on_resize=self.on_resize,
            on_mouse_press=self.on_mouse_press, on_mouse_drag=self.on_mouse_drag,
            on_mouse_release=self.on_mouse_release, on_mouse_scroll=self.on_mouse_scroll,
            on_mouse_motion=self.on_mouse_motion, on_key_press=self.on_key_press,
            on_text=self.on_text, on_text_motion=self.on_text_motion, on_close=self.on_close)
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)
        self.keys = key.KeyStateHandler()
        self.win.push_handlers(self.keys)

    # ----------------------------------------------------------- loading --
    def _boot(self, frac, text, to=None):
        """The start has got to frac: the loading window says what it is
        doing (nothing without one)."""
        self.loader.step(frac, text, to)

    def _build_with_progress(self, f0, f1):
        """The sky's patterns built before the window shows, the loading
        window counting them (Sky.prepare): this thread builds one at a time
        while the worker builds too; the sync update after it then builds
        nothing.  Without a loading window the sync update builds them, as
        before."""
        if not self.loader.alive():
            return
        t0 = time.time()
        while True:
            have, need = self.sky.prepare(self.when)
            n = max(need, 1)
            self._boot(f0 + (f1 - f0) * have / n,
                       f"Building the cloud patterns  ·  {have} of {need}",
                       f0 + (f1 - f0) * min(have + 1, n) / n)
            if have >= need or time.time() - t0 > self.JOB_GIVE_UP_S:
                return
            if self.sky.build_now(limit=1) == 0:
                time.sleep(0.01)              # the worker is building the last ones

    def warm_up(self, frames: int = 3):
        """The first frames, drawn while the window is still hidden - the
        shaders' first use, the patterns' upload, the first accumulation -
        so that it shows a finished sky."""
        if self.win.visible:
            return
        for i in range(frames):
            self._boot(0.90 + 0.10 * i / frames, "Rendering the first frames",
                       0.90 + 0.10 * (i + 1) / frames)
            self.update(1 / 60.0)
            self.win.switch_to()
            self.on_draw()
            self.ctx.finish()
        self._boot(1.0, "Ready")

    def show(self):
        """Show the window with a frame drawn; the loading window closes as
        soon as the window has drawn its first frame on the screen (at the
        latest a second later)."""
        if not self.win.visible:
            self.win.set_visible(True)
            try:
                self.win.activate()
            except Exception:                                    # noqa: BLE001
                pass
            self.win.switch_to()
            self.on_draw()
            self.win.flip()
            self._close_loader = dict(frames=2, by=time.time() + 1.0)
        else:
            self.loader.close()

    def _loader_check(self, drew=False):
        """The loading window goes once the window has drawn two frames of
        its own after showing (the first may fall before the window is on
        the screen), or a second after it showed, whichever comes first."""
        c = getattr(self, "_close_loader", None)
        if c is None:
            return
        if drew:
            c["frames"] -= 1
        if c["frames"] <= 0 or time.time() >= c["by"]:
            self._close_loader = None
            self.loader.close()

    #: a load shows its card once it has gone on this long; a shorter one
    #: passes unseen (a card flashing up for a frame is only noise)
    CARD_DELAY_S = 0.15
    #: a sky still not built after this long is shown as it is
    JOB_GIVE_UP_S = 120.0

    def _watch_sky(self, title, sky=None, then=None, ahead=False, place=False):
        """Follow a sky being built on the loading card: `sky` (the one drawn
        by default) at the clock, until every pattern it needs is built
        (Sky.prepare; ahead: the next realizations too, so that the sync
        update then() makes builds nothing).  then() - the cut to a sky made
        ready in the background - is called the moment it is, and the card
        closes.  A pending cut is never dropped for a plain watch."""
        old = self._sky_job
        if old is not None and old["then"] is not None and then is None:
            return
        if old is not None and old["sky"] is not self.sky and old["sky"] is not sky:
            old["sky"].close()                # a sky made ready for nothing
        self._sky_job = dict(title=title, sky=sky or self.sky, then=then, ahead=ahead,
                             place=place, t0=time.time(), have=0, need=0)

    def _poll_sky_job(self):
        """Every frame: how far the sky being followed has got; its cut, and
        the card's end, the frame it is built."""
        job = self._sky_job
        if job is None:
            return
        try:
            have, need = job["sky"].prepare(self.when, ahead=job["ahead"])
        except Exception as e:                                   # noqa: BLE001
            traceback.print_exc()
            have = need = 0
            self.status = f"could not build the sky: {type(e).__name__}: {e}"
        job["have"], job["need"] = have, need
        if have >= need or time.time() - job["t0"] > self.JOB_GIVE_UP_S:
            self._sky_job = None
            if job["then"] is not None:
                try:
                    job["then"]()
                except Exception as e:                           # noqa: BLE001
                    traceback.print_exc()
                    self.status = f"could not build the sky: {type(e).__name__}: {e}"
                    self.build_panel()

    def loading(self):
        """What the loading card says now: (title, detail, fraction - None
        for a bar that sweeps), or None when nothing that was asked for is
        loading.  A fetch asked for (a place, a date, the live weather) and
        then the sky it brings are one load: the card stays up between."""
        if self._fetch_card and (self._fetching or self._pending_forecast is not None):
            t = time.time() - self._last_fetch
            return (self._fetch_title, f"Open-Meteo forecast  ·  {t:.0f} s", None)
        job = self._sky_job
        if job is not None:
            have, need = job["have"], job["need"]
            return (job["title"], f"cloud patterns  {have} of {need}" if need else "",
                    have / need if need else None)
        return None

    def rebuild_sky(self):
        """R: the sky built again from its source - made ready in the
        background while this one is drawn (the loading card follows it),
        then cut to (as set_source(merge=False) on this sky did, in one
        blocking step)."""
        sky = self.sky.fresh()
        self._watch_sky("Rebuilding the sky", sky=sky, ahead=True,
                        then=lambda: self._cut_to_sky(sky))

    def _cut_to_sky(self, sky):
        if sky is not self.sky:
            old, self.sky = self.sky, sky
            old.close()
        self.rebuild_scene(sync=True, merge=False)

    def toggle_directions(self, v=None):
        self.show_directions = (not self.show_directions) if v is None else bool(v)
        self.build_panel()

    def draw_directions(self):
        """The compass points on the horizon (viewinfo.horizon_marks), with a
        dark shadow so they read on a bright sky as on the ground."""
        marks = viewinfo.horizon_marks(
            self.cam, self.renderer.rw / float(self.renderer.rh), self.win.width,
            self.win.height, self.renderer.observer_alt,
            ground_r=viewinfo.RG + float(getattr(self.renderer, "ground_elev", 0.0) or 0.0))
        for name, x, y, cardinal in marks:
            labs = self._compass_labels.get(name)
            if labs is None:
                size = 17 if cardinal else 12
                col = (204, 51, 26, 255) if cardinal else (222, 120, 96, 255)
                labs = (ui.label(name, bold=cardinal, font_name=ui.FONT, font_size=size,
                                 anchor_x="center", anchor_y="bottom", color=(0, 0, 0, 170)),
                        ui.label(name, bold=cardinal, font_name=ui.FONT, font_size=size,
                                 anchor_x="center", anchor_y="bottom", color=col))
                self._compass_labels[name] = labs
            # on whole pixels (ui.px), or the letters jump about as the view turns
            x, y = ui.px(x), ui.px(y)
            labs[0].x, labs[0].y = x + 1, y + 2
            labs[1].x, labs[1].y = x, y + 3
            labs[0].draw()
            labs[1].draw()

    def draw_card(self, now):
        """The loading card, once a load has gone on CARD_DELAY_S; gone the
        frame the load is done."""
        st = self.loading()
        if st is None:
            self._card_t0 = None
            self.card.reset()
            return
        if self._card_t0 is None:
            self._card_t0 = now
        if now - self._card_t0 < self.CARD_DELAY_S:
            return
        vis = self.panel.visible
        x0 = self.panel.width if vis else 0
        x1 = self.win.width - (self.info.width if vis else 0)
        if x1 - x0 < 200:
            x0, x1 = 0, self.win.width
        self.card.layout(x0, x1, self.win.height, st[0], st[1], st[2], now)
        self.card.draw()

    # ------------------------------------------------------------- scene --
    def _set_clock(self, clock):
        """The place's clock: an IANA zone name (its summer time followed), or
        a fixed number of hours from UTC."""
        if isinstance(clock, str):
            self.zone, self.fixed_s = clock, None
        else:
            self.zone, self.fixed_s = None, int(round(float(clock) * 3600.0))
        self.tz = self.utc_offset_s(datetime.now(timezone.utc)) / 3600.0

    def utc_offset_s(self, when=None) -> int:
        """The place's offset from UTC at an instant (the clock's by default)."""
        when = when or getattr(self, "when", None) or datetime.now(timezone.utc)
        if self.zone:
            off = places.offset_s(self.zone, when)
            if off is not None:
                return off
        return self.fixed_s if self.fixed_s is not None else 0

    @property
    def local(self) -> datetime:
        return self.when.astimezone(timezone(timedelta(seconds=self.utc_offset_s())))

    def _visibility(self):
        vis = None
        if self.snd is not None and self.snd.surface:
            vis = self.snd.surface.get("visibility")
        if vis is None or vis <= 0:
            rh = self.snd.levels[0].rh if self.snd else 70.0
            vis = gl_sky.visibility_guess(rh)
        return float(vis)

    def rebuild_scene(self, rebuild_decks=True, sync=False, merge=True):
        """Take up the present source of the sky (the forecast, or the layers
        chosen by hand).  With merge the layers showing are carried over to
        the new ones where they match (the patterns keep living); sync builds
        the patterns now instead of letting them form as they are made."""
        if rebuild_decks:
            if self.mode == "auto":
                self.sky.set_source(forecast=self.forecast, merge=merge)
            else:
                self.sky.set_source(manual=list(self.manual_decks), merge=merge)
        self.snd = self.forecast.sounding_at(self.when)
        self.geom = astro.sky_geometry(self.when, self.lat, self.lon)
        self._refresh_environment(force=True)
        self.update_upper()
        self._update_sky(0.0, sync=sync)
        self.renderer.accum_frames = 0
        self.renderer.shadow_dirty = True
        if rebuild_decks and not sync:
            # the layers' patterns form as they are made: the loading card
            # follows them (and does not show if they are made at once)
            self._watch_sky("Building the sky")
        self.build_panel()

    def _apply_decks(self):
        """Kept for the self-test's older rule: the layers the model draws."""
        self._set_model_shares()
        self._update_sky(0.0)

    def _set_model_shares(self):
        """While the model is growing a kind of cloud, the Atlas layers that
        stand for the same cloud give way to it - as much as the model is
        shown (sim_fade), so a handover dissolves one into the other and
        nothing is drawn twice."""
        shares = {}
        if self.sim_key is not None and self.mode == "auto" and self.sim.sc is not None:
            g = self.sim.sc.grid
            lo, hi = g.z0, g.top
            deep = self.sim.sc.key in ("weather", "supercell", "squall", "bomex", "cumulus")
            for st in self.sky.states:
                d = st.deck
                if (deep and d.spec.genus in ("Cu", "Cb")) or \
                        (not deep and d.base_m < hi and d.top_m > lo):
                    shares[st.track] = float(self.sim_fade)
        self.sky.model_share = shares

    def _update_sky(self, real_dt: float, sync: bool = False):
        """Move the Atlas layers to the clock and hand them to the renderer."""
        self._set_model_shares()
        self.sky.update(self.when, real_dt, sync=sync)
        self.renderer.set_deck_states(self.sky.states, self.sky.take_uploads())
        self.decks = [s.deck for s in self.sky.states]
        self.all_decks = self.decks
        self.selected = min(self.selected, max(len(self.decks) - 1, 0))
        if self.sky.overflow:
            self.status = (f"the renderer draws {gl_sky.MAXDECKS} layers at once; waiting: "
                           + ", ".join(self.sky.overflow))

    def _refresh_environment(self, force: bool = False):
        """The atmosphere of the moment: the sounding interpolated to the
        clock, and what follows from it - the haze, the ground (ten times a
        second at most, when the clock has moved five seconds: steps of the
        haze too small to see; a sounding takes a millisecond) - and the
        regional maps (every frame)."""
        now = time.time()
        moved = self._env_when is None or abs((self.when - self._env_when).total_seconds()) >= 5.0
        if force or (moved and now - self._env_t > 0.1):
            self._env_t = now
            self._env_when = self.when
            self.snd = self.forecast.sounding_at(self.when)
            self.renderer.haze = gl_sky.haze_from_visibility(self._visibility())
            # the ground the sky stands on: its height above the sea, and snow
            self.renderer.ground_elev = sounding.station_elevation(self.snd)
            self.renderer.snow = sounding.snow_cover(self.snd)
            self.renderer.dryness = sounding.ground_dryness(self.snd)
        # the regional maps: continuous in time on the GPU; the CPU only
        # makes maps when the clock enters another forecast hour
        self._refresh_region(in_place=not force)

    def fetch_weather(self, when=None, force=False, status="fetching weather...", live=False,
                      card=""):
        """Fetch the forecast for the place and the clock (or `when`) in the
        background.  force: a newer request (a place or a date chosen) goes
        ahead of one already on the way, whose answer is then dropped.
        live: the "fetch live weather" button - the live forecast again
        after an offline profile was chosen (never with --offline).
        card: the loading card's title while it comes (a fetch that was
        asked for); the program's own refreshes show none."""
        if (self.busy or self._fetching) and not force:
            return
        if live:
            self.offline = self.no_network
            card = card or f"Fetching the live weather for {self.place_name}"
        self._fetch_card = bool(card)
        self._fetch_title = card
        self.busy = True
        self._fetching = True
        self._last_fetch = time.time()
        self._fetch_id = getattr(self, "_fetch_id", 0) + 1
        fid = self._fetch_id
        self.status = status
        self.build_panel()
        when = when or self.when
        lat, lon = self.lat, self.lon
        # the hours the answer will hold (a date chosen meanwhile asks again
        # only if they do not reach it)
        try:
            self._fetch_span = sounding.request_plan(when)[2:]
        except Exception:                                        # noqa: BLE001
            self._fetch_span = None

        def work():
            f, note = forecast.get_forecast(lat, lon, when,
                                            allow_network=not self.offline,
                                            fallback_profile=self.profile)
            if fid == self._fetch_id:
                self._pending_forecast = (f, note)
                self.busy = False
                self._fetching = False

        threading.Thread(target=work, daemon=True).start()

    def _take_forecast(self, f, note):
        """A new forecast: the sky carries its layers over to the new ones
        (timeline.Sky.set_source); the model goes on unless the weather now
        calls for a different kind of cloud."""
        asked = self._fetch_card                 # a fetch that was asked for: the card goes on
        self._fetch_card = False
        job = self._sky_job
        if self._new_place or (job is not None and job.get("place")):
            # the new place's weather (or a newer forecast for it while its
            # sky is still being made: that sky is made again from it)
            self._new_place = False
            self._adopt_place(f, note)
            return
        if f.is_static and not self.forecast.is_static and not self.offline:
            # the fetch failed: keep the forecast there is, and say so
            self.status = note + " - keeping the forecast already here"
            if not self.forecast.covers(self.when):
                self.status = self._coverage_status = self._no_forecast_note(note)
            self.build_panel()
            return
        old_series = getattr(self.forecast, "region", None)
        self.forecast, self.note = f, note
        self.status = note
        if not f.covers(self.when):
            self.status = self._coverage_status = self._no_forecast_note(note, fetched=True)
        if self.mode == "auto":
            self.sky.set_source(forecast=f, merge=True)
            if asked:
                # the layers the new forecast brings form as their patterns
                # are made: the card that followed the fetch follows them
                self._watch_sky("Building the sky of the new forecast")
        # the regional maps of the forecast before blend into the new ones
        old_maps = self.rmaps
        self.rmaps = region.RegionMaps(old_maps.n)
        self.rmaps.tau_th, self.rmaps.keep_map = old_maps.tau_th, old_maps.keep_map
        if old_maps.version > 0:
            self._region_old = dict(maps=old_maps, series=old_series, t0=time.time())
        self._refresh_environment(force=True)
        self._regime_hour = None           # the next update asks again what the weather calls for
        self.build_panel()

    def _request_restart(self, key, play=False):
        """Restart the model (key) or stop it (None) - once its field has
        been handed over to the Atlas layers, so the sky never switches.
        play: start the clock with it, as a click on a model does; a restart
        the program makes by itself (the weather changed, the clock went
        back) leaves a paused clock paused."""
        self._restart_pending = ("restart", key, play)

    def _update_model_fade(self, real_dt: float):
        """How much of the model's field is shown.  It is shown once it has
        caught up with the clock (within ten minutes) and is not about to be
        restarted; it dissolves out and forms in over a few seconds.  A
        pending restart (or stop) happens when it is fully out."""
        target = 0.0
        if self.sim_key is not None and self.sim.sc is not None and self.sim.model is not None \
                and self._restart_pending is None:
            lag = self._sim_target() - self.sim.model_time
            if -120.0 <= lag <= 600.0:
                target = 1.0
            elif lag < -120.0 and self._anim is None:
                # the clock went back: the model cannot follow, it starts again
                self._request_restart(self.sim_key)
        rate = (1.0 / 3.0) if target > self.sim_fade else (1.0 / 1.2)
        step = rate * max(real_dt, 0.0)
        if target > self.sim_fade:
            self.sim_fade = min(target, self.sim_fade + step)
        else:
            self.sim_fade = max(target, self.sim_fade - step)
        if self._restart_pending is not None and self.sim_fade <= 1e-3 and self._anim is None:
            _, key_, play = self._restart_pending
            self._restart_pending = None
            if key_ is None:
                self.stop_sim(now=True)
            else:
                self.start_sim(key_, now=True, play=play)
        self.renderer.sim_fade = self.sim_fade

    def add_manual_deck(self, replace=False):
        err, warn = self.manual.validate()
        if err:
            self.status = "cannot build: " + "; ".join(err)
            self.build_panel()
            return
        reasons = ["chosen by hand"]
        if warn:
            reasons.append("outside the Atlas tables: " + "; ".join(warn))
        d = scene.deck_from_spec(self.manual, self.snd, coverage=self.manual_cover,
                                 base_m=self.manual_base,
                                 seed=int(time.time() * 1000) & 0x7FFFFFFF,
                                 reasons=reasons)
        if replace:
            self.manual_decks = [d]
        else:
            self.manual_decks.append(d)
            self.manual_decks.sort(key=lambda x: x.base_m)
            self.manual_decks = self.manual_decks[:gl_sky.MAXDECKS]
        self.mode = "manual"
        self.status = f"{d.label} at {d.base_m:.0f} m"
        self.rebuild_scene()

    # -------------------------------------------------------- simulation --
    def start_sim(self, key_, now: bool = False, play: bool = True):
        if key_ is None:
            self.stop_sim()
            return
        if not now and self.sim_key is not None and self.sim_fade > 1e-3:
            # hand the model's clouds over to the Atlas layers first
            self._request_restart(key_, play=play)
            self.status = "handing the model's clouds over to the layers, then starting again"
            self.build_panel()
            return
        size = self.sim_size
        spinup = 0.0
        try:
            if key_ == "auto":
                # the regime the weather data call for, built from the sounding
                self.regime = regimes.choose(self.snd, self.geom.sun_alt, size, self.lat)
                sc = self.regime.scenario
                spinup = self.regime.spinup_s
                if sc is None:
                    self.sim.stop()
                    self.sim_key = "auto"
                    self.renderer.set_sim(None)
                    self._apply_decks()
                    self._update_region(None)
                    self.status = self.regime.title + " - nothing for the model to grow"
                    self.build_panel()
                    return
            elif key_ == "weather":
                sun_up = self.geom.sun_alt > 5.0
                sc = crm.scenario_weather(self.snd, size, sun_up=sun_up)
                # start the model a while back, so the first thermals have
                # organised themselves by the time the sky catches up with now
                spinup = 45 * 60.0 if sun_up else 0.0
            else:
                sc = crm.scenario_by_key(key_, size, self.snd)
                spinup = 30 * 60.0 if sc.grid.dx <= 125.0 else 0.0
        except Exception as e:                                   # noqa: BLE001
            self.status = f"cannot build the model: {type(e).__name__}: {e}"
            self.build_panel()
            return
        self.sim.start(sc)
        if self.sim.error:
            self.status = "model failed: " + self.sim.error
            self.sim_key = None
            self.renderer.set_sim(None)
            self.build_panel()
            return
        self.sim_key = key_
        if key_ != "auto":
            self.regime = None
        self.sim_start_when = self.when - timedelta(seconds=spinup)
        self.renderer.set_sim(self.sim.view)
        # the new model is hidden while it spins up (the Atlas layers show
        # the sky) and forms in once it has caught up with the clock
        self.sim_fade = 0.0
        self.renderer.sim_fade = 0.0
        self._tau_t = 0.0
        self._regime_hour = self.when.replace(minute=0, second=0, microsecond=0)
        self._apply_decks()
        self._update_region(sc, key_)
        if play:
            self.playing = True
        self.status = f"{sc.name}: {sc.summary()} ({self.sim.backend})"
        self.build_panel()

    def update_upper(self):
        """The clouds above the weather: present by date, latitude and the
        stratosphere's temperature (or by request from the browser)."""
        self.upper.decide(self.when, self.lat, self.snd, self.geom.sun_alt, lon=self.lon)
        self.renderer.upper = self.upper.uniforms(self.sim_time)

    def toggle_upper(self, key):
        if key in self.upper.forced:
            self.upper.forced.discard(key)
        else:
            self.upper.forced.add(key)
        self.update_upper()
        self.renderer.accum_frames = 0
        self.build_panel()

    def _update_region(self, sc, key=None):
        """The satellite's view around the place: where the forecast has the
        model's kind of cloud, and how the field is carried apart.  The
        forecast's cover belongs to the sky chosen from the data (and the
        deep convection grown from the sounding); a reference case shows its
        mechanism with its own composition, or the same everywhere."""
        self._region_sc = (sc, key)
        self.renderer.set_keep_map(None)
        self._refresh_region(in_place=False)

    def _refresh_region(self, in_place: bool = True):
        """The regional maps at the clock (every frame).  From the forecast's
        grid, both hours around the clock carried along the wind and blended
        on the GPU (RegionMaps.build_series: maps are made only when the
        clock enters another hour, otherwise only the moment is set); beyond
        the grid's hours its edge hour holds.  A reference case's own
        composition does not change.  While a refreshed forecast takes over,
        the maps of the one before blend into the new ones (REFRESH_S)."""
        sc, key = getattr(self, "_region_sc", (None, None))
        from_data = sc is None or key in (None, "auto", "weather")
        zb = None
        period = 5000.0
        if sc is not None:
            zb = sc.random_zmin if sc.random_zmin > 0 else (sc.grid.z0 + 0.3 * sc.grid.nz * sc.grid.dz)
            period = sc.grid.lx
        series = getattr(self.forecast, "region", None)
        if from_data and series is not None:
            self.rmaps.build_series(series, self.when, zb, period,
                                    high_wind=self._high_wind_of(series))
        else:
            skey = (id(self.forecast), id(sc), key, zb, period)
            if not in_place or skey != getattr(self, "_region_static_key", None):
                src = self.snd if sc is None else region.source_for(key, sc, self.snd)
                self.rmaps.build(src, zb, period)
                self._region_static_key = skey
        old, w = None, 1.0
        ro = getattr(self, "_region_old", None)
        if ro is not None:
            k = (time.time() - ro["t0"]) / timeline.Sky.REFRESH_S
            if k >= 1.0 or ro["maps"].n != self.rmaps.n:
                self._region_old = None
            else:
                old, w = ro["maps"], k * k * (3.0 - 2.0 * k)
                if ro["series"] is not None and from_data:
                    old.build_series(ro["series"], self.when, zb, period,
                                     high_wind=self._high_wind_of(ro["series"]))
        self.renderer.set_region(self.rmaps, old, self.rmaps.tau_th, w)

    def _high_wind_of(self, series):
        """The wind of the high étage at a regional hour (the grid has none
        up there): the local sounding's, 9 km above the ground."""
        def wind(k):
            s = self.forecast.sounding_at(series.times[min(max(k, 0), len(series.times) - 1)])
            return s.wind_at(s.levels[0].z + 9000.0)
        return wind

    def stop_sim(self, now: bool = False):
        if not now and self.sim_key is not None and self.sim_fade > 1e-3:
            # dissolve the model's clouds into the Atlas layers first
            self._request_restart(None)
            self.status = "handing the model's clouds over to the layers"
            self.build_panel()
            return
        self.sim.stop()
        self.sim_key = None
        self.regime = None
        self.sim_fade = 0.0
        self.renderer.sim_fade = 0.0
        self.status = "cloud model off: the sky is drawn from Atlas layers only"
        self.renderer.set_sim(None)
        self._region_sc = (None, None)
        self._apply_decks()
        self.build_panel()

    def set_sim_size(self, s):
        self.sim_size = s
        if self.sim_key:
            self.start_sim(self.sim_key)
        else:
            self.build_panel()

    def _sim_target(self) -> float:
        return (self.when - self.sim_start_when).total_seconds()

    # -------------------------------------------------------------- panel --
    def _panel_state(self):
        """What the left panel shows that can change with no click."""
        g = getattr(self, "geom", None)
        sky = getattr(self, "sky", None)
        return (self.when.replace(microsecond=0),
                None if g is None else (round(g.sun_alt, 1), round(g.moon_alt, 1)),
                self.status, round(self.sim_fade, 2),
                tuple((d.spec.abbrev(), round(d.coverage * 8), round(d.base_m)) for d in self.decks),
                None if sky is None else (tuple((s.trend, s.later, round(s.base_cover * 8, 1))
                                                for s in sky.states), sky.pending()))

    def build_panel(self):
        self._panel_seen = self._panel_state()
        p = self.panel
        p.begin(self.win.height)
        g = self.geom if hasattr(self, "geom") else astro.sky_geometry(
            self.when, self.lat, self.lon)

        p.text(APP_NAME, size=15, bold=True)
        coords = f"{self.lat:.3f}, {self.lon:.3f}"
        p.text(coords if self.place_name == coords else f"{self.place_name}   {coords}",
               colour=ui.DIM, size=9, wrap=True)
        kind = getattr(self, "picker_kind", None)
        p.buttons([("world cities  ▾", self.open_city_picker, kind == "city", True),
                   ("latitude, longitude", lambda: self.open_city_picker(coords=True),
                    kind == "coords", True)], per_row=2, height=20)
        loc = self.local
        p.text(loc.strftime("%Y-%m-%d  %H:%M:%S")
               + f"  ({places.utc_label(self.utc_offset_s())})", size=11)
        p.buttons([(loc.strftime("%a %d %b %Y") + "  ▾", self.open_date_picker, kind == "date", True),
                   (loc.strftime("%H:%M") + "  ▾", self.open_hour_picker, kind == "hour", True)],
                  per_row=2, height=20)
        p.text(f"sun {g.sun_alt:+.1f}° az {g.sun_az:.0f}°    "
               f"moon {g.moon_alt:+.1f}°  {g.moon_illum*100:.0f}% lit",
               colour=ui.DIM, size=9)
        p.gap(4)

        p.buttons([("-1h", lambda: self.shift_time(-3600), False, True),
                   ("-10m", lambda: self.shift_time(-600), False, True),
                   ("+10m", lambda: self.shift_time(600), False, True),
                   ("+1h", lambda: self.shift_time(3600), False, True),
                   ("now", self.reset_time, False, True)], per_row=5)
        p.buttons([("▶ play" if not self.playing else "‖ pause",
                    self.toggle_play, self.playing, True),
                   ("x1", lambda: self.set_speed(1), self.time_speed == 1, True),
                   ("x10", lambda: self.set_speed(10), self.time_speed == 10, True),
                   ("x60", lambda: self.set_speed(60), self.time_speed == 60, True),
                   ("x300", lambda: self.set_speed(300), self.time_speed == 300, True)],
                  per_row=5)
        p.gap(6)

        p.text("ATMOSPHERE", colour=ui.DIM, size=9, bold=True)
        p.text(self.status or self.note, colour=ui.WARN if self.busy else ui.DIM,
               size=9, wrap=True)
        p.buttons([("fetch live weather", lambda: self.fetch_weather(live=True), False,
                    not self.busy)], per_row=1)
        p.buttons([(n, (lambda pr=pf: self.use_profile(pr)), self.offline
                    and self.profile == pf, True) for n, pf in OFFLINE_PROFILES],
                  per_row=2, height=20)
        if self.snd:
            cape, cin, zlcl, lfc, el, _ = self.snd.parcel(mixed_depth=400)
            p.text(f"CAPE {cape:.0f} J/kg   {sounding.inhibition_text(cin, lfc, unit=False)}"
                   f"   LCL {zlcl:.0f} m", size=9, colour=ui.DIM)
            fz = self.snd.freezing_level()
            fzs = f"0 °C at {fz:.0f} m" if fz else "no freezing level"
            p.text(f"{fzs}   tropopause {self.snd.tropopause():.0f} m",
                   size=9, colour=ui.DIM)
            u, v = self.snd.wind_at(1000.0)
            u2, v2 = self.snd.wind_at(9000.0)
            p.text(f"wind 1 km {math.hypot(u,v):.0f} m/s   9 km {math.hypot(u2,v2):.0f} m/s"
                   f"   visibility {self._visibility()/1000:.0f} km",
                   size=9, colour=ui.DIM)
        p.gap(6)

        self.build_sim_section(p)

        p.text("SKY", colour=ui.DIM, size=9, bold=True)
        p.buttons([("from the weather", lambda: self.set_mode("auto"),
                    self.mode == "auto", True),
                   ("by hand", lambda: self.set_mode("manual"),
                    self.mode == "manual", True)], per_row=2)
        states = list(self.sky.states)
        if not self.decks:
            # a sky still forming (its patterns being made after a jump or a
            # new place) is not a clear sky
            forming = self.sky.pending() > 0 or any(t.keys for t in self.sky.tracks.values()
                                                   if t.alive(self.sky.x))
            p.text("the sky is forming ..." if forming else
                   ("no Atlas decks" if self.sim_key else "clear sky"), colour=ui.DIM, size=10)
        for i, d in enumerate(self.decks):
            act = (i == self.selected)
            st = states[i] if i < len(states) else None
            trend = (st.trend if st is not None else "")
            p.buttons([(f"{d.spec.abbrev()}   {d.coverage*8:.0f}/8   {d.base_m:.0f} m"
                        + (f"   {trend}" if trend else ""),
                        (lambda k=i: self.select(k)), act, True)], per_row=1, height=20)
            if act:
                p.text(d.label, colour=ui.OK, size=10, wrap=True, indent=8)
                if st is not None and st.later:
                    nxt = (self.local.replace(minute=0, second=0, microsecond=0)
                           + timedelta(hours=1))
                    p.text(f"by {nxt:%H:%M} the forecast makes it {st.later}", colour=ui.DIM,
                           size=9, wrap=True, indent=8)
                if st is not None and st.base_cover > d.coverage + 0.02:
                    p.text(f"the forecast's cover {st.base_cover*8:.1f}/8; "
                           + ("the model draws the rest" if self.sim_fade > 0.02 else
                              "forming as its pattern is built"), colour=ui.DIM, size=9,
                           wrap=True, indent=8)
                odds = getattr(d, "odds", "")
                if odds:
                    # how likely each species is (scene.SPECIES_DATA)
                    p.text(odds, colour=ui.OK, size=9, wrap=True, indent=8)
                # an element's width (the cloudy part of its cell) and how wide
                # it looks 30 degrees up; a layer with no cloud has cells only
                ew = float(getattr(d, "element_w_m", 0.0) or 0.0)
                size = (f"elements {ew:.0f} m wide = {min(d.element_deg, 90):.1f}° seen 30° up"
                        if ew > 0.0 else f"cells {d.element_m:.0f} m apart")
                p.text(f"base {d.base_m:.0f} m, top {d.top_m:.0f} m, "
                       f"optical depth {d.optical_depth:.1f}, {size}",
                       colour=ui.DIM, size=9, wrap=True, indent=8)
                for r in [x for x in d.reasons if x != odds][:9]:
                    p.text("· " + r, colour=ui.DIM, size=9, wrap=True, indent=8)
                if self.mode == "manual":
                    p.buttons([("remove", (lambda k=i: self.remove_deck(k)), False, True)],
                              per_row=1, height=18)
        p.gap(6)

        if self.mode == "manual":
            self.build_browser(p)

        p.text("VIEW", colour=ui.DIM, size=9, bold=True)
        p.buttons([(q, (lambda k=q: self.set_quality(k)),
                    self.renderer.quality == q, True)
                   for q in ("low", "medium", "high", "photo")], per_row=4, height=20)
        p.buttons([(t, (lambda k=i: self.set_tone(k)), self.renderer.tone == i, True)
                   for i, t in enumerate(gl_sky.TONES)], per_row=3, height=20)
        p.slider("exposure", lambda: self.renderer.exposure_bias,
                 self.set_exposure, -3.0, 3.0, "{:+.1f} EV")
        p.slider("field of view", lambda: self.cam.fov,
                 self.set_fov, 4.0, 140.0, "{:.0f}°")
        p.slider(f"eye height  {self.renderer.observer_alt:.0f} m",
                 lambda: math.log10(max(self.renderer.observer_alt, 1.0)),
                 self.set_eye, 0.0, 4.35, "")
        p.buttons([("ground", lambda: self.set_eye(0.0), False, True),
                   ("in the cloud", self.eye_in_cloud, False, True),
                   ("above it", self.eye_above, False, True),
                   ("satellite", self.eye_satellite, False, True)], per_row=4, height=19)
        p.toggles([("stars", lambda: self.renderer.show_stars,
                    self.set_stars, True),
                   ("directions  (q)", lambda: self.show_directions,
                    self.toggle_directions, True)], per_row=2)
        p.buttons([("save screenshot", self.screenshot, False, True)], per_row=1)
        p.text(f"{self.fps:.0f} fps   {self.renderer.accum_frames} samples   "
               f"h for help", colour=ui.DIM, size=9)
        p.gap(10)
        p.end()

    def build_sim_section(self, p):
        p.text("CLOUD-RESOLVING MODEL", colour=ui.DIM, size=9, bold=True)
        backend = ("on the GPU (compute shaders)" if self.sim.gpu_ok
                   else "on the CPU - this graphics driver has no OpenGL 4.3")
        p.text(backend, colour=ui.DIM, size=9)
        p.buttons([(n, (lambda k=k: self.start_sim(k)), self.sim_key == k, True)
                   for n, k in SCENARIO_BUTTONS], per_row=2, height=20)
        p.text("or one cloud form's physics:", colour=ui.DIM, size=9)
        p.buttons([(n, (lambda k=k: self.start_sim(k)), self.sim_key == k, True)
                   for n, k in REFERENCE_CASES], per_row=3, height=19)
        if self.regime is not None:
            p.text("the weather data call for: " + self.regime.title
                   + " (the reasons are in the panel on the right)", colour=ui.OK, size=9, wrap=True)
        p.buttons([(s, (lambda k=s: self.set_sim_size(k)), self.sim_size == s, True)
                   for s in ("fast", "standard", "fine")], per_row=3, height=18)
        if self.sim_key is None or self.sim.sc is None or self.sim.model is None:
            if self.sim_key is None:
                p.text("off - the Atlas decks draw every cloud", colour=ui.DIM, size=9, wrap=True)
            p.gap(6)
            return
        sc = self.sim.sc
        d = self.sim.diag or {}
        mt = self.sim.model_time
        lag = self._sim_target() - mt
        hh, rem = divmod(int(mt), 3600)
        mm, ss = divmod(rem, 60)
        p.text(f"{sc.name}", colour=ui.OK, size=10, wrap=True)
        state = "catching up" if lag > 120 else ("running" if self.playing else "paused")
        p.text(f"model {hh:d}:{mm:02d}:{ss:02d}   {self.sim.steps_per_s:.0f} steps/s   "
               f"{state}", colour=ui.DIM, size=9)
        if d:
            p.text(f"updraught {d.get('w_max', 0):.0f} m/s   tops {d.get('cloud_top', 0)/1000:.1f} km"
                   f"   base {d.get('cloud_base', 0):.0f} m", colour=ui.DIM, size=9)
            p.text(f"cover {d.get('cloud_cover', 0)*100:.0f}%   rain {d.get('rain_rate_max', 0):.0f} mm/h"
                   f"   ice {d.get('ice_frac', 0)*100:.0f}%", colour=ui.DIM, size=9)
        name, why = self.sim.atlas_name()
        if name:
            p.text(name, colour=ui.OK, size=10, wrap=True)
            for r in why[:5]:
                p.text("· " + r, colour=ui.DIM, size=9, wrap=True, indent=8)
        if self.sim.flashes:
            p.text("lightning", colour=ui.WARN, size=9)
        if self.sim.error:
            p.text("model error: " + self.sim.error, colour=ui.WARN, size=9, wrap=True)
        p.buttons([("restart the model", lambda: self.start_sim(self.sim_key), False, True)],
                  per_row=1, height=18)
        p.gap(6)

    def build_browser(self, p):
        m = self.manual
        opts = atlas.options_for(m.genus)
        p.text("CLOUD BROWSER", colour=ui.DIM, size=9, bold=True)
        p.buttons([(gg, (lambda k=gg: self.set_genus(k)), m.genus == gg, True)
                   for gg in atlas.GENUS_ORDER], per_row=5, height=20)
        p.text(atlas.GENERA[m.genus]["name"] + " — "
               + atlas.CONSTITUTION[m.genus]["phase"] + ", "
               + atlas.etage_of(m.genus) + " étage", colour=ui.DIM, size=9)

        def row(title, keys, table, chosen, single=False):
            """Buttons are labelled with the Atlas abbreviation - the full
            Latin name appears in the assembled name below."""
            if not keys:
                return
            p.text(title, colour=ui.DIM, size=9)
            items = []
            for k in keys:
                if single:
                    items.append((k, (lambda kk=k: self.set_species(kk)),
                                  chosen == k, True))
                else:
                    items.append((k,
                                  (lambda kk=k: kk in chosen),
                                  (lambda v, kk=k: self.toggle_multi(title, kk, v)),
                                  True))
            if single:
                p.buttons(items, per_row=6, height=19)
            else:
                p.toggles(items, per_row=6, height=19)

        row("species", opts["species"], atlas.SPECIES, m.species, single=True)
        row("varieties", opts["varieties"], atlas.VARIETIES, m.varieties)
        row("supplementary", opts["supplementary"], atlas.SUPPLEMENTARY, m.supplementary)
        row("accessory", opts["accessory"], atlas.ACCESSORY, m.accessory)
        if opts["special"]:
            p.text("special clouds", colour=ui.DIM, size=9)
            p.buttons([(atlas.SPECIAL[s]["abbr"],
                        (lambda k=s: self.set_special(k)), m.special == s, True)
                       for s in opts["special"]], per_row=5, height=19)
        if opts["genitus"] or opts["mutatus"]:
            p.text("mother cloud", colour=ui.DIM, size=9)
            items = [("none", lambda: self.set_mother(None), m.mother is None, True)]
            items += [(f"{k} gen", (lambda kk=k: self.set_mother((kk, "genitus"))),
                       m.mother == (k, "genitus"), True) for k in opts["genitus"]]
            items += [(f"{k} mut", (lambda kk=k: self.set_mother((kk, "mutatus"))),
                       m.mother == (k, "mutatus"), True) for k in opts["mutatus"]]
            p.buttons(items, per_row=4, height=19)

        p.text("above the weather", colour=ui.DIM, size=9)
        p.toggles([("noctilucent", (lambda: "nlc" in self.upper.forced),
                    (lambda v: self.toggle_upper("nlc")), True),
                   ("nacreous PSC", (lambda: "psc" in self.upper.forced),
                    (lambda v: self.toggle_upper("psc")), True),
                   ("nitric PSC", (lambda: "nat" in self.upper.forced),
                    (lambda v: self.toggle_upper("nat")), True)], per_row=3, height=19)
        p.text("(shown when the Sun is 1-16 deg below the horizon, as in nature)",
               colour=ui.DIM, size=8, wrap=True)
        p.text(m.latin(), colour=ui.OK, size=10, wrap=True)
        err, warn = m.validate()
        if err:
            p.text("invalid: " + "; ".join(err), colour=(240, 120, 120, 255),
                   size=9, wrap=True)
        elif warn:
            p.text("unusual: " + "; ".join(warn), colour=ui.WARN, size=9, wrap=True)
        p.slider("cover", lambda: self.manual_cover * 8.0,
                 lambda v: self.set_cover(v / 8.0), 0.0, 8.0, "{:.0f}/8")
        lo, hi = atlas.etage_range_m(m.genus, self.lat)
        p.slider("base height", lambda: (self.manual_base if self.manual_base
                                         else (lo + hi) * 0.45),
                 self.set_base, max(lo, 50.0), hi, "{:.0f} m")
        p.buttons([("add to sky", lambda: self.add_manual_deck(False), False, True),
                   ("replace sky", lambda: self.add_manual_deck(True), False, True),
                   ("clear", self.clear_decks, False, True)], per_row=3)
        p.gap(6)

    # ------------------------------------------------------------ actions --
    def select(self, i):
        self.selected = i
        self.build_panel()

    def remove_deck(self, i):
        if 0 <= i < len(self.manual_decks):
            self.manual_decks.pop(i)
            self.selected = max(0, self.selected - 1)
            self.rebuild_scene()

    def clear_decks(self):
        self.manual_decks = []
        self.mode = "manual"
        self.rebuild_scene()

    def set_mode(self, m):
        self.mode = m
        self.rebuild_scene()

    def set_genus(self, gg):
        self.manual = CloudSpec(gg)
        self.build_panel()

    def set_species(self, s):
        self.manual.species = None if self.manual.species == s else s
        self.build_panel()

    def set_special(self, s):
        self.manual.special = None if self.manual.special == s else s
        if self.manual.special:
            self.manual.mother = None
        self.build_panel()

    def set_mother(self, m):
        self.manual.mother = m
        if m:
            self.manual.special = None
        self.build_panel()

    def toggle_multi(self, title, k, v):
        target = {"varieties": self.manual.varieties,
                  "supplementary": self.manual.supplementary,
                  "accessory": self.manual.accessory}[title]
        if v and k not in target:
            target.append(k)
            if title == "varieties":
                other = "op" if k == "tr" else ("tr" if k == "op" else None)
                if other in target:
                    target.remove(other)
        elif not v and k in target:
            target.remove(k)
        self.build_panel()

    def set_cover(self, v):
        self.manual_cover = min(max(v, 0.0), 1.0)
        self.build_panel()

    def set_base(self, v):
        self.manual_base = float(v)
        self.build_panel()

    def set_quality(self, q):
        self.renderer.set_quality(q)
        self.build_panel()

    def set_tone(self, t):
        self.renderer.tone = int(t)
        self.build_panel()

    def set_exposure(self, v):
        self.renderer.exposure_bias = float(v)
        self.renderer.accum_frames = 0
        self.build_panel()

    def set_fov(self, v):
        self.cam.fov = float(v)
        self.build_panel()

    def set_eye(self, v):
        self.renderer.observer_alt = float(10.0 ** v) if v > 0.23 else 1.7
        self.renderer.accum_frames = 0
        self.renderer.shadow_dirty = True
        self.build_panel()

    def _model_heights(self):
        d = self.sim.diag or {}
        if self.sim.sc is not None and d.get("cloud_top", 0) > 0:
            return d.get("cloud_base", 0.0), d.get("cloud_top", 0.0)
        tops = [(dk.base_m, dk.top_m) for dk in self.decks]
        return tops[0] if tops else (1000.0, 1500.0)

    def eye_in_cloud(self):
        b, t = self._model_heights()
        self.set_eye(math.log10(max(0.5 * (b + t), 2.0)))

    def eye_above(self):
        b, t = self._model_heights()
        self.set_eye(math.log10(t + 1500.0))
        self.cam.alt = -35.0

    def eye_satellite(self):
        self.set_eye(math.log10(20000.0))
        self.cam.alt = -89.0
        self.cam.fov = 90.0

    def set_stars(self, v):
        self.renderer.show_stars = bool(v)
        self.renderer.accum_frames = 0
        self.build_panel()

    def use_profile(self, pf):
        self.offline = True
        self.profile = pf
        f, note = forecast.get_forecast(self.lat, self.lon, self.when, allow_network=False,
                                        fallback_profile=pf)
        self.forecast, self.note = f, note
        self.status = self.note
        self.mode = "auto"
        self.rebuild_scene()
        if self.sim_key in ("weather", "auto"):
            self.start_sim(self.sim_key)

    #: a time shift further than this is not played through: the sky
    #: dissolves and the sky of the new moment forms
    PLAY_THROUGH_S = 3 * 3600.0

    def shift_time(self, seconds):
        # a second press while a shift plays adds to where that one goes
        base = self.when
        if self._anim is not None:
            base = self._anim["start"] + timedelta(seconds=self._anim["span"])
        self._goto(base + timedelta(seconds=seconds))

    def reset_time(self):
        # as a date chosen from the list: the weather of now fetched if the
        # forecast here (an archived one, say) does not reach it
        self.goto_time(datetime.now(timezone.utc))

    def _goto(self, target: datetime):
        """Move the clock to target.  Within three hours the clock runs there
        over a second or three and the sky passes through every moment in
        between; further, the sky dissolves and the new one forms."""
        base = self.when
        span = (target - base).total_seconds()
        if abs(span) < 1.0:
            self._anim = None
            return
        if span < 0 and self.sim_key is not None and self.sim.sc is not None:
            # the model cannot run backwards: it dissolves now and starts
            # again from the new time
            self._request_restart(self.sim_key)
        if abs(span) > self.PLAY_THROUGH_S:
            self._anim = None
            self.sky.jump(target)
            self.when = target
            self.sim_time += span
            # the new moment's layers form as their patterns are made
            self._watch_sky(f"Forming the sky of {self._local_of(target):%a %d %b  %H:%M}")
            if self.sim_key is not None and span > 0:
                self._request_restart(self.sim_key)
            self.geom = astro.sky_geometry(self.when, self.lat, self.lon)
            self._refresh_environment(force=True)
            self.update_upper()
            self._maybe_refetch()
            self._covered(target)
            self.build_panel()
            return
        dur = min(max(0.9 + 0.55 * math.log2(1.0 + abs(span) / 600.0), 0.9), 3.0)
        self._anim = dict(t0=time.time(), dur=dur, start=base, span=span, done=0.0)
        self._covered(target)
        self.build_panel()

    # ------------------------------------------------------ place and time --
    def set_place(self, name, lat, lon, zone=None, fixed_s=None):
        """Another place on the Earth: its name and clock now, its weather
        fetched; when that arrives the sky of the new place is built (a new
        place is a cut, not a dissolve) and the model starts again there."""
        self.place_name, self.lat, self.lon = str(name), float(lat), float(lon)
        self._set_clock(zone if zone else (fixed_s or 0) / 3600.0)
        self.geom = astro.sky_geometry(self.when, self.lat, self.lon)
        self._new_place = True
        job = self._sky_job
        if job is not None and job.get("place"):
            # the sky of a place chosen before is not wanted any more
            if job["sky"] is not self.sky:
                job["sky"].close()
            self._sky_job = None
        self.fetch_weather(force=True, status=f"fetching the weather for {self.place_name} ...",
                           card=f"Fetching the weather for {self.place_name}")

    def _adopt_place(self, f, note):
        """The new place's forecast has come: its own sky from the start,
        made ready in the background while the sky there is drawn (the
        loading card follows it), then cut to - a new place is a cut."""
        sky = timeline.Sky(f, lat=self.lat)
        if self.mode != "auto":
            sky.set_source(manual=list(self.manual_decks), merge=False)
        self.status = f"building the sky of {self.place_name} ..."
        self._watch_sky(f"Building the sky of {self.place_name}", sky=sky, ahead=True,
                        place=True, then=lambda: self._cut_to_place(sky, f, note))
        self.build_panel()

    def _cut_to_place(self, sky, f, note):
        """The new place's sky is built: it replaces the one drawn."""
        self.forecast, self.note = f, note
        old, self.sky = self.sky, sky
        if old is not sky:
            old.close()
        self.rmaps = region.RegionMaps(self.rmaps.n)
        self._region_old = None
        self._region_static_key = None
        self._regime_hour = None
        self.selected = 0
        self.rebuild_scene(sync=True, merge=False)
        failed = None
        if self.sim_key is not None:
            self._restart_pending = None
            self.start_sim(self.sim_key, now=True, play=False)
            if self.sim_key is None:
                failed = self.status
        self.status = f"{self.place_name}: {note}" + (f" - {failed}" if failed else "")
        if not f.is_static and not f.covers(self.when):
            self.status = self._coverage_status = (f"{self.place_name}: "
                                                   + self._no_forecast_note(note, fetched=True))
        self.update_census()
        self.build_panel()

    def _clock_target(self) -> datetime:
        """Where the clock is going: the end of a time shift being played
        through, else the clock."""
        a = self._anim
        return self.when if a is None else a["start"] + timedelta(seconds=a["span"])

    def _local_of(self, when: datetime) -> datetime:
        return when.astimezone(timezone(timedelta(seconds=self.utc_offset_s(when))))

    def goto_time(self, target):
        """The clock to a date and hour chosen from the dropdowns: played
        through within three hours, else dissolved and formed again.  The
        forecast is fetched for that time when the one here does not reach
        it (and a fetch on its way will not either)."""
        target = target.astimezone(timezone.utc)
        self._goto(target)
        if self.offline:
            return
        need = self.forecast.is_static or not self.forecast.covers(target, margin_h=2.0)
        if self._fetching:
            span = getattr(self, "_fetch_span", None)
            need = span is None or not (span[0] + timedelta(hours=2) <= target
                                        <= span[1] - timedelta(hours=2))
        if need:
            loc = self._local_of(target)
            self.fetch_weather(when=target, force=True,
                               status=f"fetching the forecast for {loc:%Y-%m-%d %H:%M} ...",
                               card=f"Fetching the forecast for {loc:%a %d %b %Y  %H:%M}")

    def _covered(self, target):
        """The clock goes back inside the forecast: the status that said it
        had left it goes."""
        said = getattr(self, "_coverage_status", None)
        if said is not None and self.status == said and self.forecast.covers(target):
            self.status = self.note
            self._coverage_status = None

    def _no_forecast_note(self, note, fetched=False):
        """The status when the forecast here does not reach the clock: none
        could be had for it, or (fetched) the one that came does not reach it."""
        loc = f"{self.local:%Y-%m-%d %H:%M}"
        tail = (f": the clouds hold the nearest hour of the forecast here; the Sun, the Moon "
                f"and the stars are those of {loc}")
        if fetched:
            return f"{note} - it does not reach {loc}" + tail
        if self.when > sounding.forecast_reach():
            why = f"Open-Meteo forecasts {sounding.FORECAST_DAYS_MAX} days ahead"
        elif self.when.date() < sounding.ARCHIVE_FROM:
            why = "Open-Meteo's archive of past forecasts starts around 2022"
        else:
            why = note
        return f"no forecast for {loc} ({why})" + tail

    # the dropdowns over the sky
    def _open_picker(self, kind, title, source, on_pick, hint, start=None, query=""):
        if getattr(self, "_toggled_off", None) == kind:
            return                    # a click on the open dropdown's own button closes it
        self.show_help = False
        self.picker = ui.Picker(title, source, on_pick, hint=hint, start=start, query=query)
        self.picker_kind = kind
        self._layout_picker()
        self.build_panel()

    def _layout_picker(self):
        if self.picker is not None:
            vis = self.panel.visible
            self.picker.layout(self.win.width, self.win.height,
                               self.panel.width if vis else 0, self.info.width if vis else 0)

    def _after_picker(self):
        if self.picker is not None and self.picker.closed:
            self.picker = None
            self.picker_kind = None
            self.build_panel()

    def _city_note(self, zone):
        off = places.offset_s(zone, self.when)
        if off is None:
            return ""
        return (self.when + timedelta(seconds=off)).strftime("%H:%M") + "  " + places.utc_label(off)

    def _city_rows(self, query, coords_first=False):
        rows = []
        c = places.parse_coords(query)
        if c is not None:
            lat, lon, hrs = c
            if hrs is None:
                z, how = places.zone_at(lat, lon)
                note = self._city_note(z)
                clock = f"its clock: {z} ({how})"
            else:
                note = places.utc_label(hrs * 3600.0)
                clock = f"its clock: {places.utc_label(hrs * 3600.0)} as typed, no summer time"
            rows.append((f"go to {lat:.4f}, {lon:.4f}   ({places.name_for(lat, lon)})",
                         ("coords", lat, lon, hrs), note))
            rows.append((clock + "  ·  or a city near it:", None, ""))
            near = sorted(places.cities(), key=lambda k: places.km(lat, lon, k.lat, k.lon))[:6]
            for k in near:
                rows.append((f"{k.label}   {places.km(lat, lon, k.lat, k.lon):.0f} km",
                             ("city", k), self._city_note(k.zone)))
            return rows
        if coords_first and not query.strip():
            return [("type latitude, longitude - e.g. 3.139, 101.687 or 3.14N 101.69E", None, "")]
        hits = places.search(query)
        if not hits:
            return [("no city by that name - or type latitude, longitude", None, "")]
        return [(k.label, ("city", k), self._city_note(k.zone)) for k in hits]

    def open_city_picker(self, coords=False):
        here, d = places.nearest(self.lat, self.lon)
        start = ("city", here) if d <= 5.0 else None
        if coords:
            # the place's own coordinates to start from: typing replaces them
            self._open_picker("coords", "Latitude, longitude", lambda q: self._city_rows(q, True),
                              self._pick_place, "e.g. 3.139, 101.687 or 3.14N 101.69E;  +8 at the "
                              "end fixes the clock",
                              query=f"{self.lat:.4f}, {self.lon:.4f}")
        else:
            self._open_picker("city", f"Place  ·  now {self.place_name}", self._city_rows,
                              self._pick_place, "type a city or a country - or latitude, longitude",
                              start=start)

    def _pick_place(self, value):
        if value[0] == "city":
            k = value[1]
            self.set_place(k.label, k.lat, k.lon, zone=k.zone)
        else:
            _, lat, lon, hrs = value
            name = places.name_for(lat, lon)
            if hrs is None:
                self.set_place(name, lat, lon, zone=places.zone_at(lat, lon)[0])
            else:
                self.set_place(name, lat, lon, fixed_s=int(round(hrs * 3600.0)))

    def _date_note(self, d, today):
        """What the weather data are on a date at the place: the forecast,
        part of the day, an archived forecast, or none (the Sun and the Moon
        are there on every date)."""
        if self.offline:
            return ""                     # no forecast is asked for
        reach = sounding.forecast_reach()
        first = places.to_utc(datetime.combine(d, dtime(0)), self.zone, self.fixed_s)
        last = places.to_utc(datetime.combine(d, dtime(23)), self.zone, self.fixed_s)
        if first > reach:
            return "beyond the forecast: Sun and Moon only"
        if last > reach:
            return f"forecast to {self._local_of(reach):%H:%M}"
        if d >= today:
            return "forecast"
        if d < sounding.ARCHIVE_FROM:
            return "before the archive (from about 2022)"
        return "archived forecast"

    def _date_rows(self, query):
        now = datetime.now(timezone.utc)
        today = self._local_of(now).date()
        rows = []
        typed = places.parse_date(query, today)
        q = places.fold(query).split()
        if typed is not None:
            # the date typed, and the days around it
            rows.append((f"go to {typed:%a %d %b %Y}", ("date", typed), self._date_note(typed, today)))
            days = [typed + timedelta(days=k) for k in range(-3, 4) if k]
        else:
            days = [today + timedelta(days=k) for k in range(-92, sounding.FORECAST_DAYS_MAX)]
            # the clock's own date too, wherever it is (a date typed before)
            at = self._local_of(self._clock_target()).date()
            if at not in days:
                days = sorted(days + [at])
        for d in days:
            k = (d - today).days
            text = d.strftime("%a %d %b %Y") + {0: "   today", 1: "   tomorrow",
                                                 -1: "   yesterday"}.get(k, "")
            if q and typed is None and not all(w in places.fold(text) for w in q):
                continue
            rows.append((text, ("date", d), self._date_note(d, today)))
        if not rows:
            rows.append(("not a date - try 2026-10-05, 5/10/2026 (day first) or 5 Oct", None, ""))
        return rows

    def open_date_picker(self):
        day = self._local_of(self._clock_target()).date()
        self._open_picker("date", f"Date  ·  {self.place_name}", self._date_rows, self._pick_date,
                          "type a date: 2026-10-05, 5/10/2026 (day first), 5 Oct, +3, tomorrow",
                          start=("date", day))

    def _pick_date(self, value):
        at = self._local_of(self._clock_target())
        local = datetime.combine(value[1], at.time().replace(microsecond=0, tzinfo=None))
        self.goto_time(places.to_utc(local, self.zone, self.fixed_s))

    def _hour_rows(self, query):
        rows = []
        typed = places.parse_time(query)
        day = self._local_of(self._clock_target()).date()
        if typed is not None:
            rows.append((f"go to {typed:%H:%M}", ("time", typed), ""))
        q = query.strip()
        for h in range(24):
            text = f"{h:02d}:00"
            if q and typed is None and q not in text:
                continue
            local = datetime.combine(day, dtime(h))
            u = places.to_utc(local, self.zone, self.fixed_s)
            g = astro.sky_geometry(u, self.lat, self.lon)
            note = f"sun {g.sun_alt:+.0f}°"
            shown = self._local_of(u).replace(tzinfo=None)
            if shown != local:
                # summer time begins that night: the clock skips the hour
                note = f"not on the clock that day - {shown:%H:%M}   " + note
            rows.append((text, ("time", dtime(h)), note))
        if not rows:
            rows.append(("not a time - try 15:30, 1530 or 3pm", None, ""))
        return rows

    def open_hour_picker(self):
        at = self._local_of(self._clock_target())
        self._open_picker("hour", f"Hour  ·  {at:%a %d %b %Y}  ·  "
                          f"{places.utc_label(self.utc_offset_s(at))}  ·  {self.place_name}",
                          self._hour_rows,
                          self._pick_hour, "type a time: 15:30, 1530, 3pm",
                          start=("time", dtime(at.hour)))

    def _pick_hour(self, value):
        day = self._local_of(self._clock_target()).date()
        self.goto_time(places.to_utc(datetime.combine(day, value[1]), self.zone, self.fixed_s))

    def _maybe_refetch(self):
        """Fetch again when the clock nears the end of the forecast there is
        (or has left it), at most every ten minutes."""
        if self.offline or self._fetching or self.forecast.is_static:
            return
        if self._sky_job is not None and self._sky_job.get("place"):
            return                        # the forecast here is the place before's
        if self.forecast.covers(self.when, margin_h=2.0):
            return
        if time.time() - self._last_fetch < 600.0:
            return
        self.fetch_weather()

    def toggle_play(self):
        self.playing = not self.playing
        self.build_panel()

    def set_speed(self, s):
        self.time_speed = s
        self.playing = True
        self.build_panel()

    def screenshot(self):
        img = self.renderer.read_image()
        name = time.strftime("cloudsim-%Y%m%d-%H%M%S.png")
        path = os.path.join(os.path.expanduser("~"), name)
        try:
            from PIL import Image
            Image.fromarray(img).save(path)
        except Exception:
            import zlib, struct

            def png(buf, w, h):
                raw = b"".join(b"\x00" + buf[y * w * 3:(y + 1) * w * 3] for y in range(h))
                def chunk(t, d):
                    c = struct.pack(">I", len(d)) + t + d
                    return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
                return (b"\x89PNG\r\n\x1a\n"
                        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                        + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
            with open(path, "wb") as f:
                f.write(png(img.tobytes(), img.shape[1], img.shape[0]))
        self.status = "saved " + path
        self.build_panel()

    # -------------------------------------------------------------- events --
    def on_resize(self, w, h):
        self.renderer.resize(w, h)
        self.build_panel()
        self.build_info()
        self._layout_picker()

    def on_mouse_press(self, x, y, button, mods):
        if self.picker is not None:
            # a dropdown is open: a click in it is its own; a click outside
            # closes it - on the sky that is all, on the panel the click goes
            # on (another dropdown's button opens that one; its own closes it)
            if self.picker.contains(x, y):
                self.picker.on_mouse_press(x, y)
                self._after_picker()
                return pyglet.event.EVENT_HANDLED
            was = self.picker_kind
            self.picker.close()
            self._after_picker()
            if not self.panel.contains(x, y):
                return pyglet.event.EVENT_HANDLED
            self._toggled_off = was
            try:
                self.panel.on_click(x, y, button)
            finally:
                self._toggled_off = None
            self.build_panel()
            return pyglet.event.EVENT_HANDLED
        if self.panel.contains(x, y):
            self.panel.on_click(x, y, button)
            self.build_panel()
            return
        if self.info.contains(x, y):
            return
        self._dragging = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, mods):
        if self.picker is not None and self.picker.contains(x, y):
            return
        if self.panel.contains(x, y) and self.panel._drag is not None:
            self.panel.on_drag(x, y)
            self.build_panel()
            return
        if self.panel.contains(x, y) or self.info.contains(x, y):
            return
        scale = self.cam.fov / self.win.height
        self.cam.az -= dx * scale
        self.cam.alt += dy * scale
        self.cam.clamp()

    def on_mouse_release(self, x, y, button, mods):
        self.panel.on_release()

    def on_mouse_scroll(self, x, y, sx, sy):
        if self.picker is not None and self.picker.contains(x, y):
            self.picker.on_mouse_scroll(sy)
            return
        if self.panel.contains(x, y):
            self.panel.on_scroll(sy)
            self.panel.end()
            return
        if self.info.contains(x, y):
            self.info.on_scroll(sy)
            self.info.end()
            return
        self.cam.fov *= math.exp(-sy * 0.14)
        self.cam.clamp()
        self.build_panel()

    def on_mouse_motion(self, x, y, dx, dy):
        if self.picker is not None:
            self.picker.on_mouse_motion(x, y)

    def on_text(self, text):
        """Typing goes to the dropdown that is open (the keys' own actions
        wait until it closes)."""
        if self.picker is not None:
            self.picker.on_text(text)
            self._after_picker()
            return pyglet.event.EVENT_HANDLED

    def on_text_motion(self, motion):
        if self.picker is not None:
            self.picker.on_text_motion(motion)
            return pyglet.event.EVENT_HANDLED

    def on_key_press(self, symbol, mods):
        if self.picker is not None:
            # a dropdown is open: the keys are its own (Esc closes it, not
            # the program; Ctrl+V pastes)
            if symbol == key.V and mods & (key.MOD_CTRL | key.MOD_ACCEL):
                try:
                    self.picker.paste(self.win.get_clipboard_text())
                except Exception:                                # noqa: BLE001
                    pass
            else:
                self.picker.on_key_press(symbol, mods)
            self._after_picker()
            return pyglet.event.EVENT_HANDLED
        if symbol == key.ESCAPE:
            self.win.close()
        elif symbol == key.TAB:
            self.panel.visible = not self.panel.visible
        elif symbol == key.H:
            self.show_help = not self.show_help
        elif symbol == key.SPACE:
            self.toggle_play()
        elif symbol == key.P:
            self.screenshot()
        elif symbol == key.F11:
            self.win.set_fullscreen(not self.win.fullscreen)
        elif symbol in (key._1, key._2, key._3, key._4):
            self.set_quality(["low", "medium", "high", "photo"][symbol - key._1])
        elif symbol == key.R:
            self.rebuild_sky()
        elif symbol == key.Q:
            self.toggle_directions()
        elif symbol == key.M:
            # cycle: off, from the weather, then every reference case
            order = [k for _, k in SCENARIO_BUTTONS] + [k for _, k in REFERENCE_CASES]
            i = order.index(self.sim_key) if self.sim_key in order else 0
            self.start_sim(order[(i + 1) % len(order)])
        elif symbol == key.T:
            self.set_tone((self.renderer.tone + 1) % len(gl_sky.TONES))

    def on_close(self):
        pyglet.clock.unschedule(self.update)
        self.sim.stop()
        self.sky.close()
        if self._sky_job is not None:
            self._sky_job["sky"].close()
        self.loader.close()
        pyglet.app.exit()

    # -------------------------------------------------------------- update --
    def update(self, dt):
        self._loader_check()
        if self._pending_forecast is not None:
            f, note = self._pending_forecast
            self._pending_forecast = None
            self._take_forecast(f, note)
        moved = False
        k = self.keys
        step = dt * self.cam.fov * 0.9
        # the keys look around - except while a dropdown is open and they type
        if self.picker is None:
            if k[key.LEFT] or k[key.A]:
                self.cam.az -= step
                moved = True
            if k[key.RIGHT] or k[key.D]:
                self.cam.az += step
                moved = True
            if k[key.UP] or k[key.W]:
                self.cam.alt += step
                moved = True
            if k[key.DOWN] or k[key.S]:
                self.cam.alt -= step
                moved = True
            if k[key.EQUAL] or k[key.PLUS]:
                self.cam.fov *= math.exp(-dt * 1.2)
                moved = True
            if k[key.MINUS]:
                self.cam.fov *= math.exp(dt * 1.2)
                moved = True
        if moved:
            self.cam.clamp()
        running = self.sim_key is not None and self.sim.running
        if running:
            # the Sun the model's ground feels is the Sun at the MODEL's time
            sg = astro.sky_geometry(self.sim_start_when + timedelta(
                seconds=self.sim.model_time), self.lat, self.lon)
            self.sim.sun_elev = math.radians(sg.sun_alt)
            self.sim.sun_az = math.radians(sg.sun_az)
        forward_model = running and self._restart_pending is None
        if self._anim is not None or self.playing:
            a = self._anim
            if a is not None:
                # a time shift played through: an eased run of the clock
                k = min((time.time() - a["t0"]) / a["dur"], 1.0)
                want = k * k * (3.0 - 2.0 * k)
                new_when = a["start"] + timedelta(seconds=a["span"] * max(want, a["done"]))
            else:
                new_when = self.when + timedelta(seconds=dt * self.time_speed)
            if forward_model and new_when > self.when:
                target = (new_when - self.sim_start_when).total_seconds()
                reached = self.sim.advance(target, budget_s=0.028)
                # While the model's clouds are shown, the world waits for the
                # model, so the Sun and the clouds stay in step whatever speed
                # was asked for.  A model still spinning up (or catching up
                # after a jump) is hidden - the Atlas layers show the sky -
                # and runs to catch the clock up; it forms in when it has.
                slack = 2.0 * max(getattr(self.sim, "_dt", 3.0), 1.0)
                if reached < target - slack and self.sim_fade > 0.5:
                    new_when = max(self.when,
                                   min(new_when, self.sim_start_when + timedelta(seconds=reached)))
            if a is not None:
                a["done"] = (new_when - a["start"]).total_seconds() / a["span"] if a["span"] else 1.0
                if a["done"] >= 1.0 - 1e-9:
                    new_when = a["start"] + timedelta(seconds=a["span"])
                    self._anim = None
            adv = (new_when - self.when).total_seconds()
            self.when = new_when
            self.sim_time += adv
            self.geom = astro.sky_geometry(self.when, self.lat, self.lon)
            self.renderer.upper["uUpperTime"] = float(self.sim_time)
            if int(self.sim_time) // 60 != int(self.sim_time - adv) // 60:
                self.update_upper()
        elif running:
            self.sim.advance(self._sim_target(), budget_s=0.02)
        # the atmosphere of the moment, the model's visibility, the weather's
        # regime, the Atlas layers
        self._refresh_environment()
        self._update_model_fade(dt)
        self._check_regime()
        self._maybe_refetch()
        self._update_sky(dt)
        self._poll_sky_job()
        self._panel_timer = getattr(self, "_panel_timer", 0.0) + dt
        # twice a second while the clock or the model runs; and, paused,
        # whenever what the panel shows has changed with no click - the clock
        # of a time shift played through, a layer that has formed
        if self._panel_timer > 0.5 and (self.playing or running
                                        or self._panel_state() != getattr(self, "_panel_seen", None)):
            self._panel_timer = 0.0
            self.build_panel()
        # the forecast-cover thinning follows the model's own columns
        now = time.time()
        if running and now - self._tau_t > 5.0:
            self._tau_t = now
            try:
                cols = self.census.columns(self.sim)
                if cols is not None:
                    tab = self.rmaps.set_tau_table(cols["tau"], self.sim.sc.grid.dx)
                    self.renderer.set_tau_table(tab)
                    self.renderer.set_keep_map(self.rmaps.keep_map)
            except Exception:                                    # noqa: BLE001
                pass
        # what is in the view, and why
        ck = (round(self.cam.az, 1), round(self.cam.alt, 1), round(self.cam.fov, 1),
              round(self.renderer.observer_alt))
        if now - self._census_t > (0.4 if ck != self._cam_key else 2.0):
            self._census_t = now
            self._cam_key = ck
            self.update_census()

    #: what a regime asks of the model: regimes in one family are grown by
    #: the same model, which goes on through a change between them
    REGIME_FAMILY = {"thunderstorm": "deep", "storm_coming": "deep", "showers": "deep",
                     "cumulus": "shallow", "stratocumulus": "layer", "fog": "layer",
                     "rain_layer": "layer", "altocumulus": "layer", "virga": "layer",
                     "cirrus": None, "clear": None}

    def _regime_at(self, when):
        snd = self.forecast.sounding_at(when)
        g = astro.sky_geometry(when, self.lat, self.lon)
        return regimes.choose(snd, g.sun_alt, self.sim_size, self.lat)

    def _check_regime(self):
        """Every hour of the clock: does the weather still call for the cloud
        the model is growing?  A change the next hour confirms (so an hour's
        flicker in the forecast does not restart the model) hands the model
        over: its clouds dissolve into the Atlas layers and the new model
        forms in once it has caught up."""
        if self.sim_key != "auto" or self._restart_pending is not None or self.regime is None:
            return
        hour = self.when.replace(minute=0, second=0, microsecond=0)
        if self._regime_hour == hour:
            return
        self._regime_hour = hour
        try:
            a = self._regime_at(self.when)
            b = self._regime_at(self.when + timedelta(hours=1))
        except Exception:                                        # noqa: BLE001
            return
        fam = self.REGIME_FAMILY.get
        cur = fam(self.regime.key)
        moved = False
        if fam(a.key) == cur == fam(b.key) and cur == "layer":
            la, lc = a.facts.get("layer"), self.regime.facts.get("layer")
            lb = b.facts.get("layer")
            moved = bool(la and lc and lb and abs(la[0] - lc[0]) > 500.0 and abs(lb[0] - lc[0]) > 500.0)
        if (fam(a.key) != cur and fam(b.key) == fam(a.key)) or moved:
            self.status = (f"the weather changes: {self.regime.title.lower()} -> "
                           f"{a.title.lower()}; the model hands over")
            self._request_restart("auto")

    # ------------------------------------------------------------ in view --
    def update_census(self):
        try:
            running = self.sim_key is not None and self.sim.model is not None
            for it in self.upper.items:
                it.sun_alt = self.geom.sun_alt
            samplers = [(lambda x, y, st=st: self.sky.frac_at(st, x, y)) for st in self.sky.states]
            self.census.update(self.cam, self.renderer.rw / float(self.renderer.rh),
                               self.renderer.observer_alt, self.sim if running else None,
                               self.regime, self.decks, self.rmaps,
                               samplers, self.sim_time,
                               special=self.upper.items, sim_fade=self.sim_fade)
        except Exception as e:                                   # noqa: BLE001
            self.census.seen = []
            self.census_error = f"{type(e).__name__}: {e}"
        self.build_info()

    def build_info(self):
        p = self.info
        w = p.width
        p.begin(self.win.height, x0=self.win.width - w)
        p.text("IN THIS VIEW", colour=ui.DIM, size=9, bold=True)
        c = self.census
        p.text(f"cloud {max(0.0, 100 * (1 - c.sky_share - c.ground_share)):.0f}%   clear sky "
               f"{100 * c.sky_share:.0f}%   ground {100 * c.ground_share:.0f}%",
               colour=ui.DIM, size=9)
        if getattr(self, "census_error", None):
            p.text(self.census_error, colour=ui.WARN, size=9, wrap=True)
        if not c.seen:
            p.text("no cloud in this direction", colour=ui.DIM, size=10, wrap=True)
        for sn in c.seen[:4]:
            p.gap(4)
            p.text(f"{sn.name}  ({sn.abbr})", colour=ui.OK, size=10, bold=True, wrap=True)
            p.text(f"{sn.share * 100:.0f}% of the view - {sn.where()}", colour=ui.FG, size=9,
                   wrap=True, indent=4)
            for line in sn.why[:6]:
                p.text("why: " + line, colour=ui.DIM, size=9, wrap=True, indent=4)
        if self.regime is not None:
            p.gap(6)
            p.text("WHY THE PROGRAM CHOSE THIS SKY", colour=ui.DIM, size=9, bold=True)
            for line in self.regime.lines()[:10]:
                p.text(line, colour=ui.DIM, size=9, wrap=True, indent=4)
            p.text(self.rmaps.note, colour=ui.DIM, size=9, wrap=True, indent=4)
        p.end()

    # ---------------------------------------------------------------- draw --
    def on_draw(self):
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self._last, 1e-4)
        self._last = now
        self.win.clear()
        self.ctx.screen.use()
        # anything moving means the picture is not accumulated over frames
        animating = bool(self.playing) or self._anim is not None or self.renderer.deck_changed \
            or 0.0 < self.sim_fade < 1.0
        if self.sim_key is not None:
            disp = min(self._sim_target(), self.sim.model_time)
            view = self.sim.frame(disp, now)
            self.renderer.sim = view if view.textures else None
            animating = animating or self.sim.model_time < self._sim_target() - 1.0
        tex = self.renderer.render(self.cam, self.geom, self.sim_time, accumulate=True,
                                   animating=animating)
        self._last_tex = tex
        self.renderer.present_to_screen(tex)
        self.ctx.disable(moderngl.DEPTH_TEST)
        if self.show_directions:
            self.draw_directions()
        self.panel.draw()
        if self.panel.visible:
            self.info.draw()
        self.draw_card(now)
        if self.show_help:
            self.draw_help()
        if self.picker is not None:
            self.picker.draw()
        self._loader_check(drew=True)

    HELP_LINES = ["drag / arrows / WASD   look around",
                  "wheel or + -           zoom (field of view)",
                  "space                  play or pause time",
                  "m                      next cloud-model scenario",
                  "t                      next tone curve",
                  "1 2 3 4                quality: low medium high photo",
                  "p                      save a screenshot to your home folder",
                  "r                      rebuild the sky",
                  "q                      compass directions on the horizon",
                  "tab                    hide or show the panel",
                  "▾ on the panel         choose a city, coordinates, date, hour",
                  "in a list              type to narrow it, Enter picks, Esc closes",
                  "h                      this help",
                  "esc                    quit"]

    def draw_help(self):
        lines = CloudSim.HELP_LINES
        b = pyglet.graphics.Batch()
        # the box as wide as its longest line in the font this computer has
        widest = max(ui.label(s, font_name=ui.MONO, font_size=11).content_width for s in lines)
        w, h = max(470, int(widest) + 36), 24 * len(lines) + 30
        x = (self.win.width - w) // 2
        y = (self.win.height - h) // 2
        bg = pyglet.shapes.Rectangle(x, y, w, h, color=(10, 14, 20), batch=b)
        bg.opacity = 238
        # held until drawn: pyglet deletes a label the moment nothing refers
        # to it, which left the box empty
        held = [bg]
        for i, s in enumerate(lines):
            held.append(ui.label(s, font_name=ui.MONO, font_size=11,
                                 x=x + 18, y=y + h - 30 - i * 24, color=ui.FG, batch=b))
        b.draw()


def run(loader=None, **kw):
    """The program: the loading window up first (cloud_sim.py starts it
    before the heavy imports; started here otherwise), the window made
    hidden and its first frames drawn, then the window shows and the
    loading window closes."""
    if loader is None:
        loader = splash.Loader.start(splash_line(kw.get("place", DEFAULT_PLACE)))
    try:
        app = CloudSim(loader=loader, **kw)
        app.warm_up()
    except BaseException:
        loader.close()
        raise
    app.show()
    pyglet.app.run()
    return app
