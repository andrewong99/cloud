# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
Runs the cloud-resolving model alongside the sky: keeps it in step with the
world clock, feeds it the Sun, hands the renderer its extinction fields,
names what it has grown in the Atlas's terms, and makes its lightning.

The model runs on the GPU when the context has compute shaders (OpenGL 4.3)
and otherwise as the NumPy reference model in a worker thread, on the small
grid.
"""

from __future__ import annotations

import math
import random
import threading
import time

import numpy as np
try:                                   # the self-test's pure parts run without OpenGL
    import moderngl
except ImportError:                    # pragma: no cover
    moderngl = None

from . import crm, gl_sky
from .atlas import CloudSpec

# Price & Rind (1992), continental: flashes per minute from the cloud-top
# height in km (as quoted by Dahl 2010): F = 3.44e-5 H^4.9
PR92_A, PR92_B = 3.44e-5, 4.9
CG_FRACTION = 0.25          # Wikipedia "Lightning": CG ~25% of all flashes


class Flash:
    """One lightning flash: 3-4 return strokes 40-50 ms apart (drawn a
    little slower so each can be seen at 60 frames a second), each a
    bright pulse that decays."""

    def __init__(self, pos, cg: bool, bolt, now: float):
        self.pos = pos
        self.cg = cg
        self.bolt = bolt
        self.t0 = now
        n = random.choice((1, 2, 3, 3, 4, 4, 5)) if cg else random.choice((1, 1, 2, 3))
        self.strokes = [0.0]
        for _ in range(n - 1):
            self.strokes.append(self.strokes[-1] + random.uniform(0.045, 0.110))
        self.power = random.uniform(0.6, 1.4)

    def intensity(self, now: float) -> float:
        dt = now - self.t0
        v = 0.0
        for s in self.strokes:
            if dt >= s:
                v = max(v, math.exp(-(dt - s) / 0.035))
        return v * self.power

    def done(self, now: float) -> bool:
        return now - self.t0 > self.strokes[-1] + 0.35


class SimDriver:
    def __init__(self, ctx: moderngl.Context, prefer_gpu: bool = True):
        self.ctx = ctx
        self.gpu_ok = False
        if prefer_gpu:
            try:
                from . import crm_gpu
                self.gpu_ok = crm_gpu.compute_supported(ctx)
            except Exception:
                self.gpu_ok = False
        self.model = None
        self.sc = None
        self.view = gl_sky.SimView()
        self.running = False
        self.status = ""
        self.diag = {}
        self.name = ""
        self.t_prev = 0.0
        self.t_curr = 0.0
        self.target = 0.0
        self.steps_per_s = 0.0
        self.flashes: list[Flash] = []
        self._flash_clock = 0.0
        self._charge = None
        self._charge_t = -1e9
        self._cpu_thread = None
        self._lock = threading.Lock()
        self._cpu_opt = None
        self._tex = None
        self.sun_elev = 0.8
        self.sun_az = 3.1
        self.error = ""

    # ----------------------------------------------------------- control --
    @property
    def backend(self) -> str:
        return "GPU" if self.gpu_ok else "CPU"

    def start(self, sc: crm.Scenario):
        self.stop()
        self.sc = sc
        self.error = ""
        try:
            init = crm.CPUModel(sc, dtype="f8" if self.gpu_ok else "f4")
            if self.gpu_ok:
                from . import crm_gpu
                self.model = crm_gpu.GPUModel(self.ctx, sc, init=init)
                self._dt = init.max_dt(0.8)
            else:
                self.model = init
                self._dt = init.max_dt(0.8)
                self._make_cpu_textures()
                self._upload_cpu(first=True)
                self._cpu_thread = threading.Thread(target=self._cpu_loop, daemon=True)
                self._cpu_stop = False
                self._cpu_thread.start()
        except Exception as e:                       # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.model = None
            self.running = False
            return
        self.running = True
        self.t_prev = self.t_curr = self.target = 0.0
        self.flashes = []
        self.diag = {}
        self.name = ""
        self._steps_window = []
        self._update_view(1.0)

    def stop(self):
        self.running = False
        if self._cpu_thread is not None:
            self._cpu_stop = True
            self._cpu_thread.join(timeout=5.0)
            self._cpu_thread = None
        if self.model is not None and hasattr(self.model, "release"):
            try:
                self.model.release()
            except Exception:
                pass
        if self._tex:
            for t in self._tex:
                t.release()
            self._tex = None
        self.model = None
        self.view.textures = None
        self.flashes = []

    # --------------------------------------------------------- CPU path --
    def _make_cpu_textures(self):
        g = self.sc.grid
        size = (g.nx, g.ny, g.nz)
        self._tex = []
        for i in range(4):
            t = self.ctx.texture3d(size, 4, dtype="f2")
            t.repeat_x = t.repeat_y = True
            t.repeat_z = False
            if i < 2:
                t.build_mipmaps()
                t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            else:
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._tex.append(t)
        self._cpu_cur = 0

    @staticmethod
    def cpu_fields(m: crm.CPUModel):
        o = m.optics()
        opt = np.stack([o["liq"], o["ice"], o["rain"], o["snow"]], -1)     # z,y,x,4
        sc = o["liq"] + o["ice"]
        dz = m.g.dz
        cum = np.cumsum(sc[::-1], axis=0)[::-1] * dz
        tau_mid = cum - 0.5 * sc * dz
        um, vm = m.sc.translate
        uc = 0.5 * (m.u + np.roll(m.u, -1, 2)) + um
        vc = 0.5 * (m.v + np.roll(m.v, -1, 1)) + vm
        wc = 0.5 * (m.w[1:] + m.w[:-1])
        aux = np.stack([tau_mid, uc, vc, wc], -1)
        return opt.astype("f2"), aux.astype("f2"), m.time

    def _upload_cpu(self, first=False):
        with self._lock:
            pack = self._cpu_opt
            self._cpu_opt = None
        if pack is None and not first:
            return False
        if pack is None:
            pack = self.cpu_fields(self.model)
        opt, aux, t = pack
        self._cpu_cur ^= 1
        a = self._cpu_cur
        self._tex[a].write(np.ascontiguousarray(opt).tobytes())
        self._tex[a].build_mipmaps()
        self._tex[2 + a].write(np.ascontiguousarray(aux).tobytes())
        if first:
            self._tex[a ^ 1].write(np.ascontiguousarray(opt).tobytes())
            self._tex[a ^ 1].build_mipmaps()
            self._tex[2 + (a ^ 1)].write(np.ascontiguousarray(aux).tobytes())
            self.t_prev = self.t_curr = t
        else:
            self.t_prev, self.t_curr = self.t_curr, t
        return True

    def _cpu_loop(self):
        m = self.model
        while not self._cpu_stop:
            if m.time >= self.target or not self.running:
                time.sleep(0.01)
                continue
            try:
                m.sun_elev, m.sun_az = self.sun_elev, self.sun_az
                dt = m.max_dt(0.8)
                t0 = time.time()
                m.step(dt)
                pack = self.cpu_fields(m)
                d = m.diagnostics()
                with self._lock:
                    self._cpu_opt = pack
                    self.diag = d
                    self._steps_window.append(time.time())
                if not self._check_health(d):
                    return
            except Exception as e:                   # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                self.running = False
                return

    # ------------------------------------------------------------ advance --
    def advance(self, model_target: float, budget_s: float = 0.030) -> float:
        """Step the model toward model_target (seconds since the start) for
        at most budget_s of wall time.  Returns the model time now reached."""
        if not self.running or self.model is None:
            return self.target
        self.target = model_target
        m = self.model
        if self.gpu_ok:
            t0 = time.time()
            n = 0
            while m.time < model_target and time.time() - t0 < budget_s and n < 6:
                m.sun_elev, m.sun_az = self.sun_elev, self.sun_az
                if n == 0 and m.steps % 4 == 0 and m.steps > 0:
                    self._dt = min(m.max_dt(0.8), self._dt * 1.1)
                self.t_prev = self.t_curr
                m.step(self._dt)
                self.t_curr = m.time
                n += 1
                self._steps_window.append(time.time())
            if n and (m.steps % 8 == 0 or not self.diag):
                self.diag = m.diagnostics()
                self._check_health(self.diag)
        else:
            self._upload_cpu()
        now = time.time()
        self._steps_window = [t for t in self._steps_window if now - t < 2.0]
        self.steps_per_s = len(self._steps_window) / 2.0
        return self.model_time

    @property
    def model_time(self) -> float:
        return self.t_curr

    def _check_health(self, d) -> bool:
        """Stop the model, with a message, if it has become numerically
        unstable (a non-finite field or an impossible updraught) rather than
        go on drawing garbage."""
        vals = [d.get(k) for k in ("w_max", "w_min", "water", "cloud_top")]
        bad = any(v is None or not math.isfinite(v) for v in vals) or \
            max(abs(d.get("w_max", 0.0)), abs(d.get("w_min", 0.0))) > 150.0
        if bad:
            self.error = (f"the model became numerically unstable at t = "
                          f"{d.get('time', 0.0) / 60:.0f} min and was stopped")
            self.running = False
        return not bad

    # ---------------------------------------------------------- the view --
    def _update_view(self, mix: float):
        v = self.view
        m = self.model
        sc = self.sc
        if m is None:
            v.textures = None
            return
        g = sc.grid
        if self.gpu_ok:
            oa, ob, aa, ab = m.optics_textures()
            v.textures = (oa, ob, aa, ab)
        else:
            a = self._cpu_cur
            v.textures = (self._tex[a ^ 1], self._tex[a], self._tex[2 + (a ^ 1)], self._tex[2 + a])
        v.mix = mix
        v.size = (g.lx, g.ly, g.nz * g.dz)
        v.cell = (g.dx, g.dy, g.dz)
        v.z0 = g.z0
        um, vm = sc.translate
        v.move = (um, vm)
        ox = sc.observer[0] * g.lx - um * self.t_curr
        oy = sc.observer[1] * g.ly - vm * self.t_curr
        if sc.tiling & 1:
            ox %= g.lx
        if sc.tiling & 2:
            oy %= g.ly
        v.obs = (ox, oy)
        v.tile = sc.tiling
        # the heights between which there is anything to see, so a ray is
        # marched only through them (from the diagnostics, with a margin for
        # the steps since they were taken)
        d = self.diag or {}
        depth = g.nz * g.dz
        pad = max(3.0 * g.dz, 150.0)
        top, base = d.get("cloud_top", 0.0), d.get("cloud_base", 0.0)
        pb = d.get("precip_bottom", -1.0)
        lows = [z for z in ((base if top > 0 else None), (pb if pb >= 0 else None)) if z is not None]
        if top > 0 or pb >= 0:
            v.zlo = max(g.z0, min(lows) - pad) if lows else g.z0
            v.zhi = min(g.z0 + depth, max(top, pb) + pad)
        else:
            v.zlo, v.zhi = g.z0, g.z0 + depth
        v.zbase = base if top > 0 else v.zlo
        v.time = self.t_curr
        v.jde = gl_sky.jde_params(crm.R_EFF_LIQ * 1e6)

    def frame(self, display_time: float, now: float):
        """Called once per rendered frame with the model time to display."""
        if self.model is None:
            self.view.textures = None
            return self.view
        span = max(self.t_curr - self.t_prev, 1e-6)
        mix = min(max((display_time - self.t_prev) / span, 0.0), 1.0)
        if not self.running:
            mix = 1.0
        self._update_view(mix)
        self._lightning(now)
        return self.view

    # --------------------------------------------------------- lightning --
    def _column_charge(self):
        m = self.model
        if self.gpu_ok:
            cd = m.column_diag()
            return cd[..., 12].astype("f8"), cd[..., 13].astype("f8")
        rho = m.b.rho0.reshape(-1, 1, 1)
        t = m._temperature() - crm.T0K
        wc = 0.5 * (m.w[1:] + m.w[:-1])
        cond = m.qc + m.qi
        ch = np.where((t < -5.0) & (t > -40.0) & (wc > 1.0), rho * m.qs * cond * 1e6, 0.0)
        charge = ch.sum(axis=0) * m.g.dz
        kz = np.argmax(ch, axis=0)
        return charge, m.g.z0 + (kz + 0.5) * m.g.dz

    def _lightning(self, now: float):
        v = self.view
        self.flashes = [f for f in self.flashes if not f.done(now)]
        d = self.diag or {}
        top_km = d.get("cloud_top", 0.0) / 1000.0
        ice = d.get("ice_frac", 0.0)
        dt_real = min(max(now - self._flash_clock, 0.0), 0.2)
        self._flash_clock = now
        if self.running and top_km > 7.0 and ice > 0.05 and d.get("w_max", 0) > 8.0:
            if now - self._charge_t > 1.5:
                self._charge = self._column_charge()
                self._charge_t = now
            charge, chz = self._charge
            if charge.max() > 0.0:
                rate_per_min = PR92_A * top_km ** PR92_B
                # model seconds per real second (time-lapse speeds the storm)
                speed = max(self.steps_per_s, 0.0) * max(self._dt if hasattr(self, "_dt") else 3.0, 0.1)
                speed = max(speed, 1.0)
                lam = rate_per_min / 60.0 * speed * dt_real
                lam = min(lam, 4.0 * dt_real)            # at most ~4 flashes a real second
                if random.random() < lam and len(self.flashes) < 3:
                    self.flashes.append(self._make_flash(charge, chz, now))
        power = 0.0
        pos = (0.0, 0.0, 0.0)
        bolt = []
        for f in self.flashes:
            i = f.intensity(now)
            if i > power:
                power, pos = i, f.pos
            if f.cg and i > 0.05:
                bolt = [(x, y, z, b * i) for (x, y, z, b) in f.bolt]
        v.flash_power = power * 40.0
        v.flash_pos = pos
        v.bolt = bolt

    def _make_flash(self, charge, chz, now):
        sc = self.sc
        g = sc.grid
        p = charge.ravel() / charge.sum()
        idx = np.random.choice(p.size, p=p)
        j, i = divmod(int(idx), g.nx)
        z = float(chz.ravel()[idx]) + random.uniform(-800.0, 800.0)
        x = (i + random.random()) * g.dx
        y = (j + random.random()) * g.dy
        ox, oy = self.view.obs
        dx, dy = x - ox, y - oy
        if sc.tiling & 1:
            dx = (dx + 0.5 * g.lx) % g.lx - 0.5 * g.lx
        if sc.tiling & 2:
            dy = (dy + 0.5 * g.ly) % g.ly - 0.5 * g.ly
        pos = world_point(dx, dy, z)
        cg = random.random() < CG_FRACTION
        bolt = make_bolt(dx, dy, z * 0.55, pos) if cg else []
        return Flash(pos, cg, bolt, now)

    # ------------------------------------------------------------- naming --
    def atlas_name(self) -> tuple[str, list[str]]:
        """The Atlas name of what the model has grown, and why."""
        d = self.diag
        if not d or d.get("cloud_cover", 0.0) <= 0.0 and d.get("cloud_top", 0.0) <= 0.0:
            return "", ["no cloud yet"]
        base, top = d.get("cloud_base", 0.0), d.get("cloud_top", 0.0)
        depth = top - base
        ice = d.get("ice_frac", 0.0)
        rain = d.get("rain_rate_max", 0.0)
        why = [f"cloud base {base:.0f} m, highest top {top:.0f} m",
               f"updraught up to {d.get('w_max', 0):.0f} m/s, ice {ice*100:.0f}% of the cloud"]
        sc = self.sc
        if sc is not None and sc.shows and (sc.layer_depth > 0.0 or sc.key == "streets"):
            # a layer case: the form it is built to grow (the in-view panel
            # adds the varieties the field actually shows)
            name, reason = sc.shows[0]
            why.append(reason)
            if d.get("precip_bottom", -1.0) > max(sc.grid.z0, 0.0) + 1.0 and "vir" not in name:
                why.append(f"its precipitation evaporates {d['precip_bottom']:.0f} m up -> virga")
            return name, why
        if ice > 0.25 and depth > 5000:
            spec = CloudSpec("Cb", "cap" if ice > 0.45 else "cal")
            why.append("glaciating top on a deep tower -> Cumulonimbus "
                       + ("capillatus" if ice > 0.45 else "calvus"))
            if ice > 0.5 and top > 9000:
                spec.supplementary.append("inc")
                why.append("ice spreading at the top -> incus")
        else:
            sp = "hum" if depth < 1000 else ("med" if depth < 2500 else "con")
            spec = CloudSpec("Cu", sp)
            why.append(f"depth {depth:.0f} m -> Cumulus "
                       + {"hum": "humilis", "med": "mediocris", "con": "congestus"}[sp])
        if rain > 0.5:
            spec.supplementary.append("pra")
            why.append(f"rain reaching the ground, up to {rain:.0f} mm/h -> praecipitatio")
        try:
            name = spec.latin()
        except Exception:
            name = spec.abbrev()
        return name, why


def world_point(dx: float, dy: float, h: float):
    """A point h metres above the ground at (dx east, dy north) of the eye, in
    the renderer's frame: x east, y up (from the eye), z north, on the curved
    Earth."""
    RG = 6360000.0
    n = np.array([dx, RG, dy], "f8")
    n /= np.linalg.norm(n)
    c = np.array([0.0, -RG, 0.0])
    p = c + (RG + h) * n
    return (float(p[0]), float(p[1]), float(p[2]))


def make_bolt(dx: float, dy: float, base_h: float, top_pos, segments: int = 36):
    """A cloud-to-ground channel: a random walk down from the cloud base with
    the tortuosity of real channels, and one fainter branch.  The branch is
    joined to the main channel by an invisible (zero-brightness) point, since
    the shader draws a single polyline."""
    pts = []
    h = max(base_h, 800.0)
    x, y = dx, dy
    step = h / segments
    walk = []
    for k in range(segments + 1):
        walk.append((x, y, h))
        pts.append((*world_point(x, y, h), 1.0))
        h = max(h - step, 0.0)
        x += random.gauss(0.0, step * 0.5)
        y += random.gauss(0.0, step * 0.5)
    k0 = random.randint(2, max(3, segments // 3))
    bx, by, bh = walk[k0]
    pts.append((*world_point(bx, by, bh), 0.0))
    ang = random.uniform(0.0, 2.0 * math.pi)
    for k in range(1, 48 - len(pts) + 1):
        bh -= step * 0.8
        if bh < base_h * 0.25:
            break
        bx += math.cos(ang) * step * 0.6 + random.gauss(0.0, step * 0.35)
        by += math.sin(ang) * step * 0.6 + random.gauss(0.0, step * 0.35)
        pts.append((*world_point(bx, by, bh), 0.35))
    return pts[:48]
