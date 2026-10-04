# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
CloudSim - the loading window.

The program's start blocks its one thread for seconds at a time: compiling
the sky's shaders, fetching the weather, building the layers' patterns,
starting the cloud model.  A window drawn by that thread could not move
while it waits (the white, frozen window), so the loading window is drawn
by a separate small process: it starts first, animates on its own, and the
program tells it how far it has got through a pipe.  The main window is
made hidden, draws its first frames, and only then shows - at which moment
the loading window closes.

    Loader.start(line)            in the program: start the loading window
    loader.step(frac, text, to)   it has got to frac (0-1) of the start and
                                  is doing text; while that lasts the bar
                                  creeps on towards `to`
    loader.close()                the program's window is up: close it

The window process (python splash.py) reads lines on its standard input:

    P <frac> <to> <text>          progress, as above
    Q                             close

and closes by itself when the pipe closes (the program has ended, or
failed), and after SAFETY_S whatever happens.  If it cannot start (no
display, no OpenGL) the program goes on without it: every Loader method is
safe to call on a loader whose window never came up.

This file runs on its own (it imports nothing of CloudSim): the window
process starts in a fraction of a second while the program is still
importing.
"""

from __future__ import annotations

import collections
import math
import os
import subprocess
import sys
import threading
import time

#: the loading window closes by itself after this long, whatever happens
SAFETY_S = 900.0
W, H = 640, 360
#: its picture: a sky drawn by CloudSim itself (sky_art when it is missing)
PICTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "splash.png")
TITLE = "CloudSim"
SUBTITLE = "Clouds and wind based on the WMO International Cloud Atlas"
FOOTER = "GPL-3.0-or-later  ·  © 2026 The CloudSim authors"
FONT = ("Segoe UI", "DejaVu Sans", "Verdana", "Arial", "sans-serif")


# ---------------------------------------------------------------------------
# The program's side

class Loader:
    """The loading window, from the program's side.  `log` keeps every step
    as (seconds since the start, frac, to, text) - what the window was told."""

    def __init__(self, proc=None):
        self.proc = proc
        self.frac = 0.0
        self.t0 = time.time()
        self.log: list = []
        self.closed = False
        self._out = collections.deque()
        self._cv = threading.Condition()
        self._last = (None, -1.0, 0.0)          # (text, frac, time) last sent
        if proc is not None:
            threading.Thread(target=self._send_loop, daemon=True).start()

    @classmethod
    def start(cls, line: str = "", python: str | None = None) -> "Loader":
        """Start the loading window (python splash.py).  A failure to start
        gives a loader with no window: the program goes on without one."""
        if os.environ.get("CLOUDSIM_NO_SPLASH"):
            return cls(None)
        cmd = [python or sys.executable, os.path.abspath(__file__)]
        if line:
            cmd += ["--line", line]
        kw = dict(stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                  stderr=None if os.environ.get("CLOUDSIM_SPLASH_DEBUG") else subprocess.DEVNULL)
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000                    # CREATE_NO_WINDOW: no console
            _allow_foreground()
        try:
            return cls(subprocess.Popen(cmd, **kw))
        except Exception:                                        # noqa: BLE001
            return cls(None)

    def alive(self) -> bool:
        """The loading window's process is running (it may still be opening)."""
        return self.proc is not None and self.proc.poll() is None and not self.closed

    def step(self, frac: float, text: str, to: float | None = None):
        """The start has got to frac (0-1) and is doing `text`; while it
        does, the bar creeps on towards `to`.  The bar never goes back."""
        frac = min(max(float(frac), self.frac), 1.0)
        to = frac if to is None else min(max(float(to), frac), 1.0)
        self.frac = frac
        text = " ".join(str(text).split())
        if not self.log or self.log[-1][1:] != (frac, to, text):
            self.log.append((time.time() - self.t0, frac, to, text))
        t, f, ts = self._last
        now = time.time()
        # a pipe is not a log: the same words a little further on are sent
        # at most twenty times a second
        if text == t and frac - f < 0.004 and now - ts < 0.05:
            return
        self._last = (text, frac, now)
        self._put(f"P {frac:.4f} {to:.4f} {text}")

    def close(self):
        """The program's window is up: the loading window goes now."""
        if self.closed:
            return
        self._put("Q")
        self._put(None)                                          # then the pipe closes
        self.closed = True

    def _put(self, msg):
        if self.proc is None:
            return
        with self._cv:
            self._out.append(msg)
            self._cv.notify()

    def _send_loop(self):
        """Writes in a thread of its own: a window process that is slow to
        read (or has died) never holds up the program."""
        pipe = self.proc.stdin
        while True:
            with self._cv:
                while not self._out:
                    self._cv.wait()
                msg = self._out.popleft()
            try:
                if msg is None:
                    pipe.close()
                    break
                pipe.write((msg + "\n").encode("utf-8"))
                pipe.flush()
            except Exception:                                    # noqa: BLE001
                break
        try:
            self.proc.wait(timeout=10.0)                         # no zombie left behind
        except Exception:                                        # noqa: BLE001
            pass


class NoLoader(Loader):
    """A loader with no window (the self-test, a script)."""

    def __init__(self):
        super().__init__(None)


def _allow_foreground():
    """Windows lets a process bring its window to the front only while it is
    the foreground process (or one allowed by it): each side allows the
    other, so the loading window can come up in front and the program's
    window can take its place."""
    try:
        import ctypes
        ctypes.windll.user32.AllowSetForegroundWindow(-1)        # ASFW_ANY
    except Exception:                                            # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# The window's side

def creep(frac: float, to: float, since: float, tau: float = 2.5) -> float:
    """Where the bar stands `since` seconds after the program said frac
    (going on to `to`): it moves on from frac towards `to` and slows as it
    nears it, so a long step (a slow network) never looks stuck and never
    runs past where the program has really got."""
    return frac + (to - frac) * (1.0 - math.exp(-max(since, 0.0) / tau))


def parse(line: str):
    """One line from the program: ("P", frac, to, text), ("Q",) or None."""
    line = line.strip()
    if line == "Q":
        return ("Q",)
    if line.startswith("P "):
        parts = line.split(" ", 3)
        try:
            f, t = float(parts[1]), float(parts[2])
        except (IndexError, ValueError):
            return None
        return ("P", min(max(f, 0.0), 1.0), min(max(t, 0.0), 1.0),
                parts[3] if len(parts) > 3 else "")
    return None


def sky_art(w: int, h: int, seed: int = 7):
    """The loading window's picture, w x h RGBA bytes from the bottom row up:
    a clear sky darkening upward, the Sun's glow, and a row of fair-weather
    cumulus along the bottom - lit from above, grey underneath (value-noise
    fbm cut at a threshold that falls toward the cloud row).  None without
    numpy."""
    try:
        import numpy as np
    except Exception:                                            # noqa: BLE001
        return None
    rng = np.random.default_rng(seed)
    yy = (np.arange(h, dtype="f4") + 0.5) / h                    # 0 at the bottom
    xx = (np.arange(w, dtype="f4") + 0.5) / w

    def noise(cells_x):
        cells_y = max(2, int(cells_x * h / w) + 2)
        g = rng.random((cells_y + 2, cells_x + 2)).astype("f4")
        xs = np.linspace(0.0, cells_x, w, endpoint=False, dtype="f4")
        ys = np.linspace(0.0, cells_y - 1, h, endpoint=False, dtype="f4")
        x0, y0 = xs.astype(int), ys.astype(int)
        fx, fy = xs - x0, ys - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        a, b = g[np.ix_(y0, x0)], g[np.ix_(y0, x0 + 1)]
        c, d = g[np.ix_(y0 + 1, x0)], g[np.ix_(y0 + 1, x0 + 1)]
        top = a + (b - a) * fx[None, :]
        bot = c + (d - c) * fx[None, :]
        return top + (bot - top) * fy[:, None]

    fbm = sum(noise(4 * 2 ** k) * 0.5 ** k for k in range(5)) / 1.9375
    # the sky: deep blue overhead to pale near the horizon
    t = (yy ** 0.8)[:, None]
    zen = np.array([28, 70, 132], "f4")
    hor = np.array([176, 208, 234], "f4")
    img = hor[None, None, :] * (1 - t[..., None]) + zen[None, None, :] * t[..., None]
    img = np.broadcast_to(img, (h, w, 3)).copy()
    # the Sun's glow, upper right
    sx, sy = 0.84, 0.80
    r2 = ((xx[None, :] - sx) * (w / h)) ** 2 + (yy[:, None] - sy) ** 2
    glow = np.exp(-r2 / 0.02) * 0.55 + np.exp(-r2 / 0.002) * 0.6
    img += glow[..., None] * np.array([255, 236, 200], "f4")[None, None, :]
    # the clouds: a row of cumulus, flat bases at 18 % of the height
    env = np.clip((yy - 0.18) / 0.05, 0, 1) * np.clip((0.62 - yy) / 0.30, 0, 1)
    dens = fbm - (0.70 - 0.32 * env[:, None])
    dens *= (yy[:, None] > 0.18)
    a = np.clip(dens / 0.08, 0, 1)
    a = a * a * (3 - 2 * a)
    # lit from above: where the cloud thins upward it is bright, inside it greys
    up = np.roll(dens, -6, axis=0)                               # 6 rows higher
    lit = np.clip(0.55 + (dens - up) * 6.0, 0, 1)
    shade = np.clip((yy[:, None] - 0.18) / 0.22, 0, 1)
    col = (np.array([150, 166, 188], "f4")[None, None, :] * (1 - lit[..., None])
           + np.array([252, 252, 255], "f4")[None, None, :] * lit[..., None])
    col = col * (0.82 + 0.18 * shade[..., None])
    img = img * (1 - a[..., None]) + col * a[..., None]
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = np.clip(img, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba.tobytes()


def run_window(line: str = "", report: bool = False):
    """The loading window's process: draw it until told to close.  report:
    say "UP" on standard output once the window is on the screen and "BYE"
    as it closes (the self-test)."""
    queue = collections.deque()

    def say(word):
        if report:
            try:
                sys.stdout.write(word + "\n")
                sys.stdout.flush()
            except Exception:                                    # noqa: BLE001
                pass

    def reader():
        try:
            for raw in sys.stdin.buffer:
                queue.append(raw.decode("utf-8", "replace"))
        except Exception:                                        # noqa: BLE001
            pass
        queue.append("EOF")

    threading.Thread(target=reader, daemon=True).start()

    os.environ.setdefault("PYGLET_SHADOW_WINDOW", "0")
    import pyglet
    pyglet.options["debug_gl"] = False
    from pyglet import shapes
    from pyglet.text import Label

    def label(text, bold=False, **kw):
        try:
            return Label(text, weight="bold" if bold else "normal", **kw)
        except TypeError:
            return Label(text, bold=bold, **kw)

    state = dict(frac=0.0, to=0.0, t=time.time(), text="Starting", shown=0.0, quit=False)

    def take():
        while queue:
            msg = queue.popleft()
            if msg == "EOF":
                state["quit"] = True
                continue
            p = parse(msg)
            if p is None:
                continue
            if p[0] == "Q":
                state["quit"] = True
            else:
                _, f, to, text = p
                state.update(frac=max(f, state["frac"]), to=max(to, f), t=time.time(),
                             text=text or state["text"])

    time.sleep(0.0)
    take()
    if state["quit"]:
        return                       # the program was ready before the window was

    style = getattr(pyglet.window.Window, "WINDOW_STYLE_BORDERLESS", None)
    win = pyglet.window.Window(W, H, caption=TITLE, style=style, visible=False)
    try:
        scr = win.screen
        win.set_location(scr.x + (scr.width - W) // 2, scr.y + (scr.height - H) // 2)
    except Exception:                                            # noqa: BLE001
        pass
    batch = pyglet.graphics.Batch()
    back = pyglet.graphics.Group(order=0)
    mid = pyglet.graphics.Group(order=1)
    front = pyglet.graphics.Group(order=2)
    held = []
    img = None
    if os.path.exists(PICTURE):
        try:                          # the program's own sky (data/splash.png)
            img = pyglet.image.load(PICTURE)
        except Exception:                                        # noqa: BLE001
            img = None
    if img is None:
        art = sky_art(W, H)
        if art is not None:
            img = pyglet.image.ImageData(W, H, "RGBA", art)
    if img is not None:
        spr = pyglet.sprite.Sprite(img, x=0, y=0, batch=batch, group=back)
        spr.scale_x, spr.scale_y = W / float(img.width), H / float(img.height)
        held.append(spr)
    else:
        for i in range(60):                                      # the sky in bands
            t = (i + 0.5) / 60.0
            c = tuple(int(a + (b - a) * t ** 0.8) for a, b in zip((176, 208, 234), (28, 70, 132)))
            held.append(shapes.Rectangle(0, H * i / 60.0, W, H / 60.0 + 1, color=c,
                                         batch=batch, group=back))
    band = shapes.Rectangle(0, 0, W, 98, color=(8, 14, 24), batch=batch, group=mid)
    band.opacity = 170
    held.append(band)
    for x, y, w, h in ((0, 0, W, 1), (0, H - 1, W, 1), (0, 0, 1, H), (W - 1, 0, 1, H)):
        r = shapes.Rectangle(x, y, w, h, color=(20, 34, 56), batch=batch, group=front)
        held.append(r)
    held.append(label(TITLE, bold=True, font_name=FONT, font_size=30, x=40, y=H - 76,
                      color=(255, 255, 255, 255), batch=batch, group=front))
    held.append(label(SUBTITLE, font_name=FONT, font_size=11, x=42, y=H - 102,
                      color=(240, 246, 255, 230), batch=batch, group=front))
    if line:
        held.append(label(line, font_name=FONT, font_size=10, x=42, y=H - 124,
                          color=(226, 236, 250, 210), batch=batch, group=front))
    status = label("Starting", font_name=FONT, font_size=10, x=40, y=58,
                   color=(236, 242, 250, 255), batch=batch, group=front)
    pct = label("0%", font_name=FONT, font_size=10, x=W - 40, y=58, anchor_x="right",
                color=(236, 242, 250, 255), batch=batch, group=front)
    held.append(label(FOOTER, font_name=FONT, font_size=8, x=40, y=16,
                      color=(170, 184, 204, 255), batch=batch, group=front))
    bx, by, bw, bh = 40, 38, W - 80, 6
    track = shapes.Rectangle(bx, by, bw, bh, color=(255, 255, 255), batch=batch, group=front)
    track.opacity = 46
    fill = shapes.Rectangle(bx, by, 1, bh, color=(122, 192, 255), batch=batch, group=front)
    gleam = shapes.Rectangle(bx, by, 28, bh, color=(214, 236, 255), batch=batch, group=front)
    gleam.opacity = 0
    held += [status, pct, track, fill, gleam]

    if sys.platform == "win32":
        _allow_foreground()
    win.set_visible(True)
    if sys.platform == "win32":
        try:                                    # above the other windows while it lasts
            import ctypes
            u32 = ctypes.WinDLL("user32")       # its own function objects, typed here
            u32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            # HWND_TOPMOST is (HWND)-1; SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE
            u32.SetWindowPos(win._hwnd, ctypes.c_void_p(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)
        except Exception:                                        # noqa: BLE001
            pass
    t_start = time.time()

    @win.event
    def on_draw():
        win.clear()
        batch.draw()

    def tick(dt):
        take()
        now = time.time()
        if state["quit"] or now - t_start > SAFETY_S:
            pyglet.clock.unschedule(tick)
            say("BYE")
            win.close()
            pyglet.app.exit()
            return
        if not state.get("up"):
            state["up"] = True
            say("UP")
        goal = creep(state["frac"], state["to"], now - state["t"])
        s = state["shown"]
        s = max(s, s + (goal - s) * (1.0 - math.exp(-dt / 0.12)))
        state["shown"] = s
        fill.width = max(1.0, bw * s)
        status.text = state["text"]
        pct.text = f"{int(s * 100 + 1e-6)}%"
        # a gleam running along the filled part
        k = ((now - t_start) / 1.6) % 1.0
        span = max(fill.width - 28, 0.0)
        gleam.x = bx + span * k
        gleam.opacity = int(70 * math.sin(math.pi * k)) if fill.width > 40 else 0

    @win.event
    def on_close():
        state["quit"] = True

    pyglet.clock.schedule_interval(tick, 1 / 60.0)
    pyglet.app.run()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    line = ""
    if "--line" in argv:
        i = argv.index("--line")
        line = argv[i + 1] if i + 1 < len(argv) else ""
    try:
        run_window(line, report="--report" in argv)
    except Exception:                                            # noqa: BLE001
        if os.environ.get("CLOUDSIM_SPLASH_DEBUG"):
            raise
    # out at once: the thread reading the pipe may still be waiting on it,
    # and an interpreter shutting down around such a thread can abort
    try:
        sys.stdout.flush()
    except Exception:                                            # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
