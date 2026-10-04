# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The CloudSim authors
# CloudSim is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3 or (at your option) any
# later version.  It comes with NO WARRANTY; see the LICENSE file.
"""
A small immediate-ish widget set drawn with pyglet, for the control panel.

Deliberately minimal: rows of buttons, toggles, radio groups and sliders in a
scrollable column.  Everything is rebuilt into a single pyglet Batch whenever
the content changes, so drawing is one call.
"""

from __future__ import annotations

import math

import pyglet
from pyglet import shapes
from pyglet.text import Label

FONT = ("DejaVu Sans", "Verdana", "Arial", "sans-serif")
MONO = ("DejaVu Sans Mono", "Consolas", "Courier New", "monospace")


def px(v) -> int:
    """The nearest whole pixel.  Text has to sit on whole pixels: pyglet
    draws each letter from its texture nearest-texel, so a label put half a
    pixel off has every letter rounded up or down on its own, and the letters
    of a word stand a pixel apart, as if dancing."""
    return int(math.floor(float(v) + 0.5))


def label(text, bold=False, **kw):
    """pyglet 2.1 spells bold as weight='bold'; 2.0 used bold=True.  The
    place is put on whole pixels (px)."""
    for k in ("x", "y"):
        if k in kw:
            kw[k] = px(kw[k])
    try:
        return Label(text, weight="bold" if bold else "normal", **kw)
    except TypeError:
        return Label(text, bold=bold, **kw)

BG = (16, 20, 28)
BG2 = (44, 52, 68)
FG = (222, 230, 240, 255)
DIM = (140, 152, 170, 255)
WARN = (232, 176, 96, 255)
ACC = (86, 156, 214)
ACC_DIM = (46, 78, 112)
OK = (126, 200, 140, 255)


_WRAP_H: dict = {}


def wrapped_height(text, size, bold, width):
    """The height a wrapped text really takes: laid out by pyglet with the
    font this computer has (a character count cannot know where the lines
    of Verdana or DejaVu break).  Kept, as the panels are rebuilt often."""
    k = (text, size, bool(bold), int(width))
    h = _WRAP_H.get(k)
    if h is None:
        if len(_WRAP_H) > 4000:
            _WRAP_H.clear()
        lab = label(text, bold=bold, font_name=FONT, font_size=size, width=int(width),
                    multiline=True)
        h = int(-(-lab.content_height // 1))
        lab.delete()
        _WRAP_H[k] = h
    return h


_TEXT_W: dict = {}


def text_width(text, size, bold=False):
    """The width of one line of text in this computer's font (kept)."""
    k = (text, size, bool(bold))
    w = _TEXT_W.get(k)
    if w is None:
        if len(_TEXT_W) > 8000:
            _TEXT_W.clear()
        lab = label(text, bold=bold, font_name=FONT, font_size=size)
        w = float(lab.content_width)
        lab.delete()
        _TEXT_W[k] = w
    return w


def fit(text, width, size, bold=False):
    """The text, shortened with an ellipsis if it is wider than width."""
    if width <= 0 or text_width(text, size, bold) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:                          # the longest start that fits with "…"
        mid = (lo + hi + 1) // 2
        if text_width(text[:mid].rstrip() + "…", size, bold) <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…"


class Widget:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def hit(self, mx, my):
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def build(self, batch, groups):
        raise NotImplementedError

    def on_click(self, mx, my, button):
        return False

    def on_drag(self, mx, my):
        return False


class Text(Widget):
    def __init__(self, x, y, w, text, colour=FG, size=10, bold=False, wrap=False):
        h = 15 if not wrap else 15
        super().__init__(x, y, w, h)
        self.text = text
        self.colour = colour
        self.size = size
        self.bold = bold
        self.wrap = wrap
        self._labels = []

    def build(self, batch, groups):
        if self.wrap:
            # the lines hang from the top of the box, which Panel.text made
            # as tall as they really are on this computer's font
            self._labels = [label(self.text, bold=self.bold, font_name=FONT,
                                  font_size=self.size, x=self.x, y=self.y + self.h - 2,
                                  width=self.w, multiline=True, anchor_y="top",
                                  color=self.colour, batch=batch, group=groups[1])]
            return
        self._labels = [label(self.text, bold=self.bold, font_name=FONT,
                              font_size=self.size, x=self.x, y=self.y + 2,
                              color=self.colour, batch=batch, group=groups[1])]


class Button(Widget):
    def __init__(self, x, y, w, text, callback, active=False, enabled=True,
                 tooltip="", size=10):
        super().__init__(x, y, w, 20)
        self.text = text
        self.callback = callback
        self.active = active
        self.enabled = enabled
        self.tooltip = tooltip
        self.size = size
        self.hover = False

    def build(self, batch, groups):
        col = ACC if self.active else (BG2 if self.enabled else (22, 26, 34))
        r = shapes.Rectangle(self.x, self.y, self.w, self.h, color=col,
                             batch=batch, group=groups[0])
        r.opacity = 255 if self.enabled else 150
        self._r = r
        c = (255, 255, 255, 255) if self.active else (FG if self.enabled else DIM)
        self._l = label(self.text, font_name=FONT, font_size=self.size,
                        x=px(self.x + self.w / 2), y=px(self.y + self.h / 2),
                        anchor_x="center", anchor_y="center", color=c,
                        batch=batch, group=groups[1])

    def on_click(self, mx, my, button):
        if self.enabled and self.callback:
            self.callback()
            return True
        return False


class Toggle(Button):
    def __init__(self, x, y, w, text, get, set_, enabled=True, tooltip=""):
        super().__init__(x, y, w, text, None, enabled=enabled, tooltip=tooltip)
        self.get = get
        self.set = set_

    def build(self, batch, groups):
        self.active = bool(self.get())
        super().build(batch, groups)

    def on_click(self, mx, my, button):
        if not self.enabled:
            return False
        self.set(not self.get())
        return True


class Slider(Widget):
    def __init__(self, x, y, w, label, get, set_, lo, hi, fmt="{:.2f}"):
        super().__init__(x, y, w, 30)
        self.label = label
        self.get = get
        self.set = set_
        self.lo, self.hi = lo, hi
        self.fmt = fmt

    def build(self, batch, groups):
        v = self.get()
        f = (v - self.lo) / max(self.hi - self.lo, 1e-9)
        f = min(max(f, 0.0), 1.0)
        # pyglet shapes vanish from the batch as soon as they are collected,
        # so every one of them has to be held onto
        self._parts = [
            label(f"{self.label}  {self.fmt.format(v)}", font_name=FONT,
                  font_size=9, x=self.x, y=self.y + 17, color=DIM,
                  batch=batch, group=groups[1]),
            shapes.Rectangle(self.x, self.y + 6, self.w, 5, color=(44, 52, 68),
                             batch=batch, group=groups[0]),
            shapes.Rectangle(self.x, self.y + 6, max(3, self.w * f), 5, color=ACC,
                             batch=batch, group=groups[0]),
            shapes.Rectangle(self.x + self.w * f - 3, self.y + 2, 7, 13,
                             color=(210, 220, 235), batch=batch, group=groups[0])]

    def on_click(self, mx, my, button):
        return self.on_drag(mx, my)

    def on_drag(self, mx, my):
        f = (mx - self.x) / max(self.w, 1)
        f = min(max(f, 0.0), 1.0)
        self.set(self.lo + f * (self.hi - self.lo))
        return True


class Panel:
    """A scrollable column of widgets on the left of the window."""

    def __init__(self, width=352):
        self.width = width
        self.x0 = 0                # left edge in the window (a panel on the right sets it)
        self.widgets: list[Widget] = []
        self.scroll = 0.0
        self.content_h = 0
        self.batch = pyglet.graphics.Batch()
        self.groups = (pyglet.graphics.Group(order=1), pyglet.graphics.Group(order=2))
        self._bgbatch = pyglet.graphics.Batch()
        self._bg = None
        self._drag = None
        self.height = 600
        self.visible = True

    # -- layout helpers ----------------------------------------------------
    def begin(self, height, x0=None):
        self.height = height
        if x0 is not None:
            self.x0 = int(x0)
        self.widgets = []
        self._y = 0            # grows downward in "content" space
        self.pad = 12

    def _place(self, h):
        y = self._y
        self._y += h
        return y

    def text(self, s, colour=FG, size=10, bold=False, wrap=False, indent=0):
        w = Text(self.pad + indent, 0, self.width - 2 * self.pad - indent, s,
                 colour, size, bold, wrap)
        if wrap:
            # measured, not guessed from a character count: the lines break
            # where this computer's font makes them break
            h = max(wrapped_height(s, size, bold, self.width - 2 * self.pad - indent), 15) + 3
            w.h = h
        else:
            h = 16
        w._top = self._place(h + 2)
        self.widgets.append(w)
        return w

    def gap(self, h=8):
        self._place(h)

    def rule(self):
        self._place(6)

    def buttons(self, items, height=22, per_row=None):
        """items: list of (text, callback, active, enabled)"""
        n = len(items)
        per_row = per_row or n
        rows = (n + per_row - 1) // per_row
        avail = self.width - 2 * self.pad
        for r in range(rows):
            row = items[r * per_row:(r + 1) * per_row]
            bw = (avail - 4 * (len(row) - 1)) / max(len(row), 1)
            top = self._place(height + 4)
            for i, it in enumerate(row):
                text, cb, active, enabled = (list(it) + [False, True])[:4]
                b = Button(self.pad + i * (bw + 4), 0, bw, text, cb,
                           active=active, enabled=enabled)
                b.h = height
                b._top = top
                self.widgets.append(b)

    def toggles(self, items, per_row=3, height=22):
        avail = self.width - 2 * self.pad
        n = len(items)
        rows = (n + per_row - 1) // per_row
        for r in range(rows):
            row = items[r * per_row:(r + 1) * per_row]
            bw = (avail - 4 * (len(row) - 1)) / per_row
            top = self._place(height + 4)
            for i, (text, get, set_, enabled) in enumerate(row):
                t = Toggle(self.pad + i * (bw + 4), 0, bw, text, get, set_, enabled)
                t.h = height
                t._top = top
                self.widgets.append(t)

    def slider(self, label, get, set_, lo, hi, fmt="{:.2f}"):
        s = Slider(self.pad, 0, self.width - 2 * self.pad, label, get, set_, lo, hi, fmt)
        s._top = self._place(34)
        self.widgets.append(s)

    def end(self):
        self.content_h = self._y
        max_scroll = max(0.0, self.content_h - (self.height - 16))
        self.scroll = min(max(self.scroll, 0.0), max_scroll)
        self.batch = pyglet.graphics.Batch()
        self._bgbatch = pyglet.graphics.Batch()
        bg = shapes.Rectangle(self.x0, 0, self.width, self.height, color=BG,
                              batch=self._bgbatch)
        bg.opacity = 232
        self._bg = bg
        for w in self.widgets:
            if not hasattr(w, "_rel_x"):
                w._rel_x = w.x
            w.x = w._rel_x + self.x0
            # scrolled by whole pixels (a touchpad scrolls by fractions)
            w.y = self.height - 8 - w._top - w.h + px(self.scroll)
            w.build(self.batch, self.groups)
        # scrollbar
        self._extra = []
        if max_scroll > 0:
            f = (self.height - 16) / self.content_h
            hh = max(24, (self.height - 16) * f)
            top = (self.scroll / max_scroll) * (self.height - 16 - hh)
            self._extra.append(
                shapes.Rectangle(self.x0 + self.width - 5, self.height - 8 - top - hh, 3, hh,
                                 color=(70, 84, 104), batch=self._bgbatch))

    # -- interaction -------------------------------------------------------
    def draw(self):
        if not self.visible:
            return
        self._bgbatch.draw()
        self.batch.draw()

    def contains(self, mx, my):
        return self.visible and self.x0 <= mx < self.x0 + self.width

    def on_click(self, mx, my, button):
        for w in self.widgets:
            if w.hit(mx, my):
                if w.on_click(mx, my, button):
                    self._drag = w if isinstance(w, Slider) else None
                    return True
        return True    # swallow clicks on the panel background

    def on_drag(self, mx, my):
        if self._drag is not None:
            self._drag.on_drag(mx, my)
            return True
        return False

    def on_release(self):
        self._drag = None

    def on_scroll(self, dy):
        self.scroll = max(0.0, min(self.scroll - dy * 42,
                                   max(0.0, self.content_h - (self.height - 16))))


class Picker:
    """A list to choose from, drawn over the sky: the dropdown of the place,
    the date and the hour.  Typing narrows the list - or gives a value it
    does not hold (coordinates, a date, an hour), which then comes first;
    arrows, Page Up/Down or the wheel move the choice; Enter or a click
    picks; Esc or a click outside closes.  Ctrl+V pastes.

    source(query) -> rows (text, value, note); a row whose value is None is a
    remark, not a choice.  on_pick(value) is called with the row picked.
    query: a text to start from, shown selected - the first key typed
    replaces it, Backspace clears it."""

    ROW = 22
    PAD = 12

    def __init__(self, title, source, on_pick, hint="", start=None, width=600, query=""):
        self.title, self.source, self.on_pick, self.hint = title, source, on_pick, hint
        self.max_width = width
        self.query = query
        self._fresh = bool(query)
        self.rows = list(source(query))
        self.sel = self._first_choice(0)
        if start is not None:
            for k, r in enumerate(self.rows):
                if r[1] == start:
                    self.sel = k
                    break
        self.top = 0
        self._centred = False             # the first layout puts the choice mid-list
        self.closed = False
        self.x = self.y = self.w = self.h = 0
        self.visible_rows = 10
        self.batch = pyglet.graphics.Batch()
        self._held = []
        self._hit_rows = []

    # -- geometry --------------------------------------------------------------
    def layout(self, win_w, win_h, left=0, right=0):
        """Centred over the sky between the panels (over the panels too
        when they leave the sky too narrow for it)."""
        avail = win_w - left - right
        if avail < 360:
            left, avail = 0, win_w
        self.w = int(max(min(self.max_width, avail - 24), min(260, win_w)))
        head = 34 + 30 + 20                 # title, the typed line, the hint
        foot = 24
        n = max(4, min(18, (win_h - 80 - head - foot) // self.ROW))
        self.visible_rows = n
        self.h = head + n * self.ROW + foot
        self.x = int(min(max(left + (avail - self.w) // 2, 0), max(win_w - self.w, 0)))
        self.y = int(max((win_h - self.h) // 2, 0))
        if not self._centred:
            self.top = self.sel - n // 2
            self._centred = True
        self._keep_sel_visible()
        self.build()

    def contains(self, mx, my):
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def _first_choice(self, k):
        for i in range(max(k, 0), len(self.rows)):
            if self.rows[i][1] is not None:
                return i
        return 0

    def _keep_sel_visible(self):
        n = self.visible_rows
        if self.sel < self.top:
            self.top = self.sel
        elif self.sel >= self.top + n:
            self.top = self.sel - n + 1
        self.top = max(0, min(self.top, max(len(self.rows) - n, 0)))

    # -- drawing ---------------------------------------------------------------
    def build(self):
        b = pyglet.graphics.Batch()
        # backgrounds, then highlights, then text
        g0, gm, g1 = (pyglet.graphics.Group(order=10), pyglet.graphics.Group(order=11),
                      pyglet.graphics.Group(order=12))
        held = []
        x, y, w, h = self.x, self.y, self.w, self.h
        inner = w - 2 * self.PAD
        bg = shapes.Rectangle(x, y, w, h, color=(12, 16, 22), batch=b, group=g0)
        bg.opacity = 248
        held.append(bg)
        for rx, ry, rw, rh in ((x, y, w, 1), (x, y + h - 1, w, 1), (x, y, 1, h), (x + w - 1, y, 1, h)):
            held.append(shapes.Rectangle(rx, ry, rw, rh, color=ACC_DIM, batch=b, group=g0))
        top = y + h
        held.append(label(fit(self.title, inner, 11, True), bold=True, font_name=FONT,
                          font_size=11, x=x + self.PAD, y=top - 22, color=FG, batch=b, group=g1))
        box_y = top - 34 - 26
        held.append(shapes.Rectangle(x + self.PAD, box_y, inner, 24, color=BG2,
                                     batch=b, group=g0))
        typed = fit(self.query, inner - 20, 11)
        if self._fresh and self.query:
            # the text to start from, selected: typing replaces it
            held.append(shapes.Rectangle(x + self.PAD + 4, box_y + 3,
                                         text_width(typed, 11) + 4, 18, color=ACC_DIM,
                                         batch=b, group=gm))
        held.append(label(typed + ("" if self._fresh else "|"), font_name=FONT, font_size=11,
                          x=x + self.PAD + 6, y=box_y + 7, color=FG, batch=b, group=g1))
        held.append(label(fit(self.hint, inner, 9), font_name=FONT, font_size=9, x=x + self.PAD,
                          y=box_y - 15, color=DIM, batch=b, group=g1))
        list_top = box_y - 22
        self._hit_rows = []
        for k in range(self.top, min(self.top + self.visible_rows, len(self.rows))):
            text, value, note = self.rows[k]
            ry = list_top - (k - self.top + 1) * self.ROW
            if k == self.sel and value is not None:
                held.append(shapes.Rectangle(x + 4, ry, w - 8, self.ROW, color=ACC_DIM,
                                             batch=b, group=gm))
            room = inner - 8
            if note:
                note = fit(note, inner * 0.55, 9)
                room -= text_width(note, 9) + 14
                held.append(label(note, font_name=FONT, font_size=9, x=x + w - self.PAD,
                                  y=ry + 6, anchor_x="right", color=DIM, batch=b, group=g1))
            held.append(label(fit(text, room, 10), font_name=FONT, font_size=10, x=x + self.PAD,
                              y=ry + 6, color=FG if value is not None else DIM, batch=b,
                              group=g1))
            self._hit_rows.append((k, ry))
        n = len([r for r in self.rows if r[1] is not None])
        more = len(self.rows) - (self.top + self.visible_rows)
        foot = (f"{n} to choose from" + (f"  ·  {more} more below" if more > 0 else "")
                + "  ·  Enter picks  ·  Esc closes")
        held.append(label(fit(foot, inner, 9), font_name=FONT, font_size=9, x=x + self.PAD,
                          y=y + 8, color=DIM, batch=b, group=g1))
        if len(self.rows) > self.visible_rows:
            track = self.visible_rows * self.ROW
            f = self.visible_rows / len(self.rows)
            hh = max(14, track * f)
            t = self.top / max(len(self.rows) - self.visible_rows, 1)
            sy = list_top - hh - t * (track - hh)
            held.append(shapes.Rectangle(x + w - 7, sy, 3, hh, color=(90, 104, 124),
                                         batch=b, group=g1))
        self.batch, self._held = b, held

    def draw(self):
        if not self.closed:
            self.batch.draw()

    # -- input -----------------------------------------------------------------
    def _refilter(self):
        self.rows = list(self.source(self.query))
        self.sel = self._first_choice(0)
        self.top = 0
        self.build()

    def on_text(self, text):
        t = "".join(c for c in text if c.isprintable() and c not in "\r\n\t")
        if t:
            if self._fresh:
                self.query, self._fresh = "", False
            self.query += t
            self._refilter()

    def paste(self, text):
        self.on_text(" ".join((text or "").split()))

    def on_text_motion(self, motion):
        from pyglet.window import key as _k
        if motion == _k.MOTION_BACKSPACE:
            if self.query:
                self.query = "" if self._fresh else self.query[:-1]
                self._fresh = False
                self._refilter()
            return
        if motion == _k.MOTION_DELETE:
            if self.query:
                self.query, self._fresh = "", False
                self._refilter()
            return
        step = {_k.MOTION_UP: -1, _k.MOTION_DOWN: 1,
                _k.MOTION_PREVIOUS_PAGE: -self.visible_rows,
                _k.MOTION_NEXT_PAGE: self.visible_rows}.get(motion)
        if step is None or not self.rows:
            return
        k = min(max(self.sel + step, 0), len(self.rows) - 1)
        d = 1 if step > 0 else -1
        while 0 <= k < len(self.rows) and self.rows[k][1] is None:
            k += d
        if 0 <= k < len(self.rows):
            self.sel = k
            self._keep_sel_visible()
            self.build()

    def on_key_press(self, symbol, modifiers):
        from pyglet.window import key as _k
        if symbol in (_k.ENTER, _k.NUM_ENTER, _k.RETURN):
            self.pick(self.sel)
        elif symbol == _k.ESCAPE:
            self.close()

    def on_mouse_press(self, mx, my):
        if not self.contains(mx, my):
            self.close()
            return
        k = self._row_at(mx, my)
        if k is not None:
            self.pick(k)

    def on_mouse_motion(self, mx, my):
        k = self._row_at(mx, my)
        if k is not None and k != self.sel and self.rows[k][1] is not None:
            self.sel = k
            self.build()

    def on_mouse_scroll(self, dy):
        n = max(len(self.rows) - self.visible_rows, 0)
        self.top = int(min(max(self.top - dy * 3, 0), n))
        if not (self.top <= self.sel < self.top + self.visible_rows):
            self.sel = self._first_choice(self.top)
        self.build()

    def _row_at(self, mx, my):
        if not (self.x <= mx <= self.x + self.w):
            return None
        for k, ry in self._hit_rows:
            if ry <= my < ry + self.ROW:
                return k
        return None

    def pick(self, k):
        if 0 <= k < len(self.rows) and self.rows[k][1] is not None:
            value = self.rows[k][1]
            self.closed = True
            self.on_pick(value)

    def close(self):
        self.closed = True


def _set(lab, text):
    """A label's text, laid out again only when it has changed."""
    if lab.text != text:
        lab.text = text


class LoadingCard:
    """The loading card drawn over the sky while something that was asked
    for loads - the weather of a place or a date, the sky of a new moment -
    and gone the frame it is done: a title, what it is doing, and a bar
    (frac None: a bar that sweeps, how long it takes is not known).  The
    sky behind is dimmed; the panels stay as they are and keep working."""

    W, H = 520, 104

    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        g0, g1, g2 = (pyglet.graphics.Group(order=k) for k in (0, 1, 2))
        b = self.batch
        self.shade = shapes.Rectangle(0, 0, 10, 10, color=(4, 8, 14), batch=b, group=g0)
        self.shade.opacity = 96
        self.edge = shapes.Rectangle(0, 0, self.W + 2, self.H + 2, color=ACC_DIM, batch=b, group=g1)
        self.card = shapes.Rectangle(0, 0, self.W, self.H, color=BG, batch=b, group=g1)
        self.card.opacity = 244
        self.track = shapes.Rectangle(0, 0, 10, 6, color=BG2, batch=b, group=g2)
        self.fill = shapes.Rectangle(0, 0, 1, 6, color=ACC, batch=b, group=g2)
        self.title = label("", bold=True, font_name=FONT, font_size=12, color=FG, batch=b, group=g2)
        self.detail = label("", font_name=FONT, font_size=9, color=DIM, batch=b, group=g2)
        self.pct = label("", font_name=FONT, font_size=9, color=DIM, anchor_x="right",
                         batch=b, group=g2)
        self.shown = 0.0
        self.t0 = None

    def reset(self):
        """A new load: the bar starts again from nothing."""
        self.shown = 0.0
        self.t0 = None

    def layout(self, x0, x1, height, title, detail, frac, now):
        """Place it in the middle of the sky between x0 and x1; frac (0-1 or
        None) as the bar.  The bar never goes back within one load."""
        if self.t0 is None:
            self.t0 = now
        w = max(x1 - x0, 1)
        self.shade.x, self.shade.y, self.shade.width, self.shade.height = x0, 0, w, height
        cw = px(min(self.W, max(w - 24, 160)))
        # on whole pixels (px): a window an odd number of pixels wide or
        # tall put the card, and its letters, half a pixel off
        cx = px(x0 + (w - cw) / 2.0)
        cy = px((height - self.H) / 2.0)
        self.card.x, self.card.y, self.card.width = cx, cy, cw
        self.edge.x, self.edge.y, self.edge.width = cx - 1, cy - 1, cw + 2
        pad = 20
        _set(self.title, fit(title, cw - 2 * pad, 12, True))
        self.title.x, self.title.y = cx + pad, cy + self.H - 34
        bx, bw = cx + pad, cw - 2 * pad
        self.track.x, self.track.y, self.track.width = bx, cy + 30, bw
        if frac is None:
            # how long is not known: a segment sweeping to and fro
            seg = 0.28 * bw
            k = 0.5 - 0.5 * math.cos((now - self.t0) * 2.4)
            self.fill.x, self.fill.width = bx + k * (bw - seg), seg
            _set(self.pct, "")
        else:
            self.shown = max(self.shown, min(max(float(frac), 0.0), 1.0))
            self.fill.x, self.fill.width = bx, max(2.0, bw * self.shown)
            _set(self.pct, f"{int(self.shown * 100 + 1e-6)}%")
        self.fill.y = cy + 30
        _set(self.detail, fit(detail, bw - 50, 9))
        self.detail.x, self.detail.y = bx, cy + 50
        self.pct.x, self.pct.y = bx + bw, cy + 50

    def draw(self):
        self.batch.draw()
