"""Simulated laser pens and scripted users.

A LaserPen has a color (multi-user attribution, Sec 4.3), an ON/OFF
state and a display-space position with hand-jitter. A ScriptedUser
produces timed gesture scripts (click / double-click / drag / paths)
that drive the pen during simulation ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

Event = Tuple[float, str, Optional[np.ndarray]]  # (t, "on"/"off"/"move", uv)


@dataclass
class LaserPen:
    pen_id: int
    color: str                      # "green", "blue", "red" ...
    radiance: float = 4000.0        # much brighter than content -> threshold works
    jitter_px: float = 3.0          # hand tremor in display px
    on: bool = False
    uv: Optional[np.ndarray] = None

    def state_at(self, rng: np.random.Generator):
        if not self.on or self.uv is None:
            return None
        return self.uv + rng.normal(0, self.jitter_px, 2)


class ScriptedUser:
    """Builds an event list; `pump(t)` applies due events to the pen."""

    def __init__(self, pen: LaserPen, seed: int = 0):
        self.pen = pen
        self.rng = np.random.default_rng(seed)
        self.events: List[Event] = []
        self._t = 0.0

    # ------------- script builders -------------
    def wait(self, dt: float):
        self._t += dt
        return self

    def click(self, uv, press: float = 0.12, gap: float = 0.8):
        uv = np.asarray(uv, float)
        self.events += [(self._t, "move", uv), (self._t, "on", None),
                        (self._t + press, "off", None)]
        self._t += press + gap
        return self

    def double_click(self, uv, press: float = 0.12, between: float = 0.25,
                     gap: float = 0.8):
        uv = np.asarray(uv, float)
        t = self._t
        self.events += [(t, "move", uv), (t, "on", None), (t + press, "off", None),
                        (t + press + between, "on", None),
                        (t + 2 * press + between, "off", None)]
        self._t = t + 2 * press + between + gap
        return self

    def drag(self, uv_from, uv_to, hold: float = 0.6, dur: float = 1.2,
             gap: float = 0.8, steps: int = 40):
        a, b = np.asarray(uv_from, float), np.asarray(uv_to, float)
        t = self._t
        self.events += [(t, "move", a), (t, "on", None)]
        for i in range(steps + 1):                      # stationary hold, then move
            tt = t + hold + dur * i / steps
            self.events.append((tt, "move", a + (b - a) * i / steps))
        self.events.append((t + hold + dur, "off", None))
        self._t = t + hold + dur + gap
        return self

    def path(self, uvs, on: bool, dur: float, gap: float = 0.5):
        """Free-form polyline (e.g. custom gesture)."""
        uvs = [np.asarray(p, float) for p in uvs]
        t = self._t
        if on:
            self.events += [(t, "move", uvs[0]), (t, "on", None)]
        n = len(uvs)
        for i, p in enumerate(uvs):
            self.events.append((t + dur * i / max(n - 1, 1), "move", p))
        if on:
            self.events.append((t + dur, "off", None))
        self._t = t + dur + gap
        return self

    # ------------- playback -------------
    def pump(self, t: float):
        while self.events and self.events[0][0] <= t:
            _, kind, uv = self.events.pop(0)
            if kind == "on":
                self.pen.on = True
            elif kind == "off":
                self.pen.on = False
            elif kind == "move":
                self.pen.uv = uv

    @property
    def done(self) -> bool:
        return not self.events

    @property
    def end_time(self) -> float:
        return self._t
