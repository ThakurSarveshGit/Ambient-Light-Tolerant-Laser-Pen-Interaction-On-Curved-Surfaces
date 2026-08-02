"""Gesture state machine (paper Fig. 4b) -- one instance per pen.

States: OFF, MAYBE_CLICK (counting clicks), DRAG.
Timing (t < x < T from the paper):
  * ON for < t then OFF                    -> click candidate
  * second ON before T elapses             -> increments click count
  * OFF gap >= T                           -> finalize SINGLE/DOUBLE/TRIPLE
  * ON and roughly stationary for >= x     -> DRAG until OFF
Emits GestureEvents consumed by the ActionMapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .config import InteractionConfig
from .tracker import TrackSample


@dataclass
class GestureEvent:
    t: float
    pen_id: int
    kind: str                    # "single_click" | "double_click" | "triple_click"
    uv: np.ndarray               # | "drag_start" | "drag_move" | "drag_end"
    path: Optional[List[np.ndarray]] = None


class PenStateMachine:
    OFF, ON_PENDING, COUNTING, DRAG = "OFF", "ON_PENDING", "COUNTING", "DRAG"

    def __init__(self, pen_id: int, cfg: InteractionConfig,
                 emit: Callable[[GestureEvent], None]):
        self.pen_id, self.cfg, self.emit = pen_id, cfg, emit
        self.state = self.OFF
        self.on_since: Optional[float] = None
        self.off_since: Optional[float] = None
        self.anchor: Optional[np.ndarray] = None
        self.last_uv: Optional[np.ndarray] = None
        self.clicks = 0
        self.drag_path: List[np.ndarray] = []

    # ------------------------------------------------------------------
    def feed(self, s: TrackSample) -> None:
        if s.on and s.uv is not None:
            self._on(s.t, s.uv)
        else:
            self._off(s.t)

    # ------------------------------------------------------------------
    def _on(self, t: float, uv: np.ndarray) -> None:
        self.last_uv = uv
        if self.state == self.OFF:
            self.state, self.on_since, self.anchor = self.ON_PENDING, t, uv
        elif self.state == self.ON_PENDING:
            moved = np.linalg.norm(uv - self.anchor) > self.cfg.drag_vicinity_px
            held = t - self.on_since >= self.cfg.x_drag_hold
            if held and not moved:
                self.state = self.DRAG
                self.drag_path = [self.anchor]
                self.emit(GestureEvent(t, self.pen_id, "drag_start", self.anchor))
            elif moved and t - self.on_since >= self.cfg.t_click_max:
                # moved too far, too long for a click -> treat as drag as well
                self.state = self.DRAG
                self.drag_path = [self.anchor, uv]
                self.emit(GestureEvent(t, self.pen_id, "drag_start", self.anchor))
        elif self.state == self.COUNTING:
            self.state, self.on_since, self.anchor = self.ON_PENDING, t, uv
        elif self.state == self.DRAG:
            self.drag_path.append(uv)
            self.emit(GestureEvent(t, self.pen_id, "drag_move", uv))

    def _off(self, t: float) -> None:
        if self.state == self.ON_PENDING:
            if self.on_since is not None and t - self.on_since <= self.cfg.t_click_max + 0.15:
                self.clicks += 1
                self.state, self.off_since = self.COUNTING, t
            else:
                self.state, self.clicks = self.OFF, 0
        elif self.state == self.COUNTING:
            if self.off_since is not None and t - self.off_since >= self.cfg.T_action_trigger:
                kind = {1: "single_click", 2: "double_click"}.get(
                    self.clicks, "triple_click")
                self.emit(GestureEvent(t, self.pen_id, kind, self.last_uv))
                self.state, self.clicks = self.OFF, 0
        elif self.state == self.DRAG:
            self.emit(GestureEvent(t, self.pen_id, "drag_end", self.last_uv,
                                   path=list(self.drag_path)))
            self.state, self.drag_path, self.clicks = self.OFF, [], 0
