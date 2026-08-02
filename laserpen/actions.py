"""Action mapping and execution (last stage of the pipeline).

Maps GestureEvents to system controls, per the paper's default mapping:
  single laser click -> mouse left click
  double laser click -> mouse double click
  triple laser click -> scroll-wheel down (imparting scroll to a laser pen)
  hold + drag        -> mouse drag (press, move, release)

Execution targets are pluggable "sinks". Two are provided:
  * VirtualDesktop  -- an in-memory desktop that logs synthesized OS
    events (stand-in for the real OS-injection code),
  * VNCMirrorSink   -- simulates the capture-card + VNC mirroring path
    (Sec 4.3): display uv is rescaled to the tethered laptop's
    resolution before injection.
Custom drag gestures can be matched against recorded templates
(paper Sec 4.2: "the drag gesture allows custom actions").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .state_machine import GestureEvent


@dataclass
class SysEvent:
    t: float
    kind: str           # mouse_click / mouse_double / mouse_press / mouse_move
    x: float            # / mouse_release / scroll_down / custom:<name>
    y: float
    pen_id: int


class VirtualDesktop:
    """Stand-in for the OS event-injection layer; records everything."""

    def __init__(self, w: int, h: int, name: str = "display"):
        self.w, self.h, self.name = w, h, name
        self.log: List[SysEvent] = []
        self.cursor = np.array([w / 2, h / 2], float)

    def inject(self, ev: SysEvent) -> None:
        self.cursor = np.array([ev.x, ev.y])
        self.log.append(ev)


class VNCMirrorSink(VirtualDesktop):
    """Mirrored laptop: rescale display uv -> laptop resolution (Sec 4.3)."""

    def __init__(self, display_w: int, display_h: int,
                 laptop_w: int = 1920, laptop_h: int = 1080):
        super().__init__(laptop_w, laptop_h, name="laptop-via-vnc")
        self.sx = laptop_w / display_w
        self.sy = laptop_h / display_h

    def inject(self, ev: SysEvent) -> None:
        super().inject(SysEvent(ev.t, ev.kind, ev.x * self.sx, ev.y * self.sy,
                                ev.pen_id))


def _resample(path: List[np.ndarray], n: int = 32) -> np.ndarray:
    p = np.asarray(path, float)
    if len(p) < 2:
        return np.repeat(p, n, axis=0)[:n]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    out = np.stack([np.interp(t, s, p[:, k]) for k in range(2)], axis=1)
    out -= out.mean(0)
    scale = np.abs(out).max() or 1.0
    return out / scale


class GestureTemplateLibrary:
    """$1-style template matcher for custom drag gestures (future-work
    section of the paper, included here as a simple baseline)."""

    def __init__(self):
        self.templates: Dict[str, np.ndarray] = {}

    def add(self, name: str, path: List[np.ndarray]) -> None:
        self.templates[name] = _resample(path)

    def match(self, path: List[np.ndarray], max_dist: float = 0.35
              ) -> Optional[str]:
        if not self.templates:
            return None
        q = _resample(path)
        best, bd = None, np.inf
        for name, t in self.templates.items():
            d = float(np.mean(np.linalg.norm(q - t, axis=1)))
            if d < bd:
                best, bd = name, d
        return best if bd <= max_dist else None


class ActionMapper:
    def __init__(self, sink: VirtualDesktop,
                 templates: Optional[GestureTemplateLibrary] = None):
        self.sink = sink
        self.templates = templates
        self.custom_hits: List[str] = []
        self.bindings: Dict[str, str] = {
            "single_click": "mouse_click",
            "double_click": "mouse_double",
            "triple_click": "scroll_down",
            "drag_start": "mouse_press",
            "drag_move": "mouse_move",
            "drag_end": "mouse_release",
        }

    def rebind(self, gesture: str, sys_kind: str) -> None:
        """User-configurable mapping (paper: 'actions triggered by each
        gesture can be modified based on user requirements')."""
        self.bindings[gesture] = sys_kind

    def handle(self, ev: GestureEvent) -> None:
        kind = self.bindings.get(ev.kind)
        if kind is None or ev.uv is None:
            return
        self.sink.inject(SysEvent(ev.t, kind, float(ev.uv[0]), float(ev.uv[1]),
                                  ev.pen_id))
        if ev.kind == "drag_end" and self.templates and ev.path:
            name = self.templates.match(ev.path)
            if name:
                self.custom_hits.append(name)
                self.sink.inject(SysEvent(ev.t, f"custom:{name}",
                                          float(ev.uv[0]), float(ev.uv[1]),
                                          ev.pen_id))
