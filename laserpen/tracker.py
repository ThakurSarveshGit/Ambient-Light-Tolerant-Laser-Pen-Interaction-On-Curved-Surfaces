"""Multi-threaded laser tracking with camera hand-off (Sec 2.2).

Architecture (mirrors the C++ multi-thread camera handling):

  * one CameraWorker thread per camera. Each tick it "captures" a frame
    (via the simulated radiometric model), black-subtracts the stored
    ambient frame, thresholds for the laser spot(s), classifies spot
    color, converts camera px -> display uv through the registration
    inverse map, and posts detections + a heartbeat.
  * one TrackingCoordinator that, per pen, polls ONLY the cameras the
    partition says are relevant for the pen's last known position
    (owner in the interior, owner+neighbor in the boundary band), fuses
    detections, and forwards (t, pen, uv, on) samples downstream.
  * a watchdog inside the coordinator detects dead cameras (missed
    heartbeats) and triggers a repartition so surviving cameras absorb
    the lost region -- interaction continues uninterrupted.

The simulation is time-stepped: SimClock ticks at the camera FPS, worker
threads synchronize on a barrier per tick so runs are reproducible while
still exercising real threads, queues and locks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional

import numpy as np

from .camera import CameraFailure
from .config import InteractionConfig
from .laser import LaserPen
from .partition import DisplayPartition
from .registration import Registration


def subpixel_peak(diff: np.ndarray, peak_xy: np.ndarray, win: int = 4) -> np.ndarray:
    """Refine an integer peak to sub-pixel via intensity-weighted centroid."""
    h, w = diff.shape
    x0, y0 = int(peak_xy[0]), int(peak_xy[1])
    xs = slice(max(0, x0 - win), min(w, x0 + win + 1))
    ys = slice(max(0, y0 - win), min(h, y0 + win + 1))
    patch = np.clip(diff[ys, xs], 0, None)
    if patch.sum() <= 0:
        return peak_xy.astype(float)
    yy, xx = np.mgrid[ys, xs]
    return np.array([(patch * xx).sum() / patch.sum(),
                     (patch * yy).sum() / patch.sum()])


@dataclass
class Detection:
    t: float
    cam_id: int
    pen_color: str
    cam_px: np.ndarray
    uv: np.ndarray
    strength: float


@dataclass
class TrackSample:
    t: float
    pen_id: int
    on: bool
    uv: Optional[np.ndarray]


class SceneState:
    """Shared world state each camera observes at a tick (thread-safe)."""

    def __init__(self, ambient_radiance: float, content_radiance: float = 15.0):
        self.lock = threading.Lock()
        self.ambient = ambient_radiance
        self.content = content_radiance  # avg app-content radiance behind laser
        self.pens: Dict[int, LaserPen] = {}
        self.t = 0.0

    def snapshot(self):
        with self.lock:
            pens = [(p.pen_id, p.color, p.radiance,
                     None if not p.on or p.uv is None else p.uv.copy())
                    for p in self.pens.values()]
            return self.t, self.ambient, self.content, pens


class CameraWorker(threading.Thread):
    """One thread per camera: capture -> subtract black -> detect -> map."""

    def __init__(self, cam_id: int, reg: Registration, scene: SceneState,
                 black_frame: np.ndarray, icfg: InteractionConfig,
                 out_q: "Queue[Detection]", tick_barrier: threading.Barrier,
                 stop: threading.Event, heartbeats: Dict[int, float],
                 hb_lock: threading.Lock, rng: np.random.Generator):
        super().__init__(daemon=True, name=f"cam-{cam_id}")
        self.cam_id, self.reg, self.scene = cam_id, reg, scene
        self.black = black_frame
        self.icfg = icfg
        self.out_q, self.barrier, self.stop = out_q, tick_barrier, stop
        self.heartbeats, self.hb_lock = heartbeats, hb_lock
        self.rng = rng
        self._bg_cache: Optional[np.ndarray] = None

    def run(self):
        cam = self.reg.cameras[self.cam_id]
        while not self.stop.is_set():
            try:
                self.barrier.wait(timeout=2.0)
            except threading.BrokenBarrierError:
                return
            t, ambient, content, pens = self.scene.snapshot()
            try:
                if self._bg_cache is None:
                    self._bg_cache = self.reg.render_for_camera(self.cam_id, content)
                img = self._capture(cam, ambient, pens)
            except CameraFailure:
                continue                       # dead: no heartbeat, no detections
            with self.hb_lock:
                self.heartbeats[self.cam_id] = t
            self._detect(t, img, pens)
            try:
                self.barrier.wait(timeout=2.0)  # end-of-tick sync
            except threading.BrokenBarrierError:
                return

    def _capture(self, cam, ambient, pens):
        img = None
        base = self._bg_cache
        for pid, color, radiance, uv in pens:
            if uv is None:
                continue
            px = self.reg.laser_pixel_in_camera(self.cam_id, uv)
            if px is None:
                continue
            # grazing-angle foreshortening: spots seen at shallow angles are
            # dimmer/smeared (the paper's edge-accuracy drop, Sec 4.2)
            n = self.reg.surface.normal_at(np.asarray(uv, float))
            w = self.reg.surface.display_to_world(np.asarray(uv, float))
            view = cam.pose.eye - w
            view = view / np.linalg.norm(view)
            cosang = abs(float(np.dot(n, view)))
            eff = radiance * (0.05 + 0.95 * cosang ** 2)
            img = cam.capture(base, ambient, laser_px=px, laser_radiance=eff,
                              laser_sigma_px=2.5 / max(cosang, 0.3))
            # NOTE: single capture per tick; multiple pens are composited by
            # adding further gaussians onto the same frame:
            base = img / (cam.gain * cam.exposure_ms) - ambient
        if img is None:
            img = cam.capture(base, ambient)
        return img

    def _detect(self, t, img, pens):
        diff = img - self.black
        thr = self.icfg.laser_detect_threshold
        if diff.max() < thr:
            return
        # connected bright peaks -> one detection per pen (color known from
        # per-pen band in the real system; here we attribute by proximity
        # to the pens' true colors via brightest-peak matching).
        from scipy.ndimage import center_of_mass, label
        mask = diff > thr
        lab, n = label(mask)
        # blob centroid is robust to the saturated plateau at the spot core
        peaks = [np.array(center_of_mass(mask, lab, i + 1)[::-1], float)
                 for i in range(n)]
        active = [(pid, color, uv) for pid, color, uv in
                  [(p[0], p[1], p[3]) for p in pens] if uv is not None]
        for px in peaks:
            uv = self.reg.cam_px_to_display(self.cam_id, px)
            if uv is None:
                continue
            # color attribution: nearest active pen in display space
            if active:
                dists = [np.linalg.norm(uv - a[2]) for a in active]
                pid, color, _ = active[int(np.argmin(dists))]
            else:
                pid, color = -1, "unknown"
            self.out_q.put(Detection(t, self.cam_id, color, px, uv,
                                     float(diff[int(px[1]), int(px[0])])))


class TrackingCoordinator:
    """Fuses detections; polls cameras per partition; handles failures."""

    def __init__(self, reg: Registration, partition: DisplayPartition,
                 icfg: InteractionConfig,
                 on_sample: Callable[[TrackSample], None]):
        self.reg, self.partition, self.icfg = reg, partition, icfg
        self.on_sample = on_sample
        self.last_uv: Dict[int, np.ndarray] = {}
        self.heartbeats: Dict[int, float] = {}
        self.polls: Dict[int, int] = {c.cam_id: 0 for c in reg.cameras}
        self.poll_history: List[tuple] = []   # (t, pen_id, [cam_ids], owner)
        self.handoffs: List[tuple] = []
        self._last_owner: Dict[int, int] = {}

    def process_tick(self, t: float, detections: List[Detection],
                     pens: List[LaserPen]) -> None:
        # ---- failure watchdog -> repartition
        active = {c.cam_id for c in self.reg.cameras if c.alive}
        if active != self.partition.active and active:
            self.partition.rebuild(active)

        by_pen: Dict[int, List[Detection]] = {}
        for d in detections:
            pid = next((p.pen_id for p in pens if p.color == d.pen_color), -1)
            by_pen.setdefault(pid, []).append(d)

        for pen in pens:
            poll = self.partition.cameras_to_poll(self.last_uv.get(pen.pen_id))
            last = self.last_uv.get(pen.pen_id)
            own = self.partition.owner_of(last) if last is not None else -1
            self.poll_history.append((t, pen.pen_id, list(poll), own))
            for c in poll:
                self.polls[c] += 1
            dets = [d for d in by_pen.get(pen.pen_id, []) if d.cam_id in poll]
            if dets:
                best = max(dets, key=lambda d: d.strength)
                uv = best.uv
                self.last_uv[pen.pen_id] = uv
                own = self.partition.owner_of(uv)
                prev = self._last_owner.get(pen.pen_id)
                if prev is not None and prev != own:
                    self.handoffs.append((t, pen.pen_id, prev, own))
                self._last_owner[pen.pen_id] = own
                self.on_sample(TrackSample(t, pen.pen_id, True, uv))
            else:
                self.on_sample(TrackSample(t, pen.pen_id, False,
                                           self.last_uv.get(pen.pen_id)))
