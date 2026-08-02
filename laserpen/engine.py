"""Simulation engine: wires the whole pipeline together.

  scripted users -> laser pens -> SceneState
  CameraWorker threads (capture, black-subtract, detect, invert-map)
  TrackingCoordinator (partition-aware polling, hand-off, failure watchdog)
  PenStateMachine (gesture recognition)
  ActionMapper -> VirtualDesktop / VNCMirrorSink

Time is simulated at the camera frame rate; camera workers are real
threads synchronized per tick (deterministic yet genuinely concurrent).
"""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Dict, List, Optional

import numpy as np

from .actions import ActionMapper, GestureTemplateLibrary, VirtualDesktop
from .ambient import AmbientCalibration, AmbientLightManager
from .config import SystemConfig
from .laser import LaserPen, ScriptedUser
from .partition import DisplayPartition
from .registration import Registration, build_camera_rig
from .state_machine import GestureEvent, PenStateMachine
from .surface import PringleSurface
from .tracker import (CameraWorker, Detection, SceneState, TrackingCoordinator,
                      TrackSample)


class Simulation:
    def __init__(self, cfg: SystemConfig, n_cameras: int = 4,
                 arrangement: str = "horizontal",
                 ambient_radiance: float = 30.0,
                 sink: Optional[VirtualDesktop] = None,
                 templates: Optional[GestureTemplateLibrary] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.rng_seed)
        self.surface = PringleSurface(cfg.screen)
        self.cameras = build_camera_rig(cfg, self.surface, n_cameras,
                                        arrangement, self.rng)
        self.reg = Registration(cfg, self.surface, self.cameras)
        self.partition = DisplayPartition(self.reg, cfg.interaction.boundary_band_px)

        # ---- ambient calibration (Sec 2.1) sets exposures + black frames
        self.ambient_mgr = AmbientLightManager(self.reg, cfg.exposure)
        self.cal: AmbientCalibration = self.ambient_mgr.calibrate(ambient_radiance)
        for cam in self.cameras:
            cam.exposure_ms = self.cal.Er_ms

        self.scene = SceneState(ambient_radiance)
        self.sink = sink or VirtualDesktop(cfg.screen.display_w, cfg.screen.display_h)
        self.mapper = ActionMapper(self.sink, templates)
        self.gestures: List[GestureEvent] = []
        self.samples: List[TrackSample] = []
        self.state_history: List[tuple] = []   # (t, pen_id, sm_state)
        self.machines: Dict[int, PenStateMachine] = {}

        def emit(ev: GestureEvent):
            self.gestures.append(ev)
            self.mapper.handle(ev)

        self._emit = emit
        self.coord = TrackingCoordinator(self.reg, self.partition,
                                         cfg.interaction, self._on_sample)
        self.users: List[ScriptedUser] = []
        self.fail_schedule: List[tuple] = []      # (t, cam_id, alive_flag)

    # ------------------------------------------------------------------
    def add_pen(self, color: str, pen_id: Optional[int] = None) -> LaserPen:
        pid = pen_id if pen_id is not None else len(self.scene.pens)
        pen = LaserPen(pid, color)
        self.scene.pens[pid] = pen
        self.machines[pid] = PenStateMachine(pid, self.cfg.interaction, self._emit)
        return pen

    def add_user(self, user: ScriptedUser) -> ScriptedUser:
        self.users.append(user)
        return user

    def schedule_camera_failure(self, t: float, cam_id: int, alive: bool = False):
        self.fail_schedule.append((t, cam_id, alive))

    def _on_sample(self, s: TrackSample) -> None:
        self.samples.append(s)
        self.machines[s.pen_id].feed(s)

    # ------------------------------------------------------------------
    def run(self, duration: Optional[float] = None) -> None:
        dur = duration or (max((u.end_time for u in self.users), default=1.0) + 1.5)
        dt = 1.0 / self.cfg.camera.fps
        n_ticks = int(np.ceil(dur / dt))

        det_q: "Queue[Detection]" = Queue()
        stop = threading.Event()
        barrier = threading.Barrier(len(self.cameras) + 1)
        hb_lock = threading.Lock()
        workers = [CameraWorker(c.cam_id, self.reg, self.scene,
                                self.cal.black_frames[c.cam_id],
                                self.cfg.interaction, det_q, barrier, stop,
                                self.coord.heartbeats, hb_lock, c.rng)
                   for c in self.cameras]
        for w in workers:
            w.start()

        try:
            for k in range(n_ticks):
                t = k * dt
                for f_t, cid, alive in self.fail_schedule:
                    if abs(f_t - t) < dt / 2:
                        self.cameras[cid].alive = alive
                with self.scene.lock:
                    self.scene.t = t
                    for u in self.users:
                        u.pump(t)
                barrier.wait(timeout=2.0)   # workers capture this tick
                barrier.wait(timeout=2.0)   # workers done posting detections
                dets: List[Detection] = []
                while True:
                    try:
                        dets.append(det_q.get_nowait())
                    except Empty:
                        break
                self.coord.process_tick(t, dets, list(self.scene.pens.values()))
                for pid, m in self.machines.items():
                    self.state_history.append((t, pid, m.state))
        finally:
            stop.set()
            try:
                barrier.abort()
            except Exception:
                pass
            for w in workers:
                w.join(timeout=1.0)
