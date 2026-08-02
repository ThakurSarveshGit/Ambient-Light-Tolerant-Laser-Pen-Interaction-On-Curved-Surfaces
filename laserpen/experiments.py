"""Experiments reproducing the paper's evaluation protocol.

  exp_exposure_table   -> Table 1 analogue: Er / Ei vs ambient light
  exp_detection_grid   -> Sec 4.2: 8x8 grid laser detection accuracy
  exp_gesture_accuracy -> Sec 4.2: single/double click recognition
  exp_location_error   -> Sec 4.2: detected-vs-true location error (px)
  exp_camera_failure   -> Sec 2.2: kill a camera mid-run, verify recovery
  exp_multi_user       -> Sec 4.3: two colored pens, balloon-pop scoring
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .ambient import AmbientLightManager, make_pattern, pattern_brightness
from .config import SystemConfig
from .engine import Simulation
from .laser import ScriptedUser
from .registration import Registration, build_camera_rig
from .surface import PringleSurface


# ---------------------------------------------------------------- Table 1
def exp_exposure_table(cfg: SystemConfig,
                       ambients=(0, 15, 30, 50, 65, 90, 115)) -> List[dict]:
    """Sweep ambient radiance; report Er and the predicted Ei for a
    calibration pattern (blobs, sparser/brighter than the reference)."""
    surface = PringleSurface(cfg.screen)
    rng = np.random.default_rng(cfg.rng_seed)
    cams = build_camera_rig(cfg, surface, 3, "vertical", rng)  # Fig. 5 rig
    reg = Registration(cfg, surface, cams)
    mgr = AmbientLightManager(reg, cfg.exposure)
    blob = make_pattern("blobs", black_fraction=0.85)
    Bs = pattern_brightness(blob)
    rows = []
    for a in ambients:
        cal = mgr.calibrate(float(a))
        rows.append({"ambient_radiance": a,
                     "captured_black_gray": round(cal.A, 1),
                     "Er_ms": cal.Er_ms,
                     "Ei_ms": cal.predict_exposure(Bs, cfg.exposure)})
    return rows


# ------------------------------------------------- Sec 4.2 detection grid
def exp_detection_grid(cfg: SystemConfig, clicks_per_cell: int = 20,
                       ambient: float = 30.0) -> np.ndarray:
    """8x8 grid; a scripted user clicks in each cell; report % detected."""
    grid = np.zeros((8, 8))
    dw, dh = cfg.screen.display_w, cfg.screen.display_h
    sim = Simulation(cfg, n_cameras=4, ambient_radiance=ambient)
    pen = sim.add_pen("green")
    user = ScriptedUser(pen, seed=3)
    cells = []          # (start_t, end_t, r, c) scheduled click windows
    for r in range(8):
        for c in range(8):
            u = (c + 0.5) / 8 * (dw - 1)
            v = (r + 0.5) / 8 * (dh - 1)
            for _ in range(clicks_per_cell):
                t0 = user.end_time
                user.click([u + user.rng.uniform(-60, 60),
                            v + user.rng.uniform(-60, 60)],
                           press=0.12, gap=0.85)
                cells.append((t0, user.end_time + 0.3, r, c))
    sim.add_user(user)
    sim.run()
    clicks = [g for g in sim.gestures if g.kind == "single_click"]
    # attribute recognized clicks back to cells in script order
    hit = np.zeros((8, 8))
    tot = np.zeros((8, 8))
    times = sorted(g.t for g in clicks)
    for (w0, w1, r, c) in cells:
        tot[r, c] += 1
        if any(w0 <= tt <= w1 + 0.7 for tt in times):
            hit[r, c] += 1
    with np.errstate(invalid="ignore"):
        grid = 100.0 * hit / np.maximum(tot, 1)
    return grid


# --------------------------------------------- Sec 4.2 gesture accuracy
def exp_gesture_accuracy(cfg: SystemConfig, n_each: int = 60,
                         ambient: float = 30.0) -> Dict[str, float]:
    sim = Simulation(cfg, n_cameras=4, ambient_radiance=ambient)
    pen = sim.add_pen("green")
    user = ScriptedUser(pen, seed=11)
    dw, dh = cfg.screen.display_w, cfg.screen.display_h
    truth: List[str] = []
    for i in range(n_each):
        uv = [user.rng.uniform(0.08, 0.92) * dw, user.rng.uniform(0.08, 0.92) * dh]
        user.click(uv); truth.append("single_click")
        uv = [user.rng.uniform(0.08, 0.92) * dw, user.rng.uniform(0.08, 0.92) * dh]
        user.double_click(uv); truth.append("double_click")
    sim.add_user(user)
    sim.run()
    recognized = [g.kind for g in sim.gestures
                  if g.kind in ("single_click", "double_click", "triple_click")]
    n = min(len(truth), len(recognized))
    acc = {"single_click": [0, 0], "double_click": [0, 0]}
    for want, got in zip(truth[:n], recognized[:n]):
        acc[want][1] += 1
        acc[want][0] += int(want == got)
    # unrecognized gestures count as misses
    for want in truth[n:]:
        acc[want][1] += 1
    return {k: 100.0 * a / max(b, 1) for k, (a, b) in acc.items()}


# --------------------------------------------- Sec 4.2 location accuracy
def exp_location_error(cfg: SystemConfig, n: int = 200,
                       ambient: float = 30.0) -> Tuple[float, float, float]:
    """Distance (display px, Manhattan) between true laser uv and tracked uv.

    Uses zero hand-jitter so the number isolates detection + mapping error
    (the paper's cursor-vs-spot measurement).
    """
    sim = Simulation(cfg, n_cameras=4, ambient_radiance=ambient)
    pen = sim.add_pen("green")
    pen.jitter_px = 0.0
    reg, part = sim.reg, sim.partition
    rng = np.random.default_rng(5)
    errs = []
    from .tracker import SceneState
    for _ in range(n):
        uv = np.array([rng.uniform(0.05, 0.95) * (cfg.screen.display_w - 1),
                       rng.uniform(0.05, 0.95) * (cfg.screen.display_h - 1)])
        cam_id = part.owner_of(uv)
        cam = reg.cameras[cam_id]
        px = reg.laser_pixel_in_camera(cam_id, uv)
        if px is None:
            continue
        content = reg.render_for_camera(cam_id, sim.scene.content)
        img = cam.capture(content, ambient, laser_px=px, laser_radiance=pen.radiance)
        diff = img - sim.cal.black_frames[cam_id]
        from scipy.ndimage import center_of_mass
        mask = diff > sim.cfg.interaction.laser_detect_threshold
        if not mask.any():
            continue
        peak = np.array(center_of_mass(mask)[::-1], float)
        got = reg.cam_px_to_display(cam_id, peak)
        if got is not None:
            errs.append(np.abs(got - uv).sum())      # Manhattan, display px
    e = np.array(errs)
    return float(e.min()), float(e.mean()), float(e.max())


# ------------------------------------------------- Sec 2.2 failure test
def exp_camera_failure(cfg: SystemConfig, ambient: float = 30.0) -> dict:
    sim = Simulation(cfg, n_cameras=4, ambient_radiance=ambient)
    pen = sim.add_pen("green")
    user = ScriptedUser(pen, seed=21)
    dw, dh = cfg.screen.display_w, cfg.screen.display_h
    # sweep left->right across all partitions, twice
    for k in range(10):
        user.click([(0.06 + 0.088 * k) * dw, 0.5 * dh], gap=0.9)
    for k in range(10):
        user.click([(0.06 + 0.088 * k) * dw, 0.35 * dh], gap=0.9)
    sim.add_user(user)
    sim.schedule_camera_failure(t=6.0, cam_id=1)     # kill cam 1 mid-run
    cov_before = sim.partition.coverage_fraction()
    sim.run()
    cov_after = sim.partition.coverage_fraction()
    clicks = sum(g.kind == "single_click" for g in sim.gestures)
    return {"clicks_expected": 20, "clicks_recognized": clicks,
            "coverage_before": cov_before, "coverage_after": cov_after,
            "active_after": sorted(sim.partition.active),
            "handoffs": len(sim.coord.handoffs)}


# ------------------------------------------------- Sec 4.3 multi-user
def exp_multi_user(cfg: SystemConfig, ambient: float = 30.0) -> dict:
    """Balloon-pop: blue and green pens click balloons; per-color scoring."""
    sim = Simulation(cfg, n_cameras=4, ambient_radiance=ambient)
    blue = sim.add_pen("blue")
    green = sim.add_pen("green")
    ub, ug = ScriptedUser(blue, seed=31), ScriptedUser(green, seed=32)
    dw, dh = cfg.screen.display_w, cfg.screen.display_h
    rngb = np.random.default_rng(41)
    balloons = [np.array([rngb.uniform(0.1, 0.9) * dw, rngb.uniform(0.1, 0.9) * dh])
                for _ in range(12)]
    for i, b in enumerate(balloons):
        (ub if i % 2 == 0 else ug).click(b, gap=0.8)
        (ug if i % 2 == 0 else ub).wait(1.0)
    sim.add_user(ub); sim.add_user(ug)
    sim.run()
    score = {"blue": 0, "green": 0}
    popped = [False] * len(balloons)
    for g in sim.gestures:
        if g.kind != "single_click":
            continue
        color = sim.scene.pens[g.pen_id].color
        for i, b in enumerate(balloons):
            if not popped[i] and np.linalg.norm(g.uv - b) < 120:
                popped[i] = True
                score[color] += 1
                break
    return {"score": score, "balloons": len(balloons),
            "popped": sum(popped)}
