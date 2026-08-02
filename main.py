#!/usr/bin/env python3
"""Run the simulated laser-pen interaction system.

  python main.py demo         interactive session demo (+ figures)
  python main.py table1       exposure vs ambient light (Table 1 analogue)
  python main.py detection    8x8 detection accuracy grid + heatmap
  python main.py gestures     single/double click recognition accuracy
  python main.py location     laser localization error (display px)
  python main.py failure      camera-failure recovery test
  python main.py multiuser    two-pen balloon-pop scoring
  python main.py all          everything
"""

import sys

import numpy as np

from laserpen import (DEFAULT, GestureTemplateLibrary, ScriptedUser,
                      Simulation, VNCMirrorSink)
from laserpen import experiments as X
from laserpen import visualize as V

OUT = "figures"


def _build_demo_sim():
    cfg = DEFAULT
    lib = GestureTemplateLibrary()
    # record a "swipe right" template for the custom-gesture path
    lib.add("swipe_right", [np.array([x, 1000.0]) for x in
                            np.linspace(500, 3000, 20)])
    sink = VNCMirrorSink(cfg.screen.display_w, cfg.screen.display_h)
    sim = Simulation(cfg, n_cameras=4, ambient_radiance=30.0,
                     sink=sink, templates=lib)
    print(f"[calib] captured-black gray A={sim.cal.A:.1f}  "
          f"Er={sim.cal.Er_ms:.0f} ms  (ref brightness Br={sim.cal.Br:.1f})")
    print(f"[partition] display coverage {sim.partition.coverage_fraction():.0%}")

    pen = sim.add_pen("green")
    u = ScriptedUser(pen, seed=1)
    u.click([600, 1000])
    u.double_click([2000, 1100])
    u.drag([800, 800], [3200, 1500])                # crosses all partitions
    u.path([[500 + 130 * i, 1000] for i in range(20)], on=True, dur=1.0)  # swipe
    sim.add_user(u)
    sim.run()
    return sim


def demo():
    import os
    os.makedirs(OUT, exist_ok=True)
    sim = _build_demo_sim()
    sink = sim.sink

    print("\n[gestures]")
    for g in sim.gestures:
        if g.kind.endswith("click") or g.kind in ("drag_start", "drag_end"):
            print(f"  t={g.t:6.2f}s  {g.kind:12s} at ({g.uv[0]:.0f},{g.uv[1]:.0f})")
    print("\n[system events on mirrored laptop 1920x1080]")
    interesting = [e for e in sink.log if e.kind not in ("mouse_move",)]
    for e in interesting:
        print(f"  t={e.t:6.2f}s  {e.kind:16s} at ({e.x:.0f},{e.y:.0f})")
    print(f"\n[hand-offs] {len(sim.coord.handoffs)}: "
          + ", ".join(f"cam{a}->cam{b}@{t:.2f}s" for t, _, a, b in sim.coord.handoffs))
    print(f"[custom gestures matched] {sim.mapper.custom_hits}")
    print(f"[camera polls] {sim.coord.polls}  "
          f"(sum {sum(sim.coord.polls.values())}; all-cam polling would be "
          f"{4 * max(sim.coord.polls.values())}-ish)")

    V.fig_rig(sim, f"{OUT}/rig.png")
    V.fig_partitions(sim, f"{OUT}/partitions.png")
    V.fig_trajectory(sim, f"{OUT}/trajectory.png")
    from laserpen.animate import animate_session
    print("\nrendering animated replay (this takes ~a minute)...")
    animate_session(sim, f"{OUT}/session.gif")
    print(f"figures + animation written to {OUT}/ (session.gif)")


def live():
    """Interactive window replay (requires a local display/backend)."""
    from laserpen.animate import animate_session
    sim = _build_demo_sim()
    animate_session(sim, None, live=True)


def table1():
    rows = X.exp_exposure_table(DEFAULT)
    print(f"{'ambient(rad)':>12} {'black gray':>10} {'Er(ms)':>7} {'Ei(ms)':>7}")
    for r in rows:
        print(f"{r['ambient_radiance']:>12} {r['captured_black_gray']:>10}"
              f" {r['Er_ms']:>7.0f} {r['Ei_ms']:>7.0f}")


def detection():
    import os
    os.makedirs(OUT, exist_ok=True)
    grid = X.exp_detection_grid(DEFAULT, clicks_per_cell=8)
    print(np.array2string(grid, precision=0, suppress_small=True))
    edge = np.ones_like(grid, bool); edge[1:-1, 1:-1] = False
    print(f"overall {grid.mean():.2f}%  center {grid[~edge].mean():.2f}%  "
          f"edges {grid[edge].mean():.2f}%")
    V.fig_heatmap(grid, "Laser click detection accuracy (%), 8x8 grid",
                  f"{OUT}/detection_heatmap.png")


def gestures():
    acc = X.exp_gesture_accuracy(DEFAULT, n_each=40)
    for k, v in acc.items():
        print(f"{k:>14}: {v:.2f}%")


def location():
    mn, mean, mx = X.exp_location_error(DEFAULT, n=150)
    print(f"localization error (display px, Manhattan): "
          f"min={mn:.2f} mean={mean:.2f} max={mx:.2f}")
    # express in camera pixels too (display is ~6x camera resolution here)
    scale = DEFAULT.screen.display_w / DEFAULT.camera.image_w
    print(f"equivalent camera-pixel error: mean ~{mean / scale:.2f} px "
          f"(paper reports 0.94 px mean in camera space)")


def failure():
    r = X.exp_camera_failure(DEFAULT)
    print(r)
    ok = r["clicks_recognized"] >= 0.9 * r["clicks_expected"]
    print("PASS: interaction continued through camera failure" if ok
          else "WARN: recognition degraded after failure")


def multiuser():
    print(X.exp_multi_user(DEFAULT))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    fns = {"demo": demo, "table1": table1, "detection": detection,
           "gestures": gestures, "location": location, "failure": failure,
           "multiuser": multiuser, "live": live}
    if cmd == "all":
        for name, fn in fns.items():
            print(f"\n========== {name} ==========")
            fn()
    else:
        fns[cmd]()
