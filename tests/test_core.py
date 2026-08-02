"""Unit tests: run with  python -m pytest tests/ -q  (or python tests/test_core.py)."""

import numpy as np

try:
    import pytest
except ImportError:      # pytest optional; file is runnable standalone
    pytest = None

from laserpen import DEFAULT, PringleSurface, ScriptedUser, Simulation
from laserpen.ambient import make_pattern, pattern_brightness
from laserpen.config import SystemConfig
from laserpen.laser import LaserPen
from laserpen.registration import Registration, build_camera_rig
from laserpen.state_machine import PenStateMachine
from laserpen.tracker import TrackSample


def test_surface_roundtrip():
    s = PringleSurface(DEFAULT.screen)
    uv = np.array([[100.0, 200.0], [3000.0, 2000.0], [1920.0, 1080.0]])
    back = s.world_to_display(s.display_to_world(uv))
    assert np.allclose(uv, back, atol=1e-6)


def test_surface_is_pringle():
    s = PringleSurface(DEFAULT.screen)
    mid = s.display_to_world(np.array([DEFAULT.screen.display_w / 2,
                                       DEFAULT.screen.display_h / 2]))
    right = s.display_to_world(np.array([DEFAULT.screen.display_w - 1.0,
                                         DEFAULT.screen.display_h / 2]))
    top = s.display_to_world(np.array([DEFAULT.screen.display_w / 2, 0.0]))
    assert right[2] > mid[2]          # curves toward viewer along width
    assert top[2] < mid[2]            # curves away along height


def test_registration_inverse_map_subpixel():
    rng = np.random.default_rng(0)
    surf = PringleSurface(DEFAULT.screen)
    cams = build_camera_rig(DEFAULT, surf, 4, "horizontal", rng)
    reg = Registration(DEFAULT, surf, cams)
    for _ in range(20):
        uv = np.array([rng.uniform(0.1, 0.9) * DEFAULT.screen.display_w,
                       rng.uniform(0.1, 0.9) * DEFAULT.screen.display_h])
        for cid in range(4):
            px = reg.laser_pixel_in_camera(cid, uv)
            if px is None:
                continue
            got = reg.cam_px_to_display(cid, px)
            assert got is not None
            assert np.abs(got - uv).max() < 2.0    # display px


def test_exposure_prediction_monotonic():
    # Eq. 1: brighter patterns need shorter exposure
    from laserpen.ambient import AmbientCalibration
    cal = AmbientCalibration(A=50.0, Er_ms=160.0, Br=100.0,
                             ambient_radiance_est=30.0)
    e_dim = cal.predict_exposure(20.0, DEFAULT.exposure)
    e_bright = cal.predict_exposure(200.0, DEFAULT.exposure)
    assert e_dim >= e_bright


def test_state_machine_single_double_drag():
    events = []
    sm = PenStateMachine(0, DEFAULT.interaction, events.append)
    uv = np.array([500.0, 500.0])

    def feed(t, on, p=uv):
        sm.feed(TrackSample(t, 0, on, p))

    # single click: ON 0.10s, OFF long
    feed(0.00, True); feed(0.10, False)
    for t in np.arange(0.12, 1.2, 0.02):
        feed(t, False)
    assert [e.kind for e in events] == ["single_click"]

    # double click
    events.clear()
    feed(2.0, True); feed(2.1, False); feed(2.3, True); feed(2.4, False)
    for t in np.arange(2.42, 3.6, 0.02):
        feed(t, False)
    assert [e.kind for e in events] == ["double_click"]

    # drag: hold stationary past x, then move
    events.clear()
    for t in np.arange(4.0, 4.5, 0.02):
        feed(t, True)
    for i, t in enumerate(np.arange(4.5, 5.0, 0.02)):
        feed(t, True, uv + np.array([i * 30.0, 0.0]))
    feed(5.0, False)
    kinds = [e.kind for e in events]
    assert kinds[0] == "drag_start" and kinds[-1] == "drag_end"
    assert "drag_move" in kinds


def test_partition_full_coverage_and_failure_recovery():
    sim = Simulation(DEFAULT, n_cameras=4, ambient_radiance=20.0)
    assert sim.partition.coverage_fraction() == 1.0
    sim.partition.rebuild(active={0, 2, 3})
    assert sim.partition.coverage_fraction() == 1.0     # survivors absorb region
    assert set(np.unique(sim.partition.owner)) <= {0, 2, 3}


def test_end_to_end_click():
    sim = Simulation(DEFAULT, n_cameras=4, ambient_radiance=30.0)
    pen = sim.add_pen("green")
    u = ScriptedUser(pen, seed=2)
    u.click([1200, 900])
    sim.add_user(u)
    sim.run()
    clicks = [g for g in sim.gestures if g.kind == "single_click"]
    assert len(clicks) == 1
    assert np.abs(clicks[0].uv - np.array([1200, 900])).max() < 30


if __name__ == "__main__":
    import sys
    if pytest is not None:
        sys.exit(pytest.main([__file__, "-q"]))
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
