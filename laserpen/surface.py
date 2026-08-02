"""Pringle-shaped display surface and its parametrization.

In the original C++ system the screen shape came from camera-based
structured-light reconstruction (refs [1-5] of the paper). Here, per the
author's note, reconstruction is *assumed available*: we generate the
surface analytically as a hyperbolic paraboloid ("pringle") and expose
the same interface the reconstruction module provided:

    * a dense 3D point cloud of the screen,
    * a parametrization mapping 3D surface points <-> 2D display
      coordinates (u, v) in pixels (the "2D display coordinates" that
      the whole interaction pipeline works in).
"""

from __future__ import annotations

import numpy as np

from .config import ScreenConfig


class PringleSurface:
    """z = cx * x^2 - cy * y^2 over a width x height rectangle.

    x spans [-W/2, W/2], y spans [-H/2, H/2]; +z points toward the viewer.
    Display coordinate (u, v):
        u in [0, display_w) maps linearly to x,
        v in [0, display_h) maps linearly to y (v = 0 is the TOP row,
        matching image conventions).
    """

    def __init__(self, cfg: ScreenConfig):
        self.cfg = cfg

    # ---------------- parametrization: display px -> 3D ----------------
    def display_to_xy(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        u = uv[..., 0]
        v = uv[..., 1]
        x = (u / (self.cfg.display_w - 1) - 0.5) * self.cfg.width_m
        y = (0.5 - v / (self.cfg.display_h - 1)) * self.cfg.height_m
        return np.stack([x, y], axis=-1)

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.cfg.curve_x * x**2 - self.cfg.curve_y * y**2

    def display_to_world(self, uv: np.ndarray) -> np.ndarray:
        """(..., 2) display px -> (..., 3) world meters."""
        xy = self.display_to_xy(uv)
        z = self.height(xy[..., 0], xy[..., 1])
        return np.concatenate([xy, z[..., None]], axis=-1)

    # ---------------- parametrization: 3D -> display px ----------------
    def world_to_display(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        u = (pts[..., 0] / self.cfg.width_m + 0.5) * (self.cfg.display_w - 1)
        v = (0.5 - pts[..., 1] / self.cfg.height_m) * (self.cfg.display_h - 1)
        return np.stack([u, v], axis=-1)

    # ---------------- sampling helpers ----------------
    def sample_grid(self, nu: int, nv: int):
        """Regular display-space grid and its 3D lift.

        Returns (uv (nv,nu,2), world (nv,nu,3)).
        """
        us = np.linspace(0, self.cfg.display_w - 1, nu)
        vs = np.linspace(0, self.cfg.display_h - 1, nv)
        uu, vv = np.meshgrid(us, vs)
        uv = np.stack([uu, vv], axis=-1)
        return uv, self.display_to_world(uv)

    def normal_at(self, uv: np.ndarray) -> np.ndarray:
        """Outward (toward +z viewer) unit normal at display coords."""
        xy = self.display_to_xy(uv)
        x, y = xy[..., 0], xy[..., 1]
        # z = cx x^2 - cy y^2  ->  n = (-dz/dx, -dz/dy, 1)
        n = np.stack(
            [-2 * self.cfg.curve_x * x, 2 * self.cfg.curve_y * y, np.ones_like(x)],
            axis=-1,
        )
        return n / np.linalg.norm(n, axis=-1, keepdims=True)
