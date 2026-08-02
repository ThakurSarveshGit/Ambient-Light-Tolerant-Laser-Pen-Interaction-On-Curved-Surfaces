"""Registration: cameras <-> parametrized display coordinates.

The original system obtained this from automated camera-based
registration (refs [1-5]). Since ground-truth geometry is available in
simulation, we compute the same products the registration produced:

  * camera extrinsics (known here by construction),
  * a forward map  display(u,v) -> camera pixel  per camera,
  * an inverse map camera pixel -> display(u,v)  per camera
    (dense lookup grids with bilinear interpolation),
  * visibility masks (which display region each camera sees).

Everything downstream (partitioning, tracking, hand-off) uses ONLY these
maps -- never the analytic surface -- exactly as the real pipeline used
only registration data.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .camera import CameraPose, SimulatedCamera, look_at
from .config import SystemConfig
from .surface import PringleSurface


def build_camera_rig(cfg: SystemConfig, surface: PringleSurface,
                     n_cameras: int = 4,
                     arrangement: str = "horizontal",
                     rng: Optional[np.random.Generator] = None
                     ) -> List[SimulatedCamera]:
    """Place n cameras behind the (rear-projected) screen in a linear array.

    Mirrors Fig. 1 (horizontal, 4 cams) / Fig. 5 (vertical, 3 cams).
    """
    rng = rng or np.random.default_rng(cfg.rng_seed)
    W, H = cfg.screen.width_m, cfg.screen.height_m
    dist = 0.9 * max(W, H)              # stand-off distance behind screen
    cams = []
    for i in range(n_cameras):
        s = (i + 0.5) / n_cameras - 0.5
        if arrangement == "horizontal":
            eye = np.array([s * 0.9 * W, 0.0, -dist])
            target = np.array([s * 0.55 * W, 0.0, 0.0])
        else:
            eye = np.array([0.0, s * 0.9 * H, -dist])
            target = np.array([0.0, s * 0.55 * H, 0.0])
        # Rear cameras look toward +z. Flip the surface z-sign handled by
        # projecting actual 3D points; looking from -z works directly.
        R, e = look_at(eye, target, up=(0, 1, 0))
        cams.append(SimulatedCamera(i, cfg.camera, CameraPose(R, e),
                                    np.random.default_rng(rng.integers(1 << 30))))
    return cams


class Registration:
    """Holds display<->camera mappings for one rig (the registration output)."""

    def __init__(self, cfg: SystemConfig, surface: PringleSurface,
                 cameras: List[SimulatedCamera],
                 grid_nu: int = 192, grid_nv: int = 108):
        self.cfg = cfg
        self.surface = surface
        self.cameras = cameras
        self.grid_nu, self.grid_nv = grid_nu, grid_nv

        # Dense display grid lifted to 3D (the "reconstruction")
        self.uv_grid, self.world_grid = surface.sample_grid(grid_nu, grid_nv)

        # Forward maps: for each camera, display-grid -> pixel, + visibility
        self.fwd_px = []      # (nv, nu, 2) per camera
        self.visible = []     # (nv, nu) bool per camera
        self.cam_dist = []    # (nv, nu) distance camera->surface point
        for cam in cameras:
            px = cam.project(self.world_grid)
            self.fwd_px.append(px)
            self.visible.append(cam.in_view(px, margin=2.0))
            self.cam_dist.append(cam.distance_to(self.world_grid))

    # ---------- camera pixel -> display coordinate (inverse map) ----------
    def cam_px_to_display(self, cam_id: int, px: np.ndarray) -> Optional[np.ndarray]:
        """Invert the forward map by local search on the registration grid.

        Real system: dense per-pixel lookup tables from structured light.
        Here: nearest grid cell + bilinear refinement (sub-pixel).
        """
        fwd = self.fwd_px[cam_id]
        vis = self.visible[cam_id]
        d2 = np.sum((fwd - px) ** 2, axis=-1)
        d2[~vis] = np.inf
        idx = np.unravel_index(np.argmin(d2), d2.shape)
        if not np.isfinite(d2[idx]):
            return None
        iv, iu = idx
        uv0 = self.uv_grid[iv, iu].copy()
        # Gauss-Newton refine uv so that project(display_to_world(uv)) == px
        cam = self.cameras[cam_id]
        for _ in range(6):
            w = self.surface.display_to_world(uv0)
            p = cam.project(w)
            r = px - p
            if np.linalg.norm(r) < 1e-3:
                break
            eps = 1.0
            J = np.zeros((2, 2))
            for k in range(2):
                duv = uv0.copy()
                duv[k] += eps
                J[:, k] = (cam.project(self.surface.display_to_world(duv)) - p) / eps
            try:
                uv0 = uv0 + np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                break
        uv0[0] = np.clip(uv0[0], 0, self.cfg.screen.display_w - 1)
        uv0[1] = np.clip(uv0[1], 0, self.cfg.screen.display_h - 1)
        return uv0

    # ---------- scene rendering into a camera (for capture simulation) ----------
    def render_for_camera(self, cam_id: int, display_radiance) -> np.ndarray:
        """Resample a display-space radiance function/image into camera pixels.

        display_radiance: callable(uv (N,2)) -> radiance (N,), or a scalar.
        Renders at registration-grid resolution then splats; adequate for
        threshold-based laser detection which only needs coarse background.
        """
        cam = self.cameras[cam_id]
        img = np.zeros((cam.cfg.image_h, cam.cfg.image_w), np.float64)
        wsum = np.zeros_like(img)
        px = self.fwd_px[cam_id][self.visible[cam_id]]
        uv = self.uv_grid[self.visible[cam_id]]
        if callable(display_radiance):
            rad = display_radiance(uv.reshape(-1, 2))
        else:
            rad = np.full(len(uv), float(display_radiance))
        xi = np.clip(px[:, 0].round().astype(int), 0, cam.cfg.image_w - 1)
        yi = np.clip(px[:, 1].round().astype(int), 0, cam.cfg.image_h - 1)
        np.add.at(img, (yi, xi), rad)
        np.add.at(wsum, (yi, xi), 1.0)
        out = np.where(wsum > 0, img / np.maximum(wsum, 1), 0.0)
        # cheap hole fill: 5x5 box blur where empty
        from scipy.ndimage import uniform_filter
        blur = uniform_filter(out, size=7)
        blurw = uniform_filter((wsum > 0).astype(float), size=7)
        fill = np.where(blurw > 0, blur / np.maximum(blurw, 1e-9), 0.0)
        return np.where(wsum > 0, out, fill)

    def laser_pixel_in_camera(self, cam_id: int, uv: np.ndarray) -> Optional[np.ndarray]:
        """Ground-truth projection of a display-space laser spot into a camera."""
        w = self.surface.display_to_world(np.asarray(uv, float))
        px = self.cameras[cam_id].project(w)
        return px if self.cameras[cam_id].in_view(px) else None
