"""Simulated feedback cameras.

Each camera is a pinhole model with a pose looking at the screen. Its
`capture()` implements a simple radiometric model so the ambient-light /
exposure logic of the paper (Sec 2.1) is exercised for real:

    pixel = clip( k * E * (L_ambient + L_content + L_laser) + noise )

where E is exposure time. Over/under-exposure therefore actually happens
in the simulation, and the exposure-prediction formula (Eq. 1) matters.

The camera does NOT render full geometry; it rasterizes only what the
tracking pipeline needs: per-pixel scene radiance sampled through the
display <-> camera mapping, plus the laser spot as a bright Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import CameraConfig


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)):
    """World->camera rotation R and translation t (x_cam = R (x - eye))."""
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    dwn = np.cross(fwd, right)
    R = np.stack([right, dwn, fwd], axis=0)
    return R, eye


@dataclass
class CameraPose:
    R: np.ndarray  # 3x3 world->camera
    eye: np.ndarray  # camera center in world


class SimulatedCamera:
    def __init__(self, cam_id: int, cfg: CameraConfig, pose: CameraPose,
                 rng: np.random.Generator):
        self.cam_id = cam_id
        self.cfg = cfg
        self.pose = pose
        self.rng = rng
        f = 0.5 * cfg.image_w / np.tan(np.deg2rad(cfg.fov_deg) / 2)
        self.K = np.array([[f, 0, cfg.image_w / 2],
                           [0, f, cfg.image_h / 2],
                           [0, 0, 1.0]])
        self.exposure_ms: float = 40.0
        self.alive: bool = True         # failure injection flag
        # radiometric gain: gray levels per (radiance-unit * ms)
        self.gain = 0.02

    # ---------------- geometry ----------------
    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """(...,3) world -> (...,2) pixel coordinates (may be off-image)."""
        p = np.asarray(pts_world, float)
        pc = (p - self.pose.eye) @ self.pose.R.T
        z = np.clip(pc[..., 2], 1e-9, None)
        x = self.K[0, 0] * pc[..., 0] / z + self.K[0, 2]
        y = self.K[1, 1] * pc[..., 1] / z + self.K[1, 2]
        return np.stack([x, y], axis=-1)

    def in_view(self, px: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return ((px[..., 0] >= margin) & (px[..., 0] < self.cfg.image_w - margin)
                & (px[..., 1] >= margin) & (px[..., 1] < self.cfg.image_h - margin))

    def distance_to(self, pts_world: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(pts_world, float) - self.pose.eye, axis=-1)

    # ---------------- radiometry ----------------
    def capture(self,
                content_radiance: np.ndarray,
                ambient_radiance: float,
                laser_px: Optional[np.ndarray] = None,
                laser_radiance: float = 0.0,
                laser_sigma_px: float = 2.5) -> np.ndarray:
        """Return a (H, W) float32 gray image.

        content_radiance: (H, W) scene radiance already resampled into
        this camera's image plane (built by Registration.render_for_camera).
        """
        if not self.alive:
            raise CameraFailure(self.cam_id)
        img = content_radiance + ambient_radiance
        if laser_px is not None and laser_radiance > 0:
            h, w = img.shape
            ys = np.arange(h)[:, None]
            xs = np.arange(w)[None, :]
            d2 = (xs - laser_px[0]) ** 2 + (ys - laser_px[1]) ** 2
            img = img + laser_radiance * np.exp(-d2 / (2 * laser_sigma_px ** 2))
        img = self.gain * self.exposure_ms * img
        img = img + self.rng.normal(0, self.cfg.read_noise_sigma, img.shape)
        return np.clip(img, 0, self.cfg.saturation).astype(np.float32)


class CameraFailure(RuntimeError):
    def __init__(self, cam_id: int):
        super().__init__(f"camera {cam_id} failed")
        self.cam_id = cam_id
