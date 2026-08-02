"""Ambient light tolerance (paper Sec 2.1).

Three steps, implemented against the simulated cameras:

  (a) Estimate ambient light A: project BLACK, capture; the captured
      gray of the screen is the ambient contribution. That black frame
      is stored and subtracted from all subsequent frames.
  (b) Find the reference exposure Er: project a reference pattern P with
      known average brightness Br (x% black, rest bright, 25<=x<=75),
      sweep the exposure ladder, pick the exposure minimizing over/under
      exposed pixels.
  (c) Predict exposure for any other pattern S with brightness Bs:

          Es = Er * (A + Br) / (A + Bs)            (Eq. 1)

      snapped to the discrete exposure ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from .config import ExposureConfig
from .registration import Registration


def make_pattern(kind: str, black_fraction: float, bright: float = 60.0):
    """Return callable(uv (N,2)) -> radiance, a structured-light pattern.

    Patterns from Fig. 2: chessboard / stripes / blobs.
    """
    period = 300.0  # display px

    def chessboard(uv):
        c = ((uv[:, 0] // period).astype(int) + (uv[:, 1] // period).astype(int)) % 2
        return np.where(c == 0, 0.0, bright)

    def stripes(uv):
        c = (uv[:, 0] % period) < period * (1 - black_fraction)
        return np.where(c, bright, 0.0)

    def blobs(uv):
        gx = (uv[:, 0] % period) - period / 2
        gy = (uv[:, 1] % period) - period / 2
        r = period * np.sqrt((1 - black_fraction) / np.pi)
        return np.where(gx**2 + gy**2 < r**2, bright, 0.0)

    return {"chessboard": chessboard, "stripes": stripes, "blobs": blobs}[kind]


def pattern_brightness(pattern: Callable, n: int = 4096,
                       dw: int = 3840, dh: int = 2160) -> float:
    rng = np.random.default_rng(0)
    uv = np.stack([rng.uniform(0, dw, n), rng.uniform(0, dh, n)], axis=1)
    return float(np.mean(pattern(uv)))


@dataclass
class AmbientCalibration:
    A: float                                   # estimated ambient (gray @ E=ref)
    Er_ms: float                               # reference exposure
    Br: float                                  # reference pattern brightness
    black_frames: Dict[int, np.ndarray] = field(default_factory=dict)
    ambient_radiance_est: float = 0.0          # in scene-radiance units

    def predict_exposure(self, Bs: float, cfg: ExposureConfig) -> float:
        """Eq. 1, snapped to the exposure ladder."""
        a = self.ambient_radiance_est
        es = self.Er_ms * (a + self.Br) / (a + max(Bs, 1e-6))
        ladder = np.asarray(cfg.ladder_ms, float)
        return float(ladder[np.argmin(np.abs(ladder - es))])


class AmbientLightManager:
    def __init__(self, reg: Registration, cfg: ExposureConfig):
        self.reg = reg
        self.cfg = cfg

    def _capture_all(self, pattern_or_scalar, ambient_radiance: float,
                     exposure_ms: float) -> Dict[int, np.ndarray]:
        out = {}
        for cam in self.reg.cameras:
            cam.exposure_ms = exposure_ms
            content = self.reg.render_for_camera(cam.cam_id, pattern_or_scalar)
            out[cam.cam_id] = cam.capture(content, ambient_radiance)
        return out

    def calibrate(self, ambient_radiance: float) -> AmbientCalibration:
        """Run steps (a) and (b); returns calibration used for prediction."""
        # (a) ambient estimation: project black at a fixed probe exposure
        probe_ms = 80.0
        blacks = self._capture_all(0.0, ambient_radiance, probe_ms)
        gray = float(np.mean([np.mean(b) for b in blacks.values()]))
        # invert the radiometric model to express ambient in radiance units
        gain = self.reg.cameras[0].gain
        ambient_est = gray / (gain * probe_ms)

        # (b) reference exposure sweep
        pat = make_pattern("stripes", self.cfg.reference_black_fraction)
        Br = pattern_brightness(pat)
        best_e, best_score = None, np.inf
        for e in self.cfg.ladder_ms:
            caps = self._capture_all(pat, ambient_radiance, float(e))
            score = 0.0
            for img in caps.values():
                over = np.mean(img >= self.cfg.over_thresh)
                under = np.mean(img <= self.cfg.under_thresh)
                mid = abs(np.mean(img) - self.cfg.target_mean) / 255.0
                score += over + under + 0.5 * mid
            if score < best_score:
                best_score, best_e = score, float(e)

        cal = AmbientCalibration(A=gray, Er_ms=best_e, Br=Br,
                                 ambient_radiance_est=ambient_est)
        # store black frames at the *reference* exposure for subtraction
        cal.black_frames = self._capture_all(0.0, ambient_radiance, best_e)
        return cal
