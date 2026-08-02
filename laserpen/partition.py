
"""Display partitioning and camera hand-off geometry (Sec 2.2).

Each display coordinate is owned by the *closest* camera that can see it.
A boundary band around each partition edge marks where the adjacent
camera must also be polled, enabling a smooth real-time hand-off.
Supports repartitioning when cameras fail (their region is absorbed by
the nearest surviving cameras).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

from .registration import Registration


class DisplayPartition:
    def __init__(self, reg: Registration, boundary_band_px: int):
        self.reg = reg
        self.boundary_band_px = boundary_band_px
        self.owner: Optional[np.ndarray] = None       # (nv, nu) camera id or -1
        self.boundary_cams: Optional[np.ndarray] = None  # (nv, nu, ncam) bool
        self.rebuild(active=set(range(len(reg.cameras))))

    # ------------------------------------------------------------------
    def rebuild(self, active: Set[int]) -> None:
        """(Re)compute ownership over the registration grid.

        Called once at startup and again whenever a camera fails/recovers.
        """
        nv, nu = self.reg.uv_grid.shape[:2]
        ncam = len(self.reg.cameras)
        dist = np.full((nv, nu, ncam), np.inf)
        for c in range(ncam):
            if c in active:
                d = self.reg.cam_dist[c].copy()
                d[~self.reg.visible[c]] = np.inf
                dist[..., c] = d
        self.owner = np.where(np.isfinite(dist.min(-1)), dist.argmin(-1), -1)

        # boundary bands: dilate each partition; overlap of dilations of two
        # different owners marks cells where both cameras are polled.
        grid_du = self.reg.uv_grid[0, 1, 0] - self.reg.uv_grid[0, 0, 0]
        it = max(1, int(round(self.boundary_band_px / max(grid_du, 1e-9))))
        self.boundary_cams = np.zeros((nv, nu, ncam), bool)
        for c in active:
            mask = self.owner == c
            grown = binary_dilation(mask, iterations=it)
            self.boundary_cams[..., c] = grown
        self.active = set(active)

    # ------------------------------------------------------------------
    def _cell_of(self, uv: np.ndarray) -> Tuple[int, int]:
        nv, nu = self.reg.uv_grid.shape[:2]
        dw = self.reg.cfg.screen.display_w
        dh = self.reg.cfg.screen.display_h
        iu = int(np.clip(round(uv[0] / (dw - 1) * (nu - 1)), 0, nu - 1))
        iv = int(np.clip(round(uv[1] / (dh - 1) * (nv - 1)), 0, nv - 1))
        return iv, iu

    def owner_of(self, uv: np.ndarray) -> int:
        iv, iu = self._cell_of(uv)
        return int(self.owner[iv, iu])

    def cameras_to_poll(self, uv: Optional[np.ndarray]) -> List[int]:
        """Cameras that must be polled for the current laser position.

        * interior of a partition -> just the owner
        * boundary band          -> owner + adjacent camera(s)
        * unknown position (laser OFF / just appeared) -> all active
        """
        if uv is None:
            return sorted(self.active)
        iv, iu = self._cell_of(uv)
        own = int(self.owner[iv, iu])
        cams = [c for c in self.active
                if self.boundary_cams[iv, iu, c] or c == own]
        return sorted(set(cams)) if cams else sorted(self.active)

    def coverage_fraction(self) -> float:
        return float(np.mean(self.owner >= 0))
