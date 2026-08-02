"""Figures analogous to the paper's: rig geometry, partitions (Fig. 3),
detection heatmaps (Fig. 7), and tracked trajectories."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from .engine import Simulation


def fig_rig(sim: Simulation, path: str) -> None:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    w = sim.reg.world_grid
    ax.plot_surface(w[..., 0], w[..., 2], w[..., 1], alpha=0.6,
                    cmap="viridis", linewidth=0)
    for cam in sim.cameras:
        e = cam.pose.eye
        ax.scatter([e[0]], [e[2]], [e[1]], s=60, marker="^",
                   label=f"cam {cam.cam_id}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("y (m)")
    ax.set_title("Pringle screen (20ft x 15ft) + rear camera rig")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_partitions(sim: Simulation, path: str) -> None:
    own = sim.partition.owner
    ncam = len(sim.cameras)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(own, cmap=cm.get_cmap("tab10", ncam), origin="upper",
                   extent=[0, sim.cfg.screen.display_w, sim.cfg.screen.display_h, 0])
    # boundary bands where >1 camera is polled
    band = sim.partition.boundary_cams.sum(-1) > 1
    ax.contour(np.linspace(0, sim.cfg.screen.display_w, band.shape[1]),
               np.linspace(0, sim.cfg.screen.display_h, band.shape[0]),
               band.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    cbar = fig.colorbar(im, ax=ax, ticks=range(ncam))
    cbar.set_label("owner camera")
    ax.set_title("Display partitions + hand-off boundary bands (cf. Fig. 3/4a)")
    ax.set_xlabel("display u (px)"); ax.set_ylabel("display v (px)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_heatmap(grid: np.ndarray, title: str, path: str,
                vmin: float = 90.0) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=100, origin="upper")
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax.text(c, r, f"{grid[r, c]:.0f}", ha="center", va="center",
                    fontsize=7)
    fig.colorbar(im, ax=ax, label="%")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_trajectory(sim: Simulation, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {p.pen_id: p.color for p in sim.scene.pens.values()}
    for pid in colors:
        pts = np.array([s.uv for s in sim.samples
                        if s.pen_id == pid and s.on and s.uv is not None])
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=3,
                    color=colors[pid] if colors[pid] in
                    ("green", "blue", "red") else None,
                    label=f"pen {pid} ({colors[pid]})")
    for g in sim.gestures:
        if g.kind == "single_click":
            ax.plot(*g.uv, "k+", ms=12, mew=1.5)
        elif g.kind == "double_click":
            ax.plot(*g.uv, "kx", ms=12, mew=1.5)
    ax.set_xlim(0, sim.cfg.screen.display_w)
    ax.set_ylim(sim.cfg.screen.display_h, 0)
    ax.set_title("Tracked laser samples (+ = single click, x = double click)")
    ax.set_xlabel("display u (px)"); ax.set_ylabel("display v (px)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
