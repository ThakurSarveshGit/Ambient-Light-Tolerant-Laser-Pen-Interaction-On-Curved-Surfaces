"""Animated replay of a simulation session.

Renders (from the recorded engine/coordinator data — nothing is faked):

  * the 2D display with partition colors and hand-off boundary bands,
  * the laser dot (per-pen color, hollow when OFF) with a fading trail,
  * which cameras are being polled *this tick* (partition-aware polling),
  * the per-pen state machine state (OFF / ON_PENDING / COUNTING / DRAG),
  * gesture banners as they fire (SINGLE CLICK, DOUBLE CLICK, DRAG ...),
  * camera hand-off flashes at the moment ownership changes,
  * the mirrored-laptop event ticker (what got injected into the OS sink).

Usage:
    animate_session(sim, "figures/session.gif")          # save GIF
    animate_session(sim, "figures/session.mp4")          # save MP4 (ffmpeg)
    animate_session(sim, None, live=True)                # interactive window
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from .engine import Simulation

_PEN_COLORS = {"green": "#18b418", "blue": "#2b6cf0", "red": "#e03131"}
_PART_CMAP = ["#dbeafe", "#fde8e8", "#fbe7f6", "#d9f6f2",
              "#fdf3d8", "#e8e6fd", "#e2f4d9", "#fbe4d5"]


def _bucket(items, key=lambda x: x[0]):
    """Group time-stamped tuples into a dict keyed by rounded time."""
    out: Dict[float, list] = {}
    for it in items:
        out.setdefault(round(key(it), 4), []).append(it)
    return out


def animate_session(sim: Simulation, path: Optional[str],
                    live: bool = False, frame_stride: int = 2,
                    trail_s: float = 1.2, fps: Optional[int] = None) -> None:
    if live:
        matplotlib.use("TkAgg", force=False)
    dw, dh = sim.cfg.screen.display_w, sim.cfg.screen.display_h
    dt = 1.0 / sim.cfg.camera.fps
    ncam = len(sim.cameras)

    # ---------- recorded data ----------
    samples = _bucket([(s.t, s) for s in sim.samples])
    states = _bucket([(t, pid, st) for t, pid, st in sim.state_history])
    polls = _bucket([(t, pid, cams, own) for t, pid, cams, own
                     in sim.coord.poll_history])
    ticks = sorted(samples.keys())
    if not ticks:
        raise RuntimeError("run the simulation before animating")
    gestures = [g for g in sim.gestures
                if g.kind in ("single_click", "double_click", "triple_click",
                              "drag_start", "drag_end")]
    handoffs = list(sim.coord.handoffs)
    syslog = list(sim.sink.log)

    # ---------- static scenery ----------
    fig = plt.figure(figsize=(11, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.0], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    axc = fig.add_subplot(gs[1])
    fig.suptitle("Laser-pen interaction on curved multi-projector display "
                 "(simulated replay)", fontsize=12)

    own = sim.partition.owner
    part_img = np.zeros(own.shape + (3,))
    for c in range(ncam):
        part_img[own == c] = matplotlib.colors.to_rgb(_PART_CMAP[c % len(_PART_CMAP)])
    ax.imshow(part_img, extent=[0, dw, dh, 0], origin="upper",
              interpolation="nearest")
    band = sim.partition.boundary_cams.sum(-1) > 1
    ax.contour(np.linspace(0, dw, band.shape[1]),
               np.linspace(0, dh, band.shape[0]), band.astype(float),
               levels=[0.5], colors="gray", linewidths=0.8, linestyles=":")
    for c in range(ncam):
        ys, xs = np.where(own == c)
        if len(xs):
            ax.text(np.median(xs) / own.shape[1] * dw,
                    np.median(ys) / own.shape[0] * dh,
                    f"cam {c}", color="#666", fontsize=9, ha="center",
                    alpha=0.8)
    ax.set_xlim(0, dw); ax.set_ylim(dh, 0)
    ax.set_xlabel("display u (px)"); ax.set_ylabel("display v (px)")

    # camera-poll indicator lamps along the top
    lamps = []
    for c in range(ncam):
        lamp = patches.Circle((dw * (0.06 + 0.04 * c), dh * 0.045),
                              radius=dh * 0.018, color="#cccccc", zorder=6)
        ax.add_patch(lamp)
        ax.text(dw * (0.06 + 0.04 * c), dh * 0.10, str(c), ha="center",
                fontsize=7, color="#444")
        lamps.append(lamp)
    ax.text(dw * 0.02, dh * 0.045, "polling:", fontsize=8, va="center",
            color="#444")

    # ---------- dynamic artists ----------
    pens = list(sim.scene.pens.values())
    dots, trails, sm_texts = {}, {}, {}
    for i, p in enumerate(pens):
        col = _PEN_COLORS.get(p.color, "black")
        dots[p.pen_id], = ax.plot([], [], "o", ms=11, mec=col, mfc=col, zorder=8)
        trails[p.pen_id], = ax.plot([], [], "-", lw=2, color=col, alpha=0.45,
                                    zorder=7)
        sm_texts[p.pen_id] = ax.text(dw * 0.62, dh * (0.045 + 0.055 * i), "",
                                     fontsize=9, color=col, zorder=9,
                                     family="monospace")
    banner = ax.text(dw / 2, dh * 0.5, "", fontsize=26, ha="center",
                     va="center", color="#111", weight="bold", zorder=10,
                     alpha=0.0)
    hoff_txt = ax.text(dw / 2, dh * 0.92, "", fontsize=11, ha="center",
                       color="#8a2be2", weight="bold", zorder=10)
    clock = ax.text(dw * 0.985, dh * 0.045, "", fontsize=9, ha="right",
                    color="#444")

    axc.set_axis_off()
    axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    axc.set_title("injected system events (mirrored laptop)", fontsize=9,
                  loc="left", color="#444")
    ticker = axc.text(0.005, 0.5, "", fontsize=8.5, family="monospace",
                      va="center")

    frame_ticks = ticks[::frame_stride]
    trail_pts: Dict[int, List] = {p.pen_id: [] for p in pens}

    def update(fi):
        t = frame_ticks[fi]
        clock.set_text(f"t = {t:5.2f} s")
        # laser dots + trails
        for s_t, s in samples.get(t, []):
            d = dots[s.pen_id]
            if s.uv is not None:
                d.set_data([s.uv[0]], [s.uv[1]])
                col = _PEN_COLORS.get(sim.scene.pens[s.pen_id].color, "k")
                d.set_markerfacecolor(col if s.on else "none")
                if s.on:
                    trail_pts[s.pen_id].append((t, s.uv))
            else:
                d.set_data([], [])
        for pid, tp in trail_pts.items():
            tp[:] = [(tt, uv) for tt, uv in tp if t - tt <= trail_s]
            if tp:
                arr = np.array([uv for _, uv in tp])
                trails[pid].set_data(arr[:, 0], arr[:, 1])
            else:
                trails[pid].set_data([], [])
        # polling lamps + owner
        active = set()
        for _, pid, cams, ownc in polls.get(t, []):
            active |= set(cams)
        for c, lamp in enumerate(lamps):
            if c not in sim.partition.active:
                lamp.set_color("#e03131")          # failed camera
            else:
                lamp.set_color("#f5b942" if c in active else "#d9d9d9")
        # state machine text
        for _, pid, st in states.get(t, []):
            sm_texts[pid].set_text(f"pen{pid} SM: {st}")
        # gesture banner (visible 0.8 s after the event)
        txt, alpha = "", 0.0
        for g in gestures:
            if 0 <= t - g.t <= 0.8:
                txt = g.kind.replace("_", " ").upper()
                alpha = max(0.0, 1.0 - (t - g.t) / 0.8)
        banner.set_text(txt); banner.set_alpha(alpha)
        # hand-off flash
        htxt = ""
        for ht, pid, a, b in handoffs:
            if 0 <= t - ht <= 0.6:
                htxt = f"hand-off cam{a} → cam{b}"
        hoff_txt.set_text(htxt)
        # event ticker: last 3 non-move events
        evs = [e for e in syslog if e.t <= t and e.kind != "mouse_move"][-3:]
        ticker.set_text("\n".join(
            f"t={e.t:5.2f}s  {e.kind:<18s} ({e.x:4.0f},{e.y:4.0f})"
            for e in evs))
        return []

    out_fps = fps or int(round(sim.cfg.camera.fps / frame_stride))
    anim = animation.FuncAnimation(fig, update, frames=len(frame_ticks),
                                   interval=1000 / out_fps, blit=False)
    if live:
        plt.show()
    elif path:
        if path.endswith(".mp4"):
            anim.save(path, writer=animation.FFMpegWriter(fps=out_fps,
                                                          bitrate=1800))
        else:
            anim.save(path, writer=animation.PillowWriter(fps=out_fps))
        plt.close(fig)
