"""Global configuration for the simulated laser-pen interaction system.

Mirrors the setup described in:
  Thakur, Urs, Ibrahim, Sidenko, Majumder,
  "Ambient Light Tolerant Laser-Pen Based Interaction with Curved
   Multi-Projector Displays", HCII 2022 (LNCS 13303).

The physical screen simulated here is the 15 ft (H) x 20 ft (W)
pringle-shaped (hyperbolic-paraboloid) rear-projection surface.
"""

from dataclasses import dataclass, field
from typing import Tuple

FT = 0.3048  # feet -> meters


@dataclass(frozen=True)
class ScreenConfig:
    width_m: float = 20 * FT          # 20 ft wide
    height_m: float = 15 * FT         # 15 ft tall
    # Pringle (hyperbolic paraboloid) curvature coefficients: z = ax*x^2 - ay*y^2
    curve_x: float = 0.28             # curvature along width  (bends toward viewer)
    curve_y: float = 0.22             # curvature along height (bends away)
    # Parametrized 2D display resolution (the "2D display coordinates" of the paper)
    display_w: int = 3840
    display_h: int = 2160


@dataclass(frozen=True)
class CameraConfig:
    image_w: int = 640
    image_h: int = 480
    fov_deg: float = 70.0             # horizontal field of view
    fps: float = 50.0                 # FLIR Blackfly ran 45-55 FPS in the paper
    read_noise_sigma: float = 1.5     # gray levels of sensor noise
    saturation: float = 255.0


@dataclass(frozen=True)
class ExposureConfig:
    # Discrete exposure ladder (ms), similar to Table 1 of the paper.
    ladder_ms: Tuple[int, ...] = (10, 20, 40, 80, 160, 320, 640)
    # Reference-pattern black fraction x in [25, 75]% (paper Sec 2.1)
    reference_black_fraction: float = 0.5
    target_mean: float = 127.0        # aim for mid-gray captures
    over_thresh: float = 250.0
    under_thresh: float = 5.0


@dataclass(frozen=True)
class InteractionConfig:
    # State-machine timing thresholds (seconds). Paper: t < x < T.
    t_click_max: float = 0.25         # laser ON shorter than t  -> click candidate
    x_drag_hold: float = 0.45         # ON, roughly stationary >= x -> drag lock
    T_action_trigger: float = 0.60    # OFF gap >= T finalizes the click count
    drag_vicinity_px: float = 40.0    # "same vicinity" radius in display px
    boundary_band_px: int = 120       # partition boundary band width (display px)
    laser_detect_threshold: float = 60.0   # gray levels above background
    heartbeat_timeout_s: float = 0.5  # camera considered failed after this silence


@dataclass(frozen=True)
class SystemConfig:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    n_projectors: int = 8             # 2 superimposed 2x2 arrays in the paper
    rng_seed: int = 7


DEFAULT = SystemConfig()
