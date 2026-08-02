"""Simulated implementation of "Ambient Light Tolerant Laser-Pen Based
Interaction with Curved Multi-Projector Displays" (HCII 2022)."""

from .actions import (ActionMapper, GestureTemplateLibrary, VNCMirrorSink,
                      VirtualDesktop)
from .ambient import AmbientLightManager, make_pattern, pattern_brightness
from .config import DEFAULT, SystemConfig
from .engine import Simulation
from .laser import LaserPen, ScriptedUser
from .partition import DisplayPartition
from .registration import Registration, build_camera_rig
from .state_machine import PenStateMachine
from .surface import PringleSurface

__all__ = [
    "ActionMapper", "GestureTemplateLibrary", "VNCMirrorSink", "VirtualDesktop",
    "AmbientLightManager", "make_pattern", "pattern_brightness",
    "DEFAULT", "SystemConfig", "Simulation", "LaserPen", "ScriptedUser",
    "DisplayPartition", "Registration", "build_camera_rig",
    "PenStateMachine", "PringleSurface",
]
