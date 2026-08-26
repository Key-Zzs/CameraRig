"""Stable configuration façade."""

from __future__ import annotations

from pathlib import Path

from camera_rig.config.loader import load_config
from camera_rig.config.models import CameraConfig

__all__ = ["CameraConfig", "load_camera_config"]


def load_camera_config(path: str | Path) -> CameraConfig:
    """Load and strictly validate one versioned single-camera configuration."""
    return load_config(path)
