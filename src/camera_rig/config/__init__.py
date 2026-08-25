"""Strict single-camera YAML configuration."""

from camera_rig.config.loader import load_config
from camera_rig.config.models import (
    CONFIG_SCHEMA_VERSION,
    CameraConfig,
    CameraSettings,
    CaptureSettings,
    StreamSettings,
)

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CameraConfig",
    "CameraSettings",
    "CaptureSettings",
    "StreamSettings",
    "load_config",
]
