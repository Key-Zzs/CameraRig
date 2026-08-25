"""Calibration-target plugin contracts."""

from camera_rig.targets.base import TargetDetector
from camera_rig.targets.io import load_target, validate_target_artifact
from camera_rig.targets.observation import TargetObservation
from camera_rig.targets.registry import TargetDetectorRegistry, registry
from camera_rig.targets.spec import TargetSpec

__all__ = [
    "TargetDetector",
    "TargetDetectorRegistry",
    "TargetObservation",
    "TargetSpec",
    "load_target",
    "registry",
    "validate_target_artifact",
]
