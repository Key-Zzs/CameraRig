"""Fixed-camera extrinsic-calibration contracts and solver."""

from camera_rig.calibration.fixed.artifact import (
    FixedCalibrationArtifact,
    load_and_validate_fixed_calibration,
    write_fixed_calibration,
)
from camera_rig.calibration.fixed.calibrator import (
    FixedCameraCalibrator,
    calibrate_fixed_camera,
)
from camera_rig.calibration.fixed.config import FixedCalibrationConfig, load_fixed_config

__all__ = [
    "FixedCalibrationArtifact",
    "FixedCalibrationConfig",
    "FixedCameraCalibrator",
    "calibrate_fixed_camera",
    "load_and_validate_fixed_calibration",
    "load_fixed_config",
    "write_fixed_calibration",
]
