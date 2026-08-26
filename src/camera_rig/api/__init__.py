"""Stable CameraRig v1 consumer API.

Only symbols listed in ``__all__`` are covered by the public compatibility policy.
Implementation modules outside :mod:`camera_rig.api` remain internal.
"""

from camera_rig.api.bundle import (
    CameraBundle,
    load_camera_bundle,
    load_provisioned_camera_bundle,
)
from camera_rig.api.calibration import (
    CameraIntrinsics,
    FactoryCalibration,
    FixedMountCalibration,
    RigidTransform,
)
from camera_rig.api.config import CameraConfig, load_camera_config
from camera_rig.api.frames import CameraFrame, StreamFrame
from camera_rig.api.replay import ReplayCameraSession
from camera_rig.api.runtime import CameraSession

__all__ = [
    "CameraBundle",
    "CameraConfig",
    "CameraFrame",
    "CameraIntrinsics",
    "CameraSession",
    "FactoryCalibration",
    "FixedMountCalibration",
    "ReplayCameraSession",
    "RigidTransform",
    "StreamFrame",
    "load_camera_bundle",
    "load_camera_config",
    "load_provisioned_camera_bundle",
]
