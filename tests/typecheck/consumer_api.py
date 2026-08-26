"""Static consumer contract checked by mypy; this file is not a pytest module."""

from pathlib import Path

from camera_rig.api import (
    CameraBundle,
    CameraConfig,
    CameraFrame,
    CameraSession,
    ReplayCameraSession,
    RigidTransform,
    load_camera_bundle,
    load_camera_config,
    load_provisioned_camera_bundle,
)


def load_config(path: Path) -> CameraConfig:
    return load_camera_config(path)


def load_bundle(path: Path) -> CameraBundle:
    return load_camera_bundle(path)


def load_provision(path: Path) -> CameraBundle:
    return load_provisioned_camera_bundle(path)


def live_frame(config: CameraConfig) -> CameraFrame:
    with CameraSession.from_config(config) as session:
        return session.capture()


def replay_frame(path: Path) -> CameraFrame:
    with ReplayCameraSession.from_artifact(path) as session:
        return session.capture()


def fixed_transform(bundle: CameraBundle) -> RigidTransform:
    fixed = bundle.fixed_mount_calibration
    if fixed is None:
        raise RuntimeError("camera is not fixed-calibrated")
    return fixed.T_parent_from_camera_reference
