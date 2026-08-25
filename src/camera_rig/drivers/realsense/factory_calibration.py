"""Factory calibration extraction from one active RealSense pipeline."""

from __future__ import annotations

import math

import numpy as np

from camera_rig.core.errors import ContractError, LifecycleError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform
from camera_rig.drivers.base import CameraLifecycleState
from camera_rig.drivers.realsense.driver import RealSenseDriver

DISTORTION_MODELS = {
    "none": "none",
    "modified_brown_conrady": "modified-brown-conrady",
    "inverse_brown_conrady": "inverse-brown-conrady",
    "ftheta": "ftheta",
    "brown_conrady": "brown-conrady",
    "kannala_brandt4": "kannala-brandt4",
}


def extract_factory_calibration(driver: RealSenseDriver) -> FactoryCalibration:
    """Read intrinsics, directed extrinsics, and depth scale from active profiles."""
    if driver.state is not CameraLifecycleState.STREAMING:
        raise LifecycleError("factory calibration requires an active RealSense driver")
    pipeline_profile = driver.pipeline_profile
    handles = driver.adapter.active_profile_handles(pipeline_profile)
    expected_names = {profile.stream_name for profile in driver.active_profiles}
    if set(handles) != expected_names:
        raise ContractError(
            f"active profile handles differ from validated profiles: {sorted(handles)}"
        )
    intrinsics: dict[str, CameraIntrinsics] = {}
    for name, handle in handles.items():
        try:
            value = driver.adapter.intrinsics_data(handle)
        except RuntimeError as error:
            raise ContractError(f"could not read {name!r} factory intrinsics: {error}") from error
        raw_model = _string(value, "distortion_model")
        try:
            stable_model = DISTORTION_MODELS[raw_model]
        except KeyError as error:
            raise ContractError(f"unsupported RealSense distortion model: {raw_model!r}") from error
        intrinsics[name] = CameraIntrinsics(
            frame=_frame(driver.config.camera.name, name),
            width=_int(value, "width"),
            height=_int(value, "height"),
            fx=_float(value, "fx"),
            fy=_float(value, "fy"),
            cx=_float(value, "cx"),
            cy=_float(value, "cy"),
            distortion_model=stable_model,
            distortion_coeffs=tuple(
                _float_item(item) for item in _tuple(value, "distortion_coeffs")
            ),
        )
    reference = driver.config.camera.output_reference_stream
    source_handle = handles[reference]
    transforms: list[RigidTransform] = []
    for target in sorted(handles):
        if target == reference:
            continue
        try:
            rotation_flat, translation = driver.adapter.extrinsics_data(
                source_handle, handles[target]
            )
        except RuntimeError as error:
            raise ContractError(
                f"could not read factory extrinsics from {reference!r} to {target!r}: {error}"
            ) from error
        transforms.append(
            _rigid_transform(
                _frame(driver.config.camera.name, reference),
                _frame(driver.config.camera.name, target),
                rotation_flat,
                translation,
            )
        )
    try:
        depth_scale = driver.adapter.depth_scale(pipeline_profile)
    except RuntimeError as error:
        raise ContractError(f"could not read RealSense depth scale: {error}") from error
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ContractError("RealSense depth scale must be finite and greater than zero")
    return FactoryCalibration(
        device=driver.get_device_info(),
        stream_profiles={profile.stream_name: profile for profile in driver.active_profiles},
        intrinsics=intrinsics,
        internal_transforms=tuple(transforms),
        depth_scale_m_per_unit=depth_scale,
    )


def _rigid_transform(
    source_frame: str,
    target_frame: str,
    rotation_flat: tuple[float, ...],
    translation: tuple[float, ...],
) -> RigidTransform:
    if len(rotation_flat) != 9 or len(translation) != 3:
        raise ContractError("RealSense extrinsics must contain 9 rotation and 3 translation values")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation_flat, dtype=np.float64).reshape((3, 3), order="F")
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return RigidTransform(source_frame=source_frame, target_frame=target_frame, matrix=matrix)


def _frame(camera_name: str, stream_name: str) -> str:
    return f"{camera_name}/{stream_name}_optical"


def _string(value: dict[str, object], name: str) -> str:
    item = value[name]
    if not isinstance(item, str):
        raise ContractError(f"{name} must be a string")
    return item


def _int(value: dict[str, object], name: str) -> int:
    item = value[name]
    if isinstance(item, bool) or not isinstance(item, int):
        raise ContractError(f"{name} must be an integer")
    return item


def _float(value: dict[str, object], name: str) -> float:
    return _float_item(value[name])


def _float_item(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError("calibration value must be numeric")
    return float(value)


def _tuple(value: dict[str, object], name: str) -> tuple[object, ...]:
    item = value[name]
    if not isinstance(item, tuple):
        raise ContractError(f"{name} must be a tuple")
    return item
