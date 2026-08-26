"""Validated OpenCV projection behind the generic pose dependency boundary."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose.camera_model import to_opencv_camera_model
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform


def project_points_px(
    object_points_m: npt.ArrayLike,
    T_camera_from_object: RigidTransform,
    intrinsics: CameraIntrinsics,
) -> npt.NDArray[np.float64]:
    """Project object-frame 3D points through a frame-checked camera pose."""
    if T_camera_from_object.target_frame != intrinsics.frame:
        raise ContractError("projection transform target must match camera intrinsics frame")
    points = np.asarray(object_points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ContractError("projection object points must have shape (N, 3) with N > 0")
    if not np.isfinite(points).all():
        raise ContractError("projection object points must be finite")
    cv2 = cv2_module()
    camera_model = to_opencv_camera_model(intrinsics)
    rvec, _jacobian = cv2.Rodrigues(T_camera_from_object.matrix[:3, :3])
    try:
        projected, _jacobian = cv2.projectPoints(
            np.ascontiguousarray(points, dtype=np.float64),
            rvec,
            T_camera_from_object.matrix[:3, 3],
            camera_model.camera_matrix,
            camera_model.distortion_coeffs,
        )
    except cv2.error as error:
        raise ContractError(f"OpenCV projection failed: {error}") from error
    result = cast(
        npt.NDArray[np.float64],
        np.asarray(projected, dtype=np.float64).reshape(-1, 2).copy(),
    )
    if not np.isfinite(result).all():
        raise ContractError("OpenCV projection returned non-finite pixels")
    result.setflags(write=False)
    return result
