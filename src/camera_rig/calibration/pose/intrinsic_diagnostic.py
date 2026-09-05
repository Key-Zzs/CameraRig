"""Pose-nuisance evaluation for diagnostic intrinsic model comparisons."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose.camera_model import to_opencv_camera_model
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics


def intrinsic_observation_pose_diversity(
    observations: tuple[tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64]], ...],
    intrinsics: CameraIntrinsics,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return nuisance-PnP distance and target-normal tilt diagnostics."""

    cv2 = cv2_module()
    model = to_opencv_camera_model(intrinsics)
    distances: list[float] = []
    tilts: list[float] = []
    for pose_id, object_points, image_points in observations:
        success, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            object_points,
            image_points,
            model.camera_matrix,
            model.distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or not rvecs or not tvecs:
            raise ContractError(f"training pose solve failed for {pose_id}")
        rotation, _jacobian = cv2.Rodrigues(rvecs[0])
        distances.append(float(np.linalg.norm(tvecs[0])))
        tilts.append(float(np.degrees(np.arccos(np.clip(abs(float(rotation[2, 2])), 0.0, 1.0)))))
    return tuple(distances), tuple(tilts)


def evaluate_intrinsic_model(
    observations: tuple[tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64]], ...],
    intrinsics: CameraIntrinsics,
) -> dict[str, object]:
    """Fit only per-pose nuisance transforms, then report frozen-model residuals."""

    cv2 = cv2_module()
    model = to_opencv_camera_model(intrinsics)
    all_errors: list[float] = []
    per_pose: dict[str, float] = {}
    per_pose_p95: dict[str, float] = {}
    for pose_id, object_points, image_points in observations:
        success, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            model.camera_matrix,
            model.distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or not rvecs or not tvecs:
            raise ContractError(f"holdout PnP failed for {pose_id}")
        rvec = rvecs[0]
        tvec = tvecs[0]
        projected, _jacobian = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            model.camera_matrix,
            model.distortion_coeffs,
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        if not np.isfinite(errors).all():
            raise ContractError("holdout reprojection returned non-finite errors")
        all_errors.extend(float(value) for value in errors)
        per_pose[pose_id] = float(np.sqrt(np.mean(np.square(errors))))
        per_pose_p95[pose_id] = float(np.percentile(errors, 95))
    values = np.asarray(all_errors, dtype=np.float64)
    return {
        "rmse_px": float(np.sqrt(np.mean(np.square(values)))),
        "p95_px": float(np.percentile(values, 95)),
        "per_pose_rmse_px": per_pose,
        "per_pose_p95_px": per_pose_p95,
        "sample_count": len(values),
    }
