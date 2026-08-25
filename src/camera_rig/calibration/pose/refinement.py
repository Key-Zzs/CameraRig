"""Shared-pose LM refinement for generic planar correspondences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose.camera_model import to_opencv_camera_model
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.calibration.pose.reprojection import ReprojectionMetrics, reprojection_metrics
from camera_rig.calibration.pose.validation import PoseValidity, validate_planar_pose
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform


@dataclass(frozen=True)
class RefinedPlanarPose:
    """LM-refined pose with post-refinement physical and reprojection evidence."""

    T_camera_from_target: RigidTransform
    reprojection: ReprojectionMetrics
    validity: PoseValidity
    camera_model_diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "T_camera_from_target": self.T_camera_from_target.to_dict(),
            "reprojection": self.reprojection.to_dict(),
            "validity": self.validity.to_dict(),
            "camera_model": dict(self.camera_model_diagnostics),
        }


def refine_planar_pose_lm(
    initial_pose: RigidTransform,
    object_points_m: npt.ArrayLike,
    image_points_px: npt.ArrayLike,
    intrinsics: CameraIntrinsics,
) -> RefinedPlanarPose:
    """Refine one target-to-camera pose against stacked planar correspondences."""
    if initial_pose.target_frame != intrinsics.frame:
        raise ContractError("initial pose target frame must match camera intrinsics frame")
    if initial_pose.source_frame == initial_pose.target_frame:
        raise ContractError("target and camera frames must be distinct")
    objects, images = _validated_correspondences(object_points_m, image_points_px)
    cv2 = cv2_module()
    camera_model = to_opencv_camera_model(intrinsics)
    initial_rvec, _jacobian = cv2.Rodrigues(initial_pose.matrix[:3, :3])
    initial_tvec = initial_pose.matrix[:3, 3].reshape(3, 1).copy()
    try:
        refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
            objects,
            images,
            camera_model.camera_matrix,
            camera_model.distortion_coeffs,
            np.asarray(initial_rvec, dtype=np.float64).reshape(3, 1),
            initial_tvec,
        )
    except cv2.error as error:
        raise ContractError(f"OpenCV LM refinement failed: {error}") from error
    rotation, _jacobian = cv2.Rodrigues(np.asarray(refined_rvec, dtype=np.float64).reshape(3, 1))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(refined_tvec, dtype=np.float64).reshape(3)
    refined = RigidTransform(
        source_frame=initial_pose.source_frame,
        target_frame=initial_pose.target_frame,
        matrix=matrix,
    )
    return RefinedPlanarPose(
        T_camera_from_target=refined,
        reprojection=reprojection_metrics(
            objects,
            images,
            refined_rvec,
            refined_tvec,
            camera_model,
            cv2=cv2,
        ),
        validity=validate_planar_pose(refined, objects),
        camera_model_diagnostics=camera_model.diagnostics,
    )


def _validated_correspondences(
    object_points_m: npt.ArrayLike, image_points_px: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    objects = np.asarray(object_points_m, dtype=np.float64)
    images = np.asarray(image_points_px, dtype=np.float64)
    if objects.ndim != 2 or objects.shape[1:] != (3,):
        raise ContractError("object_points_m must have shape (N, 3)")
    if images.ndim != 2 or images.shape[1:] != (2,):
        raise ContractError("image_points_px must have shape (N, 2)")
    if len(objects) != len(images) or len(objects) < 4:
        raise ContractError("LM refinement requires at least four matched correspondences")
    if not np.isfinite(objects).all() or not np.isfinite(images).all():
        raise ContractError("LM refinement correspondences must be finite")
    if not np.allclose(objects[:, 2], 0.0, atol=1e-9, rtol=0.0):
        raise ContractError("planar target object points must lie on target z=0")
    centered = objects[:, :2] - np.mean(objects[:, :2], axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-12) != 2:
        raise ContractError("planar target points must span two dimensions")
    return (
        np.ascontiguousarray(objects, dtype=np.float64),
        np.ascontiguousarray(images, dtype=np.float64),
    )
