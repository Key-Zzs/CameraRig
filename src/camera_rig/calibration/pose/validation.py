"""Physical validity checks for target-to-camera planar poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from camera_rig.core.errors import ContractError
from camera_rig.core.transforms import RigidTransform


@dataclass(frozen=True)
class PoseValidity:
    """Cheirality and printed-face checks with explicit failure reasons."""

    finite: bool
    cheirality: bool
    printed_face_orientation: bool
    minimum_depth_m: float
    face_orientation_dot: float
    failure_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.finite and self.cheirality and self.printed_face_orientation

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "finite": self.finite,
            "cheirality": self.cheirality,
            "printed_face_orientation": self.printed_face_orientation,
            "minimum_depth_m": self.minimum_depth_m,
            "face_orientation_dot": self.face_orientation_dot,
            "failure_reasons": list(self.failure_reasons),
        }


def validate_planar_pose(transform: RigidTransform, object_points_m: npt.ArrayLike) -> PoseValidity:
    """Require every target point in front of the camera and +Z facing the camera."""
    points = np.asarray(object_points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ContractError("pose validation object points must have shape (N, 3) with N > 0")
    camera_points = transform.transform_points(points)
    target_centroid_camera = np.mean(camera_points, axis=0)
    normal_camera = transform.matrix[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    depths = camera_points[:, 2]
    finite = bool(
        np.isfinite(camera_points).all()
        and np.isfinite(target_centroid_camera).all()
        and np.isfinite(normal_camera).all()
    )
    minimum_depth = float(np.min(depths)) if len(depths) else float("-inf")
    orientation_dot = (
        float(np.dot(normal_camera, -target_centroid_camera)) if finite else float("-inf")
    )
    cheirality = finite and bool(np.all(depths > 0.0))
    face_orientation = finite and orientation_dot > 0.0
    reasons: list[str] = []
    if not finite:
        reasons.append("non_finite_pose_geometry")
    if not cheirality:
        reasons.append("target_points_not_strictly_in_front_of_camera")
    if not face_orientation:
        reasons.append("printed_face_not_facing_camera")
    return PoseValidity(
        finite=finite,
        cheirality=cheirality,
        printed_face_orientation=face_orientation,
        minimum_depth_m=minimum_depth,
        face_orientation_dot=orientation_dot,
        failure_reasons=tuple(reasons),
    )
