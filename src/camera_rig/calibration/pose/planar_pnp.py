"""Target-agnostic planar IPPE pose estimation with full candidate diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose.camera_model import OpenCVCameraModel, to_opencv_camera_model
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.calibration.pose.observability import (
    PoseAmbiguityCandidate,
    PoseObservabilityMetrics,
    UncertaintyValidatedThresholds,
    evaluate_pose_observability,
)
from camera_rig.calibration.pose.refinement import refine_planar_pose_lm
from camera_rig.calibration.pose.reprojection import ReprojectionMetrics, reprojection_metrics
from camera_rig.calibration.pose.validation import PoseValidity, validate_planar_pose
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation


@dataclass(frozen=True)
class PoseCandidateDiagnostic:
    """One IPPE candidate, including rejected physical solutions."""

    index: int
    T_camera_from_target: RigidTransform
    validity: PoseValidity
    reprojection: ReprojectionMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "T_camera_from_target": self.T_camera_from_target.to_dict(),
            "validity": self.validity.to_dict(),
            "reprojection": self.reprojection.to_dict(),
        }


@dataclass(frozen=True)
class CandidateSeparation:
    left_index: int
    right_index: int
    translation_m: float
    rotation_rad: float

    def to_dict(self) -> dict[str, object]:
        return {
            "left_index": self.left_index,
            "right_index": self.right_index,
            "translation_m": self.translation_m,
            "rotation_rad": self.rotation_rad,
            "rotation_deg": math.degrees(self.rotation_rad),
        }


@dataclass(frozen=True)
class PlanarPoseEstimate:
    """Refined target-to-camera pose and auditable IPPE selection evidence."""

    T_camera_from_target: RigidTransform
    selected_candidate_index: int
    candidates: tuple[PoseCandidateDiagnostic, ...]
    candidate_separations: tuple[CandidateSeparation, ...]
    refined_validity: PoseValidity
    reprojection: ReprojectionMetrics
    observability: PoseObservabilityMetrics
    camera_model_diagnostics: dict[str, object]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            "T_camera_from_target": self.T_camera_from_target.to_dict(),
            "selected_candidate_index": self.selected_candidate_index,
            "candidate_count": self.candidate_count,
            "candidates": [item.to_dict() for item in self.candidates],
            "candidate_separations": [item.to_dict() for item in self.candidate_separations],
            "refined_validity": self.refined_validity.to_dict(),
            "reprojection": self.reprojection.to_dict(),
            "observability": self.observability.to_dict(),
            "camera_model": dict(self.camera_model_diagnostics),
        }


class PlanarPoseEstimator:
    """Estimate one shared planar target pose from a generic observation."""

    def __init__(
        self, observability_thresholds: UncertaintyValidatedThresholds | None = None
    ) -> None:
        self._observability_thresholds = (
            observability_thresholds or UncertaintyValidatedThresholds()
        )

    def estimate(
        self, observation: TargetObservation, intrinsics: CameraIntrinsics
    ) -> PlanarPoseEstimate:
        cv2 = cv2_module()
        camera_model = to_opencv_camera_model(intrinsics)
        _validate_inputs(observation, intrinsics)
        objects = np.ascontiguousarray(observation.object_points_m, dtype=np.float64)
        images = np.ascontiguousarray(observation.image_points_px, dtype=np.float64)
        try:
            success, rvecs_value, tvecs_value, _opencv_errors = cv2.solvePnPGeneric(
                objects,
                images,
                camera_model.camera_matrix,
                camera_model.distortion_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error as error:
            raise ContractError(f"OpenCV IPPE solve failed: {error}") from error
        rvecs = tuple(rvecs_value)
        tvecs = tuple(tvecs_value)
        if not success or not rvecs or len(rvecs) != len(tvecs):
            raise ContractError("OpenCV IPPE returned no consistent pose candidates")
        candidates = tuple(
            _candidate(index, rvec, tvec, observation, camera_model, cv2)
            for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs, strict=True))
        )
        valid_candidates = [item for item in candidates if item.validity.valid]
        if not valid_candidates:
            reasons = sorted(
                {reason for item in candidates for reason in item.validity.failure_reasons}
            )
            raise ContractError(f"IPPE returned no physically valid candidate: {reasons}")
        selected = min(valid_candidates, key=lambda item: (item.reprojection.rmse_px, item.index))
        refined = refine_planar_pose_lm(
            selected.T_camera_from_target,
            objects,
            images,
            intrinsics,
        )
        if not refined.validity.valid:
            raise ContractError(
                "LM-refined pose failed physical validation: "
                f"{list(refined.validity.failure_reasons)}"
            )
        ambiguity_candidates = tuple(
            PoseAmbiguityCandidate(
                index=item.index,
                T_camera_from_target=item.T_camera_from_target,
                valid=item.validity.valid,
                reprojection_sse_px2=float(np.sum(np.square(item.reprojection.residuals_px))),
            )
            for item in candidates
        )
        observability = evaluate_pose_observability(
            object_points_m=objects,
            image_points_px=images,
            T_camera_from_target=refined.T_camera_from_target,
            intrinsics=intrinsics,
            ambiguity_candidates=ambiguity_candidates,
            thresholds=self._observability_thresholds,
            scope="frame",
        )
        return PlanarPoseEstimate(
            T_camera_from_target=refined.T_camera_from_target,
            selected_candidate_index=selected.index,
            candidates=candidates,
            candidate_separations=_candidate_separations(candidates),
            refined_validity=refined.validity,
            reprojection=refined.reprojection,
            observability=observability,
            camera_model_diagnostics=camera_model.diagnostics,
        )


def _validate_inputs(observation: TargetObservation, intrinsics: CameraIntrinsics) -> None:
    if observation.image_size != (intrinsics.width, intrinsics.height):
        raise ContractError("observation image size does not match camera intrinsics")
    if observation.target_frame == intrinsics.frame:
        raise ContractError("target and camera frames must be distinct")
    points = observation.object_points_m
    if len(points) < 4:
        raise ContractError("planar pose estimation requires at least four correspondences")
    if not np.allclose(points[:, 2], 0.0, atol=1e-9, rtol=0.0):
        raise ContractError("planar target object points must lie on target z=0")
    centered = points[:, :2] - np.mean(points[:, :2], axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-12) != 2:
        raise ContractError("planar target points must span two dimensions")


def _candidate(
    index: int,
    rvec: npt.ArrayLike,
    tvec: npt.ArrayLike,
    observation: TargetObservation,
    camera_model: OpenCVCameraModel,
    cv2: Any,
) -> PoseCandidateDiagnostic:
    transform = _transform_from_vectors(
        rvec,
        tvec,
        observation.target_frame,
        _camera_frame(camera_model, observation),
        cv2,
    )
    return PoseCandidateDiagnostic(
        index=index,
        T_camera_from_target=transform,
        validity=validate_planar_pose(transform, observation.object_points_m),
        reprojection=reprojection_metrics(
            observation.object_points_m,
            observation.image_points_px,
            rvec,
            tvec,
            camera_model,
            cv2=cv2,
        ),
    )


def _camera_frame(camera_model: OpenCVCameraModel, observation: TargetObservation) -> str:
    value = camera_model.diagnostics.get("camera_frame")
    if isinstance(value, str):
        return value
    raise ContractError(
        f"internal camera model for target {observation.target_frame!r} lacks a camera frame"
    )


def _transform_from_vectors(
    rvec: npt.ArrayLike,
    tvec: npt.ArrayLike,
    target_frame: str,
    camera_frame: str,
    cv2: Any,
) -> RigidTransform:
    rotation, _jacobian = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return RigidTransform(source_frame=target_frame, target_frame=camera_frame, matrix=matrix)


def _candidate_separations(
    candidates: tuple[PoseCandidateDiagnostic, ...],
) -> tuple[CandidateSeparation, ...]:
    result: list[CandidateSeparation] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            translation = float(
                np.linalg.norm(
                    left.T_camera_from_target.matrix[:3, 3]
                    - right.T_camera_from_target.matrix[:3, 3]
                )
            )
            relative = (
                left.T_camera_from_target.matrix[:3, :3].T
                @ right.T_camera_from_target.matrix[:3, :3]
            )
            cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            result.append(
                CandidateSeparation(
                    left_index=left.index,
                    right_index=right.index,
                    translation_m=translation,
                    rotation_rad=math.acos(cosine),
                )
            )
    return tuple(result)
