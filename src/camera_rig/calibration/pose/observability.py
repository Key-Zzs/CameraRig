"""Conditional local pose uncertainty and planar-candidate observability diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose.camera_model import to_opencv_camera_model
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform

FloatArray = npt.NDArray[np.float64]
ObservabilityScope = Literal["frame", "final"]


@dataclass(frozen=True)
class UncertaintyValidatedThresholds:
    """Frozen release preset for the fixed-camera uncertainty policy."""

    preset: str = "uncertainty_validated_v1"
    pixel_noise_floor_px: float = 0.25
    maximum_frame_translation_worst_std_mm: float = 5.0
    maximum_frame_rotation_worst_std_deg: float = 2.0
    maximum_final_translation_worst_std_mm: float = 2.0
    maximum_final_rotation_worst_std_deg: float = 0.5
    maximum_scaled_condition_number: float = 100.0
    minimum_pose_solve_ratio: float = 0.95
    minimum_observable_frame_ratio: float = 0.90
    maximum_ambiguous_frame_ratio: float = 0.05
    ambiguity_minimum_delta_chi2: float = 9.0
    ambiguity_material_translation_mm: float = 5.0
    ambiguity_material_rotation_deg: float = 5.0

    def __post_init__(self) -> None:
        if self.preset != "uncertainty_validated_v1":
            raise ContractError("unsupported pose-observability release preset")
        positive = (
            self.pixel_noise_floor_px,
            self.maximum_frame_translation_worst_std_mm,
            self.maximum_frame_rotation_worst_std_deg,
            self.maximum_final_translation_worst_std_mm,
            self.maximum_final_rotation_worst_std_deg,
            self.maximum_scaled_condition_number,
            self.ambiguity_minimum_delta_chi2,
            self.ambiguity_material_translation_mm,
            self.ambiguity_material_rotation_deg,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ContractError("pose-observability thresholds must be finite and positive")
        ratios = (
            self.minimum_pose_solve_ratio,
            self.minimum_observable_frame_ratio,
            self.maximum_ambiguous_frame_ratio,
        )
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in ratios):
            raise ContractError("pose-observability ratios must lie in [0, 1]")

    def uncertainty_limits(self, scope: ObservabilityScope) -> tuple[float, float]:
        if scope == "frame":
            return (
                self.maximum_frame_translation_worst_std_mm,
                self.maximum_frame_rotation_worst_std_deg,
            )
        return (
            self.maximum_final_translation_worst_std_mm,
            self.maximum_final_rotation_worst_std_deg,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "pixel_noise_floor_px": self.pixel_noise_floor_px,
            "maximum_frame_translation_worst_std_mm": (self.maximum_frame_translation_worst_std_mm),
            "maximum_frame_rotation_worst_std_deg": self.maximum_frame_rotation_worst_std_deg,
            "maximum_final_translation_worst_std_mm": (self.maximum_final_translation_worst_std_mm),
            "maximum_final_rotation_worst_std_deg": self.maximum_final_rotation_worst_std_deg,
            "maximum_scaled_condition_number": self.maximum_scaled_condition_number,
            "minimum_pose_solve_ratio": self.minimum_pose_solve_ratio,
            "minimum_observable_frame_ratio": self.minimum_observable_frame_ratio,
            "maximum_ambiguous_frame_ratio": self.maximum_ambiguous_frame_ratio,
            "ambiguity_minimum_delta_chi2": self.ambiguity_minimum_delta_chi2,
            "ambiguity_material_translation_mm": self.ambiguity_material_translation_mm,
            "ambiguity_material_rotation_deg": self.ambiguity_material_rotation_deg,
        }


@dataclass(frozen=True)
class PoseAmbiguityCandidate:
    """Equal-basis IPPE candidate input for the ambiguity comparison."""

    index: int
    T_camera_from_target: RigidTransform
    valid: bool
    reprojection_sse_px2: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ContractError("pose ambiguity candidate index must be non-negative")
        if not math.isfinite(self.reprojection_sse_px2) or self.reprojection_sse_px2 < 0:
            raise ContractError("pose ambiguity candidate SSE must be finite and non-negative")


@dataclass(frozen=True)
class PoseAmbiguityMetrics:
    """Statistical competitiveness and physical separation of an IPPE alternative."""

    comparison_basis: str
    valid_candidate_count: int
    second_candidate_available: bool
    best_candidate_index: int | None
    second_candidate_index: int | None
    delta_chi2: float | None
    translation_separation_mm: float | None
    rotation_separation_deg: float | None
    materially_distinct: bool
    statistically_competitive: bool
    ambiguous: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison_basis": self.comparison_basis,
            "valid_candidate_count": self.valid_candidate_count,
            "second_candidate_available": self.second_candidate_available,
            "best_candidate_index": self.best_candidate_index,
            "second_candidate_index": self.second_candidate_index,
            "delta_chi2": self.delta_chi2,
            "translation_separation_mm": self.translation_separation_mm,
            "rotation_separation_deg": self.rotation_separation_deg,
            "materially_distinct": self.materially_distinct,
            "statistically_competitive": self.statistically_competitive,
            "ambiguous": self.ambiguous,
        }


@dataclass(frozen=True)
class PoseObservabilityMetrics:
    """Serializable local uncertainty in a physical pose tangent space."""

    parameterization: str
    scope: ObservabilityScope
    target_scale_m: float
    pixel_noise_sigma_px: float
    pixel_noise_floor_px: float
    residual_dof: int
    effective_rank: int
    rank_tolerance: float
    scaled_singular_values: tuple[float, ...]
    scaled_condition_number: float | None
    minimum_scaled_singular_value: float
    covariance_6x6: tuple[tuple[float, ...], ...] | None
    rotation_std_xyz_deg: tuple[float, float, float] | None
    translation_std_xyz_mm: tuple[float, float, float] | None
    rotation_worst_axis_std_deg: float | None
    translation_worst_axis_std_mm: float | None
    candidate_ambiguity: PoseAmbiguityMetrics
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameterization": self.parameterization,
            "scope": self.scope,
            "target_scale_m": self.target_scale_m,
            "pixel_noise_sigma_px": self.pixel_noise_sigma_px,
            "pixel_noise_floor_px": self.pixel_noise_floor_px,
            "residual_dof": self.residual_dof,
            "effective_rank": self.effective_rank,
            "rank_tolerance": self.rank_tolerance,
            "scaled_singular_values": list(self.scaled_singular_values),
            "scaled_condition_number": self.scaled_condition_number,
            "minimum_scaled_singular_value": self.minimum_scaled_singular_value,
            "covariance_6x6": (
                [list(row) for row in self.covariance_6x6]
                if self.covariance_6x6 is not None
                else None
            ),
            "rotation_std_xyz_deg": (
                list(self.rotation_std_xyz_deg) if self.rotation_std_xyz_deg is not None else None
            ),
            "translation_std_xyz_mm": (
                list(self.translation_std_xyz_mm)
                if self.translation_std_xyz_mm is not None
                else None
            ),
            "rotation_worst_axis_std_deg": self.rotation_worst_axis_std_deg,
            "translation_worst_axis_std_mm": self.translation_worst_axis_std_mm,
            "candidate_ambiguity": self.candidate_ambiguity.to_dict(),
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "conditional_uncertainty": {
                "fixed_intrinsics_and_distortion": True,
                "fixed_target_geometry": True,
                "correct_correspondence_association": True,
                "local_gaussian_pixel_approximation": True,
                "absolute_system_uncertainty": False,
            },
        }


def evaluate_pose_observability(
    *,
    object_points_m: npt.ArrayLike,
    image_points_px: npt.ArrayLike,
    T_camera_from_target: RigidTransform,
    intrinsics: CameraIntrinsics,
    ambiguity_candidates: tuple[PoseAmbiguityCandidate, ...] = (),
    thresholds: UncertaintyValidatedThresholds | None = None,
    scope: ObservabilityScope = "frame",
) -> PoseObservabilityMetrics:
    """Evaluate scaled-Jacobian rank, covariance, and planar ambiguity."""
    release = thresholds or UncertaintyValidatedThresholds()
    objects, observed = _validated_inputs(
        object_points_m, image_points_px, T_camera_from_target, intrinsics
    )
    cv2 = cv2_module()
    camera_model = to_opencv_camera_model(intrinsics)
    rvec, _rodrigues_jacobian = cv2.Rodrigues(T_camera_from_target.matrix[:3, :3])
    tvec = T_camera_from_target.matrix[:3, 3].reshape(3, 1)
    try:
        projected, jacobian_value = cv2.projectPoints(
            objects,
            rvec,
            tvec,
            camera_model.camera_matrix,
            camera_model.distortion_coeffs,
        )
    except cv2.error as error:
        raise ContractError(f"OpenCV observability projection failed: {error}") from error
    projected_points = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    jacobian = np.asarray(jacobian_value, dtype=np.float64)
    if jacobian.shape[0] != 2 * len(objects) or jacobian.shape[1] < 6:
        raise ContractError("OpenCV projectPoints returned an unexpected Jacobian shape")
    projection_jacobian = np.ascontiguousarray(jacobian[:, :6], dtype=np.float64)
    if not np.isfinite(projected_points).all() or not np.isfinite(projection_jacobian).all():
        raise ContractError("pose-observability projection or Jacobian is non-finite")

    centered = objects - np.mean(objects, axis=0, keepdims=True)
    target_scale_m = float(np.sqrt(np.mean(np.sum(np.square(centered), axis=1))))
    if not math.isfinite(target_scale_m) or target_scale_m <= 0:
        raise ContractError("pose-observability target characteristic scale must be positive")
    parameter_scale = np.diag([1.0, 1.0, 1.0, target_scale_m, target_scale_m, target_scale_m])
    scaled_jacobian = projection_jacobian @ parameter_scale
    _left, singular_values, right_transpose = np.linalg.svd(scaled_jacobian, full_matrices=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.shape != (6,) or not np.isfinite(singular_values).all():
        raise ContractError("pose-observability SVD did not return six finite singular values")
    rank_tolerance = float(
        max(scaled_jacobian.shape) * np.finfo(np.float64).eps * singular_values[0]
    )
    effective_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    minimum_singular = float(singular_values[-1])
    condition = (
        float(singular_values[0] / minimum_singular)
        if effective_rank == 6 and minimum_singular > 0
        else None
    )

    component_residuals = projected_points - observed
    residual_dof = max(2 * len(objects) - 6, 1)
    sigma_fit_squared = float(np.sum(np.square(component_residuals)) / residual_dof)
    sigma_px = max(release.pixel_noise_floor_px, math.sqrt(max(sigma_fit_squared, 0.0)))
    ambiguity = evaluate_pose_ambiguity(
        ambiguity_candidates,
        pixel_noise_sigma_px=sigma_px,
        thresholds=release,
    )

    covariance: FloatArray | None = None
    rotation_std: tuple[float, float, float] | None = None
    translation_std: tuple[float, float, float] | None = None
    rotation_worst: float | None = None
    translation_worst: float | None = None
    if effective_rank == 6:
        inverse_squared = np.diag(1.0 / np.square(singular_values))
        covariance_q = (sigma_px**2) * (right_transpose.T @ inverse_squared @ right_transpose)
        covariance_rvec_tvec = parameter_scale @ covariance_q @ parameter_scale.T
        covariance_rvec_tvec = (covariance_rvec_tvec + covariance_rvec_tvec.T) / 2.0
        if not np.isfinite(covariance_rvec_tvec).all():
            raise ContractError("pose-observability covariance is non-finite")
        # OpenCV differentiates in additive Rodrigues-vector coordinates.  Convert
        # their covariance to the left-invariant camera-frame rotation tangent used
        # by physical pose-error metrics: Exp(r + dr) ~= Exp(J_l(r) dr) Exp(r).
        tangent_from_parameters = np.eye(6, dtype=np.float64)
        tangent_from_parameters[:3, :3] = _so3_left_jacobian(rvec.reshape(3))
        covariance = tangent_from_parameters @ covariance_rvec_tvec @ tangent_from_parameters.T
        covariance = (covariance + covariance.T) / 2.0
        rotation_block = covariance[:3, :3]
        translation_block = covariance[3:, 3:]
        rotation_variances = np.maximum(np.diag(rotation_block), 0.0)
        translation_variances = np.maximum(np.diag(translation_block), 0.0)
        rotation_std = (
            float(math.degrees(math.sqrt(rotation_variances[0]))),
            float(math.degrees(math.sqrt(rotation_variances[1]))),
            float(math.degrees(math.sqrt(rotation_variances[2]))),
        )
        translation_std = (
            float(1000.0 * math.sqrt(translation_variances[0])),
            float(1000.0 * math.sqrt(translation_variances[1])),
            float(1000.0 * math.sqrt(translation_variances[2])),
        )
        rotation_worst = float(
            math.degrees(math.sqrt(max(float(np.max(np.linalg.eigvalsh(rotation_block))), 0.0)))
        )
        translation_worst = float(
            1000.0 * math.sqrt(max(float(np.max(np.linalg.eigvalsh(translation_block))), 0.0))
        )

    maximum_translation, maximum_rotation = release.uncertainty_limits(scope)
    failures: list[str] = []
    if effective_rank < 6:
        failures.append("POSE_OBSERVABILITY_RANK_DEFICIENT")
    if translation_worst is None or translation_worst > maximum_translation:
        failures.append("POSE_TRANSLATION_UNCERTAINTY_EXCEEDED")
    if rotation_worst is None or rotation_worst > maximum_rotation:
        failures.append("POSE_ROTATION_UNCERTAINTY_EXCEEDED")
    if condition is None or condition > release.maximum_scaled_condition_number:
        failures.append("POSE_CONDITION_NUMBER_EXCEEDED")
    if ambiguity.ambiguous:
        failures.append("POSE_AMBIGUOUS")

    covariance_tuple = (
        tuple(tuple(float(value) for value in row) for row in covariance)
        if covariance is not None
        else None
    )
    return PoseObservabilityMetrics(
        parameterization=(
            "linearization p=[opencv_rvec_rad,tvec_m]; "
            "reported covariance=[left_invariant_camera_rotation_rad,tvec_m]; "
            "q=[opencv_rvec_rad,tvec_m/target_scale_m]"
        ),
        scope=scope,
        target_scale_m=target_scale_m,
        pixel_noise_sigma_px=sigma_px,
        pixel_noise_floor_px=release.pixel_noise_floor_px,
        residual_dof=residual_dof,
        effective_rank=effective_rank,
        rank_tolerance=rank_tolerance,
        scaled_singular_values=tuple(float(value) for value in singular_values),
        scaled_condition_number=condition,
        minimum_scaled_singular_value=minimum_singular,
        covariance_6x6=covariance_tuple,
        rotation_std_xyz_deg=rotation_std,
        translation_std_xyz_mm=translation_std,
        rotation_worst_axis_std_deg=rotation_worst,
        translation_worst_axis_std_mm=translation_worst,
        candidate_ambiguity=ambiguity,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def _so3_left_jacobian(rotation_vector: FloatArray) -> FloatArray:
    """Map additive Rodrigues increments to left-invariant SO(3) increments."""
    vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    skew = np.asarray(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        result = np.eye(3) + 0.5 * skew + (1.0 / 6.0) * (skew @ skew)
    else:
        angle_squared = angle * angle
        result = (
            np.eye(3)
            + ((1.0 - math.cos(angle)) / angle_squared) * skew
            + ((angle - math.sin(angle)) / (angle_squared * angle)) * (skew @ skew)
        )
    if not np.isfinite(result).all():
        raise ContractError("SO(3) left Jacobian is non-finite")
    return np.asarray(result, dtype=np.float64)


def evaluate_pose_ambiguity(
    candidates: tuple[PoseAmbiguityCandidate, ...],
    *,
    pixel_noise_sigma_px: float,
    thresholds: UncertaintyValidatedThresholds,
) -> PoseAmbiguityMetrics:
    """Compare the two best physically valid raw IPPE candidates on an equal basis."""
    if not math.isfinite(pixel_noise_sigma_px) or pixel_noise_sigma_px <= 0:
        raise ContractError("pose ambiguity requires positive finite pixel noise")
    valid = sorted(
        (candidate for candidate in candidates if candidate.valid),
        key=lambda candidate: (candidate.reprojection_sse_px2, candidate.index),
    )
    if len(valid) < 2:
        return PoseAmbiguityMetrics(
            comparison_basis="raw_ippe_candidates_equal_basis",
            valid_candidate_count=len(valid),
            second_candidate_available=False,
            best_candidate_index=valid[0].index if valid else None,
            second_candidate_index=None,
            delta_chi2=None,
            translation_separation_mm=None,
            rotation_separation_deg=None,
            materially_distinct=False,
            statistically_competitive=False,
            ambiguous=False,
        )
    best, second = valid[:2]
    delta_chi2 = max(
        0.0,
        (second.reprojection_sse_px2 - best.reprojection_sse_px2) / (pixel_noise_sigma_px**2),
    )
    translation_mm = float(
        1000.0
        * np.linalg.norm(
            best.T_camera_from_target.matrix[:3, 3] - second.T_camera_from_target.matrix[:3, 3]
        )
    )
    relative_rotation = (
        best.T_camera_from_target.matrix[:3, :3].T @ second.T_camera_from_target.matrix[:3, :3]
    )
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = math.degrees(math.acos(cosine))
    materially_distinct = (
        translation_mm >= thresholds.ambiguity_material_translation_mm
        or rotation_deg >= thresholds.ambiguity_material_rotation_deg
    )
    statistically_competitive = delta_chi2 < thresholds.ambiguity_minimum_delta_chi2
    return PoseAmbiguityMetrics(
        comparison_basis="raw_ippe_candidates_equal_basis",
        valid_candidate_count=len(valid),
        second_candidate_available=True,
        best_candidate_index=best.index,
        second_candidate_index=second.index,
        delta_chi2=float(delta_chi2),
        translation_separation_mm=translation_mm,
        rotation_separation_deg=rotation_deg,
        materially_distinct=materially_distinct,
        statistically_competitive=statistically_competitive,
        ambiguous=materially_distinct and statistically_competitive,
    )


def projection_jacobian_first_six(
    *,
    object_points_m: npt.ArrayLike,
    T_camera_from_target: RigidTransform,
    intrinsics: CameraIntrinsics,
) -> FloatArray:
    """Return OpenCV's analytic ``d(pixel)/d(rvec,tvec)`` columns for parity tests."""
    objects = np.ascontiguousarray(np.asarray(object_points_m, dtype=np.float64).reshape(-1, 3))
    if not len(objects) or not np.isfinite(objects).all():
        raise ContractError("analytic projection Jacobian requires finite object points")
    if T_camera_from_target.target_frame != intrinsics.frame:
        raise ContractError("projection Jacobian transform target must match intrinsics frame")
    cv2: Any = cv2_module()
    camera_model = to_opencv_camera_model(intrinsics)
    rvec, _jacobian = cv2.Rodrigues(T_camera_from_target.matrix[:3, :3])
    _projected, jacobian = cv2.projectPoints(
        objects,
        rvec,
        T_camera_from_target.matrix[:3, 3],
        camera_model.camera_matrix,
        camera_model.distortion_coeffs,
    )
    result: FloatArray = np.asarray(jacobian, dtype=np.float64)[:, :6].copy()
    result.setflags(write=False)
    return result


def _validated_inputs(
    object_points_m: npt.ArrayLike,
    image_points_px: npt.ArrayLike,
    transform: RigidTransform,
    intrinsics: CameraIntrinsics,
) -> tuple[FloatArray, FloatArray]:
    objects = np.asarray(object_points_m, dtype=np.float64)
    images = np.asarray(image_points_px, dtype=np.float64)
    if objects.ndim != 2 or objects.shape[1:] != (3,):
        raise ContractError("pose-observability object points must have shape (N, 3)")
    if images.ndim != 2 or images.shape[1:] != (2,):
        raise ContractError("pose-observability image points must have shape (N, 2)")
    if len(objects) != len(images) or len(objects) < 4:
        raise ContractError("pose-observability requires at least four matched points")
    if not np.isfinite(objects).all() or not np.isfinite(images).all():
        raise ContractError("pose-observability correspondences must be finite")
    if transform.target_frame != intrinsics.frame:
        raise ContractError("pose-observability transform target must match intrinsics frame")
    if not np.allclose(objects[:, 2], 0.0, atol=1e-9, rtol=0.0):
        raise ContractError("pose-observability target points must lie on target z=0")
    centered_xy = objects[:, :2] - np.mean(objects[:, :2], axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered_xy, tol=1e-12) != 2:
        raise ContractError("pose-observability target points must span two dimensions")
    return (
        np.ascontiguousarray(objects, dtype=np.float64),
        np.ascontiguousarray(images, dtype=np.float64),
    )
