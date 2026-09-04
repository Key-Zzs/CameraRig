"""Offline sensitivity analysis over retained real target detections."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.artifacts.target_detection import TargetDetectionArtifact
from camera_rig.calibration.fixed.aggregation import distribution, pose_delta
from camera_rig.calibration.fixed.structured_residuals import (
    StructuredReprojectionPolicy,
    evaluate_final_shared_structured_residuals,
    evaluate_observation_structured_residuals,
)
from camera_rig.calibration.pose import PlanarPoseEstimator
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

COUNTERFACTUAL_SCHEMA_VERSION = "camera-rig.model-counterfactual-evaluation.v1"
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class _Variant:
    name: str
    category: str
    description: str
    intrinsics: Callable[[CameraIntrinsics], CameraIntrinsics] | None = None
    geometry: Callable[[Mapping[int, FloatArray]], dict[int, FloatArray]] | None = None


def evaluate_model_counterfactuals(
    detection: TargetDetectionArtifact,
    factory: FactoryCalibrationArtifact,
    *,
    detection_report_sha256: str,
    factory_calibration_sha256: str,
) -> dict[str, object]:
    """Evaluate wrong-model sensitivity without modifying production artifacts.

    Pose deltas are sensitivities relative to the retained-data baseline, not ground-truth
    pose errors. The report therefore cannot release a production policy by itself.
    """
    if not detection.is_capture or detection.stream is None:
        raise ContractError("model counterfactuals require a capture target-detection report")
    if detection.stream not in factory.calibration.intrinsics:
        raise ContractError("target-detection stream has no factory intrinsics")
    observations = tuple(frame.observation for frame in detection.per_frame if frame.success)
    if len(observations) < 3:
        raise ContractError("model counterfactuals require at least three successful frames")
    intrinsics = factory.calibration.intrinsics[detection.stream]
    geometry = _canonical_geometry(observations)
    policy = StructuredReprojectionPolicy()
    baseline = _evaluate_variant(
        _Variant("baseline", "correct_model", "retained factory and target assumptions"),
        observations,
        intrinsics,
        geometry,
        baseline_poses=None,
        policy=policy,
    )
    baseline_poses = baseline.pop("_poses")
    assert isinstance(baseline_poses, dict)
    variants = [
        _evaluate_variant(
            variant,
            observations,
            intrinsics,
            geometry,
            baseline_poses=baseline_poses,
            policy=policy,
        )
        for variant in _variants()
    ]
    for result in variants:
        result.pop("_poses", None)
    return {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "status": "ANALYSIS_ONLY_NO_GROUND_TRUTH",
        "interpretation": (
            "pose deltas are sensitivity relative to the retained-data baseline, not pose bias"
        ),
        "production_artifact_mutation": False,
        "source": {
            "target_detection_sha256": detection_report_sha256,
            "factory_calibration_sha256": factory_calibration_sha256,
            "stream": detection.stream,
            "successful_frame_count": len(observations),
        },
        "candidate_policy": policy.to_dict(),
        "baseline": baseline,
        "counterfactuals": variants,
    }


def _evaluate_variant(
    variant: _Variant,
    observations: Sequence[TargetObservation],
    base_intrinsics: CameraIntrinsics,
    geometry: Mapping[int, FloatArray],
    *,
    baseline_poses: Mapping[int, RigidTransform] | None,
    policy: StructuredReprojectionPolicy,
) -> dict[str, object]:
    intrinsics = variant.intrinsics(base_intrinsics) if variant.intrinsics else base_intrinsics
    mapped = variant.geometry(geometry) if variant.geometry else dict(geometry)
    changed = tuple(_replace_geometry(observation, mapped) for observation in observations)
    board_reference = np.asarray([mapped[point_id] for point_id in sorted(mapped)])
    estimator = PlanarPoseEstimator()
    poses: dict[int, RigidTransform] = {}
    frame_rmse: list[float] = []
    frame_p95: list[float] = []
    translation: list[float] = []
    rotation: list[float] = []
    structured_rejections = 0
    observable = 0
    failures: list[dict[str, object]] = []
    for index, observation in enumerate(changed):
        try:
            estimate = estimator.estimate(observation, intrinsics)
            poses[index] = estimate.T_camera_from_target
            frame_rmse.append(estimate.reprojection.rmse_px)
            frame_p95.append(estimate.reprojection.p95_px)
            observable += int(estimate.observability.passed)
            structured = evaluate_observation_structured_residuals(
                observation,
                estimate.T_camera_from_target,
                intrinsics,
                thresholds=policy.thresholds,
                board_reference_points_m=board_reference,
            )
            structured_rejections += int(not structured.passed)
            if baseline_poses is not None and index in baseline_poses:
                reference = baseline_poses[index]
                delta = pose_delta(estimate.T_camera_from_target, reference)
                translation.append(delta.translation_mm)
                rotation.append(delta.rotation_deg)
        except ContractError as error:
            failures.append({"successful_observation_index": index, "reason": str(error)})
    if not poses:
        return {
            "name": variant.name,
            "category": variant.category,
            "description": variant.description,
            "solved_frame_count": 0,
            "failed_frame_count": len(failures),
            "failures": failures,
            "_poses": poses,
        }
    result: dict[str, object] = {
        "name": variant.name,
        "category": variant.category,
        "description": variant.description,
        "solved_frame_count": len(poses),
        "failed_frame_count": len(failures),
        "failures": failures,
        "frame_gross_reprojection": {
            "rmse_px": distribution(frame_rmse),
            "p95_px": distribution(frame_p95),
            "maximum_allowed_rmse_px": 1.5,
            "maximum_allowed_p95_px": 2.0,
        },
        "frame_observable_ratio": observable / len(poses),
        "frame_structured_reject_count": structured_rejections,
        "frame_structured_reject_ratio": structured_rejections / len(poses),
        "_poses": poses,
    }
    try:
        final_observation = _mean_corner_observation(changed)
        final_estimate = estimator.estimate(final_observation, intrinsics)
        result["final_evaluation_status"] = "EVALUATED"
        result["final_corner_mean_structured"] = evaluate_final_shared_structured_residuals(
            changed,
            final_estimate.T_camera_from_target,
            intrinsics,
            thresholds=policy.thresholds,
        )
    except ContractError as error:
        result["final_evaluation_status"] = "FAIL_CLOSED"
        result["final_failure_reason"] = str(error)
    if translation:
        result["pose_sensitivity_from_baseline"] = {
            "translation_mm": distribution(translation),
            "rotation_deg": distribution(rotation),
        }
    return result


def _canonical_geometry(observations: Sequence[TargetObservation]) -> dict[int, FloatArray]:
    result: dict[int, FloatArray] = {}
    for observation in observations:
        for index, point_id in enumerate(observation.point_ids):
            point = np.asarray(observation.object_points_m[index], dtype=np.float64)
            previous = result.setdefault(point_id, point.copy())
            if not np.allclose(previous, point, rtol=0.0, atol=1e-12):
                raise ContractError("one corner ID maps to inconsistent retained geometry")
    return result


def _replace_geometry(
    observation: TargetObservation, geometry: Mapping[int, FloatArray]
) -> TargetObservation:
    return replace(
        observation,
        object_points_m=np.asarray([geometry[point_id] for point_id in observation.point_ids]),
    )


def _mean_corner_observation(observations: Sequence[TargetObservation]) -> TargetObservation:
    grouped: dict[int, list[FloatArray]] = {}
    geometry = _canonical_geometry(observations)
    for observation in observations:
        for index, point_id in enumerate(observation.point_ids):
            grouped.setdefault(point_id, []).append(observation.image_points_px[index])
    ids = tuple(sorted(grouped))
    template = observations[0]
    return replace(
        template,
        point_ids=ids,
        image_points_px=np.asarray([np.mean(grouped[point_id], axis=0) for point_id in ids]),
        object_points_m=np.asarray([geometry[point_id] for point_id in ids]),
        metadata={**template.metadata, "purpose": "counterfactual_final_corner_mean"},
    )


def _variants() -> tuple[_Variant, ...]:
    return (
        _Variant(
            "focal_minus_1pct",
            "intrinsics",
            "solve with fx and fy reduced by 1 percent",
            intrinsics=lambda value: replace(value, fx=value.fx * 0.99, fy=value.fy * 0.99),
        ),
        _Variant(
            "focal_plus_1pct",
            "intrinsics",
            "solve with fx and fy increased by 1 percent",
            intrinsics=lambda value: replace(value, fx=value.fx * 1.01, fy=value.fy * 1.01),
        ),
        _Variant(
            "principal_plus_5px",
            "intrinsics",
            "solve with principal point shifted by +5 px in both axes",
            intrinsics=lambda value: replace(value, cx=value.cx + 5.0, cy=value.cy + 5.0),
        ),
        _Variant(
            "brown_k1_plus_0p005",
            "distortion",
            "solve with an added OpenCV Brown k1 coefficient of 0.005",
            intrinsics=_distortion_k1_variant,
        ),
        _Variant(
            "brown_p1_plus_0p002",
            "distortion",
            "solve with an added OpenCV Brown tangential p1 coefficient of 0.002",
            intrinsics=_distortion_p1_variant,
        ),
        _Variant(
            "target_scale_minus_1pct",
            "target_geometry",
            "solve with target X/Y scale reduced by 1 percent",
            geometry=lambda value: _scaled_geometry(value, 0.99),
        ),
        _Variant(
            "target_scale_plus_1pct",
            "target_geometry",
            "solve with target X/Y scale increased by 1 percent",
            geometry=lambda value: _scaled_geometry(value, 1.01),
        ),
        _Variant(
            "board_quadratic_xy_1mm",
            "target_geometry",
            "solve with a coherent planar quadratic coordinate error up to 1 mm",
            geometry=_quadratic_geometry,
        ),
        _Variant(
            "combined_focal_plus_0p5pct_scale_minus_0p5pct",
            "combined_mild",
            "solve with focal +0.5 percent and target scale -0.5 percent",
            intrinsics=lambda value: replace(value, fx=value.fx * 1.005, fy=value.fy * 1.005),
            geometry=lambda value: _scaled_geometry(value, 0.995),
        ),
    )


def _distortion_with_delta(
    value: CameraIntrinsics, coefficient_index: int, delta: float
) -> CameraIntrinsics:
    coefficients = list(value.distortion_coeffs)
    if len(coefficients) != 5:
        coefficients = [0.0] * 5
    coefficients[coefficient_index] += delta
    return replace(
        value,
        distortion_model="brown-conrady",
        distortion_coeffs=tuple(coefficients),
    )


def _distortion_k1_variant(value: CameraIntrinsics) -> CameraIntrinsics:
    return _distortion_with_delta(value, 0, 0.005)


def _distortion_p1_variant(value: CameraIntrinsics) -> CameraIntrinsics:
    return _distortion_with_delta(value, 2, 0.002)


def _scaled_geometry(geometry: Mapping[int, FloatArray], factor: float) -> dict[int, FloatArray]:
    return {
        point_id: np.asarray([point[0] * factor, point[1] * factor, point[2]])
        for point_id, point in geometry.items()
    }


def _quadratic_geometry(geometry: Mapping[int, FloatArray]) -> dict[int, FloatArray]:
    ids = tuple(sorted(geometry))
    points = np.asarray([geometry[point_id] for point_id in ids], dtype=np.float64)
    centered = points[:, :2] - np.mean(points[:, :2], axis=0)
    scale = np.max(np.abs(centered), axis=0)
    normalized = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)
    warped = points.copy()
    warped[:, 0] += 0.001 * (np.square(normalized[:, 0]) - 0.5)
    warped[:, 1] += 0.001 * normalized[:, 0] * normalized[:, 1]
    return {point_id: warped[index] for index, point_id in enumerate(ids)}
