"""Robust capture-level aggregation for the uncertainty-validated target policy."""

from __future__ import annotations

import math

import numpy as np

from camera_rig.calibration.pose import (
    PlanarPoseEstimator,
    UncertaintyValidatedThresholds,
)
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.targets.observation import TargetObservation


def pose_frame_diagnostic(
    observation: TargetObservation,
    intrinsics: CameraIntrinsics,
    *,
    estimator: PlanarPoseEstimator | None = None,
) -> dict[str, object]:
    """Solve one observation and preserve precise fail-closed reasons."""
    if not observation.quality.passed:
        return {
            "solve_success": False,
            "observable": False,
            "estimate": None,
            "observability": None,
            "failure_reasons": ["DETECTION_INTEGRITY_FAILED"],
        }
    try:
        estimate = (estimator or PlanarPoseEstimator()).estimate(observation, intrinsics)
    except ContractError as error:
        return {
            "solve_success": False,
            "observable": False,
            "estimate": None,
            "observability": None,
            "failure_reasons": [f"POSE_SOLVE_FAILED: {error}"],
        }
    observability = estimate.observability
    return {
        "solve_success": True,
        "observable": observability.passed,
        "estimate": estimate.to_dict(),
        "observability": observability.to_dict(),
        "failure_reasons": list(observability.failure_reasons),
    }


def aggregate_pose_diagnostics(diagnostics: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate solve/observable ratios and robust uncertainty distributions."""
    if not diagnostics:
        raise ArtifactError("pose-observability aggregation requires at least one frame")
    solved = [item for item in diagnostics if item.get("solve_success") is True]
    observable = [item for item in diagnostics if item.get("observable") is True]
    translations: list[float] = []
    rotations: list[float] = []
    conditions: list[float] = []
    ambiguous_count = 0
    for item in solved:
        metrics = _mapping(item.get("observability"), "observability")
        translation = metrics.get("translation_worst_axis_std_mm")
        if translation is not None:
            translations.append(_finite_number(translation, "translation_worst_axis_std_mm"))
        rotation = metrics.get("rotation_worst_axis_std_deg")
        if rotation is not None:
            rotations.append(_finite_number(rotation, "rotation_worst_axis_std_deg"))
        condition = metrics.get("scaled_condition_number")
        if condition is not None:
            conditions.append(_finite_number(condition, "scaled_condition_number"))
        ambiguity = _mapping(metrics.get("candidate_ambiguity"), "candidate_ambiguity")
        ambiguous_count += int(ambiguity.get("ambiguous") is True)
    frame_count = len(diagnostics)
    solved_count = len(solved)
    return {
        "frame_count": frame_count,
        "solve_success_count": solved_count,
        "solve_success_ratio": solved_count / frame_count,
        "observable_frame_count": len(observable),
        "observable_frame_ratio": len(observable) / frame_count,
        "translation_worst_axis_std_mm": _distribution(translations),
        "rotation_worst_axis_std_deg": _distribution(rotations),
        "scaled_condition_number": _distribution(conditions),
        "ambiguous_frame_count": ambiguous_count,
        "ambiguous_frame_ratio": ambiguous_count / solved_count if solved_count else 1.0,
    }


def uncertainty_capture_acceptance(
    *,
    aggregate: dict[str, object],
    frame_count: int,
    thresholds: UncertaintyValidatedThresholds | None = None,
    minimum_frames: int = 60,
    minimum_detection_success_ratio: float = 0.95,
    minimum_median_corners: float = 12.0,
    minimum_median_corner_fraction: float = 0.50,
    maximum_median_jitter_px: float = 0.5,
    maximum_p95_jitter_px: float = 1.0,
) -> dict[str, object]:
    """Apply robust capture gates without a coverage or image-span hard dependency."""
    release = thresholds or UncertaintyValidatedThresholds()
    corners = _mapping(aggregate.get("detected_charuco_corner_count"), "corner count")
    fractions = _mapping(aggregate.get("corner_fraction"), "corner fraction")
    jitter = _mapping(aggregate.get("temporal_jitter"), "temporal jitter")
    pose = _mapping(aggregate.get("pose_observability"), "pose observability")
    translation = _mapping(pose.get("translation_worst_axis_std_mm"), "translation uncertainty")
    rotation = _mapping(pose.get("rotation_worst_axis_std_deg"), "rotation uncertainty")
    condition = _mapping(pose.get("scaled_condition_number"), "condition number")
    checks = {
        "frame_count_is_60": frame_count == minimum_frames,
        "detection_success_ratio_at_least_0_95": (
            _finite_number(aggregate.get("success_ratio"), "success_ratio")
            >= minimum_detection_success_ratio
        ),
        "median_corners_at_least_threshold": (
            _finite_number(corners.get("median"), "corner median") >= minimum_median_corners
        ),
        "median_corner_fraction_at_least_threshold": (
            _finite_number(fractions.get("median"), "fraction median")
            >= minimum_median_corner_fraction
        ),
        "median_jitter_at_most_0_5_px": (
            _finite_number(jitter.get("median_radial_std_px"), "median jitter")
            <= maximum_median_jitter_px
        ),
        "p95_jitter_at_most_1_0_px": (
            _finite_number(jitter.get("p95_radial_std_px"), "p95 jitter") <= maximum_p95_jitter_px
        ),
        "temporal_jitter_has_eligible_corners": (
            _finite_number(jitter.get("eligible_corner_count"), "eligible corner count") >= 1
        ),
        "pose_solve_ratio_at_least_threshold": (
            _finite_number(pose.get("solve_success_ratio"), "pose solve ratio")
            >= release.minimum_pose_solve_ratio
        ),
        "observable_frame_ratio_at_least_threshold": (
            _finite_number(pose.get("observable_frame_ratio"), "observable ratio")
            >= release.minimum_observable_frame_ratio
        ),
        "p95_translation_uncertainty_at_most_threshold": (
            _optional_p95(translation) <= release.maximum_frame_translation_worst_std_mm
        ),
        "p95_rotation_uncertainty_at_most_threshold": (
            _optional_p95(rotation) <= release.maximum_frame_rotation_worst_std_deg
        ),
        "p95_condition_number_at_most_threshold": (
            _optional_p95(condition) <= release.maximum_scaled_condition_number
        ),
        "ambiguous_frame_ratio_at_most_threshold": (
            _finite_number(pose.get("ambiguous_frame_ratio"), "ambiguous ratio")
            <= release.maximum_ambiguous_frame_ratio
        ),
    }
    coverage = _mapping(aggregate.get("coverage_ratio"), "coverage")
    return {
        "passed": all(checks.values()),
        "policy": "uncertainty_validated",
        "thresholds": {
            "frame_count": minimum_frames,
            "detection_success_ratio": minimum_detection_success_ratio,
            "median_charuco_corners": minimum_median_corners,
            "median_corner_fraction": minimum_median_corner_fraction,
            "median_jitter_px": maximum_median_jitter_px,
            "p95_jitter_px": maximum_p95_jitter_px,
            **release.to_dict(),
        },
        "checks": checks,
        "coverage": {
            "observed_median": _finite_number(coverage.get("median"), "coverage median"),
            "recommended": 0.05,
            "hard_gate": False,
        },
        "recommendations": {
            "median_coverage_at_least_0_05": (
                _finite_number(coverage.get("median"), "coverage median") >= 0.05
            )
        },
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ArtifactError("pose-observability distribution must be finite")
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactError(f"target pose aggregate {name} must be an object")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactError(f"target pose aggregate {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactError(f"target pose aggregate {name} must be finite")
    return result


def _optional_p95(distribution: dict[str, object]) -> float:
    value = distribution.get("p95")
    return _finite_number(value, "distribution p95") if value is not None else math.inf
