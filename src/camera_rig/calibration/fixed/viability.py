"""Canonical reprojection and observability decisions for fixed-pose calibration."""

from __future__ import annotations

from typing import Literal

from camera_rig.calibration.fixed.config import FixedCalibrationConfig, FixedSolverThresholds
from camera_rig.calibration.pose import PlanarPoseEstimate, UncertaintyValidatedThresholds
from camera_rig.targets.observation import TargetObservation

PosePolicy = Literal["legacy_strict", "pose_validated", "uncertainty_validated"]


def reprojection_policy_name(pose_policy: str) -> str:
    """Return the persisted reprojection policy selected by a pose policy."""
    if pose_policy == "uncertainty_validated":
        return "uncertainty_gross_model_consistency"
    return "legacy_precision"


def evaluate_fixed_pose_frame_viability(
    *,
    frame_index: int,
    detection_success: bool,
    observation: TargetObservation,
    estimate: PlanarPoseEstimate | None,
    config: FixedCalibrationConfig,
    pose_policy: str,
    pose_solve_error: str | None = None,
    uncertainty_thresholds: UncertaintyValidatedThresholds | None = None,
) -> dict[str, object]:
    """Evaluate one solved or rejected frame with the canonical fixed-pose gates."""
    uncertainty = pose_policy == "uncertainty_validated"
    release = uncertainty_thresholds or UncertaintyValidatedThresholds()
    reasons: list[str] = []
    checks: dict[str, bool] = {}
    corner_count = len(observation.point_ids)

    checks["detection_integrity"] = detection_success
    if not detection_success:
        reasons.append("DETECTION_INTEGRITY_FAILED" if uncertainty else "target_detection_failed")
    checks["minimum_corners"] = corner_count >= config.solver.minimum_corners_per_frame
    if not checks["minimum_corners"]:
        reasons.append("INSUFFICIENT_CORNERS" if uncertainty else "insufficient_corners")

    if estimate is None and not reasons:
        checks["pose_solve"] = False
        detail = pose_solve_error or "pose estimator returned no result"
        reasons.append(
            f"POSE_SOLVE_FAILED: {detail}" if uncertainty else f"pose_solve_failed: {detail}"
        )
    else:
        checks["pose_solve"] = estimate is not None

    if uncertainty:
        applied_rmse = release.maximum_gross_frame_rmse_px
        applied_p95 = release.maximum_gross_frame_p95_px
        rmse_reason = "GROSS_REPROJECTION_RMSE_EXCEEDED"
        p95_reason = "GROSS_REPROJECTION_P95_EXCEEDED"
    else:
        applied_rmse = config.solver.maximum_frame_rmse_px
        applied_p95 = config.solver.maximum_frame_p95_px
        rmse_reason = "frame_reprojection_rmse_exceeded"
        p95_reason = "frame_reprojection_p95_exceeded"

    rmse: float | None = None
    p95: float | None = None
    observability: dict[str, object] | None = None
    if estimate is not None:
        rmse = estimate.reprojection.rmse_px
        p95 = estimate.reprojection.p95_px
        checks["reprojection_rmse"] = rmse <= applied_rmse
        checks["reprojection_p95"] = p95 <= applied_p95
        if not checks["reprojection_rmse"]:
            reasons.append(rmse_reason)
        if not checks["reprojection_p95"]:
            reasons.append(p95_reason)
        if uncertainty:
            observability = estimate.observability.to_dict()
            checks["pose_observability"] = estimate.observability.passed
            reasons.extend(estimate.observability.failure_reasons)
    else:
        checks["reprojection_rmse"] = False
        checks["reprojection_p95"] = False
        if uncertainty:
            checks["pose_observability"] = False

    policy = reprojection_policy_name(pose_policy)
    return {
        "frame_index": frame_index,
        "accepted": not reasons,
        "failure_reasons": reasons,
        "reprojection_decision": {
            "policy": policy,
            "checks": {
                "rmse_within_applied_threshold": checks["reprojection_rmse"],
                "p95_within_applied_threshold": checks["reprojection_p95"],
            },
            "metrics": {"rmse_px": rmse, "p95_px": p95},
            "applied_thresholds": {
                "maximum_frame_rmse_px": applied_rmse,
                "maximum_frame_p95_px": applied_p95,
            },
            "legacy_precision_thresholds": {
                "maximum_frame_rmse_px": config.solver.maximum_frame_rmse_px,
                "maximum_frame_p95_px": config.solver.maximum_frame_p95_px,
            },
        },
        "observability_decision": {
            "policy": pose_policy,
            "evaluated": uncertainty and estimate is not None,
            "passed": checks.get("pose_observability"),
            "metrics": observability,
        },
        "checks": checks,
    }


def evaluate_fixed_pose_final_reprojection(
    *,
    global_reprojection: dict[str, object],
    thresholds: FixedSolverThresholds,
    pose_policy: str,
    uncertainty_thresholds: UncertaintyValidatedThresholds | None = None,
) -> dict[str, object]:
    """Evaluate the shared-pose reprojection with policy-specific thresholds."""
    release = uncertainty_thresholds or UncertaintyValidatedThresholds()
    rmse = _number(global_reprojection.get("rmse_px"))
    p95 = _number(global_reprojection.get("p95_px"))
    if pose_policy == "uncertainty_validated":
        maximum_rmse = release.maximum_gross_final_rmse_px
        maximum_p95 = release.maximum_gross_final_p95_px
    else:
        maximum_rmse = thresholds.maximum_frame_rmse_px
        maximum_p95 = thresholds.maximum_frame_p95_px
    checks = {
        "rmse_within_applied_threshold": rmse is not None and rmse <= maximum_rmse,
        "p95_within_applied_threshold": p95 is not None and p95 <= maximum_p95,
    }
    return {
        "policy": reprojection_policy_name(pose_policy),
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {"rmse_px": rmse, "p95_px": p95},
        "applied_thresholds": {
            "maximum_final_rmse_px": maximum_rmse,
            "maximum_final_p95_px": maximum_p95,
        },
        "legacy_precision_thresholds": {
            "maximum_frame_rmse_px": thresholds.maximum_frame_rmse_px,
            "maximum_frame_p95_px": thresholds.maximum_frame_p95_px,
        },
    }


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
