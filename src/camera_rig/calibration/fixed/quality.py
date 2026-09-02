"""Fail-closed numerical quality gates for fixed-camera calibration."""

from __future__ import annotations

from camera_rig.calibration.fixed.config import FixedSolverThresholds
from camera_rig.calibration.fixed.viability import evaluate_fixed_pose_final_reprojection
from camera_rig.calibration.pose import UncertaintyValidatedThresholds
from camera_rig.core.quality import QualityReport


def evaluate_fixed_calibration_quality(
    *,
    thresholds: FixedSolverThresholds,
    frame_count: int,
    accepted_frames: int,
    global_reprojection: dict[str, object],
    pose_repeatability: dict[str, object],
    split_half: dict[str, object],
    native_depth_sanity: dict[str, object],
    pose_policy: str = "legacy_strict",
    final_pose_observability: dict[str, object] | None = None,
    observable_frame_ratio: float | None = None,
    ambiguous_frame_ratio: float | None = None,
    require_native_depth_pass: bool = False,
) -> QualityReport:
    """Evaluate every persisted hard gate without silently skipping unavailable evidence."""
    accepted_ratio = accepted_frames / frame_count if frame_count else 0.0
    translation = _mapping(pose_repeatability.get("translation_mm"))
    rotation = _mapping(pose_repeatability.get("rotation_deg"))
    depth_status = native_depth_sanity.get("status")
    global_rmse = _number(global_reprojection.get("rmse_px"))
    global_p95 = _number(global_reprojection.get("p95_px"))
    translation_p95 = _number(translation.get("p95"))
    rotation_p95 = _number(rotation.get("p95"))
    split_translation = _number(split_half.get("translation_delta_mm"))
    split_rotation = _number(split_half.get("rotation_delta_deg"))
    reprojection_decision = evaluate_fixed_pose_final_reprojection(
        global_reprojection=global_reprojection,
        thresholds=thresholds,
        pose_policy=pose_policy,
    )
    reprojection_checks = _mapping(reprojection_decision.get("checks"))
    checks: dict[str, bool] = {
        "minimum_accepted_frames": accepted_frames >= thresholds.minimum_accepted_frames,
        "minimum_accepted_ratio": accepted_ratio >= thresholds.minimum_accepted_ratio,
    }
    release = None
    if pose_policy == "uncertainty_validated":
        release = UncertaintyValidatedThresholds()
        checks.update(
            {
                "gross_global_reprojection_rmse": (
                    reprojection_checks.get("rmse_within_applied_threshold") is True
                ),
                "gross_global_reprojection_p95": (
                    reprojection_checks.get("p95_within_applied_threshold") is True
                ),
            }
        )
    else:
        # Preserve the historical order of legacy failure reasons in persisted artifacts.
        checks.update(
            {
                "global_reprojection_rmse": (
                    reprojection_checks.get("rmse_within_applied_threshold") is True
                ),
                "global_reprojection_p95": (
                    reprojection_checks.get("p95_within_applied_threshold") is True
                ),
            }
        )
    checks.update(
        {
            "pose_translation_p95": translation_p95 is not None
            and translation_p95 <= thresholds.maximum_pose_translation_p95_mm,
            "pose_rotation_p95": rotation_p95 is not None
            and rotation_p95 <= thresholds.maximum_pose_rotation_p95_deg,
            "split_translation_delta": split_translation is not None
            and split_translation <= thresholds.maximum_split_translation_delta_mm,
            "split_rotation_delta": split_rotation is not None
            and split_rotation <= thresholds.maximum_split_rotation_delta_deg,
            "native_depth_sanity": (
                depth_status == "PASS"
                if require_native_depth_pass
                else depth_status in {"PASS", "SKIPPED_WITH_WARNING"}
            ),
        }
    )
    if pose_policy == "uncertainty_validated":
        assert release is not None
        final = _mapping(final_pose_observability)
        final_failures = final.get("failure_reasons")
        final_ambiguity = _mapping(final.get("candidate_ambiguity"))
        final_valid_candidate_count = _integer(final_ambiguity.get("valid_candidate_count"))
        checks.update(
            {
                "observable_frame_ratio": observable_frame_ratio is not None
                and observable_frame_ratio >= release.minimum_observable_frame_ratio,
                "ambiguous_frame_ratio": ambiguous_frame_ratio is not None
                and ambiguous_frame_ratio <= release.maximum_ambiguous_frame_ratio,
                "final_pose_observability": final.get("passed") is True,
                "final_pose_full_rank": final.get("effective_rank") == 6,
                "final_pose_translation_uncertainty": (
                    "POSE_TRANSLATION_UNCERTAINTY_EXCEEDED" not in final_failures
                    if isinstance(final_failures, list)
                    else False
                ),
                "final_pose_rotation_uncertainty": (
                    "POSE_ROTATION_UNCERTAINTY_EXCEEDED" not in final_failures
                    if isinstance(final_failures, list)
                    else False
                ),
                "final_pose_condition_number": (
                    "POSE_CONDITION_NUMBER_EXCEEDED" not in final_failures
                    if isinstance(final_failures, list)
                    else False
                ),
                "final_pose_unambiguous": (
                    final_valid_candidate_count is not None
                    and final_valid_candidate_count >= 1
                    and "POSE_AMBIGUOUS" not in final_failures
                    if isinstance(final_failures, list)
                    else False
                ),
            }
        )
    failure_reason_list: list[str] = []
    for name, passed in checks.items():
        if passed:
            continue
        if name == "gross_global_reprojection_rmse":
            failure_reason_list.append("GROSS_FINAL_REPROJECTION_RMSE_EXCEEDED")
        elif name == "gross_global_reprojection_p95":
            failure_reason_list.append("GROSS_FINAL_REPROJECTION_P95_EXCEEDED")
        else:
            failure_reason_list.append(name)
    if release is not None:
        if observable_frame_ratio is None or (
            observable_frame_ratio < release.minimum_observable_frame_ratio
        ):
            failure_reason_list.append("POSE_OBSERVABLE_FRAME_RATIO_BELOW_THRESHOLD")
        if ambiguous_frame_ratio is None or (
            ambiguous_frame_ratio > release.maximum_ambiguous_frame_ratio
        ):
            failure_reason_list.append("POSE_AMBIGUOUS_FRAME_RATIO_EXCEEDED")
        final_reasons = final.get("failure_reasons")
        if isinstance(final_reasons, list):
            failure_reason_list.extend(
                reason for reason in final_reasons if isinstance(reason, str)
            )
    failure_reasons = tuple(dict.fromkeys(failure_reason_list))
    warnings: tuple[str, ...] = ()
    if depth_status == "SKIPPED_WITH_WARNING":
        warning = native_depth_sanity.get("warning", native_depth_sanity.get("reason"))
        warning_text = (
            warning if isinstance(warning, str) and warning.strip() else "native depth skipped"
        )
        warnings = (warning_text,)
    metrics: dict[str, object] = {
        "frame_count": frame_count,
        "accepted_frames": accepted_frames,
        "accepted_ratio": accepted_ratio,
        "global_reprojection_rmse_px": global_rmse,
        "global_reprojection_p95_px": global_p95,
        "reprojection_decision": reprojection_decision,
        "pose_translation_p95_mm": translation_p95,
        "pose_rotation_p95_deg": rotation_p95,
        "split_translation_delta_mm": split_translation,
        "split_rotation_delta_deg": split_rotation,
        "native_depth_status": depth_status,
        "require_native_depth_pass": require_native_depth_pass,
        "checks": checks,
    }
    persisted_thresholds = thresholds.to_dict()
    if release is not None:
        metrics.update(
            {
                "pose_policy": pose_policy,
                "observable_frame_ratio": observable_frame_ratio,
                "ambiguous_frame_ratio": ambiguous_frame_ratio,
                "final_pose_observability": final_pose_observability,
            }
        )
        persisted_thresholds = {**persisted_thresholds, "pose_observability": release.to_dict()}
    return QualityReport(
        passed=not failure_reasons,
        metrics=metrics,
        thresholds=persisted_thresholds,
        warnings=warnings,
        failure_reasons=failure_reasons,
    )


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
