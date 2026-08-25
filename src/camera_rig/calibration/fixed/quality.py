"""Fail-closed numerical quality gates for fixed-camera calibration."""

from __future__ import annotations

from camera_rig.calibration.fixed.config import FixedSolverThresholds
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
    checks = {
        "minimum_accepted_frames": accepted_frames >= thresholds.minimum_accepted_frames,
        "minimum_accepted_ratio": accepted_ratio >= thresholds.minimum_accepted_ratio,
        "global_reprojection_rmse": global_rmse is not None
        and global_rmse <= thresholds.maximum_frame_rmse_px,
        "global_reprojection_p95": global_p95 is not None
        and global_p95 <= thresholds.maximum_frame_p95_px,
        "pose_translation_p95": translation_p95 is not None
        and translation_p95 <= thresholds.maximum_pose_translation_p95_mm,
        "pose_rotation_p95": rotation_p95 is not None
        and rotation_p95 <= thresholds.maximum_pose_rotation_p95_deg,
        "split_translation_delta": split_translation is not None
        and split_translation <= thresholds.maximum_split_translation_delta_mm,
        "split_rotation_delta": split_rotation is not None
        and split_rotation <= thresholds.maximum_split_rotation_delta_deg,
        "native_depth_sanity": depth_status in {"PASS", "SKIPPED_WITH_WARNING"},
    }
    failure_reasons = tuple(name for name, passed in checks.items() if not passed)
    warnings: tuple[str, ...] = ()
    if depth_status == "SKIPPED_WITH_WARNING":
        warning = native_depth_sanity.get("warning", native_depth_sanity.get("reason"))
        warning_text = (
            warning if isinstance(warning, str) and warning.strip() else "native depth skipped"
        )
        warnings = (warning_text,)
    return QualityReport(
        passed=not failure_reasons,
        metrics={
            "frame_count": frame_count,
            "accepted_frames": accepted_frames,
            "accepted_ratio": accepted_ratio,
            "global_reprojection_rmse_px": global_rmse,
            "global_reprojection_p95_px": global_p95,
            "pose_translation_p95_mm": translation_p95,
            "pose_rotation_p95_deg": rotation_p95,
            "split_translation_delta_mm": split_translation,
            "split_rotation_delta_deg": split_rotation,
            "native_depth_status": depth_status,
            "checks": checks,
        },
        thresholds=thresholds.to_dict(),
        warnings=warnings,
        failure_reasons=failure_reasons,
    )


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
