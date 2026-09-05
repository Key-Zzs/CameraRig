"""Independent native-depth diagnostic for a solved fixed target pose."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.hashing import sha256_bytes
from camera_rig.artifacts.io import (
    JsonValue,
    atomic_write_json,
    deterministic_json_bytes,
    load_json,
)
from camera_rig.calibration.pose import project_points_px
from camera_rig.calibration.pose.camera_model import to_opencv_camera_model
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.core.errors import ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget

METRIC_DEPTH_RECEIPT_SCHEMA_VERSION = "camera-rig.metric-depth-integrity.v1"
_REGIONS = {"top_left", "top_right", "bottom_left", "bottom_right"}
_EVALUATION_CHECKS = {
    "valid_samples_at_least_minimum",
    "valid_frames_at_least_minimum",
    "valid_sample_ratio_at_least_minimum",
    "all_depth_values_finite",
    "all_board_regions_have_support",
    "passing_frames_at_least_minimum",
    "passing_frame_ratio_at_least_minimum",
    "all_evaluated_frames_pass_geometry",
    "median_absolute_error_within_limit",
    "p95_absolute_error_within_limit",
    "plane_offset_within_limit",
    "plane_normal_error_within_limit",
    "metric_scale_ratio_within_limit",
    "worst_frame_plane_offset_within_limit",
    "worst_frame_plane_normal_error_within_limit",
    "worst_frame_metric_scale_within_limit",
}
_FRAME_CHECKS = {
    "valid_support_at_least_minimum",
    "all_board_regions_have_support",
    "median_absolute_error_within_limit",
    "p95_absolute_error_within_limit",
    "plane_offset_within_limit",
    "plane_normal_error_within_limit",
    "metric_scale_ratio_within_limit",
}


def evaluate_native_depth_sanity(
    *,
    target: ResolvedCharucoTarget,
    calibration: FactoryCalibration,
    T_detection_from_target: RigidTransform,
    detection_stream: str,
    frames: Sequence[CameraFrame],
    frame_indices: Sequence[int],
    window_radius: int = 2,
    minimum_valid_samples: int = 10,
    maximum_median_error_mm: float = 20.0,
    maximum_p95_error_mm: float = 40.0,
    minimum_valid_frames: int = 1,
    minimum_valid_sample_ratio: float = 0.1,
    maximum_plane_offset_mm: float = 20.0,
    maximum_plane_normal_error_deg: float = 3.0,
    maximum_scale_ratio_error: float = 0.03,
    minimum_region_valid_samples: int = 1,
    minimum_frame_valid_samples: int = 10,
    minimum_passing_frames: int = 1,
    minimum_passing_frame_ratio: float = 1.0,
    threshold_policy: dict[str, object] | None = None,
    fail_closed: bool = False,
) -> dict[str, object]:
    """Compare predicted target-plane depth with raw native depth without fitting pose."""
    if len(set(frame_indices)) != len(frame_indices):
        raise ContractError("native-depth frame indices must be unique")
    _validate_threshold_arguments(
        window_radius=window_radius,
        minimum_valid_samples=minimum_valid_samples,
        minimum_valid_frames=minimum_valid_frames,
        minimum_region_valid_samples=minimum_region_valid_samples,
        minimum_frame_valid_samples=minimum_frame_valid_samples,
        minimum_passing_frames=minimum_passing_frames,
        minimum_valid_sample_ratio=minimum_valid_sample_ratio,
        minimum_passing_frame_ratio=minimum_passing_frame_ratio,
        maximum_median_error_mm=maximum_median_error_mm,
        maximum_p95_error_mm=maximum_p95_error_mm,
        maximum_plane_offset_mm=maximum_plane_offset_mm,
        maximum_plane_normal_error_deg=maximum_plane_normal_error_deg,
        maximum_scale_ratio_error=maximum_scale_ratio_error,
    )
    detection = calibration.intrinsics.get(detection_stream)
    depth = calibration.intrinsics.get("depth")
    if detection is None or depth is None:
        return _unavailable(
            "detection or native-depth intrinsics are unavailable", fail_closed=fail_closed
        )
    if T_detection_from_target.target_frame != detection.frame:
        raise ContractError("solved pose does not target the configured detection frame")
    graph = TransformGraph()
    for transform in calibration.internal_transforms:
        graph.add(transform)
    try:
        T_depth_from_detection = graph.resolve(detection.frame, depth.frame)
        T_depth_from_target = T_depth_from_detection.compose(T_detection_from_target)
        samples_target = _sample_target_plane(target)
        pixels = project_points_px(samples_target, T_depth_from_target, depth)
    except ContractError as error:
        return _unavailable(
            f"native-depth projection is unsupported: {error}", fail_closed=fail_closed
        )
    predicted_points = T_depth_from_target.transform_points(samples_target)
    predicted_depth = predicted_points[:, 2]
    predicted_normal = T_depth_from_target.matrix[:3, :3] @ np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float64
    )
    predicted_normal /= np.linalg.norm(predicted_normal)
    predicted_plane_distance_m = abs(float(np.dot(predicted_normal, predicted_points[0])))
    errors_mm: list[float] = []
    point_to_plane_mm: list[float] = []
    frame_statistics: list[dict[str, object]] = []
    region_counts = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
    requested = 0
    non_finite_depth_values = 0
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(frames):
            raise ContractError("native-depth frame index is outside captured frames")
        stream = frames[frame_index].depth
        if stream is None:
            requested += len(samples_target)
            frame_statistics.append(_failed_frame(frame_index, predicted_plane_distance_m))
            continue
        raw = np.asarray(stream.data)
        if raw.ndim != 2:
            raise ContractError("native depth image must be two-dimensional")
        frame_errors: list[float] = []
        frame_point_to_plane: list[float] = []
        frame_points: list[npt.NDArray[np.float64]] = []
        frame_regions = {name: 0 for name in _REGIONS}
        for target_point, pixel, expected_m in zip(
            samples_target, pixels, predicted_depth, strict=True
        ):
            requested += 1
            observed_units, invalid_non_finite = _window_median(raw, pixel, window_radius)
            non_finite_depth_values += invalid_non_finite
            if observed_units is None:
                continue
            observed_m = observed_units * calibration.depth_scale_m_per_unit
            if not math.isfinite(observed_m) or observed_m <= 0.0:
                non_finite_depth_values += 1
                continue
            error_mm = (observed_m - float(expected_m)) * 1000.0
            errors_mm.append(error_mm)
            frame_errors.append(error_mm)
            point = _deproject_depth_pixel(pixel, observed_m, depth)
            frame_points.append(point)
            signed_plane_mm = float(np.dot(predicted_normal, point - predicted_points[0])) * 1000.0
            point_to_plane_mm.append(signed_plane_mm)
            frame_point_to_plane.append(signed_plane_mm)
            region = _board_region(target_point, target)
            region_counts[region] += 1
            frame_regions[region] += 1
        frame_statistics.append(
            _evaluate_frame_geometry(
                frame_index=frame_index,
                frame_errors=frame_errors,
                frame_point_to_plane=frame_point_to_plane,
                frame_points=frame_points,
                frame_regions=frame_regions,
                predicted_normal=predicted_normal,
                predicted_plane_distance_m=predicted_plane_distance_m,
                minimum_frame_valid_samples=minimum_frame_valid_samples,
                minimum_region_valid_samples=minimum_region_valid_samples,
                maximum_median_error_mm=maximum_median_error_mm,
                maximum_p95_error_mm=maximum_p95_error_mm,
                maximum_plane_offset_mm=maximum_plane_offset_mm,
                maximum_plane_normal_error_deg=maximum_plane_normal_error_deg,
                maximum_scale_ratio_error=maximum_scale_ratio_error,
            )
        )
    valid_ratio = len(errors_mm) / requested if requested else 0.0
    absolute = np.abs(np.asarray(errors_mm, dtype=np.float64))
    median = float(np.median(absolute)) if len(absolute) else float("inf")
    p95 = float(np.percentile(absolute, 95)) if len(absolute) else float("inf")
    bias = float(np.median(np.asarray(errors_mm))) if errors_mm else float("inf")
    passed_frames = [item for item in frame_statistics if item.get("status") == "PASS"]
    plane_offsets = _frame_values(frame_statistics, "plane_offset_mm")
    normal_errors = _frame_values(frame_statistics, "plane_normal_error_deg")
    scale_ratios = _frame_values(frame_statistics, "distance_scale_ratio")
    plane_offset = float(np.median(plane_offsets)) if len(plane_offsets) else float("inf")
    normal_error = float(np.median(normal_errors)) if len(normal_errors) else float("inf")
    scale_ratio = float(np.median(scale_ratios)) if len(scale_ratios) else float("inf")
    passing_frame_ratio = len(passed_frames) / len(frame_statistics) if frame_statistics else 0.0
    worst_plane_offset = _finite_max_abs(plane_offsets)
    worst_normal_error = _finite_max_abs(normal_errors)
    worst_scale_error = _finite_max_abs(scale_ratios - 1.0)
    point_plane_absolute = np.abs(np.asarray(point_to_plane_mm, dtype=np.float64))
    checks = {
        "valid_samples_at_least_minimum": len(errors_mm) >= minimum_valid_samples,
        "valid_frames_at_least_minimum": len(frame_statistics) >= minimum_valid_frames,
        "valid_sample_ratio_at_least_minimum": valid_ratio >= minimum_valid_sample_ratio,
        "all_depth_values_finite": non_finite_depth_values == 0,
        "all_board_regions_have_support": all(
            value >= minimum_region_valid_samples for value in region_counts.values()
        ),
        "passing_frames_at_least_minimum": len(passed_frames) >= minimum_passing_frames,
        "passing_frame_ratio_at_least_minimum": passing_frame_ratio >= minimum_passing_frame_ratio,
        "all_evaluated_frames_pass_geometry": len(passed_frames) == len(frame_statistics),
        "median_absolute_error_within_limit": median <= maximum_median_error_mm,
        "p95_absolute_error_within_limit": p95 <= maximum_p95_error_mm,
        "plane_offset_within_limit": abs(plane_offset) <= maximum_plane_offset_mm,
        "plane_normal_error_within_limit": normal_error <= maximum_plane_normal_error_deg,
        "metric_scale_ratio_within_limit": abs(scale_ratio - 1.0) <= maximum_scale_ratio_error,
        "worst_frame_plane_offset_within_limit": worst_plane_offset <= maximum_plane_offset_mm,
        "worst_frame_plane_normal_error_within_limit": worst_normal_error
        <= maximum_plane_normal_error_deg,
        "worst_frame_metric_scale_within_limit": worst_scale_error <= maximum_scale_ratio_error,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sample_point_count": len(samples_target),
        "requested_samples": requested,
        "valid_samples": len(errors_mm),
        "valid_sample_ratio": valid_ratio,
        "valid_frame_count": len(frame_statistics),
        "passing_frame_count": len(passed_frames),
        "passing_frame_ratio": passing_frame_ratio,
        "non_finite_depth_values": non_finite_depth_values,
        "predicted_plane_distance_m": predicted_plane_distance_m,
        "measured_plane_distance_m": (
            predicted_plane_distance_m + plane_offset / 1000.0
            if math.isfinite(plane_offset)
            else float("inf")
        ),
        "median_absolute_error_mm": median,
        "p95_absolute_error_mm": p95,
        "signed_bias_mm": bias,
        "median_absolute_point_to_plane_error_mm": (
            float(np.median(point_plane_absolute)) if len(point_plane_absolute) else float("inf")
        ),
        "plane_offset_mm": plane_offset,
        "plane_normal_error_deg": normal_error,
        "distance_scale_ratio": scale_ratio,
        "worst_frame_plane_offset_mm": worst_plane_offset,
        "worst_frame_plane_normal_error_deg": worst_normal_error,
        "worst_frame_scale_ratio_error": worst_scale_error,
        "board_region_valid_support": region_counts,
        "per_frame": frame_statistics,
        "thresholds": {
            "minimum_valid_samples": minimum_valid_samples,
            "maximum_median_absolute_error_mm": maximum_median_error_mm,
            "maximum_p95_absolute_error_mm": maximum_p95_error_mm,
            "minimum_valid_frames": minimum_valid_frames,
            "minimum_region_valid_samples": minimum_region_valid_samples,
            "minimum_frame_valid_samples": minimum_frame_valid_samples,
            "minimum_passing_frames": minimum_passing_frames,
            "minimum_valid_sample_ratio": minimum_valid_sample_ratio,
            "minimum_passing_frame_ratio": minimum_passing_frame_ratio,
            "maximum_plane_offset_mm": maximum_plane_offset_mm,
            "maximum_plane_normal_error_deg": maximum_plane_normal_error_deg,
            "maximum_scale_ratio_error": maximum_scale_ratio_error,
            "window_size": window_radius * 2 + 1,
        },
        "threshold_policy": dict(
            threshold_policy
            or {
                "schema_version": "camera-rig.metric-depth-threshold-policy.generic.v1",
                "source": "explicit_function_arguments",
            }
        ),
        "checks": checks,
        "role": "independent metric-depth integrity gate; not used by pose optimization",
    }


def _sample_target_plane(target: ResolvedCharucoTarget) -> npt.NDArray[np.float64]:
    margin = target.square_length_m * 0.5
    x_values = np.linspace(margin, target.board_width_m - margin, target.squares_x)
    y_values = np.linspace(margin, target.board_height_m - margin, target.squares_y)
    return np.asarray([(x, y, 0.0) for y in y_values for x in x_values], dtype=np.float64)


def _window_median(
    raw: npt.NDArray[np.generic], pixel: npt.NDArray[np.float64], radius: int
) -> tuple[float | None, int]:
    x = round(float(pixel[0]))
    y = round(float(pixel[1]))
    if x < 0 or y < 0 or x >= raw.shape[1] or y >= raw.shape[0]:
        return None, 0
    x0, x1 = max(0, x - radius), min(raw.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(raw.shape[0], y + radius + 1)
    values = np.asarray(raw[y0:y1, x0:x1], dtype=np.float64)
    non_finite = int(np.count_nonzero(~np.isfinite(values)))
    valid = values[np.isfinite(values) & (values > 0)]
    return (float(np.median(valid)) if len(valid) else None), non_finite


def _unavailable(reason: str, *, fail_closed: bool) -> dict[str, object]:
    return {
        "status": "FAIL" if fail_closed else "SKIPPED_WITH_WARNING",
        "failure_reason" if fail_closed else "warning": reason,
        "role": "independent metric-depth integrity gate; not used by pose optimization",
    }


def _deproject_depth_pixel(
    pixel: npt.NDArray[np.float64], depth_m: float, intrinsics: object
) -> npt.NDArray[np.float64]:
    from camera_rig.core.intrinsics import CameraIntrinsics

    if not isinstance(intrinsics, CameraIntrinsics):
        raise ContractError("native-depth intrinsics are invalid")
    model = to_opencv_camera_model(intrinsics)
    cv2 = cv2_module()
    normalized = cv2.undistortPoints(
        np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
        model.camera_matrix,
        model.distortion_coeffs,
    ).reshape(2)
    return np.asarray([normalized[0] * depth_m, normalized[1] * depth_m, depth_m])


def _fit_plane(points: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], float]:
    if points.shape[0] < 3 or not np.isfinite(points).all():
        raise ContractError("metric-depth plane fit requires at least three finite points")
    retained = points
    for _iteration in range(3):
        centroid = np.median(retained, axis=0)
        _u, singular, vh = np.linalg.svd(retained - centroid, full_matrices=False)
        if len(singular) < 3 or singular[1] <= 1e-12:
            raise ContractError("metric-depth plane support is degenerate")
        normal = np.asarray(vh[-1], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        residuals = np.abs((points - centroid) @ normal)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_sigma = 1.4826 * mad
        keep = residuals <= median + max(3.0 * robust_sigma, 0.001)
        if int(np.count_nonzero(keep)) < 3 or np.array_equal(retained, points[keep]):
            retained = points[keep]
            break
        retained = points[keep]
    centroid = np.mean(retained, axis=0)
    _u, singular, vh = np.linalg.svd(retained - centroid, full_matrices=False)
    if len(singular) < 3 or singular[1] <= 1e-12:
        raise ContractError("metric-depth robust plane support is degenerate")
    normal = np.asarray(vh[-1], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return normal, abs(float(np.dot(normal, centroid)))


def _board_region(point: npt.NDArray[np.float64], target: ResolvedCharucoTarget) -> str:
    """Classify support in target-local coordinates, independent of image placement."""

    horizontal = "left" if float(point[0]) < target.board_width_m / 2.0 else "right"
    vertical = "bottom" if float(point[1]) < target.board_height_m / 2.0 else "top"
    return f"{vertical}_{horizontal}"


def _frame_values(values: list[dict[str, object]], key: str) -> npt.NDArray[np.float64]:
    numeric: list[float] = []
    for item in values:
        value = item.get(key)
        if not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value):
            numeric.append(float(value))
    return np.asarray(numeric, dtype=np.float64)


def _finite_max_abs(values: npt.NDArray[np.float64]) -> float:
    return float(np.max(np.abs(values))) if len(values) else float("inf")


def _failed_frame(frame_index: int, predicted_distance_m: float) -> dict[str, object]:
    checks = {name: False for name in _FRAME_CHECKS}
    return {
        "frame_index": frame_index,
        "status": "FAIL",
        "valid_depth_support": 0,
        "board_region_valid_support": {name: 0 for name in _REGIONS},
        "predicted_plane_distance_m": predicted_distance_m,
        "measured_plane_distance_m": None,
        "signed_depth_residual_mm": None,
        "median_absolute_depth_residual_mm": None,
        "p95_absolute_depth_residual_mm": None,
        "plane_offset_mm": None,
        "plane_normal_error_deg": None,
        "distance_scale_ratio": None,
        "depth_residuals_mm": [],
        "point_to_plane_residuals_mm": [],
        "observed_points_m": [],
        "predicted_plane_normal": [],
        "checks": checks,
    }


def _evaluate_frame_geometry(
    *,
    frame_index: int,
    frame_errors: list[float],
    frame_point_to_plane: list[float],
    frame_points: list[npt.NDArray[np.float64]],
    frame_regions: dict[str, int],
    predicted_normal: npt.NDArray[np.float64],
    predicted_plane_distance_m: float,
    minimum_frame_valid_samples: int,
    minimum_region_valid_samples: int,
    maximum_median_error_mm: float,
    maximum_p95_error_mm: float,
    maximum_plane_offset_mm: float,
    maximum_plane_normal_error_deg: float,
    maximum_scale_ratio_error: float,
) -> dict[str, object]:
    if len(frame_points) < 3:
        failed = _failed_frame(frame_index, predicted_plane_distance_m)
        failed["valid_depth_support"] = len(frame_points)
        failed["board_region_valid_support"] = frame_regions
        failed["depth_residuals_mm"] = frame_errors
        failed["point_to_plane_residuals_mm"] = frame_point_to_plane
        failed["observed_points_m"] = [point.tolist() for point in frame_points]
        failed["predicted_plane_normal"] = predicted_normal.tolist()
        return failed
    try:
        observed_normal, observed_distance_m = _fit_plane(
            np.asarray(frame_points, dtype=np.float64)
        )
    except ContractError:
        failed = _failed_frame(frame_index, predicted_plane_distance_m)
        failed["valid_depth_support"] = len(frame_points)
        failed["board_region_valid_support"] = frame_regions
        failed["depth_residuals_mm"] = frame_errors
        failed["point_to_plane_residuals_mm"] = frame_point_to_plane
        failed["observed_points_m"] = [point.tolist() for point in frame_points]
        failed["predicted_plane_normal"] = predicted_normal.tolist()
        return failed
    if float(np.dot(observed_normal, predicted_normal)) < 0.0:
        observed_normal = -observed_normal
    normal_error_deg = math.degrees(
        math.acos(float(np.clip(np.dot(observed_normal, predicted_normal), -1.0, 1.0)))
    )
    plane_offset_mm = (observed_distance_m - predicted_plane_distance_m) * 1000.0
    scale_ratio = (
        observed_distance_m / predicted_plane_distance_m
        if predicted_plane_distance_m > 0.0
        else float("inf")
    )
    absolute_frame = np.abs(np.asarray(frame_errors, dtype=np.float64))
    median = float(np.median(absolute_frame))
    p95 = float(np.percentile(absolute_frame, 95))
    checks = {
        "valid_support_at_least_minimum": len(frame_points) >= minimum_frame_valid_samples,
        "all_board_regions_have_support": all(
            value >= minimum_region_valid_samples for value in frame_regions.values()
        ),
        "median_absolute_error_within_limit": median <= maximum_median_error_mm,
        "p95_absolute_error_within_limit": p95 <= maximum_p95_error_mm,
        "plane_offset_within_limit": abs(plane_offset_mm) <= maximum_plane_offset_mm,
        "plane_normal_error_within_limit": normal_error_deg <= maximum_plane_normal_error_deg,
        "metric_scale_ratio_within_limit": abs(scale_ratio - 1.0) <= maximum_scale_ratio_error,
    }
    return {
        "frame_index": frame_index,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "valid_depth_support": len(frame_points),
        "board_region_valid_support": frame_regions,
        "predicted_plane_distance_m": predicted_plane_distance_m,
        "measured_plane_distance_m": observed_distance_m,
        "signed_depth_residual_mm": float(np.median(frame_errors)),
        "median_absolute_depth_residual_mm": median,
        "p95_absolute_depth_residual_mm": p95,
        "plane_offset_mm": plane_offset_mm,
        "plane_normal_error_deg": normal_error_deg,
        "distance_scale_ratio": scale_ratio,
        "depth_residuals_mm": frame_errors,
        "point_to_plane_residuals_mm": frame_point_to_plane,
        "observed_points_m": [point.tolist() for point in frame_points],
        "predicted_plane_normal": predicted_normal.tolist(),
        "checks": checks,
    }


def _validate_threshold_arguments(**values: int | float) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ContractError(f"metric-depth threshold {name} must be finite")
        if name.startswith("minimum_") and name.endswith(("samples", "frames")):
            if not isinstance(value, int) or value < 1:
                raise ContractError(f"metric-depth threshold {name} must be a positive integer")
        elif name.endswith("ratio"):
            if not 0.0 < float(value) <= 1.0:
                raise ContractError(f"metric-depth threshold {name} must be in (0,1]")
        elif float(value) <= 0.0:
            raise ContractError(f"metric-depth threshold {name} must be positive")


def build_metric_depth_receipt(
    *,
    evaluation: dict[str, object],
    camera_identity_sha256: str,
    target_identity_sha256: str,
    factory_calibration_sha256: str,
    capture_manifest_sha256: str,
) -> dict[str, object]:
    """Bind the independent metric-depth decision to its immutable inputs."""
    validate_native_depth_evaluation(
        evaluation, require_pass=True, require_fixed_bootstrap_policy=True
    )
    for name, digest in (
        ("camera_identity_sha256", camera_identity_sha256),
        ("target_identity_sha256", target_identity_sha256),
        ("factory_calibration_sha256", factory_calibration_sha256),
        ("capture_manifest_sha256", capture_manifest_sha256),
    ):
        _digest(digest, name)
    report: dict[str, object] = {
        "schema_version": METRIC_DEPTH_RECEIPT_SCHEMA_VERSION,
        "status": evaluation.get("status"),
        "camera_identity_sha256": camera_identity_sha256,
        "target_identity_sha256": target_identity_sha256,
        "factory_calibration_sha256": factory_calibration_sha256,
        "capture_manifest_sha256": capture_manifest_sha256,
        "evaluation": dict(evaluation),
        "role": "independent_metric_depth_integrity",
    }
    report["receipt_fingerprint"] = sha256_bytes(deterministic_json_bytes(report))
    return report


def write_metric_depth_receipt(path: str | Path, report: dict[str, object]) -> None:
    """Write a passed, hash-bound metric-depth receipt."""
    validate_metric_depth_receipt_data(report, require_pass=True)
    output = Path(path)
    if output.exists():
        raise ContractError("metric-depth receipt is immutable and already exists")
    atomic_write_json(output, report)


def load_metric_depth_receipt(path: str | Path, *, require_pass: bool = False) -> dict[str, object]:
    return validate_metric_depth_receipt_data(load_json(path), require_pass=require_pass)


def validate_metric_depth_receipt_data(
    value: JsonValue | dict[str, object], *, require_pass: bool = False
) -> dict[str, object]:
    required = {
        "schema_version",
        "status",
        "camera_identity_sha256",
        "target_identity_sha256",
        "factory_calibration_sha256",
        "capture_manifest_sha256",
        "evaluation",
        "role",
        "receipt_fingerprint",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ContractError("metric-depth receipt fields are incomplete")
    if value.get("schema_version") != METRIC_DEPTH_RECEIPT_SCHEMA_VERSION:
        raise ContractError("metric-depth receipt schema is invalid")
    if value.get("role") != "independent_metric_depth_integrity":
        raise ContractError("metric-depth receipt role is invalid")
    for name in (
        "camera_identity_sha256",
        "target_identity_sha256",
        "factory_calibration_sha256",
        "capture_manifest_sha256",
    ):
        digest = value.get(name)
        if not isinstance(digest, str):
            raise ContractError(f"metric-depth receipt {name} must be a digest")
        _digest(digest, name)
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ContractError("metric-depth receipt evaluation must be an object")
    validate_native_depth_evaluation(
        evaluation,
        require_pass=require_pass,
        require_fixed_bootstrap_policy=require_pass,
    )
    if value.get("status") != evaluation.get("status"):
        raise ContractError("metric-depth receipt status differs from evaluation")
    if require_pass and value.get("status") != "PASS":
        raise ContractError("metric-depth receipt is not passed")
    fingerprint = value.get("receipt_fingerprint")
    payload = {key: item for key, item in value.items() if key != "receipt_fingerprint"}
    if fingerprint != sha256_bytes(deterministic_json_bytes(payload)):
        raise ContractError("metric-depth receipt fingerprint differs")
    return dict(value)


def validate_native_depth_evaluation(
    evaluation: dict[str, object],
    *,
    require_pass: bool = False,
    require_fixed_bootstrap_policy: bool = False,
) -> dict[str, object]:
    required = {
        "status",
        "sample_point_count",
        "requested_samples",
        "valid_samples",
        "valid_sample_ratio",
        "valid_frame_count",
        "passing_frame_count",
        "passing_frame_ratio",
        "non_finite_depth_values",
        "predicted_plane_distance_m",
        "measured_plane_distance_m",
        "median_absolute_error_mm",
        "p95_absolute_error_mm",
        "signed_bias_mm",
        "median_absolute_point_to_plane_error_mm",
        "plane_offset_mm",
        "plane_normal_error_deg",
        "distance_scale_ratio",
        "worst_frame_plane_offset_mm",
        "worst_frame_plane_normal_error_deg",
        "worst_frame_scale_ratio_error",
        "board_region_valid_support",
        "per_frame",
        "thresholds",
        "threshold_policy",
        "checks",
        "role",
    }
    if set(evaluation) != required:
        raise ContractError("metric-depth evaluation fields are incomplete")
    if evaluation.get("role") != (
        "independent metric-depth integrity gate; not used by pose optimization"
    ):
        raise ContractError("metric-depth evaluation role is invalid")
    thresholds = evaluation.get("thresholds")
    threshold_policy = evaluation.get("threshold_policy")
    checks = evaluation.get("checks")
    frames = evaluation.get("per_frame")
    regions = evaluation.get("board_region_valid_support")
    if not isinstance(thresholds, dict) or not isinstance(checks, dict):
        raise ContractError("metric-depth thresholds/checks must be objects")
    if (
        not isinstance(threshold_policy, dict)
        or set(threshold_policy) != {"schema_version", "source"}
        or not all(isinstance(value, str) and value for value in threshold_policy.values())
    ):
        raise ContractError("metric-depth threshold policy is invalid")
    threshold_names = {
        "minimum_valid_samples",
        "maximum_median_absolute_error_mm",
        "maximum_p95_absolute_error_mm",
        "minimum_valid_frames",
        "minimum_region_valid_samples",
        "minimum_frame_valid_samples",
        "minimum_passing_frames",
        "minimum_valid_sample_ratio",
        "minimum_passing_frame_ratio",
        "maximum_plane_offset_mm",
        "maximum_plane_normal_error_deg",
        "maximum_scale_ratio_error",
        "window_size",
    }
    if set(thresholds) != threshold_names:
        raise ContractError("metric-depth threshold set is invalid")
    threshold_values = {name: thresholds[name] for name in threshold_names if name != "window_size"}
    _validate_threshold_arguments(**threshold_values)
    window_size = thresholds.get("window_size")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size < 3
        or window_size % 2 != 1
    ):
        raise ContractError("metric-depth window size must be an odd integer at least three")
    if require_fixed_bootstrap_policy:
        _validate_fixed_bootstrap_threshold_contract(thresholds, threshold_policy)
    if set(checks) != _EVALUATION_CHECKS or not all(
        isinstance(item, bool) for item in checks.values()
    ):
        raise ContractError("metric-depth check set is invalid")
    if not isinstance(frames, list) or not all(isinstance(item, dict) for item in frames):
        raise ContractError("metric-depth per-frame records are invalid")
    if not isinstance(regions, dict) or set(regions) != _REGIONS:
        raise ContractError("metric-depth board-region support is invalid")
    integer_names = (
        "sample_point_count",
        "requested_samples",
        "valid_samples",
        "valid_frame_count",
        "passing_frame_count",
        "non_finite_depth_values",
    )
    ints = {name: _nonnegative_integer(evaluation.get(name), name) for name in integer_names}
    if ints["sample_point_count"] < 1 or ints["requested_samples"] < 1:
        raise ContractError("metric-depth sample counts must be positive")
    if ints["valid_frame_count"] != len(frames):
        raise ContractError("metric-depth valid_frame_count differs from records")
    if ints["requested_samples"] != ints["sample_point_count"] * len(frames):
        raise ContractError("metric-depth requested sample count differs from frames")
    valid_ratio = _finite(evaluation.get("valid_sample_ratio"), "valid_sample_ratio")
    passing_ratio = _finite(evaluation.get("passing_frame_ratio"), "passing_frame_ratio")
    if not math.isclose(
        valid_ratio,
        ints["valid_samples"] / ints["requested_samples"],
        abs_tol=1e-12,
    ):
        raise ContractError("metric-depth valid sample ratio differs")
    passed_frames = 0
    total_frame_support = 0
    summed_regions = {name: 0 for name in _REGIONS}
    frame_offsets: list[float] = []
    frame_normals: list[float] = []
    frame_scales: list[float] = []
    frame_predicted_distances: list[float] = []
    all_depth_residuals: list[float] = []
    all_point_to_plane_residuals: list[float] = []
    for frame in frames:
        if set(frame) != {
            "frame_index",
            "status",
            "valid_depth_support",
            "board_region_valid_support",
            "predicted_plane_distance_m",
            "measured_plane_distance_m",
            "signed_depth_residual_mm",
            "median_absolute_depth_residual_mm",
            "p95_absolute_depth_residual_mm",
            "plane_offset_mm",
            "plane_normal_error_deg",
            "distance_scale_ratio",
            "depth_residuals_mm",
            "point_to_plane_residuals_mm",
            "observed_points_m",
            "predicted_plane_normal",
            "checks",
        }:
            raise ContractError("metric-depth frame fields are incomplete")
        frame_checks = frame.get("checks")
        if not isinstance(frame_checks, dict) or set(frame_checks) != _FRAME_CHECKS:
            raise ContractError("metric-depth frame check set is invalid")
        expected_status = "PASS" if all(frame_checks.values()) else "FAIL"
        if frame.get("status") != expected_status:
            raise ContractError("metric-depth frame status differs from checks")
        support = _nonnegative_integer(frame.get("valid_depth_support"), "frame support")
        total_frame_support += support
        frame_regions = frame.get("board_region_valid_support")
        if not isinstance(frame_regions, dict) or set(frame_regions) != _REGIONS:
            raise ContractError("metric-depth frame region support is invalid")
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in frame_regions.values()
            )
            or sum(frame_regions.values()) != support
        ):
            raise ContractError("metric-depth frame region support differs from frame support")
        for name in _REGIONS:
            summed_regions[name] += frame_regions[name]
        depth_residuals = _finite_number_list(
            frame.get("depth_residuals_mm"), "frame depth residuals"
        )
        point_plane_residuals = _finite_number_list(
            frame.get("point_to_plane_residuals_mm"), "frame point-to-plane residuals"
        )
        observed_points = _finite_points(frame.get("observed_points_m"))
        if not (
            len(depth_residuals) == len(point_plane_residuals) == len(observed_points) == support
        ):
            raise ContractError("metric-depth frame evidence support differs")
        all_depth_residuals.extend(depth_residuals)
        all_point_to_plane_residuals.extend(point_plane_residuals)
        if expected_status == "PASS":
            passed_frames += 1
        numeric_frame = all(
            frame.get(name) is not None
            for name in (
                "median_absolute_depth_residual_mm",
                "p95_absolute_depth_residual_mm",
                "plane_offset_mm",
                "plane_normal_error_deg",
                "distance_scale_ratio",
            )
        )
        if numeric_frame:
            predicted_distance = _finite(
                frame.get("predicted_plane_distance_m"), "frame predicted distance"
            )
            frame_predicted_distances.append(predicted_distance)
            predicted_normal = np.asarray(
                _finite_number_list(frame.get("predicted_plane_normal"), "frame predicted normal"),
                dtype=np.float64,
            )
            if (
                predicted_distance <= 0.0
                or predicted_normal.shape != (3,)
                or not math.isclose(float(np.linalg.norm(predicted_normal)), 1.0, abs_tol=1e-9)
                or support < 3
            ):
                raise ContractError("metric-depth frame prediction evidence is invalid")
            observed_normal, observed_distance = _fit_plane(
                np.asarray(observed_points, dtype=np.float64)
            )
            if float(np.dot(observed_normal, predicted_normal)) < 0.0:
                observed_normal = -observed_normal
            signed_predicted_distance = math.copysign(
                predicted_distance,
                float(np.dot(predicted_normal, observed_points[0])),
            )
            recomputed_point_plane = [
                (float(np.dot(predicted_normal, point)) - signed_predicted_distance) * 1000.0
                for point in observed_points
            ]
            if any(
                not math.isclose(stored, expected, abs_tol=1e-9)
                for stored, expected in zip(
                    point_plane_residuals, recomputed_point_plane, strict=True
                )
            ):
                raise ContractError("metric-depth point-to-plane evidence differs")
            frame_median = float(np.median(np.abs(depth_residuals)))
            frame_p95 = float(np.percentile(np.abs(depth_residuals), 95))
            frame_offset = (observed_distance - predicted_distance) * 1000.0
            frame_normal = math.degrees(
                math.acos(float(np.clip(np.dot(observed_normal, predicted_normal), -1.0, 1.0)))
            )
            frame_scale = observed_distance / predicted_distance
            recomputed_frame_metrics = {
                "measured_plane_distance_m": observed_distance,
                "signed_depth_residual_mm": float(np.median(depth_residuals)),
                "median_absolute_depth_residual_mm": frame_median,
                "p95_absolute_depth_residual_mm": frame_p95,
                "plane_offset_mm": frame_offset,
                "plane_normal_error_deg": frame_normal,
                "distance_scale_ratio": frame_scale,
            }
            if any(
                not math.isclose(_finite(frame.get(name), name), expected, abs_tol=1e-9)
                for name, expected in recomputed_frame_metrics.items()
            ):
                raise ContractError("metric-depth frame metrics differ from measurement evidence")
            frame_offsets.append(frame_offset)
            frame_normals.append(frame_normal)
            frame_scales.append(frame_scale)
            expected_frame_checks = {
                "valid_support_at_least_minimum": support
                >= int(thresholds["minimum_frame_valid_samples"]),
                "all_board_regions_have_support": all(
                    value >= int(thresholds["minimum_region_valid_samples"])
                    for value in frame_regions.values()
                ),
                "median_absolute_error_within_limit": frame_median
                <= float(thresholds["maximum_median_absolute_error_mm"]),
                "p95_absolute_error_within_limit": frame_p95
                <= float(thresholds["maximum_p95_absolute_error_mm"]),
                "plane_offset_within_limit": abs(frame_offset)
                <= float(thresholds["maximum_plane_offset_mm"]),
                "plane_normal_error_within_limit": frame_normal
                <= float(thresholds["maximum_plane_normal_error_deg"]),
                "metric_scale_ratio_within_limit": abs(frame_scale - 1.0)
                <= float(thresholds["maximum_scale_ratio_error"]),
            }
        else:
            expected_frame_checks = {name: False for name in _FRAME_CHECKS}
            if support >= 3:
                raise ContractError("metric-depth frame with sufficient geometry lacks metrics")
        if frame_checks != expected_frame_checks:
            raise ContractError("metric-depth frame checks differ from metrics")
    if total_frame_support != ints["valid_samples"] or summed_regions != regions:
        raise ContractError("metric-depth aggregate support differs from frames")
    if not frame_offsets or not frame_normals or not frame_scales:
        raise ContractError("metric-depth evaluation has no finite frame geometry")
    recomputed_geometry = {
        "plane_offset_mm": float(np.median(frame_offsets)),
        "plane_normal_error_deg": float(np.median(frame_normals)),
        "distance_scale_ratio": float(np.median(frame_scales)),
        "worst_frame_plane_offset_mm": max(abs(value) for value in frame_offsets),
        "worst_frame_plane_normal_error_deg": max(abs(value) for value in frame_normals),
        "worst_frame_scale_ratio_error": max(abs(value - 1.0) for value in frame_scales),
    }
    for name, expected in recomputed_geometry.items():
        if not math.isclose(_finite(evaluation.get(name), name), expected, abs_tol=1e-12):
            raise ContractError(f"metric-depth aggregate geometry differs: {name}")
    if not all_depth_residuals or not all_point_to_plane_residuals:
        raise ContractError("metric-depth aggregate measurement evidence is empty")
    recomputed_residuals = {
        "median_absolute_error_mm": float(np.median(np.abs(all_depth_residuals))),
        "p95_absolute_error_mm": float(np.percentile(np.abs(all_depth_residuals), 95)),
        "signed_bias_mm": float(np.median(all_depth_residuals)),
        "median_absolute_point_to_plane_error_mm": float(
            np.median(np.abs(all_point_to_plane_residuals))
        ),
    }
    for name, expected in recomputed_residuals.items():
        if not math.isclose(_finite(evaluation.get(name), name), expected, abs_tol=1e-9):
            raise ContractError(f"metric-depth aggregate residual differs: {name}")
    predicted_distance = _finite(
        evaluation.get("predicted_plane_distance_m"), "predicted plane distance"
    )
    measured_distance = _finite(
        evaluation.get("measured_plane_distance_m"), "measured plane distance"
    )
    if (
        predicted_distance <= 0.0
        or any(
            not math.isclose(value, predicted_distance, abs_tol=1e-12)
            for value in frame_predicted_distances
        )
        or not math.isclose(
            measured_distance,
            predicted_distance + recomputed_geometry["plane_offset_mm"] / 1000.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            recomputed_geometry["distance_scale_ratio"],
            measured_distance / predicted_distance,
            abs_tol=1e-12,
        )
    ):
        raise ContractError("metric-depth aggregate distance identity differs")
    if ints["passing_frame_count"] != passed_frames or not math.isclose(
        passing_ratio, passed_frames / len(frames) if frames else 0.0, abs_tol=1e-12
    ):
        raise ContractError("metric-depth passing-frame aggregate differs")
    expected_status = "PASS" if all(checks.values()) else "FAIL"
    aggregate_expected = {
        "valid_samples_at_least_minimum": ints["valid_samples"]
        >= int(thresholds["minimum_valid_samples"]),
        "valid_frames_at_least_minimum": ints["valid_frame_count"]
        >= int(thresholds["minimum_valid_frames"]),
        "valid_sample_ratio_at_least_minimum": valid_ratio
        >= float(thresholds["minimum_valid_sample_ratio"]),
        "all_depth_values_finite": ints["non_finite_depth_values"] == 0,
        "all_board_regions_have_support": all(
            value >= int(thresholds["minimum_region_valid_samples"]) for value in regions.values()
        ),
        "passing_frames_at_least_minimum": passed_frames
        >= int(thresholds["minimum_passing_frames"]),
        "passing_frame_ratio_at_least_minimum": passing_ratio
        >= float(thresholds["minimum_passing_frame_ratio"]),
        "all_evaluated_frames_pass_geometry": passed_frames == len(frames),
        "median_absolute_error_within_limit": _finite(
            evaluation.get("median_absolute_error_mm"), "median error"
        )
        <= float(thresholds["maximum_median_absolute_error_mm"]),
        "p95_absolute_error_within_limit": _finite(
            evaluation.get("p95_absolute_error_mm"), "p95 error"
        )
        <= float(thresholds["maximum_p95_absolute_error_mm"]),
        "plane_offset_within_limit": abs(_finite(evaluation.get("plane_offset_mm"), "plane offset"))
        <= float(thresholds["maximum_plane_offset_mm"]),
        "plane_normal_error_within_limit": _finite(
            evaluation.get("plane_normal_error_deg"), "plane normal"
        )
        <= float(thresholds["maximum_plane_normal_error_deg"]),
        "metric_scale_ratio_within_limit": abs(
            _finite(evaluation.get("distance_scale_ratio"), "scale ratio") - 1.0
        )
        <= float(thresholds["maximum_scale_ratio_error"]),
        "worst_frame_plane_offset_within_limit": _finite(
            evaluation.get("worst_frame_plane_offset_mm"), "worst offset"
        )
        <= float(thresholds["maximum_plane_offset_mm"]),
        "worst_frame_plane_normal_error_within_limit": _finite(
            evaluation.get("worst_frame_plane_normal_error_deg"), "worst normal"
        )
        <= float(thresholds["maximum_plane_normal_error_deg"]),
        "worst_frame_metric_scale_within_limit": _finite(
            evaluation.get("worst_frame_scale_ratio_error"), "worst scale"
        )
        <= float(thresholds["maximum_scale_ratio_error"]),
    }
    if checks != aggregate_expected:
        raise ContractError("metric-depth checks differ from recomputed decision")
    if evaluation.get("status") != expected_status:
        raise ContractError("metric-depth evaluation status differs from checks")
    if require_pass and expected_status != "PASS":
        raise ContractError("metric-depth evaluation is not passed")
    for name in (
        "predicted_plane_distance_m",
        "measured_plane_distance_m",
        "median_absolute_error_mm",
        "p95_absolute_error_mm",
        "signed_bias_mm",
        "median_absolute_point_to_plane_error_mm",
        "plane_offset_mm",
        "plane_normal_error_deg",
        "distance_scale_ratio",
        "worst_frame_plane_offset_mm",
        "worst_frame_plane_normal_error_deg",
        "worst_frame_scale_ratio_error",
    ):
        _finite(evaluation.get(name), name)
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in regions.values()
    ):
        raise ContractError("metric-depth region counts must be non-negative integers")
    return evaluation


def _validate_fixed_bootstrap_threshold_contract(
    thresholds: dict[str, object], threshold_policy: dict[str, object]
) -> None:
    from camera_rig.provision.config import (
        BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
        BOOTSTRAP_METRIC_DEPTH_THRESHOLDS,
    )

    expected = {
        "minimum_valid_samples": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_valid_samples"],
        "maximum_median_absolute_error_mm": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS[
            "maximum_median_error_mm"
        ],
        "maximum_p95_absolute_error_mm": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["maximum_p95_error_mm"],
        "minimum_valid_frames": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_valid_frames"],
        "minimum_region_valid_samples": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS[
            "minimum_region_valid_samples"
        ],
        "minimum_frame_valid_samples": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS[
            "minimum_frame_valid_samples"
        ],
        "minimum_passing_frames": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_passing_frames"],
        "minimum_valid_sample_ratio": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS[
            "minimum_valid_sample_ratio"
        ],
        "minimum_passing_frame_ratio": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS[
            "minimum_passing_frame_ratio"
        ],
        "maximum_plane_offset_mm": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["maximum_plane_offset_mm"],
        "maximum_plane_normal_error_deg": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS[
            "maximum_plane_normal_error_deg"
        ],
        "maximum_scale_ratio_error": BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["maximum_scale_ratio_error"],
        "window_size": 5,
    }
    expected_policy = {
        "schema_version": BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
        "source": "immutable_fixed_provision_contract",
    }
    if thresholds != expected or threshold_policy != expected_policy:
        raise ContractError("metric-depth bootstrap threshold policy differs from frozen contract")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ContractError(f"metric-depth {name} must be finite")
    return float(value)


def _finite_number_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ContractError(f"metric-depth {name} must be an array")
    return [_finite(item, f"{name}[]") for item in value]


def _finite_points(value: object) -> list[npt.NDArray[np.float64]]:
    if not isinstance(value, list):
        raise ContractError("metric-depth observed points must be an array")
    points: list[npt.NDArray[np.float64]] = []
    for item in value:
        coordinates = _finite_number_list(item, "observed point")
        if len(coordinates) != 3:
            raise ContractError("metric-depth observed point must have three coordinates")
        points.append(np.asarray(coordinates, dtype=np.float64))
    return points


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"metric-depth {name} must be a non-negative integer")
    return value


def _digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"metric-depth {name} must be a lowercase SHA-256 digest")
