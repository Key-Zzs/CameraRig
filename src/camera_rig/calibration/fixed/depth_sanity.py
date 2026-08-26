"""Independent native-depth diagnostic for a solved fixed target pose."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose import project_points_px
from camera_rig.core.errors import ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget


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
) -> dict[str, object]:
    """Compare predicted target-plane depth with raw native depth without fitting pose."""
    if window_radius < 1:
        raise ContractError("native depth window radius must be positive")
    detection = calibration.intrinsics.get(detection_stream)
    depth = calibration.intrinsics.get("depth")
    if detection is None or depth is None:
        return _skipped("detection or native-depth intrinsics are unavailable")
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
        return _skipped(f"native-depth projection is unsupported: {error}")
    predicted_depth = T_depth_from_target.transform_points(samples_target)[:, 2]
    errors_mm: list[float] = []
    requested = 0
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(frames):
            raise ContractError("native-depth frame index is outside captured frames")
        stream = frames[frame_index].depth
        if stream is None:
            continue
        raw = np.asarray(stream.data)
        if raw.ndim != 2:
            raise ContractError("native depth image must be two-dimensional")
        for pixel, expected_m in zip(pixels, predicted_depth, strict=True):
            requested += 1
            observed_units = _window_median(raw, pixel, window_radius)
            if observed_units is None:
                continue
            observed_m = observed_units * calibration.depth_scale_m_per_unit
            errors_mm.append((observed_m - float(expected_m)) * 1000.0)
    valid_ratio = len(errors_mm) / requested if requested else 0.0
    absolute = np.abs(np.asarray(errors_mm, dtype=np.float64))
    median = float(np.median(absolute)) if len(absolute) else float("inf")
    p95 = float(np.percentile(absolute, 95)) if len(absolute) else float("inf")
    bias = float(np.median(np.asarray(errors_mm))) if errors_mm else float("inf")
    checks = {
        "valid_samples_at_least_minimum": len(errors_mm) >= minimum_valid_samples,
        "median_absolute_error_within_limit": median <= maximum_median_error_mm,
        "p95_absolute_error_within_limit": p95 <= maximum_p95_error_mm,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sample_point_count": len(samples_target),
        "requested_samples": requested,
        "valid_samples": len(errors_mm),
        "valid_sample_ratio": valid_ratio,
        "median_absolute_error_mm": median,
        "p95_absolute_error_mm": p95,
        "signed_bias_mm": bias,
        "thresholds": {
            "minimum_valid_samples": minimum_valid_samples,
            "maximum_median_absolute_error_mm": maximum_median_error_mm,
            "maximum_p95_absolute_error_mm": maximum_p95_error_mm,
            "window_size": window_radius * 2 + 1,
        },
        "checks": checks,
        "role": "independent diagnostic only; not used by pose optimization",
    }


def _sample_target_plane(target: ResolvedCharucoTarget) -> npt.NDArray[np.float64]:
    margin = target.square_length_m * 0.5
    x_values = np.linspace(margin, target.board_width_m - margin, target.squares_x)
    y_values = np.linspace(margin, target.board_height_m - margin, target.squares_y)
    return np.asarray([(x, y, 0.0) for y in y_values for x in x_values], dtype=np.float64)


def _window_median(
    raw: npt.NDArray[np.generic], pixel: npt.NDArray[np.float64], radius: int
) -> float | None:
    x = round(float(pixel[0]))
    y = round(float(pixel[1]))
    if x < 0 or y < 0 or x >= raw.shape[1] or y >= raw.shape[0]:
        return None
    x0, x1 = max(0, x - radius), min(raw.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(raw.shape[0], y + radius + 1)
    values = np.asarray(raw[y0:y1, x0:x1], dtype=np.float64)
    valid = values[values > 0]
    return float(np.median(valid)) if len(valid) else None


def _skipped(reason: str) -> dict[str, object]:
    return {
        "status": "SKIPPED_WITH_WARNING",
        "warning": reason,
        "role": "independent diagnostic only; not used by pose optimization",
    }
