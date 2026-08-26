"""Pose-free ChArUco detection quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from camera_rig.core.quality import QualityReport


@dataclass(frozen=True)
class CharucoQualityThresholds:
    policy: str = "legacy_strict"
    minimum_charuco_corners: int = 12
    minimum_corner_fraction: float = 0.50
    minimum_span_x_ratio: float = 0.10
    minimum_span_y_ratio: float = 0.10
    minimum_marker_perimeter_px: float = 20.0
    absolute_minimum_coverage_ratio: float = 0.05
    recommended_coverage_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.policy not in {"legacy_strict", "pose_validated"}:
            raise ValueError(f"unsupported ChArUco quality policy: {self.policy!r}")

    @classmethod
    def pose_validated(cls) -> CharucoQualityThresholds:
        """Return the deployment policy whose 5% coverage threshold is advisory."""
        return cls(
            policy="pose_validated",
            absolute_minimum_coverage_ratio=0.01,
            recommended_coverage_ratio=0.05,
        )


def detection_quality(
    *,
    image_gray: npt.NDArray[np.uint8],
    image_points: npt.NDArray[np.float64],
    detected_marker_count: int,
    marker_corners: tuple[npt.NDArray[np.float64], ...],
    total_corner_count: int,
    thresholds: CharucoQualityThresholds,
    cv2: Any,
) -> QualityReport:
    """Compute geometric image-space metrics without pose or camera intrinsics."""
    height, width = image_gray.shape
    corner_count = len(image_points)
    fraction = corner_count / total_corner_count
    if corner_count >= 3:
        hull = cv2.convexHull(np.asarray(image_points, dtype=np.float32))
        coverage = float(cv2.contourArea(hull)) / float(width * height)
        x_span = float(np.ptp(image_points[:, 0])) / float(width)
        y_span = float(np.ptp(image_points[:, 1])) / float(height)
        minimum = np.min(image_points, axis=0)
        maximum = np.max(image_points, axis=0)
        bounding_box_area_ratio = float(np.prod(maximum - minimum)) / float(width * height)
        centered = image_points - np.mean(image_points, axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        distribution_condition = (
            float(singular_values[-1] / singular_values[0])
            if len(singular_values) >= 2 and singular_values[0] > 0
            else 0.0
        )
    else:
        coverage = 0.0
        x_span = 0.0
        y_span = 0.0
        bounding_box_area_ratio = 0.0
        distribution_condition = 0.0
    if corner_count:
        u = image_points[:, 0]
        v = image_points[:, 1]
        border_distance = float(np.min(np.stack((u, v, width - 1 - u, height - 1 - v))))
    else:
        border_distance = 0.0
    perimeters = [
        float(cv2.arcLength(np.asarray(corners, dtype=np.float32).reshape(-1, 2), True))
        for corners in marker_corners
    ]
    failures: list[str] = []
    if corner_count < thresholds.minimum_charuco_corners:
        failures.append("insufficient ChArUco corners")
    if fraction < thresholds.minimum_corner_fraction:
        failures.append("corner fraction below threshold")
    minimum_perimeter = float(np.min(perimeters)) if perimeters else 0.0
    if thresholds.policy == "pose_validated":
        if x_span < thresholds.minimum_span_x_ratio:
            failures.append("image x span below threshold")
        if y_span < thresholds.minimum_span_y_ratio:
            failures.append("image y span below threshold")
        if minimum_perimeter < thresholds.minimum_marker_perimeter_px:
            failures.append("marker perimeter below threshold")
    if coverage < thresholds.absolute_minimum_coverage_ratio:
        failures.append("board coverage below absolute minimum")
    warnings: list[str] = []
    if coverage < thresholds.recommended_coverage_ratio:
        warnings.append("board coverage below recommended deployment coverage")
    return QualityReport(
        passed=not failures,
        metrics={
            "detected_marker_count": detected_marker_count,
            "detected_charuco_corner_count": corner_count,
            "corner_fraction": fraction,
            "coverage_ratio": coverage,
            "convex_hull_coverage_ratio": coverage,
            "image_span_x_ratio": x_span,
            "image_span_y_ratio": y_span,
            "bounding_box_area_ratio": bounding_box_area_ratio,
            "minimum_border_distance_px": border_distance,
            "mean_marker_perimeter_px": float(np.mean(perimeters)) if perimeters else 0.0,
            "minimum_marker_perimeter_px": minimum_perimeter,
            "corner_distribution_condition": distribution_condition,
            "image_contrast": float(np.std(image_gray, dtype=np.float64)),
        },
        thresholds={
            "policy": thresholds.policy,
            "minimum_charuco_corners": thresholds.minimum_charuco_corners,
            "minimum_corner_fraction": thresholds.minimum_corner_fraction,
            "minimum_span_x_ratio": thresholds.minimum_span_x_ratio,
            "minimum_span_y_ratio": thresholds.minimum_span_y_ratio,
            "minimum_marker_perimeter_px": thresholds.minimum_marker_perimeter_px,
            "absolute_minimum_coverage_ratio": thresholds.absolute_minimum_coverage_ratio,
            "recommended_coverage_ratio": thresholds.recommended_coverage_ratio,
        },
        warnings=tuple(warnings),
        failure_reasons=tuple(failures),
    )
