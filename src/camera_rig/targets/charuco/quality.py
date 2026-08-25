"""Pose-free ChArUco detection quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from camera_rig.core.quality import QualityReport


@dataclass(frozen=True)
class CharucoQualityThresholds:
    minimum_charuco_corners: int = 12
    minimum_corner_fraction: float = 0.50
    minimum_coverage_ratio: float = 0.05


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
    else:
        coverage = 0.0
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
    if coverage < thresholds.minimum_coverage_ratio:
        failures.append("board coverage below threshold")
    return QualityReport(
        passed=not failures,
        metrics={
            "detected_marker_count": detected_marker_count,
            "detected_charuco_corner_count": corner_count,
            "corner_fraction": fraction,
            "coverage_ratio": coverage,
            "minimum_border_distance_px": border_distance,
            "mean_marker_perimeter_px": float(np.mean(perimeters)) if perimeters else 0.0,
            "image_contrast": float(np.std(image_gray, dtype=np.float64)),
        },
        thresholds={
            "minimum_charuco_corners": thresholds.minimum_charuco_corners,
            "minimum_corner_fraction": thresholds.minimum_corner_fraction,
            "minimum_coverage_ratio": thresholds.minimum_coverage_ratio,
        },
        failure_reasons=tuple(failures),
    )
