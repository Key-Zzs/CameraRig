"""Diagnostic-only ChArUco overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from camera_rig.core.errors import ArtifactError
from camera_rig.targets.charuco.dependencies import cv2_module
from camera_rig.targets.observation import TargetObservation


def render_overlay(
    image_rgb: npt.NDArray[np.uint8], observation: TargetObservation
) -> npt.NDArray[np.uint8]:
    """Draw IDs and a coverage hull without affecting detector output."""
    cv2 = cv2_module()
    source = np.asarray(image_rgb)
    if source.ndim == 2:
        overlay = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    elif source.ndim == 3 and source.shape[2] == 3:
        overlay = source.copy()
    else:
        raise ArtifactError("overlay expects grayscale or RGB uint8 image")
    points = np.asarray(observation.image_points_px, dtype=np.float64)
    if len(points) >= 3:
        hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).astype(np.int32)
        cv2.polylines(overlay, [hull], True, (0, 255, 255), 2, cv2.LINE_AA)
    for point_id, point in zip(observation.point_ids, points, strict=True):
        center = (round(point[0]), round(point[1]))
        cv2.circle(overlay, center, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(point_id),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return np.asarray(overlay, dtype=np.uint8)


def write_overlay(
    path: str | Path,
    image_rgb: npt.NDArray[np.uint8],
    observation: TargetObservation,
) -> None:
    cv2 = cv2_module()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay = render_overlay(image_rgb, observation)
    if not cv2.imwrite(str(target), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise ArtifactError(f"could not write overlay: {target}")
