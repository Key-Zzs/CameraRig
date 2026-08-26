"""Fixed-pose reprojection and canonical-axis diagnostic overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose import project_points_px
from camera_rig.core.errors import MissingOptionalDependencyError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation


def write_fixed_pose_overlay(
    path: str | Path,
    *,
    image_rgb: npt.NDArray[np.uint8],
    observation: TargetObservation,
    T_camera_from_target: RigidTransform,
    intrinsics: CameraIntrinsics,
    board_width_m: float,
    board_height_m: float,
    axis_length_m: float = 0.06,
) -> None:
    """Draw detections, reprojection residuals, boundary, and canonical axes."""
    image_module, draw_module = _pillow()
    image = image_module.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    draw = draw_module.Draw(image)
    reprojected = project_points_px(observation.object_points_m, T_camera_from_target, intrinsics)
    for point_id, detected, projected in zip(
        observation.point_ids,
        observation.image_points_px,
        reprojected,
        strict=True,
    ):
        detected_xy = (float(detected[0]), float(detected[1]))
        projected_xy = (float(projected[0]), float(projected[1]))
        draw.line((detected_xy, projected_xy), fill=(255, 0, 255), width=2)
        _circle(draw, detected_xy, 3, (255, 255, 0))
        _circle(draw, projected_xy, 2, (0, 255, 255))
        draw.text((detected_xy[0] + 4, detected_xy[1] + 2), str(point_id), fill=(255, 255, 0))

    boundary_target = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [board_width_m, 0.0, 0.0],
            [board_width_m, board_height_m, 0.0],
            [0.0, board_height_m, 0.0],
        ],
        dtype=np.float64,
    )
    boundary = project_points_px(boundary_target, T_camera_from_target, intrinsics)
    boundary_points = [tuple(map(float, point)) for point in boundary]
    draw.line([*boundary_points, boundary_points[0]], fill=(255, 165, 0), width=3)

    axes_target = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, axis_length_m],
        ],
        dtype=np.float64,
    )
    axes = project_points_px(axes_target, T_camera_from_target, intrinsics)
    origin = (float(axes[0, 0]), float(axes[0, 1]))
    colors = ((255, 0, 0), (0, 255, 0), (0, 128, 255))
    labels = ("+X", "+Y", "+Z")
    for endpoint, color, label in zip(axes[1:], colors, labels, strict=True):
        endpoint_xy = (float(endpoint[0]), float(endpoint[1]))
        draw.line((origin, endpoint_xy), fill=color, width=4)
        draw.text((endpoint_xy[0] + 4, endpoint_xy[1] + 2), label, fill=color)
    _circle(draw, origin, 4, (255, 255, 255))
    draw.text((origin[0] + 5, origin[1] + 5), "canonical origin", fill=(255, 255, 255))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def select_overlay_frames(per_frame: tuple[dict[str, object], ...]) -> dict[str, int]:
    """Choose worst, median, and best accepted frames by final-pose RMSE."""
    accepted: list[tuple[float, int]] = []
    for item in per_frame:
        if item.get("accepted") is not True:
            continue
        frame_index = item.get("frame_index")
        rmse = item.get("final_pose_reprojection_rmse_px")
        if (
            isinstance(frame_index, int)
            and not isinstance(frame_index, bool)
            and isinstance(rmse, int | float)
        ):
            accepted.append((float(rmse), frame_index))
    if not accepted:
        return {}
    ranked = sorted(accepted)
    return {
        "best": ranked[0][1],
        "median_quality": ranked[len(ranked) // 2][1],
        "worst_accepted": ranked[-1][1],
    }


def _circle(
    draw: Any,
    point: tuple[float, float],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)


def _pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'fixed calibration overlays require: pip install "camera-rig[viz]"'
        ) from error
    return Image, ImageDraw
