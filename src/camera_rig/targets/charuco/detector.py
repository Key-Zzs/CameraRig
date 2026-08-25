"""ChArUco detector plugin producing persisted canonical object geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from camera_rig.core.errors import ContractError
from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget
from camera_rig.targets.charuco.geometry import create_board
from camera_rig.targets.charuco.quality import CharucoQualityThresholds, detection_quality
from camera_rig.targets.observation import TargetObservation


class CharucoDetector:
    """Detect 2D ChArUco corners and look up immutable CameraRig canonical points."""

    plugin_name = "charuco"

    def __init__(
        self,
        target_spec: object,
        *,
        thresholds: CharucoQualityThresholds | None = None,
    ) -> None:
        if not isinstance(target_spec, ResolvedCharucoTarget):
            raise ContractError("CharucoDetector requires a ResolvedCharucoTarget artifact")
        self.target_spec = target_spec
        self.thresholds = thresholds or CharucoQualityThresholds()
        board, _dictionary, cv2 = create_board(target_spec)
        detector_parameters = cv2.aruco.DetectorParameters()
        detector_parameters.markerBorderBits = target_spec.border_bits
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._cv2: Any = cv2
        self._detector_parameters: Any = detector_parameters
        self._detector: Any = cv2.aruco.CharucoDetector(
            board,
            cv2.aruco.CharucoParameters(),
            detector_parameters,
        )

    def detect(self, image: npt.NDArray[np.generic]) -> TargetObservation:
        """Detect from uint8 grayscale or explicitly RGB input."""
        source = np.asarray(image)
        if source.dtype != np.uint8:
            raise ContractError("ChArUco detection requires a uint8 image")
        if source.ndim == 2:
            if source.shape[0] == 0 or source.shape[1] == 0:
                raise ContractError("ChArUco detection image must be non-empty")
            gray = source.copy()
        elif source.ndim == 3 and source.shape[2] == 3:
            if source.shape[0] == 0 or source.shape[1] == 0:
                raise ContractError("ChArUco detection image must be non-empty")
            gray = self._cv2.cvtColor(source, self._cv2.COLOR_RGB2GRAY)
        else:
            raise ContractError("ChArUco detection expects grayscale [H,W] or RGB [H,W,3]")
        charuco_corners, charuco_ids, marker_corners, marker_ids = self._detector.detectBoard(gray)
        points, point_ids = _normalized_corners(charuco_corners, charuco_ids)
        order = np.argsort(np.asarray(point_ids, dtype=np.int64))
        sorted_ids = tuple(point_ids[index] for index in order)
        sorted_points = points[order] if len(order) else np.empty((0, 2), dtype=np.float64)
        object_points = self.target_spec.object_points_for(sorted_ids)
        normalized_marker_corners = tuple(
            np.asarray(corners, dtype=np.float64).reshape(-1, 2) for corners in marker_corners
        )
        normalized_marker_ids = (
            tuple(sorted(int(value) for value in np.asarray(marker_ids).reshape(-1)))
            if marker_ids is not None
            else ()
        )
        quality = detection_quality(
            image_gray=gray,
            image_points=sorted_points,
            detected_marker_count=len(normalized_marker_ids),
            marker_corners=normalized_marker_corners,
            total_corner_count=len(self.target_spec.corner_points),
            thresholds=self.thresholds,
            cv2=self._cv2,
        )
        height, width = gray.shape
        return TargetObservation(
            plugin_name=self.plugin_name,
            target_frame=self.target_spec.target_frame,
            point_ids=sorted_ids,
            image_points_px=sorted_points,
            object_points_m=object_points,
            image_size=(width, height),
            quality=quality,
            metadata={
                "target_spec_sha256": self.target_spec.artifact_sha256,
                "dictionary": self.target_spec.dictionary,
                "opencv_version": self.target_spec.opencv_version,
                "marker_ids": list(normalized_marker_ids),
                "detector_parameters": {
                    "marker_border_bits": int(self._detector_parameters.markerBorderBits),
                    "corner_refinement": "subpixel",
                    "input_color_order": "RGB",
                },
            },
        )


def _normalized_corners(
    corners: object, ids: object
) -> tuple[npt.NDArray[np.float64], tuple[int, ...]]:
    if corners is None or ids is None:
        return np.empty((0, 2), dtype=np.float64), ()
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    point_ids = tuple(int(value) for value in np.asarray(ids).reshape(-1))
    if len(points) != len(point_ids) or len(set(point_ids)) != len(point_ids):
        raise ContractError("OpenCV returned inconsistent ChArUco corner IDs")
    return points, point_ids
