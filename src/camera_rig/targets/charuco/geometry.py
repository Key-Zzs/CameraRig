"""OpenCV-board construction and one-time canonical geometry conversion."""

from __future__ import annotations

from typing import Any

import numpy as np

from camera_rig.core.errors import ArtifactError, ConfigurationError
from camera_rig.targets.charuco.dependencies import cv2_module
from camera_rig.targets.charuco.spec import CharucoTargetSpec

SUPPORTED_DICTIONARIES = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
)


def create_board(spec: CharucoTargetSpec | Any) -> tuple[Any, Any, Any]:
    """Build an explicitly configured OpenCV board, dictionary, and cv2 module."""
    cv2 = cv2_module()
    if spec.dictionary not in SUPPORTED_DICTIONARIES:
        raise ConfigurationError(f"unsupported ChArUco dictionary: {spec.dictionary!r}")
    dictionary_id = getattr(cv2.aruco, spec.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )
    board.setLegacyPattern(spec.legacy_pattern)
    return board, dictionary, cv2


def canonical_corners_from_board(
    spec: CharucoTargetSpec, board: Any
) -> tuple[tuple[int, tuple[float, float, float]], ...]:
    """Convert OpenCV's top-left/down local geometry to CameraRig bottom-left/up."""
    local = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    if local.shape != (spec.charuco_corner_count, 3):
        raise ArtifactError("OpenCV returned unexpected ChArUco corner geometry")
    result: list[tuple[int, tuple[float, float, float]]] = []
    for point_id, point in enumerate(local):
        canonical = (float(point[0]), spec.board_height_m - float(point[1]), 0.0)
        row, column = divmod(point_id, spec.squares_x - 1)
        expected = (
            (column + 1) * spec.square_length_m,
            spec.board_height_m - (row + 1) * spec.square_length_m,
            0.0,
        )
        if not np.allclose(canonical, expected, rtol=0.0, atol=1e-7):
            raise ArtifactError(
                f"OpenCV corner ID {point_id} does not match CameraRig canonical mapping"
            )
        result.append((point_id, canonical))
    return tuple(result)
