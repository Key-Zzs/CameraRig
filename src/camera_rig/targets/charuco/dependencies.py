"""Lazy optional dependency boundaries for the ChArUco plugin."""

from __future__ import annotations

from typing import Any

from camera_rig.core.errors import MissingOptionalDependencyError


def cv2_module() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'ChArUco support requires: pip install "camera-rig[charuco]"'
        ) from error
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise MissingOptionalDependencyError(
            'ChArUco support requires the official extra: pip install "camera-rig[charuco]"'
        )
    return cv2
