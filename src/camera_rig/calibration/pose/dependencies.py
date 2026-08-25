"""Lazy optional dependency boundary for generic OpenCV pose operations."""

from __future__ import annotations

from typing import Any

from camera_rig.core.errors import MissingOptionalDependencyError


def cv2_module() -> Any:
    """Import OpenCV only when a pose operation is requested."""
    try:
        import cv2
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'Planar pose estimation requires: pip install "camera-rig[charuco]"'
        ) from error
    required = (
        "Rodrigues",
        "SOLVEPNP_IPPE",
        "projectPoints",
        "solvePnPGeneric",
        "solvePnPRefineLM",
    )
    missing = [name for name in required if not hasattr(cv2, name)]
    if missing:
        raise MissingOptionalDependencyError(
            "Planar pose estimation requires an OpenCV build with calib3d support"
        )
    return cv2
