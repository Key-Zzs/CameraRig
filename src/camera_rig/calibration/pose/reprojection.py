"""Reprojection residual calculation independent of target type."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose.camera_model import OpenCVCameraModel
from camera_rig.core.errors import ContractError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ReprojectionMetrics:
    """Complete pixel residual distribution for one pose and correspondence set."""

    residuals_px: FloatArray
    rmse_px: float
    median_px: float
    p95_px: float
    maximum_px: float

    def __post_init__(self) -> None:
        residuals = np.asarray(self.residuals_px, dtype=np.float64).reshape(-1).copy()
        values = (self.rmse_px, self.median_px, self.p95_px, self.maximum_px)
        if not len(residuals) or not np.isfinite(residuals).all():
            raise ContractError("reprojection residuals must be non-empty and finite")
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ContractError("reprojection statistics must be finite and non-negative")
        residuals.setflags(write=False)
        object.__setattr__(self, "residuals_px", residuals)

    def to_dict(self) -> dict[str, object]:
        return {
            "residuals_px": self.residuals_px.tolist(),
            "rmse_px": self.rmse_px,
            "median_px": self.median_px,
            "p95_px": self.p95_px,
            "maximum_px": self.maximum_px,
        }


def reprojection_metrics(
    object_points_m: npt.ArrayLike,
    image_points_px: npt.ArrayLike,
    rvec: npt.ArrayLike,
    tvec: npt.ArrayLike,
    camera_model: OpenCVCameraModel,
    *,
    cv2: Any,
) -> ReprojectionMetrics:
    """Project all points and summarize Euclidean pixel residuals."""
    objects = np.asarray(object_points_m, dtype=np.float64).reshape(-1, 3)
    observed = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
    if len(objects) != len(observed) or not len(objects):
        raise ContractError("reprojection correspondence counts must match and be non-empty")
    projected, _jacobian = cv2.projectPoints(
        objects,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        camera_model.camera_matrix,
        camera_model.distortion_coeffs,
    )
    projected_points = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    residuals = np.linalg.norm(projected_points - observed, axis=1)
    return ReprojectionMetrics(
        residuals_px=residuals,
        rmse_px=float(np.sqrt(np.mean(np.square(residuals)))),
        median_px=float(np.median(residuals)),
        p95_px=float(np.percentile(residuals, 95)),
        maximum_px=float(np.max(residuals)),
    )
