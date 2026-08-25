"""Fail-closed conversion from CameraRig intrinsics to OpenCV parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OpenCVCameraModel:
    """OpenCV camera matrix, coefficients, and an auditable mapping decision."""

    camera_matrix: FloatArray
    distortion_coeffs: FloatArray
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64).copy()
        coefficients = np.asarray(self.distortion_coeffs, dtype=np.float64).reshape(-1).copy()
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            raise ContractError("OpenCV camera matrix must be a finite 3x3 matrix")
        if not np.isfinite(coefficients).all():
            raise ContractError("OpenCV distortion coefficients must be finite")
        camera_matrix.setflags(write=False)
        coefficients.setflags(write=False)
        object.__setattr__(self, "camera_matrix", camera_matrix)
        object.__setattr__(self, "distortion_coeffs", coefficients)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
            "diagnostics": dict(self.diagnostics),
        }


def to_opencv_camera_model(intrinsics: CameraIntrinsics) -> OpenCVCameraModel:
    """Map only distortion models whose OpenCV semantics are explicitly established."""
    camera_matrix = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    source = np.asarray(intrinsics.distortion_coeffs, dtype=np.float64).reshape(-1)
    model = intrinsics.distortion_model
    identity_only = False
    if model == "none":
        if source.size and not np.equal(source, 0.0).all():
            raise ContractError("distortion model 'none' cannot contain non-zero coefficients")
        coefficients = np.zeros(5, dtype=np.float64)
        decision = "none-as-opencv-identity"
        opencv_model = "identity"
    elif model == "brown-conrady":
        if source.shape != (5,):
            raise ContractError("brown-conrady requires exactly five RealSense coefficients")
        coefficients = source.copy()
        decision = "realsense-brown-conrady-to-opencv-k1-k2-p1-p2-k3"
        opencv_model = "brown-conrady"
    elif model == "inverse-brown-conrady":
        if source.shape != (5,) or not np.equal(source, 0.0).all():
            raise ContractError(
                "inverse-brown-conrady is supported only for exactly five zero coefficients"
            )
        coefficients = source.copy()
        decision = "inverse-brown-conrady-zero-coefficients-as-opencv-identity"
        opencv_model = "identity"
        identity_only = True
    else:
        raise ContractError(f"distortion model {model!r} has no proven OpenCV mapping")
    return OpenCVCameraModel(
        camera_matrix=camera_matrix,
        distortion_coeffs=coefficients,
        diagnostics={
            "camera_frame": intrinsics.frame,
            "source_distortion_model": model,
            "source_distortion_coeffs": source.tolist(),
            "opencv_distortion_model": opencv_model,
            "mapping_decision": decision,
            "inverse_brown_identity_only": identity_only,
        },
    )
