"""Camera intrinsic-parameter contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

from camera_rig.core._validation import (
    decoded_float,
    decoded_int,
    decoded_string,
    require_non_empty,
    require_positive_int,
)
from camera_rig.core.errors import ContractError


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole parameters and stream-specific distortion metadata."""

    frame: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coeffs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.frame, "frame")
        require_positive_int(self.width, "width")
        require_positive_int(self.height, "height")
        require_non_empty(self.distortion_model, "distortion_model")
        values = (self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs)
        if not all(math.isfinite(float(value)) for value in values):
            raise ContractError("intrinsic parameters must all be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ContractError("fx and fy must be greater than zero")
        if not 0 <= self.cx < self.width:
            raise ContractError("cx must lie inside the image width")
        if not 0 <= self.cy < self.height:
            raise ContractError("cy must lie inside the image height")
        object.__setattr__(
            self, "distortion_coeffs", tuple(float(v) for v in self.distortion_coeffs)
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe-compatible representation."""
        return {
            "frame": self.frame,
            "width": self.width,
            "height": self.height,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "distortion_model": self.distortion_model,
            "distortion_coeffs": list(self.distortion_coeffs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CameraIntrinsics:
        """Reconstruct intrinsics from decoded JSON data."""
        coefficients = data.get("distortion_coeffs", [])
        if not isinstance(coefficients, list):
            raise TypeError("distortion_coeffs must be an array")
        return cls(
            frame=decoded_string(data["frame"], "frame"),
            width=decoded_int(data["width"], "width"),
            height=decoded_int(data["height"], "height"),
            fx=decoded_float(data["fx"], "fx"),
            fy=decoded_float(data["fy"], "fy"),
            cx=decoded_float(data["cx"], "cx"),
            cy=decoded_float(data["cy"], "cy"),
            distortion_model=decoded_string(data["distortion_model"], "distortion_model"),
            distortion_coeffs=tuple(
                decoded_float(value, "distortion_coeffs[]") for value in coefficients
            ),
        )
