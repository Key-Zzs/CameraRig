"""SDK- and detector-independent target observation contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from camera_rig.core._validation import require_non_empty, require_positive_int, string_keyed_copy
from camera_rig.core.errors import ContractError
from camera_rig.core.quality import QualityReport

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TargetObservation:
    """Matched 2D image and 3D target points from a detector plugin."""

    plugin_name: str
    target_frame: str
    point_ids: tuple[int, ...]
    image_points_px: FloatArray
    object_points_m: FloatArray
    image_size: tuple[int, int]
    quality: QualityReport
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.plugin_name, "plugin_name")
        require_non_empty(self.target_frame, "target_frame")
        ids = tuple(self.point_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise ContractError("point_ids must contain integers")
        if len(set(ids)) != len(ids):
            raise ContractError("point_ids must be unique")
        image_points = np.asarray(self.image_points_px, dtype=np.float64).copy()
        object_points = np.asarray(self.object_points_m, dtype=np.float64).copy()
        if image_points.ndim != 2 or image_points.shape[1:] != (2,):
            raise ContractError("image_points_px must have shape (N, 2)")
        if object_points.ndim != 2 or object_points.shape[1:] != (3,):
            raise ContractError("object_points_m must have shape (N, 3)")
        if len(ids) != len(image_points) or len(ids) != len(object_points):
            raise ContractError("point_ids, image_points_px, and object_points_m counts must match")
        if not np.isfinite(image_points).all() or not np.isfinite(object_points).all():
            raise ContractError("target points must contain only finite values")
        if len(self.image_size) != 2:
            raise ContractError("image_size must be (width, height)")
        require_positive_int(self.image_size[0], "image_size width")
        require_positive_int(self.image_size[1], "image_size height")
        image_points.setflags(write=False)
        object_points.setflags(write=False)
        object.__setattr__(self, "point_ids", ids)
        object.__setattr__(self, "image_points_px", image_points)
        object.__setattr__(self, "object_points_m", object_points)
        object.__setattr__(self, "image_size", tuple(self.image_size))
        object.__setattr__(self, "metadata", string_keyed_copy(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, object]:
        """Serialize the observation without pickle or detector-specific types."""
        return {
            "plugin_name": self.plugin_name,
            "target_frame": self.target_frame,
            "point_ids": list(self.point_ids),
            "image_points_px": self.image_points_px.tolist(),
            "object_points_m": self.object_points_m.tolist(),
            "image_size": list(self.image_size),
            "quality": self.quality.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TargetObservation:
        """Reconstruct a persisted observation for later pose consumers."""
        try:
            expected_fields = {
                "plugin_name",
                "target_frame",
                "point_ids",
                "image_points_px",
                "object_points_m",
                "image_size",
                "quality",
                "metadata",
            }
            if set(data) != expected_fields:
                raise TypeError("observation has missing or unknown fields")
            point_ids_value = data["point_ids"]
            image_size_value = data["image_size"]
            quality_value = data["quality"]
            metadata_value = data.get("metadata", {})
            if not isinstance(point_ids_value, list):
                raise TypeError("point_ids must be an array")
            if not isinstance(image_size_value, list) or len(image_size_value) != 2:
                raise TypeError("image_size must be a two-element array")
            if not isinstance(quality_value, dict) or not isinstance(metadata_value, dict):
                raise TypeError("quality and metadata must be objects")
            if not all(isinstance(key, str) for key in metadata_value):
                raise TypeError("metadata keys must be strings")
            return cls(
                plugin_name=_decoded_string(data["plugin_name"], "plugin_name"),
                target_frame=_decoded_string(data["target_frame"], "target_frame"),
                point_ids=tuple(_decoded_int(value, "point_ids[]") for value in point_ids_value),
                image_points_px=np.asarray(data["image_points_px"], dtype=np.float64),
                object_points_m=np.asarray(data["object_points_m"], dtype=np.float64),
                image_size=(
                    _decoded_int(image_size_value[0], "image_size[0]"),
                    _decoded_int(image_size_value[1], "image_size[1]"),
                ),
                quality=QualityReport.from_dict(dict(quality_value)),
                metadata=dict(metadata_value),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"invalid persisted target observation: {error}") from error


def _decoded_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _decoded_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value
