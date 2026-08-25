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
