"""Frame-explicit SE(3) rigid transform contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from camera_rig.core._validation import decoded_string
from camera_rig.core.errors import TransformError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RigidTransform:
    """A homogeneous transform with ``T_target_from_source`` semantics."""

    source_frame: str
    target_frame: str
    matrix: FloatArray

    def __post_init__(self) -> None:
        _require_frame(self.source_frame, "source_frame")
        _require_frame(self.target_frame, "target_frame")
        matrix = np.asarray(self.matrix, dtype=np.float64).copy()
        if matrix.shape != (4, 4):
            raise TransformError("rigid transform matrix must have shape (4, 4)")
        if not np.isfinite(matrix).all():
            raise TransformError("rigid transform matrix must contain only finite values")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
            raise TransformError("rigid transform last row must be [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=1e-7):
            raise TransformError("rigid transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=1e-7):
            raise TransformError("rigid transform rotation determinant must be +1")
        if self.source_frame == self.target_frame and not np.allclose(
            matrix, np.eye(4), atol=1e-9, rtol=0.0
        ):
            raise TransformError("a same-frame transform must be identity")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls, frame: str) -> RigidTransform:
        """Construct the identity transform for one named frame."""
        return cls(source_frame=frame, target_frame=frame, matrix=np.eye(4))

    def inverse(self) -> RigidTransform:
        """Return ``T_source_from_target``."""
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]
        result = np.eye(4)
        result[:3, :3] = rotation.T
        result[:3, 3] = -(rotation.T @ translation)
        return RigidTransform(self.target_frame, self.source_frame, result)

    def compose(self, other: RigidTransform) -> RigidTransform:
        """Return ``self @ other`` after checking the intermediate frame.

        If ``self`` is ``T_a_from_b``, ``other`` must be ``T_b_from_c`` and the result
        is ``T_a_from_c``.
        """
        if self.source_frame != other.target_frame:
            raise TransformError(
                "cannot compose transforms: "
                f"{self.source_frame!r} != {other.target_frame!r} at the intermediate frame"
            )
        return RigidTransform(
            source_frame=other.source_frame,
            target_frame=self.target_frame,
            matrix=self.matrix @ other.matrix,
        )

    def transform_points(self, points: npt.ArrayLike) -> FloatArray:
        """Transform a 3-vector or an ``(N, 3)`` array of column-vector points."""
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 1:
            if array.shape != (3,):
                raise TransformError("a point must have shape (3,)")
            if not np.isfinite(array).all():
                raise TransformError("points must contain only finite values")
            return self.matrix[:3, :3] @ array + self.matrix[:3, 3]
        if array.ndim != 2 or array.shape[1] != 3:
            raise TransformError("points must have shape (N, 3)")
        if not np.isfinite(array).all():
            raise TransformError("points must contain only finite values")
        return np.asarray(array @ self.matrix[:3, :3].T + self.matrix[:3, 3], dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        """Serialize the matrix as nested JSON arrays."""
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RigidTransform:
        """Reconstruct and revalidate an SE(3) transform."""
        return cls(
            source_frame=decoded_string(data["source_frame"], "source_frame"),
            target_frame=decoded_string(data["target_frame"], "target_frame"),
            matrix=np.asarray(data["matrix"], dtype=np.float64),
        )


def compose(left: RigidTransform, right: RigidTransform) -> RigidTransform:
    """Explicit free-function spelling of ``left.compose(right)``."""
    return left.compose(right)


def _require_frame(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TransformError(f"{field_name} must be a non-empty string")
