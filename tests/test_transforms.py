from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from camera_rig.core.errors import TransformError
from camera_rig.core.transforms import RigidTransform, compose


def test_identity() -> None:
    transform = RigidTransform.identity("color")
    np.testing.assert_allclose(transform.matrix, np.eye(4))


def test_inverse_and_point_transform(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    transform = make_transform("color", "workspace", (1.0, 2.0, 3.0))
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    transformed = transform.transform_points(points)
    np.testing.assert_allclose(transformed, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    np.testing.assert_allclose(transform.inverse().transform_points(transformed), points)
    np.testing.assert_allclose(transform.transform_points([1.0, 0.0, 0.0]), [2.0, 2.0, 3.0])


def test_compose_checks_and_preserves_frames(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    workspace_from_color = make_transform("color", "workspace", (1.0, 0.0, 0.0))
    color_from_ir = make_transform("ir_left", "color", (0.0, 2.0, 0.0))
    result = compose(workspace_from_color, color_from_ir)
    assert (result.source_frame, result.target_frame) == ("ir_left", "workspace")
    np.testing.assert_allclose(result.matrix[:3, 3], [1.0, 2.0, 0.0])


def test_compose_rejects_frame_mismatch(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    with pytest.raises(TransformError, match="intermediate frame"):
        make_transform("color", "workspace").compose(make_transform("ir_left", "depth"))


def test_wrong_shape_is_rejected() -> None:
    with pytest.raises(TransformError, match="shape"):
        RigidTransform("a", "b", np.eye(3))


def test_nan_is_rejected() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = np.nan
    with pytest.raises(TransformError, match="finite"):
        RigidTransform("a", "b", matrix)


def test_non_orthonormal_rotation_is_rejected() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = 2.0
    with pytest.raises(TransformError, match="orthonormal"):
        RigidTransform("a", "b", matrix)


def test_negative_rotation_determinant_is_rejected() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = -1.0
    with pytest.raises(TransformError, match="determinant"):
        RigidTransform("a", "b", matrix)


def test_invalid_last_row_is_rejected() -> None:
    matrix = np.eye(4)
    matrix[3, 0] = 1.0
    with pytest.raises(TransformError, match="last row"):
        RigidTransform("a", "b", matrix)


def test_same_frame_non_identity_is_rejected() -> None:
    matrix = np.eye(4)
    matrix[0, 3] = 1.0
    with pytest.raises(TransformError, match="same-frame"):
        RigidTransform("a", "a", matrix)


def test_transform_matrix_is_read_only(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    transform = make_transform("a", "b")
    with pytest.raises(ValueError, match="read-only"):
        transform.matrix[0, 0] = 2.0
