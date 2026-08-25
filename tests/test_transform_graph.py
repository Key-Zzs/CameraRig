from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from camera_rig.core.errors import TransformError
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.core.transforms import RigidTransform


def test_direct_and_inverse_resolution(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    graph = TransformGraph()
    graph.add(make_transform("color", "workspace", (1.0, 0.0, 0.0)))
    np.testing.assert_allclose(graph.resolve("color", "workspace").matrix[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(graph.resolve("workspace", "color").matrix[:3, 3], [-1, 0, 0])


def test_multi_hop_resolution(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    graph = TransformGraph()
    graph.add(make_transform("ir_left", "color", (0.0, 2.0, 0.0)))
    graph.add(make_transform("color", "workspace", (1.0, 0.0, 0.0)))
    resolved = graph.resolve("ir_left", "workspace")
    assert (resolved.source_frame, resolved.target_frame) == ("ir_left", "workspace")
    np.testing.assert_allclose(resolved.matrix[:3, 3], [1, 2, 0])


def test_missing_path_is_rejected() -> None:
    with pytest.raises(TransformError, match="no transform path"):
        TransformGraph().resolve("a", "b")


def test_duplicate_and_conflicting_edges_are_rejected(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    graph = TransformGraph()
    graph.add(make_transform("a", "b", (1.0, 0.0, 0.0)))
    with pytest.raises(TransformError, match="duplicate"):
        graph.add(make_transform("a", "b", (1.0, 0.0, 0.0)))
    with pytest.raises(TransformError, match="conflicting"):
        graph.add(make_transform("a", "b", (2.0, 0.0, 0.0)))
    with pytest.raises(TransformError, match="duplicate reverse"):
        graph.add(make_transform("b", "a", (-1.0, 0.0, 0.0)))


def test_equal_length_paths_are_deterministic(
    make_transform: Callable[[str, str, tuple[float, float, float]], RigidTransform],
) -> None:
    graph = TransformGraph()
    graph.add(make_transform("source", "b", (20.0, 0.0, 0.0)))
    graph.add(make_transform("b", "target"))
    graph.add(make_transform("source", "a", (10.0, 0.0, 0.0)))
    graph.add(make_transform("a", "target"))
    np.testing.assert_allclose(graph.resolve("source", "target").matrix[:3, 3], [10.0, 0.0, 0.0])
