"""Deterministic transform resolution without a graph dependency."""

from __future__ import annotations

from collections import deque

import numpy as np

from camera_rig.core.errors import TransformError
from camera_rig.core.transforms import RigidTransform


class TransformGraph:
    """A small frame graph that resolves direct, inverse, and multi-hop paths."""

    def __init__(self) -> None:
        self._transforms: dict[tuple[str, str], RigidTransform] = {}

    def add(self, transform: RigidTransform) -> None:
        """Add one transform, rejecting duplicate or conflicting directed edges."""
        key = (transform.source_frame, transform.target_frame)
        reverse_key = (transform.target_frame, transform.source_frame)
        existing = self._transforms.get(key)
        reverse = self._transforms.get(reverse_key)
        if existing is not None:
            relation = (
                "duplicate" if np.allclose(existing.matrix, transform.matrix) else "conflicting"
            )
            raise TransformError(f"{relation} transform for {key[0]!r} -> {key[1]!r}")
        if reverse is not None:
            relation = (
                "duplicate"
                if np.allclose(reverse.matrix, transform.inverse().matrix)
                else "conflicting"
            )
            raise TransformError(f"{relation} reverse transform for {key[0]!r} -> {key[1]!r}")
        self._transforms[key] = transform

    def resolve(self, source_frame: str, target_frame: str) -> RigidTransform:
        """Resolve the deterministic shortest path from source to target."""
        if not isinstance(source_frame, str) or not source_frame.strip():
            raise TransformError("source_frame must be a non-empty string")
        if not isinstance(target_frame, str) or not target_frame.strip():
            raise TransformError("target_frame must be a non-empty string")
        if source_frame == target_frame:
            return RigidTransform.identity(source_frame)

        queue: deque[tuple[str, RigidTransform]] = deque(
            [(source_frame, RigidTransform.identity(source_frame))]
        )
        visited = {source_frame}
        while queue:
            current_frame, current_from_source = queue.popleft()
            for neighbor, edge in self._neighbors(current_frame):
                if neighbor in visited:
                    continue
                neighbor_from_source = edge.compose(current_from_source)
                if neighbor == target_frame:
                    return neighbor_from_source
                visited.add(neighbor)
                queue.append((neighbor, neighbor_from_source))
        raise TransformError(f"no transform path from {source_frame!r} to {target_frame!r}")

    def _neighbors(self, frame: str) -> list[tuple[str, RigidTransform]]:
        neighbors: list[tuple[str, RigidTransform]] = []
        for transform in self._transforms.values():
            if transform.source_frame == frame:
                neighbors.append((transform.target_frame, transform))
            elif transform.target_frame == frame:
                neighbors.append((transform.source_frame, transform.inverse()))
        return sorted(neighbors, key=lambda item: item[0])
