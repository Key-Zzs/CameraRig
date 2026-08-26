"""Robust aggregation utilities for a fixed camera and fixed planar target."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from camera_rig.core.errors import ContractError
from camera_rig.core.transforms import RigidTransform


@dataclass(frozen=True)
class PoseDelta:
    """Frame-explicit translation and rotation distance between two poses."""

    translation_m: float
    rotation_rad: float

    @property
    def translation_mm(self) -> float:
        return self.translation_m * 1000.0

    @property
    def rotation_deg(self) -> float:
        return math.degrees(self.rotation_rad)


def pose_delta(left: RigidTransform, right: RigidTransform) -> PoseDelta:
    """Return translation distance and SO(3) geodesic angle."""
    if (left.source_frame, left.target_frame) != (right.source_frame, right.target_frame):
        raise ContractError("pose distance requires matching transform frames")
    translation = float(np.linalg.norm(left.matrix[:3, 3] - right.matrix[:3, 3]))
    relative = left.matrix[:3, :3].T @ right.matrix[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return PoseDelta(translation_m=translation, rotation_rad=math.acos(cosine))


def pose_medoid_index(
    poses: list[RigidTransform],
    *,
    translation_scale_m: float = 0.005,
    rotation_scale_rad: float = math.radians(0.5),
) -> int:
    """Choose a deterministic robust medoid using normalized SE(3) distances."""
    if not poses:
        raise ContractError("pose medoid requires at least one pose")
    if translation_scale_m <= 0 or rotation_scale_rad <= 0:
        raise ContractError("pose medoid scales must be positive")
    costs = np.zeros(len(poses), dtype=np.float64)
    for left_index, left in enumerate(poses):
        for right_index in range(left_index + 1, len(poses)):
            delta = pose_delta(left, poses[right_index])
            cost = (
                delta.translation_m / translation_scale_m + delta.rotation_rad / rotation_scale_rad
            )
            costs[left_index] += cost
            costs[right_index] += cost
    return int(np.argmin(costs))


def pose_inlier_indices(
    poses: list[RigidTransform],
    medoid_index: int,
    *,
    maximum_translation_mm: float,
    maximum_rotation_deg: float,
) -> list[int]:
    """Apply the persisted pose-level outlier policy relative to a medoid."""
    if medoid_index < 0 or medoid_index >= len(poses):
        raise ContractError("pose medoid index is out of range")
    medoid = poses[medoid_index]
    return [
        index
        for index, pose in enumerate(poses)
        if (delta := pose_delta(pose, medoid)).translation_mm <= maximum_translation_mm
        and delta.rotation_deg <= maximum_rotation_deg
    ]


def distribution(values: list[float]) -> dict[str, float]:
    """Persist a complete deterministic scalar distribution."""
    if not values:
        raise ContractError("cannot summarize an empty distribution")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ContractError("distribution values must be finite")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
        "std": float(np.std(array)),
    }


def even_odd_partition(indices: list[int]) -> tuple[list[int], list[int]]:
    """Split frame indices deterministically while preserving temporal coverage."""
    first = [index for index in indices if index % 2 == 0]
    second = [index for index in indices if index % 2 == 1]
    if not first or not second:
        raise ContractError("split-half validation requires both even and odd frames")
    return first, second
