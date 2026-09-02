"""Diagnostic-only reprojection residual vector-field summaries."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose import project_points_px
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

FloatArray = npt.NDArray[np.float64]


def evaluate_residual_vector_field(
    observation: TargetObservation,
    pose: RigidTransform,
    intrinsics: CameraIntrinsics,
) -> dict[str, object]:
    """Describe residual magnitude and structure without adding a hard acceptance gate."""
    projected = project_points_px(observation.object_points_m, pose, intrinsics)
    observed = np.asarray(observation.image_points_px, dtype=np.float64)
    vectors = observed - projected
    norms = np.linalg.norm(vectors, axis=1)
    center = np.asarray([intrinsics.cx, intrinsics.cy], dtype=np.float64)
    centered = projected - center
    radii_px = np.linalg.norm(centered, axis=1)
    maximum_radius_px = max(
        math.hypot(x - intrinsics.cx, y - intrinsics.cy)
        for x in (0.0, float(intrinsics.width - 1))
        for y in (0.0, float(intrinsics.height - 1))
    )
    normalized_radius = radii_px / maximum_radius_px
    radial_unit = np.divide(
        centered,
        radii_px[:, None],
        out=np.zeros_like(centered),
        where=radii_px[:, None] > 1e-12,
    )
    tangential_unit = np.stack((-radial_unit[:, 1], radial_unit[:, 0]), axis=1)
    radial_component = np.sum(vectors * radial_unit, axis=1)
    tangential_component = np.sum(vectors * tangential_unit, axis=1)
    per_corner = [
        {
            "point_id": point_id,
            "observed_uv_px": [float(observed[index, 0]), float(observed[index, 1])],
            "projected_uv_px": [float(projected[index, 0]), float(projected[index, 1])],
            "du_px": float(vectors[index, 0]),
            "dv_px": float(vectors[index, 1]),
            "norm_px": float(norms[index]),
            "normalized_image_radius": float(normalized_radius[index]),
            "radial_component_px": float(radial_component[index]),
            "tangential_component_px": float(tangential_component[index]),
        }
        for index, point_id in enumerate(observation.point_ids)
    ]
    return {
        "role": "diagnostic_only_not_a_hard_gate",
        "per_corner": per_corner,
        "aggregate": {
            "count": len(norms),
            "mean_du_px": float(np.mean(vectors[:, 0])),
            "mean_dv_px": float(np.mean(vectors[:, 1])),
            "rmse_px": float(np.sqrt(np.mean(np.square(norms)))),
            "median_px": float(np.median(norms)),
            "p95_px": float(np.percentile(norms, 95)),
            "maximum_px": float(np.max(norms)),
            "median_absolute_deviation_px": float(np.median(np.abs(norms - np.median(norms)))),
            "fraction_above_1_px": float(np.mean(norms > 1.0)),
            "fraction_above_2_px": float(np.mean(norms > 2.0)),
            "mean_radial_component_px": float(np.mean(radial_component)),
            "mean_tangential_component_px": float(np.mean(tangential_component)),
        },
        "normalized_radius_bins": _radius_bins(normalized_radius, norms, radial_component, vectors),
        "trends": {
            "norm_vs_radius_pearson": _pearson(normalized_radius, norms),
            "norm_vs_radius_spearman": _spearman(normalized_radius, norms),
            "radial_component_vs_radius_pearson": _pearson(normalized_radius, radial_component),
            "radial_component_vs_radius_spearman": _spearman(normalized_radius, radial_component),
            "tangential_component_vs_radius_pearson": _pearson(
                normalized_radius, tangential_component
            ),
            "tangential_component_vs_radius_spearman": _spearman(
                normalized_radius, tangential_component
            ),
        },
        "board_coordinate_polynomial": _board_coordinate_polynomial(
            np.asarray(observation.object_points_m, dtype=np.float64), vectors
        ),
        "quadrants": _quadrants(centered, vectors, norms, radial_component),
    }


def _radius_bins(
    radius: FloatArray,
    norms: FloatArray,
    radial: FloatArray,
    vectors: FloatArray,
) -> list[dict[str, object]]:
    edges = (0.0, 0.25, 0.5, 0.75, math.inf)
    bins: list[dict[str, object]] = []
    for index, (lower, upper) in enumerate(pairwise(edges)):
        mask = (radius >= lower) & (radius < upper if index < 3 else radius <= upper)
        bins.append(
            {
                "minimum_normalized_radius": lower,
                "maximum_normalized_radius": None if math.isinf(upper) else upper,
                **_masked_statistics(mask, norms, radial, vectors),
            }
        )
    return bins


def _quadrants(
    centered: FloatArray,
    vectors: FloatArray,
    norms: FloatArray,
    radial: FloatArray,
) -> dict[str, object]:
    masks = {
        "upper_left": (centered[:, 0] < 0) & (centered[:, 1] < 0),
        "upper_right": (centered[:, 0] >= 0) & (centered[:, 1] < 0),
        "lower_left": (centered[:, 0] < 0) & (centered[:, 1] >= 0),
        "lower_right": (centered[:, 0] >= 0) & (centered[:, 1] >= 0),
    }
    return {name: _masked_statistics(mask, norms, radial, vectors) for name, mask in masks.items()}


def _board_coordinate_polynomial(points: FloatArray, vectors: FloatArray) -> dict[str, object]:
    coordinates = points[:, :2]
    centered = coordinates - np.mean(coordinates, axis=0)
    scale = np.max(np.abs(centered), axis=0)
    normalized = np.divide(
        centered,
        scale,
        out=np.zeros_like(centered),
        where=scale > 1e-12,
    )
    x = normalized[:, 0]
    y = normalized[:, 1]
    design = np.column_stack(
        (
            np.ones(len(x)),
            x,
            y,
            x * x,
            x * y,
            y * y,
            x * x * x,
            x * x * y,
            x * y * y,
            y * y * y,
        )
    )
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design, vectors, rcond=None)
    predicted = design @ coefficients
    centered_vectors = vectors - np.mean(vectors, axis=0)
    total = float(np.sum(np.square(centered_vectors)))
    unexplained = float(np.sum(np.square(vectors - predicted)))
    r_squared = None if total <= 1e-15 else float(1.0 - unexplained / total)
    return {
        "basis": ["1", "x", "y", "x2", "xy", "y2", "x3", "x2y", "xy2", "y3"],
        "coordinate_normalization": "centered_and_per_axis_max_abs_scaled",
        "rank": int(rank),
        "du_coefficients_px": [float(value) for value in coefficients[:, 0]],
        "dv_coefficients_px": [float(value) for value in coefficients[:, 1]],
        "vector_r_squared": r_squared,
        "fitted_vector_rmse_px": float(np.sqrt(np.mean(np.sum(np.square(predicted), axis=1)))),
        "unexplained_vector_rmse_px": float(
            np.sqrt(np.mean(np.sum(np.square(vectors - predicted), axis=1)))
        ),
    }


def _masked_statistics(
    mask: npt.NDArray[np.bool_],
    norms: FloatArray,
    radial: FloatArray,
    vectors: FloatArray,
) -> dict[str, object]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "count": 0,
            "mean_du_px": None,
            "mean_dv_px": None,
            "rmse_px": None,
            "p95_px": None,
            "mean_radial_component_px": None,
        }
    selected_norms = norms[mask]
    return {
        "count": count,
        "mean_du_px": float(np.mean(vectors[mask, 0])),
        "mean_dv_px": float(np.mean(vectors[mask, 1])),
        "rmse_px": float(np.sqrt(np.mean(np.square(selected_norms)))),
        "p95_px": float(np.percentile(selected_norms, 95)),
        "mean_radial_component_px": float(np.mean(radial[mask])),
    }


def _pearson(left: FloatArray, right: FloatArray) -> float | None:
    if len(left) < 2 or float(np.std(left)) <= 1e-15 or float(np.std(right)) <= 1e-15:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        raise ContractError("residual trend correlation is non-finite")
    return value


def _spearman(left: FloatArray, right: FloatArray) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks
