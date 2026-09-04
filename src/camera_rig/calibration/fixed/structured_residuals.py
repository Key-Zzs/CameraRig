"""Cross-validated spatial structure tests for reprojection residual vectors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from camera_rig.calibration.pose import project_points_px
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StructuredScope = Literal["frame", "final"]
ReleaseState = Literal["CANDIDATE", "HOLD", "RELEASED"]


@dataclass(frozen=True)
class StructuredResidualThresholds:
    """Candidate v1 thresholds; release values remain HOLD until preregistered."""

    model_name: str = "minimal_image_board_union"
    model_version: str = "structured_residual_v1"
    fold_count: int = 4
    regularization: float = 1e-3
    minimum_corner_count: int = 20
    minimum_test_corners_per_fold: int = 3
    minimum_training_dof_margin: int = 4
    minimum_training_feature_rank: int | None = None
    maximum_regularized_condition_number: float = 1e6
    minimum_observations_per_final_corner: int = 3
    maximum_final_observation_count_ratio: float = 1.25
    permutation_count: int = 999
    permutation_seed: int = 20260904
    maximum_permutation_p_value: float = 0.05
    minimum_cv_explained_fraction: float = 0.20
    minimum_structured_amplitude_px: float = 0.15
    insufficient_support_policy: str = "fail_closed"

    def __post_init__(self) -> None:
        if self.model_name not in _MODEL_COLUMNS:
            raise ContractError("unsupported structured-residual model")
        if self.model_version != "structured_residual_v1":
            raise ContractError("unsupported structured-residual model version")
        if self.fold_count != 4:
            raise ContractError("structured residual v1 requires deterministic 4-fold CV")
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise ContractError("structured-residual regularization must be finite and positive")
        if self.minimum_corner_count < 4:
            raise ContractError("structured-residual minimum corner count is invalid")
        if self.minimum_test_corners_per_fold < 1:
            raise ContractError("structured-residual fold support must be positive")
        if self.minimum_training_dof_margin < 0:
            raise ContractError("structured-residual DOF margin must be non-negative")
        required_rank = self.required_training_feature_rank
        if not 1 <= required_rank <= len(_MODEL_COLUMNS[self.model_name]):
            raise ContractError("structured-residual minimum feature rank is invalid")
        if (
            not math.isfinite(self.maximum_regularized_condition_number)
            or self.maximum_regularized_condition_number <= 1
        ):
            raise ContractError("structured-residual condition limit must exceed one")
        if self.minimum_observations_per_final_corner < 2:
            raise ContractError("final structured corners require repeated observations")
        if (
            not math.isfinite(self.maximum_final_observation_count_ratio)
            or self.maximum_final_observation_count_ratio < 1
        ):
            raise ContractError("final structured observation-count ratio must be at least one")
        if self.permutation_count < 999:
            raise ContractError("structured-residual permutation count must be at least 999")
        if self.permutation_seed < 0:
            raise ContractError("structured-residual permutation seed must be non-negative")
        ratios = (
            self.maximum_permutation_p_value,
            self.minimum_cv_explained_fraction,
        )
        if not all(math.isfinite(value) and 0 < value < 1 for value in ratios):
            raise ContractError("structured-residual probability/effect thresholds are invalid")
        if (
            not math.isfinite(self.minimum_structured_amplitude_px)
            or self.minimum_structured_amplitude_px <= 0
        ):
            raise ContractError("structured-residual amplitude floor must be positive")
        if self.insufficient_support_policy != "fail_closed":
            raise ContractError("structured residual v1 must fail closed on insufficient support")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "fold_count": self.fold_count,
            "regularization": self.regularization,
            "minimum_corner_count": self.minimum_corner_count,
            "minimum_test_corners_per_fold": self.minimum_test_corners_per_fold,
            "minimum_training_dof_margin": self.minimum_training_dof_margin,
            "minimum_training_feature_rank": self.required_training_feature_rank,
            "maximum_regularized_condition_number": (self.maximum_regularized_condition_number),
            "minimum_observations_per_final_corner": (self.minimum_observations_per_final_corner),
            "maximum_final_observation_count_ratio": (self.maximum_final_observation_count_ratio),
            "permutation_count": self.permutation_count,
            "permutation_seed": self.permutation_seed,
            "maximum_permutation_p_value": self.maximum_permutation_p_value,
            "minimum_cv_explained_fraction": self.minimum_cv_explained_fraction,
            "minimum_structured_amplitude_px": self.minimum_structured_amplitude_px,
            "insufficient_support_policy": self.insufficient_support_policy,
        }

    @property
    def required_training_feature_rank(self) -> int:
        return self.minimum_training_feature_rank or len(_MODEL_COLUMNS[self.model_name])


@dataclass(frozen=True)
class StructuredReprojectionPolicy:
    """Explicit v2 HOLD identity; release support is intentionally not implemented."""

    preset: str = "uncertainty_validated_v2"
    release_state: ReleaseState = "HOLD"
    release_criteria_version: str = "structured_reprojection_release_v1"
    release_manifest_sha256: str | None = None
    thresholds: StructuredResidualThresholds = field(default_factory=StructuredResidualThresholds)

    def __post_init__(self) -> None:
        if self.preset != "uncertainty_validated_v2":
            raise ContractError("unsupported structured reprojection preset")
        if self.release_state not in {"CANDIDATE", "HOLD", "RELEASED"}:
            raise ContractError("unsupported structured reprojection release state")
        if not self.release_criteria_version.strip():
            raise ContractError("structured release criteria version must be non-empty")
        digest = self.release_manifest_sha256
        if digest is not None and (
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError("structured release manifest SHA-256 must be lowercase hexadecimal")
        if self.release_state == "RELEASED":
            raise ContractError(
                "uncertainty_validated_v2 is not release-enabled without an authenticated "
                "criteria and passing-holdout loader"
            )

    @property
    def production_eligible(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "release_state": self.release_state,
            "release_criteria_version": self.release_criteria_version,
            "release_manifest_sha256": self.release_manifest_sha256,
            "production_eligible": self.production_eligible,
            "thresholds": self.thresholds.to_dict(),
        }


@dataclass(frozen=True)
class StructuredResidualMetrics:
    """Versioned, JSON-safe structured-residual decision and its full provenance."""

    scope: StructuredScope
    residual_convention: str
    corner_count: int
    model_name: str
    model_version: str
    feature_definition: tuple[str, ...]
    parameter_count: int
    cross_validation_scheme: str
    fold_count: int
    fold_corner_counts: tuple[int, ...]
    fold_training_feature_ranks: tuple[int, ...]
    fold_regularized_condition_numbers: tuple[float | None, ...]
    regularization: float
    baseline_definition: str
    mean_du_px: float
    mean_dv_px: float
    cv_explained_fraction: float | None
    cv_predicted_rmse_px: float | None
    unexplained_rmse_px: float | None
    structured_effect_fraction: float | None
    structured_amplitude_px: float | None
    permutation_count: int
    permutation_seed: int
    permutation_p_value: float | None
    observed_statistic: float | None
    null_median: float | None
    null_p95: float | None
    sufficient_support: bool
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "residual_convention": self.residual_convention,
            "corner_count": self.corner_count,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_definition": list(self.feature_definition),
            "parameter_count": self.parameter_count,
            "cross_validation_scheme": self.cross_validation_scheme,
            "fold_count": self.fold_count,
            "fold_corner_counts": list(self.fold_corner_counts),
            "fold_training_feature_ranks": list(self.fold_training_feature_ranks),
            "fold_regularized_condition_numbers": list(self.fold_regularized_condition_numbers),
            "regularization": self.regularization,
            "baseline_definition": self.baseline_definition,
            "mean_du_px": self.mean_du_px,
            "mean_dv_px": self.mean_dv_px,
            "cv_explained_fraction": self.cv_explained_fraction,
            "cv_predicted_rmse_px": self.cv_predicted_rmse_px,
            "unexplained_rmse_px": self.unexplained_rmse_px,
            "structured_effect_fraction": self.structured_effect_fraction,
            "structured_amplitude_px": self.structured_amplitude_px,
            "permutation_count": self.permutation_count,
            "permutation_seed": self.permutation_seed,
            "permutation_p_value": self.permutation_p_value,
            "observed_statistic": self.observed_statistic,
            "null_median": self.null_median,
            "null_p95": self.null_p95,
            "sufficient_support": self.sufficient_support,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass
class _CornerResidualGroup:
    object_point_m: FloatArray
    projected_uv_px: FloatArray
    residuals: list[FloatArray]


_ALL_FEATURE_DEFINITION = (
    "image_radial_k1:[x*r2,y*r2]",
    "image_tangential_p1:[2xy,r2+2y2]",
    "image_tangential_p2:[r2+2x2,2xy]",
    "image_focal_x:[x,0]",
    "image_focal_y:[0,y]",
    "board_du_x2:[X2,0]",
    "board_du_xy:[XY,0]",
    "board_du_y2:[Y2,0]",
    "board_dv_x2:[0,X2]",
    "board_dv_xy:[0,XY]",
    "board_dv_y2:[0,Y2]",
)
_MODEL_COLUMNS = {
    "image_physical": tuple(range(5)),
    "board_quadratic": tuple(range(5, 11)),
    "minimal_image_board_union": tuple(range(11)),
}
_CV_SCHEME = "deterministic_board_checkerboard_4fold"
_BASELINE = "training_fold_mean_residual_vector"
_RESIDUAL_CONVENTION = "observed_minus_projected"


def evaluate_structured_residuals(
    *,
    object_points_m: npt.ArrayLike,
    projected_points_px: npt.ArrayLike,
    residual_vectors_px: npt.ArrayLike,
    intrinsics: CameraIntrinsics,
    scope: StructuredScope,
    thresholds: StructuredResidualThresholds | None = None,
    board_reference_points_m: npt.ArrayLike | None = None,
) -> StructuredResidualMetrics:
    """Test whether residual vectors are predictably related to image/board location."""
    release = thresholds or StructuredResidualThresholds()
    objects, projected, residuals = _validated_inputs(
        object_points_m, projected_points_px, residual_vectors_px
    )
    count = len(objects)
    mean_residual = np.mean(residuals, axis=0)
    board_reference = _validated_board_reference(board_reference_points_m, objects)
    fold_ids = _spatial_fold_ids(objects[:, :2], board_reference[:, :2])
    fold_counts = tuple(int(np.count_nonzero(fold_ids == fold)) for fold in range(4))
    feature_columns = _MODEL_COLUMNS[release.model_name]
    feature_definition = tuple(_ALL_FEATURE_DEFINITION[index] for index in feature_columns)
    feature_matrix = _feature_matrix(objects, projected, intrinsics, board_reference)[
        :, feature_columns
    ]
    support_failure = _support_failure(count, fold_ids, feature_matrix, release)
    support_diagnostics = _fold_support_diagnostics(
        feature_matrix, fold_ids, release.regularization
    )
    fold_ranks = tuple(item[0] for item in support_diagnostics)
    fold_conditions = tuple(item[1] for item in support_diagnostics)
    if support_failure is not None:
        return StructuredResidualMetrics(
            scope=scope,
            residual_convention=_RESIDUAL_CONVENTION,
            corner_count=count,
            model_name=release.model_name,
            model_version=release.model_version,
            feature_definition=feature_definition,
            parameter_count=len(feature_columns),
            cross_validation_scheme=_CV_SCHEME,
            fold_count=release.fold_count,
            fold_corner_counts=fold_counts,
            fold_training_feature_ranks=fold_ranks,
            fold_regularized_condition_numbers=fold_conditions,
            regularization=release.regularization,
            baseline_definition=_BASELINE,
            mean_du_px=float(mean_residual[0]),
            mean_dv_px=float(mean_residual[1]),
            cv_explained_fraction=None,
            cv_predicted_rmse_px=None,
            unexplained_rmse_px=None,
            structured_effect_fraction=None,
            structured_amplitude_px=None,
            permutation_count=release.permutation_count,
            permutation_seed=release.permutation_seed,
            permutation_p_value=None,
            observed_statistic=None,
            null_median=None,
            null_p95=None,
            sufficient_support=False,
            passed=False,
            failure_reasons=(support_failure,),
        )

    operators = _fold_operators(feature_matrix, fold_ids, release.regularization)
    observed = _cross_validated_statistics(residuals[None, :, :], operators)
    observed_r2 = float(observed[0][0])
    predicted_rmse = float(observed[1][0])
    unexplained_rmse = float(observed[2][0])
    amplitude = float(observed[3][0])

    rng = np.random.default_rng(release.permutation_seed)
    permutations = np.stack(
        [rng.permutation(count) for _ in range(release.permutation_count)], axis=0
    )
    permuted = residuals[permutations]
    null_r2 = _cross_validated_statistics(permuted, operators)[0]
    p_value = float((1 + np.count_nonzero(null_r2 >= observed_r2)) / (len(null_r2) + 1))
    null_median = float(np.median(null_r2))
    null_p95 = float(np.percentile(null_r2, 95))
    effect = float(max(0.0, min(1.0, observed_r2)))

    significant = p_value <= release.maximum_permutation_p_value
    material_effect = effect >= release.minimum_cv_explained_fraction
    material_amplitude = amplitude >= release.minimum_structured_amplitude_px
    structured_failure = significant and material_effect and material_amplitude
    reasons = ("STRUCTURED_RESIDUAL_MODEL_MISMATCH",) if structured_failure else ()
    return StructuredResidualMetrics(
        scope=scope,
        residual_convention=_RESIDUAL_CONVENTION,
        corner_count=count,
        model_name=release.model_name,
        model_version=release.model_version,
        feature_definition=feature_definition,
        parameter_count=len(feature_columns),
        cross_validation_scheme=_CV_SCHEME,
        fold_count=release.fold_count,
        fold_corner_counts=fold_counts,
        fold_training_feature_ranks=fold_ranks,
        fold_regularized_condition_numbers=fold_conditions,
        regularization=release.regularization,
        baseline_definition=_BASELINE,
        mean_du_px=float(mean_residual[0]),
        mean_dv_px=float(mean_residual[1]),
        cv_explained_fraction=observed_r2,
        cv_predicted_rmse_px=predicted_rmse,
        unexplained_rmse_px=unexplained_rmse,
        structured_effect_fraction=effect,
        structured_amplitude_px=amplitude,
        permutation_count=release.permutation_count,
        permutation_seed=release.permutation_seed,
        permutation_p_value=p_value,
        observed_statistic=observed_r2,
        null_median=null_median,
        null_p95=null_p95,
        sufficient_support=True,
        passed=not structured_failure,
        failure_reasons=reasons,
    )


def evaluate_observation_structured_residuals(
    observation: TargetObservation,
    pose: RigidTransform,
    intrinsics: CameraIntrinsics,
    *,
    scope: StructuredScope = "frame",
    thresholds: StructuredResidualThresholds | None = None,
    board_reference_points_m: npt.ArrayLike | None = None,
) -> StructuredResidualMetrics:
    """Evaluate one observation using the canonical observed-minus-projected sign."""
    projected = project_points_px(observation.object_points_m, pose, intrinsics)
    observed = np.asarray(observation.image_points_px, dtype=np.float64)
    return evaluate_structured_residuals(
        object_points_m=observation.object_points_m,
        projected_points_px=projected,
        residual_vectors_px=observed - projected,
        intrinsics=intrinsics,
        scope=scope,
        thresholds=thresholds,
        board_reference_points_m=board_reference_points_m,
    )


def evaluate_final_shared_structured_residuals(
    observations: Sequence[TargetObservation],
    pose: RigidTransform,
    intrinsics: CameraIntrinsics,
    *,
    thresholds: StructuredResidualThresholds | None = None,
) -> dict[str, object]:
    """Average repeated residuals by physical corner ID before the final structure test."""
    release = thresholds or StructuredResidualThresholds()
    if not observations:
        raise ContractError("final structured residual evaluation requires observations")
    grouped: dict[int, _CornerResidualGroup] = {}
    for observation in observations:
        projected = project_points_px(observation.object_points_m, pose, intrinsics)
        observed = np.asarray(observation.image_points_px, dtype=np.float64)
        residuals = observed - projected
        for index, point_id in enumerate(observation.point_ids):
            point = np.asarray(observation.object_points_m[index], dtype=np.float64)
            projection = np.asarray(projected[index], dtype=np.float64)
            group = grouped.setdefault(
                point_id,
                _CornerResidualGroup(point, projection, []),
            )
            if not np.allclose(group.object_point_m, point, rtol=0.0, atol=1e-12):
                raise ContractError("one corner ID maps to inconsistent target geometry")
            group.residuals.append(np.asarray(residuals[index], dtype=np.float64))

    statistics: list[dict[str, object]] = []
    eligible_objects: list[FloatArray] = []
    eligible_projected: list[FloatArray] = []
    eligible_residuals: list[FloatArray] = []
    eligible_counts: list[int] = []
    for point_id, group in sorted(grouped.items()):
        values = np.asarray(group.residuals, dtype=np.float64)
        count = len(values)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=1) if count > 1 else np.zeros(2, dtype=np.float64)
        standard_error = std / math.sqrt(count)
        eligible = count >= release.minimum_observations_per_final_corner
        statistics.append(
            {
                "point_id": point_id,
                "count": count,
                "mean_du_px": float(mean[0]),
                "mean_dv_px": float(mean[1]),
                "std_u_px": float(std[0]),
                "std_v_px": float(std[1]),
                "standard_error_u_px": float(standard_error[0]),
                "standard_error_v_px": float(standard_error[1]),
                "eligible": eligible,
            }
        )
        if eligible:
            eligible_objects.append(group.object_point_m)
            eligible_projected.append(group.projected_uv_px)
            eligible_residuals.append(mean)
            eligible_counts.append(count)

    count_ratio = max(eligible_counts) / min(eligible_counts) if eligible_counts else None
    balanced_support = (
        count_ratio is not None and count_ratio <= release.maximum_final_observation_count_ratio
    )
    evaluation_objects = eligible_objects or [group.object_point_m for group in grouped.values()]
    evaluation_projected = eligible_projected or [
        group.projected_uv_px for group in grouped.values()
    ]
    evaluation_residuals = eligible_residuals or [
        np.mean(group.residuals, axis=0) for group in grouped.values()
    ]
    board_reference = np.asarray(
        [group.object_point_m for _point_id, group in sorted(grouped.items())],
        dtype=np.float64,
    )
    metrics = evaluate_structured_residuals(
        object_points_m=np.asarray(evaluation_objects, dtype=np.float64),
        projected_points_px=np.asarray(evaluation_projected, dtype=np.float64),
        residual_vectors_px=np.asarray(evaluation_residuals, dtype=np.float64),
        intrinsics=intrinsics,
        scope="final",
        thresholds=release,
        board_reference_points_m=board_reference,
    )
    forced_reason = None
    if not eligible_objects:
        forced_reason = "STRUCTURED_RESIDUAL_INSUFFICIENT_REPEAT_SUPPORT"
    elif not balanced_support:
        forced_reason = "STRUCTURED_RESIDUAL_UNBALANCED_FINAL_SUPPORT"
    if forced_reason is not None:
        metrics = replace(
            metrics,
            sufficient_support=False,
            passed=False,
            failure_reasons=tuple(dict.fromkeys((*metrics.failure_reasons, forced_reason))),
        )
    return {
        "aggregation": "equal_weight_corner_mean_over_accepted_inlier_frames",
        "minimum_observations_per_corner": release.minimum_observations_per_final_corner,
        "maximum_observation_count_ratio": release.maximum_final_observation_count_ratio,
        "observed_corner_count": len(grouped),
        "eligible_corner_count": len(eligible_objects),
        "eligible_observation_count_ratio": count_ratio,
        "balanced_repeat_support": balanced_support,
        "corner_statistics": statistics,
        "structured_metrics": metrics.to_dict(),
    }


def _validated_inputs(
    object_points_m: npt.ArrayLike,
    projected_points_px: npt.ArrayLike,
    residual_vectors_px: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    objects = np.asarray(object_points_m, dtype=np.float64)
    projected = np.asarray(projected_points_px, dtype=np.float64)
    residuals = np.asarray(residual_vectors_px, dtype=np.float64)
    if objects.ndim != 2 or objects.shape[1] != 3:
        raise ContractError("structured residual object points must have shape (N, 3)")
    if projected.shape != (len(objects), 2) or residuals.shape != (len(objects), 2):
        raise ContractError("structured residual image arrays must have shape (N, 2)")
    if not len(objects) or not (
        np.isfinite(objects).all() and np.isfinite(projected).all() and np.isfinite(residuals).all()
    ):
        raise ContractError("structured residual inputs must be non-empty and finite")
    return objects, projected, residuals


def _spatial_fold_ids(board_xy: FloatArray, reference_xy: FloatArray) -> IntArray:
    """Assign deterministic checkerboard folds so adjacent corners are held apart."""
    rounded = np.round(board_xy, decimals=12)
    reference = np.round(reference_xy, decimals=12)
    x_values = np.unique(reference[:, 0])
    y_values = np.unique(reference[:, 1])
    x_rank = np.searchsorted(x_values, rounded[:, 0])
    y_rank = np.searchsorted(y_values, rounded[:, 1])
    if (
        np.any(x_rank >= len(x_values))
        or np.any(y_rank >= len(y_values))
        or not np.array_equal(x_values[x_rank], rounded[:, 0])
        or not np.array_equal(y_values[y_rank], rounded[:, 1])
    ):
        raise ContractError("observed board points are absent from the canonical reference")
    return np.asarray(2 * (x_rank % 2) + (y_rank % 2), dtype=np.int64)


def _feature_matrix(
    objects: FloatArray,
    projected: FloatArray,
    intrinsics: CameraIntrinsics,
    board_reference: FloatArray,
) -> FloatArray:
    x = (projected[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (projected[:, 1] - intrinsics.cy) / intrinsics.fy
    focal_reference = math.sqrt(intrinsics.fx * intrinsics.fy)
    x_pixel_scale = intrinsics.fx / focal_reference
    y_pixel_scale = intrinsics.fy / focal_reference
    radius_squared = x * x + y * y
    board_center = np.mean(board_reference[:, :2], axis=0, keepdims=True)
    reference_centered = board_reference[:, :2] - board_center
    board = objects[:, :2] - board_center
    board_scale = float(np.sqrt(np.mean(np.sum(np.square(reference_centered), axis=1))))
    if not math.isfinite(board_scale) or board_scale <= 1e-12:
        return np.zeros((2 * len(objects), 11), dtype=np.float64)
    board /= board_scale
    board_x = board[:, 0]
    board_y = board[:, 1]
    matrix = np.zeros((len(objects), 2, 11), dtype=np.float64)
    matrix[:, 0, 0] = x_pixel_scale * x * radius_squared
    matrix[:, 1, 0] = y_pixel_scale * y * radius_squared
    matrix[:, 0, 1] = x_pixel_scale * 2.0 * x * y
    matrix[:, 1, 1] = y_pixel_scale * (radius_squared + 2.0 * y * y)
    matrix[:, 0, 2] = x_pixel_scale * (radius_squared + 2.0 * x * x)
    matrix[:, 1, 2] = y_pixel_scale * 2.0 * x * y
    matrix[:, 0, 3] = x_pixel_scale * x
    matrix[:, 1, 4] = y_pixel_scale * y
    board_terms = (board_x * board_x, board_x * board_y, board_y * board_y)
    for offset, term in enumerate(board_terms):
        matrix[:, 0, 5 + offset] = term
        matrix[:, 1, 8 + offset] = term
    return matrix.reshape(2 * len(objects), 11)


def _validated_board_reference(value: npt.ArrayLike | None, observed: FloatArray) -> FloatArray:
    reference = observed if value is None else np.asarray(value, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 3 or not len(reference):
        raise ContractError("structured residual board reference must have shape (N, 3)")
    if not np.isfinite(reference).all():
        raise ContractError("structured residual board reference must be finite")
    return reference


@dataclass(frozen=True)
class _FoldOperator:
    train_indices: IntArray
    test_indices: IntArray
    train_operator: FloatArray
    test_matrix: FloatArray


def _fold_operators(
    features: FloatArray, fold_ids: IntArray, regularization: float
) -> tuple[_FoldOperator, ...]:
    point_features = features.reshape(len(fold_ids), 2, features.shape[1])
    values: list[_FoldOperator] = []
    for fold in range(4):
        test = np.flatnonzero(fold_ids == fold)
        train = np.flatnonzero(fold_ids != fold)
        train_features = point_features[train]
        test_features = point_features[test]
        feature_mean = np.mean(train_features, axis=0, keepdims=True)
        train_matrix = (train_features - feature_mean).reshape(2 * len(train), -1)
        test_matrix = (test_features - feature_mean).reshape(2 * len(test), -1)
        scale = np.sqrt(np.mean(np.square(train_matrix), axis=0))
        scale = np.where(scale > 1e-12, scale, 1.0)
        train_scaled = train_matrix / scale
        test_scaled = test_matrix / scale
        left, singular, right_transpose = np.linalg.svd(train_scaled, full_matrices=False)
        shrinkage = singular / (np.square(singular) + regularization)
        operator = (right_transpose.T * shrinkage) @ left.T
        values.append(_FoldOperator(train, test, operator, test_scaled))
    return tuple(values)


def _cross_validated_statistics(
    residual_sets: FloatArray, operators: tuple[_FoldOperator, ...]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    batch = len(residual_sets)
    total_error = np.zeros(batch, dtype=np.float64)
    total_baseline = np.zeros(batch, dtype=np.float64)
    total_predicted = np.zeros(batch, dtype=np.float64)
    total_structured = np.zeros(batch, dtype=np.float64)
    point_count = 0
    for fold in operators:
        train_values = residual_sets[:, fold.train_indices, :]
        test_values = residual_sets[:, fold.test_indices, :]
        baseline = np.mean(train_values, axis=1)
        centered_train = train_values - baseline[:, None, :]
        coefficients = centered_train.reshape(batch, -1) @ fold.train_operator.T
        predicted_centered = coefficients @ fold.test_matrix.T
        predicted = predicted_centered.reshape(batch, len(fold.test_indices), 2)
        predicted += baseline[:, None, :]
        error = test_values - predicted
        baseline_error = test_values - baseline[:, None, :]
        total_error += np.sum(np.square(error), axis=(1, 2))
        total_baseline += np.sum(np.square(baseline_error), axis=(1, 2))
        total_predicted += np.sum(np.square(predicted), axis=(1, 2))
        total_structured += np.sum(np.square(predicted - baseline[:, None, :]), axis=(1, 2))
        point_count += len(fold.test_indices)
    explained = np.divide(
        total_baseline - total_error,
        total_baseline,
        out=np.zeros_like(total_baseline),
        where=total_baseline > 1e-15,
    )
    predicted_rmse = np.sqrt(total_predicted / point_count)
    unexplained_rmse = np.sqrt(total_error / point_count)
    amplitude = np.sqrt(total_structured / point_count)
    return explained, predicted_rmse, unexplained_rmse, amplitude


def _support_failure(
    corner_count: int,
    fold_ids: IntArray,
    features: FloatArray,
    thresholds: StructuredResidualThresholds,
) -> str | None:
    if corner_count < thresholds.minimum_corner_count:
        return "STRUCTURED_RESIDUAL_INSUFFICIENT_SUPPORT"
    for fold in range(thresholds.fold_count):
        test_count = int(np.count_nonzero(fold_ids == fold))
        train_count = corner_count - test_count
        if test_count < thresholds.minimum_test_corners_per_fold:
            return "STRUCTURED_RESIDUAL_INSUFFICIENT_SUPPORT"
        available_dof = 2 * train_count - 2
        if available_dof < features.shape[1] + thresholds.minimum_training_dof_margin:
            return "STRUCTURED_RESIDUAL_INSUFFICIENT_SUPPORT"
    diagnostics = _fold_support_diagnostics(features, fold_ids, thresholds.regularization)
    if any(rank < thresholds.required_training_feature_rank for rank, _condition in diagnostics):
        return "STRUCTURED_RESIDUAL_INSUFFICIENT_SUPPORT"
    if any(
        condition is None or condition > thresholds.maximum_regularized_condition_number
        for _rank, condition in diagnostics
    ):
        return "STRUCTURED_RESIDUAL_INSUFFICIENT_SUPPORT"
    return None


def _fold_support_diagnostics(
    features: FloatArray, fold_ids: IntArray, regularization: float
) -> tuple[tuple[int, float | None], ...]:
    point_features = features.reshape(len(fold_ids), 2, features.shape[1])
    values: list[tuple[int, float | None]] = []
    for fold in range(4):
        train = np.flatnonzero(fold_ids != fold)
        if not len(train):
            values.append((0, None))
            continue
        train_features = point_features[train]
        feature_mean = np.mean(train_features, axis=0, keepdims=True)
        matrix = (train_features - feature_mean).reshape(2 * len(train), -1)
        scale = np.sqrt(np.mean(np.square(matrix), axis=0))
        scale = np.where(scale > 1e-12, scale, 1.0)
        singular = np.linalg.svd(matrix / scale, compute_uv=False)
        tolerance = (
            max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
            if len(singular) and singular[0] > 0
            else 0.0
        )
        rank = int(np.count_nonzero(singular > tolerance))
        maximum = float(singular[0]) if len(singular) else 0.0
        minimum = float(singular[-1]) if len(singular) else 0.0
        condition = math.sqrt(
            (maximum * maximum + regularization) / (minimum * minimum + regularization)
        )
        values.append((rank, condition))
    return tuple(values)
