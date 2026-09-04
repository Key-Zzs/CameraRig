from __future__ import annotations

import math

import numpy as np
import pytest

from camera_rig.calibration.fixed.structured_residuals import (
    StructuredReprojectionPolicy,
    StructuredResidualThresholds,
    _feature_matrix,
    _spatial_fold_ids,
    evaluate_final_shared_structured_residuals,
    evaluate_observation_structured_residuals,
    evaluate_structured_residuals,
)
from camera_rig.calibration.pose import project_points_px
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

pytest.importorskip("cv2")


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        frame="camera/color_optical",
        width=1280,
        height=720,
        fx=900.0,
        fy=905.0,
        cx=639.5,
        cy=359.5,
        distortion_model="none",
    )


def _points(square_m: float = 0.03) -> np.ndarray:
    return np.asarray(
        [[square_m * column, square_m * row, 0.0] for row in range(5) for column in range(7)],
        dtype=np.float64,
    )


def _pose(scale: float = 1.0) -> RigidTransform:
    angle = math.radians(25.0)
    tilt = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = tilt @ np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = scale * np.asarray([-0.09, 0.06, 0.72])
    return RigidTransform("target", "camera/color_optical", matrix)


def _observation(points: np.ndarray, pixels: np.ndarray) -> TargetObservation:
    return TargetObservation(
        plugin_name="synthetic-grid",
        target_frame="target",
        point_ids=tuple(range(len(points))),
        image_points_px=pixels,
        object_points_m=points,
        image_size=(1280, 720),
        quality=QualityReport(True),
    )


def _structured(pattern: str, points: np.ndarray, projected: np.ndarray) -> np.ndarray:
    intrinsics = _intrinsics()
    x = (projected[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (projected[:, 1] - intrinsics.cy) / intrinsics.fy
    radius_squared = x * x + y * y
    if pattern == "radial":
        values = np.stack((x * radius_squared, y * radius_squared), axis=1)
    elif pattern == "tangential":
        values = np.stack(
            (2.0 * x * y, radius_squared + 2.0 * y * y),
            axis=1,
        )
    elif pattern == "board_warp":
        board = points[:, :2] - np.mean(points[:, :2], axis=0)
        scale = np.sqrt(np.mean(np.sum(np.square(board), axis=1)))
        X, Y = (board / scale).T
        values = np.stack((X * X + 0.5 * X * Y, Y * Y - 0.3 * X * Y), axis=1)
    elif pattern == "local":
        board = points[:, :2]
        center = np.median(board, axis=0)
        smooth = 1.0 / (
            1.0
            + np.exp(
                -30.0
                * (
                    (board[:, 0] - center[0]) / np.ptp(board[:, 0])
                    + (board[:, 1] - center[1]) / np.ptp(board[:, 1])
                    - 0.35
                )
            )
        )
        values = np.stack((smooth, -0.6 * smooth), axis=1)
    else:
        raise ValueError(pattern)
    return values * (0.5 / np.sqrt(np.mean(np.sum(np.square(values), axis=1))))


def _evaluate(points: np.ndarray, projected: np.ndarray, residuals: np.ndarray):
    return evaluate_structured_residuals(
        object_points_m=points,
        projected_points_px=projected,
        residual_vectors_px=residuals,
        intrinsics=_intrinsics(),
        scope="frame",
    )


def test_random_residual_is_not_called_structured_and_is_deterministic() -> None:
    points = _points()
    projected = project_points_px(points, _pose(), _intrinsics())
    residuals = np.random.default_rng(71).normal(0.0, 0.25, (len(points), 2))
    first = _evaluate(points, projected, residuals)
    second = _evaluate(points, projected, residuals)
    assert first == second
    assert first.sufficient_support is True
    assert first.passed is True
    assert first.permutation_count == 999
    assert first.permutation_p_value is not None and first.permutation_p_value > 0.05
    assert first.cv_explained_fraction is not None and first.cv_explained_fraction < 0.2


@pytest.mark.parametrize("pattern", ["radial", "tangential", "board_warp", "local"])
def test_material_structured_patterns_are_detected(pattern: str) -> None:
    points = _points()
    projected = project_points_px(points, _pose(), _intrinsics())
    metrics = _evaluate(points, projected, _structured(pattern, points, projected))
    assert metrics.sufficient_support is True
    assert metrics.passed is False
    assert metrics.permutation_p_value == pytest.approx(0.001)
    assert metrics.structured_effect_fraction is not None
    assert metrics.structured_effect_fraction >= 0.2
    assert metrics.structured_amplitude_px is not None
    assert metrics.structured_amplitude_px >= 0.15
    assert metrics.failure_reasons == ("STRUCTURED_RESIDUAL_MODEL_MISMATCH",)


def test_target_scale_normalization_is_invariant_for_a4_and_large_geometry() -> None:
    a4 = _points(0.03)
    large = _points(0.10)
    a4_projected = project_points_px(a4, _pose(), _intrinsics())
    large_projected = project_points_px(large, _pose(0.10 / 0.03), _intrinsics())
    residuals = _structured("board_warp", a4, a4_projected)
    a4_metrics = _evaluate(a4, a4_projected, residuals)
    large_metrics = _evaluate(large, large_projected, residuals)
    assert large_projected == pytest.approx(a4_projected)
    assert large_metrics.cv_explained_fraction == pytest.approx(
        a4_metrics.cv_explained_fraction, abs=1e-12
    )
    assert large_metrics.permutation_p_value == a4_metrics.permutation_p_value
    assert large_metrics.passed == a4_metrics.passed


def test_canonical_board_features_and_folds_do_not_shift_with_partial_visibility() -> None:
    points = _points()
    projected = project_points_px(points, _pose(), _intrinsics())
    selected = np.arange(len(points)) % 7 != 0
    full_features = _feature_matrix(points, projected, _intrinsics(), points).reshape(
        len(points), 2, -1
    )
    partial_features = _feature_matrix(
        points[selected], projected[selected], _intrinsics(), points
    ).reshape(np.count_nonzero(selected), 2, -1)
    assert partial_features == pytest.approx(full_features[selected])
    full_folds = _spatial_fold_ids(points[:, :2], points[:, :2])
    partial_folds = _spatial_fold_ids(points[selected, :2], points[:, :2])
    assert np.array_equal(partial_folds, full_folds[selected])


def test_observation_wrapper_locks_observed_minus_projected_sign() -> None:
    points = _points()
    projected = project_points_px(points, _pose(), _intrinsics())
    residuals = _structured("tangential", points, projected)
    wrapped = evaluate_observation_structured_residuals(
        _observation(points, projected + residuals), _pose(), _intrinsics()
    )
    direct = _evaluate(points, projected, residuals)
    assert wrapped.permutation_p_value == direct.permutation_p_value
    assert wrapped.cv_explained_fraction == pytest.approx(direct.cv_explained_fraction)
    assert wrapped.structured_amplitude_px == pytest.approx(direct.structured_amplitude_px)
    assert wrapped.passed == direct.passed
    assert wrapped.residual_convention == "observed_minus_projected"
    assert wrapped.mean_du_px == pytest.approx(float(np.mean(residuals[:, 0])))
    assert wrapped.mean_dv_px == pytest.approx(float(np.mean(residuals[:, 1])))
    reversed_metrics = _evaluate(points, projected, -residuals)
    assert reversed_metrics.mean_du_px == pytest.approx(-wrapped.mean_du_px)
    assert reversed_metrics.mean_dv_px == pytest.approx(-wrapped.mean_dv_px)


def test_small_n_and_degenerate_geometry_fail_closed_without_nonfinite_values() -> None:
    points = _points()[:12]
    projected = np.column_stack(
        (np.linspace(400.0, 800.0, len(points)), np.linspace(250.0, 500.0, len(points)))
    )
    metrics = _evaluate(points, projected, np.zeros((len(points), 2)))
    assert metrics.sufficient_support is False
    assert metrics.passed is False
    assert metrics.failure_reasons == ("STRUCTURED_RESIDUAL_INSUFFICIENT_SUPPORT",)
    assert metrics.cv_explained_fraction is None
    collapsed = np.zeros((20, 3), dtype=np.float64)
    collapsed_metrics = _evaluate(
        collapsed,
        np.tile([640.0, 360.0], (20, 1)),
        np.zeros((20, 2)),
    )
    assert collapsed_metrics.sufficient_support is False
    assert "NaN" not in str(collapsed_metrics.to_dict())
    assert "inf" not in str(collapsed_metrics.to_dict()).lower()


def test_final_shared_pose_averages_by_corner_id_without_pseudoreplication() -> None:
    points = _points()
    pose = _pose()
    projected = project_points_px(points, pose, _intrinsics())
    structure = _structured("radial", points, projected)
    observations = []
    for seed in range(5):
        noise = np.random.default_rng(seed).normal(0.0, 0.1, structure.shape)
        observations.append(_observation(points, projected + structure + noise))
    report = evaluate_final_shared_structured_residuals(observations, pose, _intrinsics())
    assert report["observed_corner_count"] == len(points)
    assert report["eligible_corner_count"] == len(points)
    statistics = report["corner_statistics"]
    assert isinstance(statistics, list)
    assert all(item["count"] == 5 for item in statistics)
    assert all(item["standard_error_u_px"] >= 0 for item in statistics)
    metrics = report["structured_metrics"]
    assert isinstance(metrics, dict)
    assert metrics["corner_count"] == len(points)
    assert metrics["scope"] == "final"
    assert metrics["passed"] is False


def test_final_repeat_support_fails_closed_without_raising() -> None:
    points = _points()
    pose = _pose()
    projected = project_points_px(points, pose, _intrinsics())
    report = evaluate_final_shared_structured_residuals(
        [_observation(points, projected), _observation(points, projected)],
        pose,
        _intrinsics(),
    )
    assert report["eligible_corner_count"] == 0
    assert report["balanced_repeat_support"] is False
    metrics = report["structured_metrics"]
    assert metrics["passed"] is False
    assert "STRUCTURED_RESIDUAL_INSUFFICIENT_REPEAT_SUPPORT" in metrics["failure_reasons"]


def test_material_radial_pattern_with_unequal_focal_lengths_is_detected() -> None:
    intrinsics = CameraIntrinsics(
        frame="camera/color_optical",
        width=1280,
        height=720,
        fx=500.0,
        fy=1000.0,
        cx=639.5,
        cy=359.5,
        distortion_model="none",
    )
    points = _points()
    projected = project_points_px(points, _pose(), intrinsics)
    x = (projected[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (projected[:, 1] - intrinsics.cy) / intrinsics.fy
    r2 = x * x + y * y
    residuals = np.stack((intrinsics.fx * x * r2, intrinsics.fy * y * r2), axis=1)
    residuals *= 0.5 / np.sqrt(np.mean(np.sum(np.square(residuals), axis=1)))
    metrics = evaluate_structured_residuals(
        object_points_m=points,
        projected_points_px=projected,
        residual_vectors_px=residuals,
        intrinsics=intrinsics,
        scope="frame",
        thresholds=StructuredResidualThresholds(model_name="image_physical"),
    )
    assert metrics.passed is False
    assert metrics.permutation_p_value == pytest.approx(0.001)


def test_threshold_contract_rejects_tiny_permutation_count() -> None:
    with pytest.raises(ContractError, match="at least 999"):
        StructuredResidualThresholds(permutation_count=99)


def test_v2_policy_is_hold_and_release_requires_manifest() -> None:
    candidate = StructuredReprojectionPolicy()
    assert candidate.preset == "uncertainty_validated_v2"
    assert candidate.release_state == "HOLD"
    assert candidate.production_eligible is False
    assert candidate.to_dict()["thresholds"]["permutation_count"] == 999
    with pytest.raises(ContractError, match="not release-enabled"):
        StructuredReprojectionPolicy(release_state="RELEASED")
    with pytest.raises(ContractError, match="not release-enabled"):
        StructuredReprojectionPolicy(release_state="RELEASED", release_manifest_sha256="a" * 64)
