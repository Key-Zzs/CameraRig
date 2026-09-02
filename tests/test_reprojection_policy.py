from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from camera_rig.calibration.fixed.config import FixedCalibrationConfig, FixedSolverThresholds
from camera_rig.calibration.fixed.quality import evaluate_fixed_calibration_quality
from camera_rig.calibration.fixed.residuals import evaluate_residual_vector_field
from camera_rig.calibration.fixed.viability import evaluate_fixed_pose_frame_viability
from camera_rig.calibration.pose import (
    PlanarPoseEstimator,
    ReprojectionMetrics,
    project_points_px,
)
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


def _pose() -> RigidTransform:
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [-0.09, 0.06, 0.72]
    return RigidTransform("target", "camera/color_optical", matrix)


def _oblique_pose() -> RigidTransform:
    angle = math.radians(35.0)
    tilt = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = tilt @ np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [-0.09, 0.06, 0.72]
    return RigidTransform("target", "camera/color_optical", matrix)


def _points() -> np.ndarray:
    return np.asarray(
        [[0.03 * column, 0.03 * row, 0.0] for row in range(5) for column in range(7)],
        dtype=np.float64,
    )


def _observation(pixels: np.ndarray | None = None) -> TargetObservation:
    points = _points()
    if pixels is None:
        pixels = project_points_px(points, _pose(), _intrinsics())
        pixels = pixels + np.random.default_rng(73).normal(0.0, 0.02, pixels.shape)
    return TargetObservation(
        plugin_name="synthetic-grid",
        target_frame="target",
        point_ids=tuple(range(len(points))),
        image_points_px=pixels,
        object_points_m=points,
        image_size=(1280, 720),
        quality=QualityReport(True),
    )


def _structured_vectors(
    pattern: str,
    points: np.ndarray,
    projected: np.ndarray,
    pose: RigidTransform,
    intrinsics: CameraIntrinsics,
    target_rmse: float,
) -> np.ndarray:
    centered = projected - np.asarray([intrinsics.cx, intrinsics.cy])
    x = centered[:, 0] / intrinsics.fx
    y = centered[:, 1] / intrinsics.fy
    radius_squared = x * x + y * y
    if pattern == "tangential":
        vectors = np.stack(
            (
                intrinsics.fx * (2.0 * x * y - 0.7 * (radius_squared + 2.0 * x * x)),
                intrinsics.fy * (radius_squared + 2.0 * y * y - 1.4 * x * y),
            ),
            axis=1,
        )
    elif pattern == "board_warp_like":
        board = points.copy()
        board_xy = board[:, :2] - np.mean(board[:, :2], axis=0)
        board_scale = np.max(np.abs(board_xy), axis=0)
        board_x = board_xy[:, 0] / board_scale[0]
        board_y = board_xy[:, 1] / board_scale[1]
        board[:, 2] = 0.005 * (board_x * board_x + 0.5 * board_x * board_y)
        board[:, 2] -= np.mean(board[:, 2])
        vectors = project_points_px(board, pose, intrinsics) - projected
    elif pattern == "local_corruption":
        vectors = np.zeros_like(projected)
        mask = (points[:, 0] >= np.quantile(points[:, 0], 0.7)) & (
            points[:, 1] >= np.quantile(points[:, 1], 0.5)
        )
        vectors[mask] = np.asarray([2.0, -1.0])
    else:
        raise ValueError(pattern)
    vector_rmse = float(np.sqrt(np.mean(np.sum(np.square(vectors), axis=1))))
    return vectors * (target_rmse / vector_rmse)


def _config() -> FixedCalibrationConfig:
    return FixedCalibrationConfig(
        workspace_frame="workspace",
        target_frame="target",
        T_workspace_from_target=RigidTransform("target", "workspace", np.eye(4)),
        detection_stream="color",
        reference_stream="color",
        solver=FixedSolverThresholds(
            method="ippe",
            refinement="lm",
            minimum_corners_per_frame=12,
            minimum_accepted_frames=4,
            minimum_accepted_ratio=0.75,
            maximum_frame_rmse_px=0.5,
            maximum_frame_p95_px=1.0,
            maximum_pose_translation_p95_mm=3.0,
            maximum_pose_rotation_p95_deg=0.3,
            maximum_split_translation_delta_mm=2.0,
            maximum_split_rotation_delta_deg=0.2,
        ),
        native_depth_check=True,
    )


def _estimate_with_residual(rmse: float, p95: float | None = None):
    observation = _observation()
    estimate = PlanarPoseEstimator().estimate(observation, _intrinsics())
    residuals = np.full(len(observation.point_ids), rmse, dtype=np.float64)
    metrics = ReprojectionMetrics(
        residuals_px=residuals,
        rmse_px=rmse,
        median_px=rmse,
        p95_px=rmse if p95 is None else p95,
        maximum_px=max(rmse, rmse if p95 is None else p95),
    )
    return observation, replace(estimate, reprojection=metrics)


@pytest.mark.parametrize("rmse", [0.2, 0.55, 0.65])
def test_uncertainty_policy_does_not_reuse_legacy_half_pixel_gate(rmse: float) -> None:
    observation, estimate = _estimate_with_residual(rmse, p95=min(1.2, 1.7 * rmse))
    result = evaluate_fixed_pose_frame_viability(
        frame_index=0,
        detection_success=True,
        observation=observation,
        estimate=estimate,
        config=_config(),
        pose_policy="uncertainty_validated",
    )
    assert result["accepted"] is True
    decision = result["reprojection_decision"]
    assert isinstance(decision, dict)
    assert decision["policy"] == "uncertainty_gross_model_consistency"
    assert decision["legacy_precision_thresholds"] == {
        "maximum_frame_rmse_px": 0.5,
        "maximum_frame_p95_px": 1.0,
    }


@pytest.mark.parametrize(("component_noise_px", "seed"), [(0.55, 811), (0.65, 932)])
def test_uncertainty_policy_uses_recomputed_noise_and_covariance(
    component_noise_px: float, seed: int
) -> None:
    points = _points()
    ideal = project_points_px(points, _oblique_pose(), _intrinsics())
    pixels = ideal + np.random.default_rng(seed).normal(0.0, component_noise_px, ideal.shape)
    observation = _observation(pixels)
    estimate = PlanarPoseEstimator().estimate(observation, _intrinsics())
    result = evaluate_fixed_pose_frame_viability(
        frame_index=0,
        detection_success=True,
        observation=observation,
        estimate=estimate,
        config=_config(),
        pose_policy="uncertainty_validated",
    )
    assert estimate.reprojection.rmse_px > 0.5
    assert estimate.observability.pixel_noise_sigma_px > 0.25
    assert estimate.observability.covariance_6x6 is not None
    assert estimate.observability.passed is True
    assert result["accepted"] is True


@pytest.mark.parametrize("rmse", [2.0, 3.0])
def test_uncertainty_policy_rejects_multi_pixel_model_inconsistency(rmse: float) -> None:
    observation, estimate = _estimate_with_residual(rmse, p95=rmse)
    result = evaluate_fixed_pose_frame_viability(
        frame_index=0,
        detection_success=True,
        observation=observation,
        estimate=estimate,
        config=_config(),
        pose_policy="uncertainty_validated",
    )
    assert result["accepted"] is False
    assert "GROSS_REPROJECTION_RMSE_EXCEEDED" in result["failure_reasons"]


@pytest.mark.parametrize("policy", ["legacy_strict", "pose_validated"])
def test_legacy_and_pose_validated_retain_half_pixel_gate(policy: str) -> None:
    observation, estimate = _estimate_with_residual(0.55, p95=0.8)
    result = evaluate_fixed_pose_frame_viability(
        frame_index=0,
        detection_success=True,
        observation=observation,
        estimate=estimate,
        config=_config(),
        pose_policy=policy,
    )
    assert result["accepted"] is False
    assert result["failure_reasons"] == ["frame_reprojection_rmse_exceeded"]


def test_low_reprojection_does_not_override_failed_observability() -> None:
    observation, estimate = _estimate_with_residual(0.2, p95=0.3)
    failed_observability = replace(
        estimate.observability,
        passed=False,
        failure_reasons=("POSE_CONDITION_NUMBER_EXCEEDED",),
    )
    result = evaluate_fixed_pose_frame_viability(
        frame_index=0,
        detection_success=True,
        observation=observation,
        estimate=replace(estimate, observability=failed_observability),
        config=_config(),
        pose_policy="uncertainty_validated",
    )
    assert result["accepted"] is False
    assert result["failure_reasons"] == ["POSE_CONDITION_NUMBER_EXCEEDED"]


def test_final_uncertainty_gate_uses_gross_thresholds_but_legacy_does_not() -> None:
    common = {
        "thresholds": _config().solver,
        "frame_count": 60,
        "accepted_frames": 60,
        "global_reprojection": {"rmse_px": 0.65, "p95_px": 1.1},
        "pose_repeatability": {
            "translation_mm": {"p95": 0.2},
            "rotation_deg": {"p95": 0.02},
        },
        "split_half": {"translation_delta_mm": 0.1, "rotation_delta_deg": 0.01},
        "native_depth_sanity": {"status": "PASS"},
    }
    _observation_value, estimate = _estimate_with_residual(0.2)
    final = replace(estimate.observability, scope="final").to_dict()
    uncertainty = evaluate_fixed_calibration_quality(
        **common,
        pose_policy="uncertainty_validated",
        final_pose_observability=final,
        observable_frame_ratio=1.0,
        ambiguous_frame_ratio=0.0,
    )
    legacy = evaluate_fixed_calibration_quality(**common, pose_policy="legacy_strict")
    assert uncertainty.passed is True
    assert legacy.passed is False
    assert "global_reprojection_rmse" in legacy.failure_reasons


def test_legacy_multi_failure_reason_order_is_backward_compatible() -> None:
    quality = evaluate_fixed_calibration_quality(
        thresholds=_config().solver,
        frame_count=60,
        accepted_frames=0,
        global_reprojection={"rmse_px": 2.0, "p95_px": 3.0},
        pose_repeatability={
            "translation_mm": {"p95": 20.0},
            "rotation_deg": {"p95": 2.0},
        },
        split_half={"translation_delta_mm": 20.0, "rotation_delta_deg": 2.0},
        native_depth_sanity={"status": "FAIL"},
        pose_policy="legacy_strict",
    )
    assert quality.failure_reasons == (
        "minimum_accepted_frames",
        "minimum_accepted_ratio",
        "global_reprojection_rmse",
        "global_reprojection_p95",
        "pose_translation_p95",
        "pose_rotation_p95",
        "split_translation_delta",
        "split_rotation_delta",
        "native_depth_sanity",
    )


def test_existing_physical_target_requires_native_depth_pass() -> None:
    _observation_value, estimate = _estimate_with_residual(0.2)
    quality = evaluate_fixed_calibration_quality(
        thresholds=_config().solver,
        frame_count=60,
        accepted_frames=60,
        global_reprojection={"rmse_px": 0.2, "p95_px": 0.3},
        pose_repeatability={
            "translation_mm": {"p95": 0.2},
            "rotation_deg": {"p95": 0.02},
        },
        split_half={"translation_delta_mm": 0.1, "rotation_delta_deg": 0.01},
        native_depth_sanity={"status": "SKIPPED_WITH_WARNING", "warning": "unsupported"},
        pose_policy="uncertainty_validated",
        final_pose_observability=replace(estimate.observability, scope="final").to_dict(),
        observable_frame_ratio=1.0,
        ambiguous_frame_ratio=0.0,
        require_native_depth_pass=True,
    )
    assert quality.passed is False
    assert "native_depth_sanity" in quality.failure_reasons


def test_radial_residual_vector_field_exposes_systematic_trend() -> None:
    points = _points()
    projected = project_points_px(points, _pose(), _intrinsics())
    center = np.asarray([_intrinsics().cx, _intrinsics().cy])
    centered = projected - center
    radius = np.linalg.norm(centered, axis=1)
    direction = np.divide(
        centered,
        radius[:, None],
        out=np.zeros_like(centered),
        where=radius[:, None] > 1e-12,
    )
    pixels = projected + direction * (0.1 + 0.012 * radius)[:, None]
    report = evaluate_residual_vector_field(_observation(pixels), _pose(), _intrinsics())
    assert report["role"] == "diagnostic_only_not_a_hard_gate"
    trends = report["trends"]
    assert isinstance(trends, dict)
    assert trends["radial_component_vs_radius_pearson"] > 0.99
    assert trends["radial_component_vs_radius_spearman"] > 0.99
    assert len(report["per_corner"]) == len(points)
    quadrants = report["quadrants"]
    assert isinstance(quadrants, dict)
    populated = [item for item in quadrants.values() if item["count"]]  # type: ignore[index]
    assert any(abs(item["mean_radial_component_px"]) > 0.01 for item in populated)  # type: ignore[arg-type]
    polynomial = report["board_coordinate_polynomial"]
    assert isinstance(polynomial, dict)
    assert polynomial["vector_r_squared"] > 0.95


def test_structured_radial_model_mismatch_is_exposed_and_rejected() -> None:
    points = _points()
    intrinsics = _intrinsics()
    projected = project_points_px(points, _pose(), intrinsics)
    x = (projected[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (projected[:, 1] - intrinsics.cy) / intrinsics.fy
    radius_squared = x * x + y * y
    vectors = np.stack(
        (intrinsics.fx * x * radius_squared, intrinsics.fy * y * radius_squared), axis=1
    )
    vector_rmse = float(np.sqrt(np.mean(np.sum(np.square(vectors), axis=1))))
    vectors *= 3.0 / vector_rmse
    observation = _observation(projected + vectors)
    estimate = PlanarPoseEstimator().estimate(observation, intrinsics)
    decision = evaluate_fixed_pose_frame_viability(
        frame_index=0,
        detection_success=True,
        observation=observation,
        estimate=estimate,
        config=_config(),
        pose_policy="uncertainty_validated",
    )
    diagnostics = evaluate_residual_vector_field(
        observation, estimate.T_camera_from_target, intrinsics
    )
    assert decision["accepted"] is False
    assert set(decision["failure_reasons"]) & {
        "GROSS_REPROJECTION_RMSE_EXCEEDED",
        "GROSS_REPROJECTION_P95_EXCEEDED",
        "POSE_AMBIGUOUS",
    }
    trends = diagnostics["trends"]
    assert isinstance(trends, dict)
    assert abs(trends["radial_component_vs_radius_pearson"]) > 0.5


@pytest.mark.parametrize("pattern", ["tangential", "board_warp_like", "local_corruption"])
def test_structured_pattern_diagnostics_are_regression_locked(pattern: str) -> None:
    points = _points()
    intrinsics = _intrinsics()
    pose = _pose()
    projected = project_points_px(points, pose, intrinsics)
    vectors = _structured_vectors(pattern, points, projected, pose, intrinsics, 2.0)
    report = evaluate_residual_vector_field(_observation(projected + vectors), pose, intrinsics)
    aggregate = report["aggregate"]
    polynomial = report["board_coordinate_polynomial"]
    assert isinstance(aggregate, dict)
    assert isinstance(polynomial, dict)
    assert aggregate["maximum_px"] > 2.0
    if pattern == "local_corruption":
        assert aggregate["fraction_above_2_px"] > 0.05
        assert polynomial["unexplained_vector_rmse_px"] > 0.25
    else:
        assert polynomial["vector_r_squared"] > 0.9
