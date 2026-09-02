from __future__ import annotations

import math

import numpy as np
import pytest

from camera_rig.calibration.pose import (
    PoseAmbiguityCandidate,
    UncertaintyValidatedThresholds,
    evaluate_pose_ambiguity,
    evaluate_pose_observability,
    projection_jacobian_first_six,
    to_opencv_camera_model,
)
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.transforms import RigidTransform

cv2 = pytest.importorskip("cv2")


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        "synthetic/color_optical",
        1280,
        720,
        900.0,
        905.0,
        639.5,
        359.5,
        "none",
    )


def _points(scale_m: float = 0.1) -> np.ndarray:
    return np.asarray(
        [[scale_m * column, scale_m * row, 0.0] for row in range(6) for column in range(4)],
        dtype=np.float64,
    )


def _pose(distance_m: float, tilt_deg: float = 30.0) -> RigidTransform:
    angle = math.radians(tilt_deg)
    tilt = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = tilt @ np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [-0.15, 0.25, distance_m]
    return RigidTransform("target", "synthetic/color_optical", matrix)


def _project(points: np.ndarray, pose: RigidTransform) -> np.ndarray:
    model = to_opencv_camera_model(_intrinsics())
    rvec, _jacobian = cv2.Rodrigues(pose.matrix[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        points,
        rvec,
        pose.matrix[:3, 3],
        model.camera_matrix,
        model.distortion_coeffs,
    )
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def _coverage(points_px: np.ndarray) -> float:
    hull = cv2.convexHull(np.asarray(points_px, dtype=np.float32))
    return float(cv2.contourArea(hull)) / (1280 * 720)


def test_analytic_projection_jacobian_matches_central_finite_difference() -> None:
    points = _points()
    pose = _pose(1.4, 37.0)
    analytic = projection_jacobian_first_six(
        object_points_m=points,
        T_camera_from_target=pose,
        intrinsics=_intrinsics(),
    )
    model = to_opencv_camera_model(_intrinsics())
    rvec, _jacobian = cv2.Rodrigues(pose.matrix[:3, :3])
    parameters = np.concatenate((rvec.reshape(3), pose.matrix[:3, 3]))
    finite_difference = np.empty_like(analytic)
    epsilon = 1e-7
    for column in range(6):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        plus_pixels, _jacobian = cv2.projectPoints(
            points,
            plus[:3],
            plus[3:],
            model.camera_matrix,
            model.distortion_coeffs,
        )
        minus_pixels, _jacobian = cv2.projectPoints(
            points,
            minus[:3],
            minus[3:],
            model.camera_matrix,
            model.distortion_coeffs,
        )
        finite_difference[:, column] = (
            np.asarray(plus_pixels).reshape(-1) - np.asarray(minus_pixels).reshape(-1)
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-7, atol=2e-5)
    assert not analytic.flags.writeable


def test_full_rank_covariance_is_finite_symmetric_psd_and_uses_declared_units() -> None:
    points = _points()
    pose = _pose(1.5)
    metrics = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=_project(points, pose),
        T_camera_from_target=pose,
        intrinsics=_intrinsics(),
    )
    assert metrics.effective_rank == 6
    assert metrics.covariance_6x6 is not None
    covariance = np.asarray(metrics.covariance_6x6)
    np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-15
    assert metrics.translation_std_xyz_mm is not None
    assert metrics.rotation_std_xyz_deg is not None
    np.testing.assert_allclose(
        metrics.translation_std_xyz_mm,
        1000.0 * np.sqrt(np.diag(covariance)[3:]),
    )
    np.testing.assert_allclose(
        metrics.rotation_std_xyz_deg,
        np.degrees(np.sqrt(np.diag(covariance)[:3])),
    )
    assert metrics.pixel_noise_sigma_px == 0.25


def test_reported_rotation_covariance_is_left_invariant_physical_tangent() -> None:
    points = _points()
    pose = _pose(1.0, 30.0)
    metrics = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=_project(points, pose),
        T_camera_from_target=pose,
        intrinsics=_intrinsics(),
    )
    assert metrics.covariance_6x6 is not None
    jacobian = projection_jacobian_first_six(
        object_points_m=points,
        T_camera_from_target=pose,
        intrinsics=_intrinsics(),
    )
    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(np.square(centered), axis=1))))
    parameter_scale = np.diag([1.0, 1.0, 1.0, scale, scale, scale])
    _left, singular, right_transpose = np.linalg.svd(
        jacobian @ parameter_scale, full_matrices=False
    )
    covariance_q = 0.25**2 * (
        right_transpose.T @ np.diag(1.0 / np.square(singular)) @ right_transpose
    )
    covariance_parameters = parameter_scale @ covariance_q @ parameter_scale.T
    rvec, _jacobian = cv2.Rodrigues(pose.matrix[:3, :3])
    vector = rvec.reshape(3)
    angle = float(np.linalg.norm(vector))
    skew = np.asarray(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    left_jacobian = (
        np.eye(3)
        + ((1.0 - math.cos(angle)) / angle**2) * skew
        + ((angle - math.sin(angle)) / angle**3) * (skew @ skew)
    )
    tangent_from_parameters = np.eye(6)
    tangent_from_parameters[:3, :3] = left_jacobian
    expected = (
        tangent_from_parameters @ covariance_parameters @ tangent_from_parameters.T
    )
    np.testing.assert_allclose(metrics.covariance_6x6, expected, rtol=2e-12, atol=1e-15)
    assert "left_invariant_camera_rotation_rad" in metrics.parameterization


def test_pixel_noise_distance_and_corner_distribution_have_expected_trends() -> None:
    points = _points()
    near = _pose(1.0)
    far = _pose(3.0, 60.0)
    clean_near = _project(points, near)
    noisy_near = clean_near + np.random.default_rng(42).normal(0.0, 1.0, clean_near.shape)
    clean_metrics = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=clean_near,
        T_camera_from_target=near,
        intrinsics=_intrinsics(),
    )
    noisy_metrics = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=noisy_near,
        T_camera_from_target=near,
        intrinsics=_intrinsics(),
    )
    far_metrics = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=_project(points, far),
        T_camera_from_target=far,
        intrinsics=_intrinsics(),
    )
    selected = np.asarray([0, 2, 3, 8, 10, 11, 12, 14, 15, 20, 22, 23])
    partial_metrics = evaluate_pose_observability(
        object_points_m=points[selected],
        image_points_px=clean_near[selected],
        T_camera_from_target=near,
        intrinsics=_intrinsics(),
    )
    assert noisy_metrics.pixel_noise_sigma_px > clean_metrics.pixel_noise_sigma_px
    assert (
        noisy_metrics.translation_worst_axis_std_mm
        > clean_metrics.translation_worst_axis_std_mm
    )
    assert far_metrics.translation_worst_axis_std_mm > clean_metrics.translation_worst_axis_std_mm
    assert (
        partial_metrics.translation_worst_axis_std_mm
        > clean_metrics.translation_worst_axis_std_mm
    )


def test_coverage_is_advisory_while_information_controls_three_key_cases() -> None:
    points = _points()

    low_good_pose = _pose(3.0, 60.0)
    low_good_pixels = _project(points, low_good_pose)
    low_good = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=low_good_pixels,
        T_camera_from_target=low_good_pose,
        intrinsics=_intrinsics(),
    )
    assert _coverage(low_good_pixels) < 0.01
    assert low_good.passed

    high_poor_pose = _pose(1.0, 15.0)
    high_pixels = _project(points, high_poor_pose)
    high_noisy = high_pixels + np.random.default_rng(4).normal(0.0, 5.0, high_pixels.shape)
    high_poor = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=high_noisy,
        T_camera_from_target=high_poor_pose,
        intrinsics=_intrinsics(),
    )
    assert _coverage(high_noisy) > 0.05
    assert not high_poor.passed
    assert "POSE_TRANSLATION_UNCERTAINTY_EXCEEDED" in high_poor.failure_reasons

    low_poor_pose = _pose(4.0, 70.0)
    low_pixels = _project(points, low_poor_pose)
    low_noisy = low_pixels + np.random.default_rng(4).normal(0.0, 0.5, low_pixels.shape)
    low_poor = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=low_noisy,
        T_camera_from_target=low_poor_pose,
        intrinsics=_intrinsics(),
    )
    assert _coverage(low_noisy) < 0.01
    assert not low_poor.passed


def test_nearly_collinear_geometry_fails_condition_gate_without_false_confidence() -> None:
    epsilon = 1e-6
    points = np.asarray(
        [[0.1 * index, epsilon * (index % 2), 0.0] for index in range(6)],
        dtype=np.float64,
    )
    pose = _pose(1.0, 0.0)
    metrics = evaluate_pose_observability(
        object_points_m=points,
        image_points_px=_project(points, pose),
        T_camera_from_target=pose,
        intrinsics=_intrinsics(),
    )
    assert metrics.scaled_condition_number is not None
    assert metrics.scaled_condition_number > 100.0
    assert "POSE_CONDITION_NUMBER_EXCEEDED" in metrics.failure_reasons


def test_material_statistically_competitive_ippe_alternative_is_ambiguous() -> None:
    best = _pose(1.0, 20.0)
    alternative_matrix = best.matrix.copy()
    alternative_matrix[0, 3] += 0.010
    alternative = RigidTransform("target", "synthetic/color_optical", alternative_matrix)
    thresholds = UncertaintyValidatedThresholds()
    metrics = evaluate_pose_ambiguity(
        (
            PoseAmbiguityCandidate(0, best, True, 1.0),
            PoseAmbiguityCandidate(1, alternative, True, 1.1),
        ),
        pixel_noise_sigma_px=0.25,
        thresholds=thresholds,
    )
    assert metrics.delta_chi2 == pytest.approx(1.6)
    assert metrics.materially_distinct
    assert metrics.statistically_competitive
    assert metrics.ambiguous


def test_larger_physical_target_improves_information_at_same_distance() -> None:
    small = _points(0.04)
    large = _points(0.10)
    pose = _pose(2.0, 30.0)
    small_metrics = evaluate_pose_observability(
        object_points_m=small,
        image_points_px=_project(small, pose),
        T_camera_from_target=pose,
        intrinsics=_intrinsics(),
    )
    large_metrics = evaluate_pose_observability(
        object_points_m=large,
        image_points_px=_project(large, pose),
        T_camera_from_target=pose,
        intrinsics=_intrinsics(),
    )
    assert (
        large_metrics.translation_worst_axis_std_mm
        < small_metrics.translation_worst_axis_std_mm
    )
