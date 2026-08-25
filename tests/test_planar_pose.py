from __future__ import annotations

import math

import numpy as np
import pytest

from camera_rig.calibration.pose import (
    PlanarPoseEstimator,
    project_points_px,
    refine_planar_pose_lm,
    to_opencv_camera_model,
    validate_planar_pose,
)
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

cv2 = pytest.importorskip("cv2")


def _intrinsics(model: str = "none", coefficients: tuple[float, ...] = ()) -> CameraIntrinsics:
    return CameraIntrinsics(
        frame="synthetic/color_optical",
        width=640,
        height=480,
        fx=800.0,
        fy=805.0,
        cx=319.5,
        cy=239.5,
        distortion_model=model,
        distortion_coeffs=coefficients,
    )


def _object_points() -> np.ndarray:
    return np.asarray(
        [[0.03 * column, 0.03 * row, 0.0] for row in range(4) for column in range(6)],
        dtype=np.float64,
    )


def _ground_truth(*, oblique: bool) -> RigidTransform:
    rotation_x = np.diag([1.0, -1.0, -1.0])
    if oblique:
        angle = math.radians(13.0)
        rotation_y = np.asarray(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        rotation = rotation_y @ rotation_x
    else:
        rotation = rotation_x
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [-0.075, 0.045, 0.72]
    return RigidTransform("target", "synthetic/color_optical", matrix)


def _observation(
    transform: RigidTransform,
    *,
    selected: np.ndarray | None = None,
    noise_px: float = 0.0,
) -> TargetObservation:
    intrinsics = _intrinsics()
    model = to_opencv_camera_model(intrinsics)
    points = _object_points()
    if selected is None:
        selected = np.arange(len(points))
    rvec, _jacobian = cv2.Rodrigues(transform.matrix[:3, :3])
    image_points, _jacobian = cv2.projectPoints(
        points[selected],
        rvec,
        transform.matrix[:3, 3],
        model.camera_matrix,
        model.distortion_coeffs,
    )
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if noise_px:
        pixels += np.random.default_rng(20260825).normal(0.0, noise_px, pixels.shape)
    return TargetObservation(
        plugin_name="synthetic-grid",
        target_frame="target",
        point_ids=tuple(int(value) for value in selected),
        image_points_px=pixels,
        object_points_m=points[selected],
        image_size=(640, 480),
        quality=QualityReport(True),
        metadata={"source": "synthetic"},
    )


def _rotation_error_deg(actual: RigidTransform, expected: RigidTransform) -> float:
    relative = expected.matrix[:3, :3].T @ actual.matrix[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def test_camera_adapter_supports_explicit_proven_models() -> None:
    none = to_opencv_camera_model(_intrinsics())
    assert none.diagnostics["mapping_decision"] == "none-as-opencv-identity"
    np.testing.assert_array_equal(none.distortion_coeffs, np.zeros(5))

    coefficients = (0.1, -0.02, 0.001, -0.002, 0.003)
    brown = to_opencv_camera_model(_intrinsics("brown-conrady", coefficients))
    assert brown.diagnostics["opencv_distortion_model"] == "brown-conrady"
    np.testing.assert_array_equal(brown.distortion_coeffs, coefficients)


def test_inverse_brown_is_identity_only_and_diagnostic_is_persistable() -> None:
    model = to_opencv_camera_model(_intrinsics("inverse-brown-conrady", (0.0,) * 5))
    assert model.diagnostics["inverse_brown_identity_only"] is True
    assert (
        model.to_dict()["diagnostics"]["mapping_decision"]  # type: ignore[index]
        == "inverse-brown-conrady-zero-coefficients-as-opencv-identity"
    )
    with pytest.raises(ContractError, match="only for exactly five zero"):
        to_opencv_camera_model(_intrinsics("inverse-brown-conrady", (0.01, 0.0, 0.0, 0.0, 0.0)))


@pytest.mark.parametrize(
    ("model", "coefficients"),
    [
        ("none", (0.1,)),
        ("brown-conrady", (0.0,) * 4),
        ("modified-brown-conrady", (0.0,) * 5),
        ("kannala-brandt4", (0.0,) * 4),
    ],
)
def test_camera_adapter_fails_closed(model: str, coefficients: tuple[float, ...]) -> None:
    with pytest.raises(ContractError):
        to_opencv_camera_model(_intrinsics(model, coefficients))


def test_front_facing_pose_recovers_target_to_camera_direction_and_two_candidates() -> None:
    expected = _ground_truth(oblique=False)
    estimate = PlanarPoseEstimator().estimate(_observation(expected), _intrinsics())
    assert estimate.T_camera_from_target.source_frame == "target"
    assert estimate.T_camera_from_target.target_frame == "synthetic/color_optical"
    assert estimate.candidate_count == 2
    assert len(estimate.candidate_separations) == 1
    assert estimate.refined_validity.valid
    selected = estimate.candidates[estimate.selected_candidate_index]
    valid_rmse = [item.reprojection.rmse_px for item in estimate.candidates if item.validity.valid]
    assert selected.reprojection.rmse_px == min(valid_rmse)
    np.testing.assert_allclose(
        estimate.T_camera_from_target.matrix[:3, 3], expected.matrix[:3, 3], atol=1e-7
    )
    assert _rotation_error_deg(estimate.T_camera_from_target, expected) < 1e-5
    assert estimate.reprojection.rmse_px < 1e-5


def test_oblique_partial_noisy_pose_recovers_with_all_residuals() -> None:
    expected = _ground_truth(oblique=True)
    selected = np.asarray([0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15])
    observation = _observation(expected, selected=selected, noise_px=0.08)
    estimate = PlanarPoseEstimator().estimate(observation, _intrinsics())
    translation_error = np.linalg.norm(
        estimate.T_camera_from_target.matrix[:3, 3] - expected.matrix[:3, 3]
    )
    assert translation_error < 0.003
    assert _rotation_error_deg(estimate.T_camera_from_target, expected) < 0.5
    assert len(estimate.reprojection.residuals_px) == len(selected)
    assert estimate.reprojection.p95_px < 0.25


def test_public_projection_and_stacked_lm_refinement_share_camera_adapter() -> None:
    expected = _ground_truth(oblique=True)
    observation = _observation(expected)
    projected = project_points_px(observation.object_points_m, expected, _intrinsics())
    np.testing.assert_allclose(projected, observation.image_points_px, atol=1e-9)
    assert not projected.flags.writeable

    initial_rvec, _jacobian = cv2.Rodrigues(expected.matrix[:3, :3])
    perturbed_rotation, _jacobian = cv2.Rodrigues(
        initial_rvec + np.asarray([[0.01], [-0.008], [0.006]])
    )
    initial_matrix = expected.matrix.copy()
    initial_matrix[:3, :3] = perturbed_rotation
    initial_matrix[:3, 3] += [0.003, -0.002, 0.006]
    initial = RigidTransform("target", "synthetic/color_optical", initial_matrix)
    stacked_objects = np.vstack([observation.object_points_m, observation.object_points_m])
    stacked_images = np.vstack([observation.image_points_px, observation.image_points_px])
    refined = refine_planar_pose_lm(initial, stacked_objects, stacked_images, _intrinsics())
    assert refined.validity.valid
    assert refined.camera_model_diagnostics["mapping_decision"] == "none-as-opencv-identity"
    np.testing.assert_allclose(
        refined.T_camera_from_target.matrix[:3, 3], expected.matrix[:3, 3], atol=1e-7
    )
    assert _rotation_error_deg(refined.T_camera_from_target, expected) < 1e-5
    assert len(refined.reprojection.residuals_px) == 48


def test_negative_depth_and_mirrored_printed_face_are_rejected() -> None:
    points = _object_points()
    valid = _ground_truth(oblique=False)
    assert validate_planar_pose(valid, points).valid

    negative_matrix = valid.matrix.copy()
    negative_matrix[2, 3] = -0.72
    negative = RigidTransform("target", "synthetic/color_optical", negative_matrix)
    negative_validity = validate_planar_pose(negative, points)
    assert not negative_validity.cheirality
    assert "target_points_not_strictly_in_front_of_camera" in negative_validity.failure_reasons

    mirrored_matrix = np.eye(4)
    mirrored_matrix[:3, 3] = [-0.075, -0.045, 0.72]
    mirrored = RigidTransform("target", "synthetic/color_optical", mirrored_matrix)
    mirrored_validity = validate_planar_pose(mirrored, points)
    assert mirrored_validity.cheirality
    assert not mirrored_validity.printed_face_orientation
    assert "printed_face_not_facing_camera" in mirrored_validity.failure_reasons


def test_pose_input_contract_rejects_non_planar_points() -> None:
    expected = _ground_truth(oblique=False)
    observation = _observation(expected)
    points = observation.object_points_m.copy()
    points[0, 2] = 0.001
    invalid = TargetObservation(
        plugin_name=observation.plugin_name,
        target_frame=observation.target_frame,
        point_ids=observation.point_ids,
        image_points_px=observation.image_points_px,
        object_points_m=points,
        image_size=observation.image_size,
        quality=observation.quality,
    )
    with pytest.raises(ContractError, match="target z=0"):
        PlanarPoseEstimator().estimate(invalid, _intrinsics())
