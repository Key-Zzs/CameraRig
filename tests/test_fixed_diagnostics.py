from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from camera_rig.calibration.fixed.depth_sanity import (
    build_metric_depth_receipt,
    evaluate_native_depth_sanity,
    load_metric_depth_receipt,
    validate_native_depth_evaluation,
    write_metric_depth_receipt,
)
from camera_rig.calibration.fixed.overlays import select_overlay_frames, write_fixed_pose_overlay
from camera_rig.calibration.pose import project_points_px
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform
from camera_rig.provision.config import (
    BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
    BOOTSTRAP_METRIC_DEPTH_THRESHOLDS,
)
from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget
from camera_rig.targets.observation import TargetObservation

pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")


def _target() -> ResolvedCharucoTarget:
    squares_x = 7
    squares_y = 5
    square = 0.03
    height = squares_y * square
    corners = []
    for point_id in range((squares_x - 1) * (squares_y - 1)):
        row, column = divmod(point_id, squares_x - 1)
        corners.append(
            (
                point_id,
                ((column + 1) * square, height - (row + 1) * square, 0.0),
            )
        )
    return ResolvedCharucoTarget(
        target_name="synthetic_charuco",
        target_frame="target",
        dictionary="DICT_5X5_100",
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square,
        marker_length_m=0.022,
        border_bits=1,
        legacy_pattern=False,
        board_width_m=squares_x * square,
        board_height_m=height,
        corner_points=tuple(corners),
        marker_ids=tuple(range((squares_x * squares_y) // 2)),
        camera_rig_version="test",
        opencv_version="test",
        source_config_sha256="1" * 64,
        board_png_sha256="2" * 64,
        print_pdf_sha256="3" * 64,
        artifact_sha256="4" * 64,
    )


def _intrinsics(frame: str, distortion_model: str = "none") -> CameraIntrinsics:
    coefficients = () if distortion_model == "none" else (0.0,) * 5
    return CameraIntrinsics(
        frame=frame,
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        cx=319.5,
        cy=239.5,
        distortion_model=distortion_model,
        distortion_coeffs=coefficients,
    )


def _factory(*, depth_distortion: str = "none") -> FactoryCalibration:
    color_frame = "synthetic/color_optical"
    depth_frame = "synthetic/depth_optical"
    return FactoryCalibration(
        device=CameraDeviceInfo("synthetic", "synthetic", "test", "test", "test"),
        stream_profiles={
            "color": StreamProfile("color", 640, 480, 30, "rgb8"),
            "depth": StreamProfile("depth", 640, 480, 30, "z16"),
        },
        intrinsics={
            "color": _intrinsics(color_frame),
            "depth": _intrinsics(depth_frame, depth_distortion),
        },
        internal_transforms=(RigidTransform(color_frame, depth_frame, np.eye(4)),),
        depth_scale_m_per_unit=0.001,
    )


def _front_pose() -> RigidTransform:
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [-0.105, 0.075, 0.75]
    return RigidTransform("target", "synthetic/color_optical", matrix)


def _oblique_pose() -> RigidTransform:
    angle = math.radians(12.0)
    rotation_y = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    matrix = _front_pose().matrix.copy()
    matrix[:3, :3] = rotation_y @ matrix[:3, :3]
    return RigidTransform("target", "synthetic/color_optical", matrix)


def _frames(raw_depth_units: int, count: int = 2) -> tuple[CameraFrame, ...]:
    return tuple(
        CameraFrame(
            camera_name="synthetic",
            serial="test",
            streams={
                "depth": StreamFrame(
                    "depth",
                    np.full((480, 640), raw_depth_units, dtype=np.uint16),
                    frame_number=index,
                )
            },
            host_receive_timestamp_ns=index,
        )
        for index in range(count)
    )


def test_native_depth_sanity_passes_known_metric_plane() -> None:
    result = evaluate_native_depth_sanity(
        target=_target(),
        calibration=_factory(),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=_frames(750),
        frame_indices=(0, 1),
    )
    assert result["status"] == "PASS"
    assert result["sample_point_count"] == 35
    assert result["requested_samples"] == 70
    assert result["valid_samples"] == 70
    assert result["valid_sample_ratio"] == 1.0
    assert result["median_absolute_error_mm"] == pytest.approx(0.0, abs=1e-9)
    assert result["p95_absolute_error_mm"] == pytest.approx(0.0, abs=1e-9)
    assert result["signed_bias_mm"] == pytest.approx(0.0, abs=1e-9)


def test_native_depth_sanity_fails_a_gross_scale_or_direction_error() -> None:
    result = evaluate_native_depth_sanity(
        target=_target(),
        calibration=_factory(),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=_frames(1000),
        frame_indices=(0, 1),
    )
    assert result["status"] == "FAIL"
    assert result["median_absolute_error_mm"] == pytest.approx(250.0)
    assert result["p95_absolute_error_mm"] == pytest.approx(250.0)
    checks = result["checks"]
    assert isinstance(checks, dict)
    assert checks["valid_samples_at_least_minimum"] is True
    assert checks["median_absolute_error_within_limit"] is False
    assert checks["p95_absolute_error_within_limit"] is False


def test_native_depth_sanity_fails_one_bad_frame_instead_of_hiding_it_in_median() -> None:
    frames = _frames(750, count=1) + _frames(1000, count=1)
    result = evaluate_native_depth_sanity(
        target=_target(),
        calibration=_factory(),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=frames,
        frame_indices=(0, 1),
    )
    assert result["status"] == "FAIL"
    assert result["passing_frame_count"] == 1
    assert result["checks"]["all_evaluated_frames_pass_geometry"] is False


def test_native_depth_sanity_fails_sparse_and_nonfinite_depth() -> None:
    sparse = np.zeros((480, 640), dtype=np.float64)
    sparse[239:242, 319:322] = np.nan
    frame = CameraFrame(
        camera_name="synthetic",
        serial="test",
        streams={"depth": StreamFrame("depth", sparse, frame_number=0)},
        host_receive_timestamp_ns=0,
    )
    result = evaluate_native_depth_sanity(
        target=_target(),
        calibration=_factory(),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=(frame,),
        frame_indices=(0,),
        fail_closed=True,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["valid_samples_at_least_minimum"] is False


def test_metric_depth_receipt_rejects_status_only_forgery() -> None:
    with pytest.raises(ContractError, match="evaluation fields"):
        build_metric_depth_receipt(
            evaluation={"status": "PASS"},
            camera_identity_sha256="a" * 64,
            target_identity_sha256="b" * 64,
            factory_calibration_sha256="c" * 64,
            capture_manifest_sha256="d" * 64,
        )


def test_metric_depth_failed_receipt_requires_explicit_nonpassing_route(tmp_path: Path) -> None:
    thresholds = BOOTSTRAP_METRIC_DEPTH_THRESHOLDS
    evaluation = evaluate_native_depth_sanity(
        target=_target(),
        calibration=_factory(),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=_frames(1000, count=30),
        frame_indices=tuple(range(30)),
        minimum_valid_samples=int(thresholds["minimum_valid_samples"]),
        minimum_valid_frames=int(thresholds["minimum_valid_frames"]),
        minimum_valid_sample_ratio=float(thresholds["minimum_valid_sample_ratio"]),
        minimum_region_valid_samples=int(thresholds["minimum_region_valid_samples"]),
        minimum_frame_valid_samples=int(thresholds["minimum_frame_valid_samples"]),
        minimum_passing_frames=int(thresholds["minimum_passing_frames"]),
        minimum_passing_frame_ratio=float(thresholds["minimum_passing_frame_ratio"]),
        maximum_median_error_mm=float(thresholds["maximum_median_error_mm"]),
        maximum_p95_error_mm=float(thresholds["maximum_p95_error_mm"]),
        maximum_plane_offset_mm=float(thresholds["maximum_plane_offset_mm"]),
        maximum_plane_normal_error_deg=float(thresholds["maximum_plane_normal_error_deg"]),
        maximum_scale_ratio_error=float(thresholds["maximum_scale_ratio_error"]),
        threshold_policy={
            "schema_version": BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
            "source": "immutable_fixed_provision_contract",
        },
        fail_closed=True,
    )
    kwargs = {
        "evaluation": evaluation,
        "camera_identity_sha256": "a" * 64,
        "target_identity_sha256": "b" * 64,
        "factory_calibration_sha256": "c" * 64,
        "capture_manifest_sha256": "d" * 64,
    }
    with pytest.raises(ContractError, match="not passed"):
        build_metric_depth_receipt(**kwargs)
    receipt = build_metric_depth_receipt(**kwargs, require_pass=False)
    path = tmp_path / "failed-metric-depth.json"
    write_metric_depth_receipt(path, receipt, require_pass=False)
    assert load_metric_depth_receipt(path, require_pass=False)["status"] == "FAIL"


def test_metric_depth_evaluation_rejects_rehashed_check_forgery() -> None:
    result = evaluate_native_depth_sanity(
        target=_target(),
        calibration=_factory(),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=_frames(750, count=1),
        frame_indices=(0,),
    )
    result["checks"]["plane_offset_within_limit"] = False
    result["status"] = "FAIL"
    with pytest.raises(ContractError, match="recomputed decision"):
        validate_native_depth_evaluation(result)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("measured_plane_distance_m", 9.0),
        ("plane_offset_mm", 0.0),
        ("distance_scale_ratio", 1.0),
        ("median_absolute_depth_residual_mm", 0.0),
    ],
)
def test_metric_depth_frame_summaries_are_recomputed_from_measurement_evidence(
    field: str, value: float
) -> None:
    result = evaluate_native_depth_sanity(
        target=_target(),
        calibration=_factory(),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=_frames(1000, count=1),
        frame_indices=(0,),
    )
    forged = deepcopy(result)
    frame = forged["per_frame"][0]
    frame[field] = value
    with pytest.raises(ContractError, match=r"measurement evidence|distance identity"):
        validate_native_depth_evaluation(forged)


def test_native_depth_sanity_rejects_duplicate_frame_indices() -> None:
    with pytest.raises(ContractError, match="frame indices must be unique"):
        evaluate_native_depth_sanity(
            target=_target(),
            calibration=_factory(),
            T_detection_from_target=_front_pose(),
            detection_stream="color",
            frames=_frames(750, count=2),
            frame_indices=(0, 0),
        )


def test_native_depth_sanity_skips_an_unsupported_projection_model() -> None:
    result = evaluate_native_depth_sanity(
        target=_target(),
        calibration=_factory(depth_distortion="ftheta"),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=_frames(750),
        frame_indices=(0,),
    )
    assert result["status"] == "SKIPPED_WITH_WARNING"
    assert "unsupported" in str(result["warning"])
    assert "not used by pose optimization" in str(result["role"])


def test_fixed_pose_overlay_writes_diagnostics_and_frame_selection(tmp_path: Path) -> None:
    target = _target()
    pose = _oblique_pose()
    intrinsics = _intrinsics("synthetic/color_optical")
    point_ids = tuple(point_id for point_id, _point in target.corner_points)
    object_points = target.object_points_for(point_ids)
    detected = project_points_px(object_points, pose, intrinsics)
    observation = TargetObservation(
        plugin_name="synthetic-target",
        target_frame=target.target_frame,
        point_ids=point_ids,
        image_points_px=detected,
        object_points_m=object_points,
        image_size=(640, 480),
        quality=QualityReport(True),
    )
    output = tmp_path / "nested" / "pose.png"
    write_fixed_pose_overlay(
        output,
        image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        observation=observation,
        T_camera_from_target=pose,
        intrinsics=intrinsics,
        board_width_m=target.board_width_m,
        board_height_m=target.board_height_m,
    )
    assert output.is_file()
    with Image.open(output) as rendered:
        assert rendered.size == (640, 480)
        pixels = np.asarray(rendered)
    assert np.count_nonzero(pixels) > 500

    per_frame: tuple[dict[str, object], ...] = (
        {"frame_index": 9, "accepted": True, "final_pose_reprojection_rmse_px": 0.30},
        {"frame_index": 2, "accepted": False, "final_pose_reprojection_rmse_px": 0.01},
        {"frame_index": 4, "accepted": True, "final_pose_reprojection_rmse_px": 0.10},
        {"frame_index": 7, "accepted": True, "final_pose_reprojection_rmse_px": 0.20},
    )
    assert select_overlay_frames(per_frame) == {
        "best": 4,
        "median_quality": 7,
        "worst_accepted": 9,
    }
    assert select_overlay_frames(({"frame_index": 1, "accepted": False},)) == {}
