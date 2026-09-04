from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.artifacts.hashing import sha256_bytes
from camera_rig.artifacts.io import deterministic_json_bytes
from camera_rig.artifacts.target_detection import (
    TargetDetectionArtifact,
    TargetDetectionFrame,
)
from camera_rig.calibration.fixed.artifact import (
    load_and_validate_fixed_calibration,
    write_fixed_calibration,
)
from camera_rig.calibration.fixed.calibrator import (
    FixedCameraCalibrator,
    _validated_print_provenance,
)
from camera_rig.calibration.fixed.config import (
    FixedCalibrationConfig,
    FixedSolverThresholds,
)
from camera_rig.calibration.fixed.quality import evaluate_fixed_calibration_quality
from camera_rig.calibration.pose import project_points_px
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

pytest.importorskip("cv2")

TARGET_SHA = "a" * 64
CAPTURE_SHA = "b" * 64


def _intrinsics(stream: str) -> CameraIntrinsics:
    return CameraIntrinsics(
        frame=f"head/{stream}_optical",
        width=640,
        height=480,
        fx=800.0,
        fy=805.0,
        cx=319.5,
        cy=239.5,
        distortion_model="none",
    )


def _factory(
    *, include_transform: bool = True, near_tolerance_rotation: bool = False
) -> FactoryCalibrationArtifact:
    profiles = {
        name: StreamProfile(name, 640, 480, 30, "rgb8" if name == "color" else "y8")
        for name in ("color", "ir_left")
    }
    internal_transforms: tuple[RigidTransform, ...] = ()
    if include_transform:
        matrix = np.eye(4)
        if near_tolerance_rotation:
            matrix[0, 0] += 4e-8
        matrix[0, 3] = 0.02
        internal_transforms = (
            RigidTransform("head/ir_left_optical", "head/color_optical", matrix),
        )
    calibration = FactoryCalibration(
        device=CameraDeviceInfo("synthetic", "head", "D435i", "D435i", "placeholder"),
        stream_profiles=profiles,
        intrinsics={name: _intrinsics(name) for name in profiles},
        internal_transforms=internal_transforms,
        depth_scale_m_per_unit=0.001,
    )
    return FactoryCalibrationArtifact(
        created_at=datetime.now(timezone.utc).isoformat(),
        calibration=calibration,
        quality=QualityReport(True),
        provenance={"source": "synthetic"},
    )


def _config() -> FixedCalibrationConfig:
    return FixedCalibrationConfig(
        workspace_frame="workspace",
        target_frame="charuco_target",
        T_workspace_from_target=RigidTransform("charuco_target", "workspace", np.eye(4)),
        detection_stream="color",
        reference_stream="ir_left",
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


def _object_points() -> np.ndarray:
    return np.asarray(
        [[0.03 * column, 0.03 * row, 0.0] for row in range(4) for column in range(6)],
        dtype=np.float64,
    )


def _pose(*, x_offset_m: float = 0.0) -> RigidTransform:
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [-0.075 + x_offset_m, 0.045, 0.72]
    return RigidTransform("charuco_target", "head/color_optical", matrix)


def _observation(pose: RigidTransform, frame_index: int) -> TargetObservation:
    points = _object_points()
    pixels = project_points_px(points, pose, _intrinsics("color")).copy()
    pixels += np.random.default_rng(1000 + frame_index).normal(0.0, 0.02, pixels.shape)
    return TargetObservation(
        plugin_name="synthetic-grid",
        target_frame="charuco_target",
        point_ids=tuple(range(len(points))),
        image_points_px=pixels,
        object_points_m=points,
        image_size=(640, 480),
        quality=QualityReport(True),
        metadata={"target_spec_sha256": TARGET_SHA},
    )


def _detection() -> TargetDetectionArtifact:
    frames = tuple(
        TargetDetectionFrame(
            frame_index=index,
            success=True,
            observation=_observation(_pose(x_offset_m=0.03 if index == 5 else 0.0), index),
            overlay=None,
        )
        for index in range(6)
    )
    return TargetDetectionArtifact(
        target_spec_sha256=TARGET_SHA,
        capture_manifest_sha256=CAPTURE_SHA,
        stream="color",
        frame_count=len(frames),
        per_frame=frames,
        aggregate={"success_ratio": 1.0},
        acceptance={"passed": True, "thresholds": {}, "checks": {"r6_passed": True}},
        selected_overlays={},
        software={"camera_rig_version": "0.3.0", "opencv_version": "4.14.0"},
    )


def _print_provenance() -> dict[str, object]:
    return {
        "horizontal_print_scale": 0.997,
        "vertical_print_scale": 0.997,
        "maximum_observed_print_scale_error": 0.003,
        "geometry_policy": "pose uses nominal persisted target geometry",
    }


def test_existing_target_print_provenance_records_measurement_identity_without_fake_scale() -> None:
    value = {
        "source_type": "existing_physical",
        "physical_measurement_sha256": "d" * 64,
        "geometry_policy": "nominal measured geometry from registered target artifact",
    }
    assert _validated_print_provenance(value) == value
    with pytest.raises(ContractError, match="missing or unknown"):
        _validated_print_provenance(
            {
                **value,
                "horizontal_print_scale": 1.0,
            }
        )


def _artifact_sha(value: object) -> str:
    return sha256_bytes(deterministic_json_bytes(value))


def _solve(**changes: object):
    detection = _detection()
    factory = _factory()
    arguments: dict[str, object] = {
        "target_spec_sha256": TARGET_SHA,
        "capture_manifest_sha256": CAPTURE_SHA,
        "factory_calibration_sha256": _artifact_sha(factory.to_dict()),
        "target_detection_sha256": _artifact_sha(detection.to_dict()),
        "print_provenance": _print_provenance(),
        "native_depth_evaluator": lambda _pose, _indices: {
            "status": "SKIPPED_WITH_WARNING",
            "warning": "synthetic test has no native depth image",
        },
        "created_at": "2026-08-25T00:00:00Z",
    }
    arguments.update(changes)
    return FixedCameraCalibrator().calibrate(
        _config(),
        detection,
        factory,
        **arguments,  # type: ignore[arg-type]
    )


def test_fixed_calibrator_recovers_pose_rejects_outlier_and_builds_chain(
    tmp_path: Path,
) -> None:
    callback_evidence: dict[str, object] = {}

    def depth_evaluator(
        final_pose: RigidTransform, inlier_indices: tuple[int, ...]
    ) -> dict[str, object]:
        callback_evidence["pose"] = final_pose
        callback_evidence["indices"] = inlier_indices
        return {"status": "PASS", "valid_sample_count": 20}

    artifact = _solve(native_depth_evaluator=depth_evaluator)

    assert artifact.quality.passed
    assert artifact.aggregate["accepted_frames"] == 5
    assert artifact.per_frame_pose_summary[5]["accepted"] is False
    assert (
        "pose_outlier_threshold_exceeded" in artifact.per_frame_pose_summary[5]["failure_reasons"]
    )
    assert artifact.per_frame_pose_summary[5]["medoid_translation_delta_mm"] > 5.0
    first_summary = artifact.per_frame_pose_summary[0]
    candidates = first_summary["candidates"]
    assert isinstance(candidates, list) and len(candidates) == 2
    assert "reprojection" in candidates[0] and "validity" in candidates[0]
    assert first_summary["final_pose_reprojection_rmse_px"] < 0.1
    assert first_summary["final_pose_reprojection_p95_px"] < 0.1
    assert callback_evidence["indices"] == (0, 1, 2, 3, 4)
    assert isinstance(callback_evidence["pose"], RigidTransform)
    np.testing.assert_allclose(
        artifact.T_detection_from_target.matrix,
        _pose().matrix,
        atol=8e-4,
    )
    expected_detection_from_reference = _factory().calibration.internal_transforms[0]
    np.testing.assert_allclose(
        artifact.T_detection_from_reference.matrix,
        expected_detection_from_reference.matrix,
    )
    expected_workspace_from_reference = artifact.T_workspace_from_detection.compose(
        expected_detection_from_reference
    )
    np.testing.assert_allclose(
        artifact.T_workspace_from_reference.matrix,
        expected_workspace_from_reference.matrix,
    )
    reprojection = artifact.aggregate["reprojection"]
    assert isinstance(reprojection, dict)
    assert len(reprojection["per_frame"]) == 5  # type: ignore[arg-type]
    assert len(reprojection["per_corner_id"]) == 24  # type: ignore[arg-type]
    assert artifact.aggregate["split_half"]["status"] == "AVAILABLE"  # type: ignore[index]

    path = tmp_path / "fixed.json"
    write_fixed_calibration(path, artifact)
    restored = load_and_validate_fixed_calibration(path)
    assert restored.quality.passed


def test_uncertainty_policy_gates_frames_and_final_shared_pose() -> None:
    legacy = _detection()
    detection = replace(
        legacy,
        acceptance={
            "passed": True,
            "policy": "uncertainty_validated",
            "thresholds": {},
            "checks": {"uncertainty_capture_passed": True},
        },
    )
    factory = _factory()
    artifact = FixedCameraCalibrator().calibrate(
        _config(),
        detection,
        factory,
        target_spec_sha256=TARGET_SHA,
        capture_manifest_sha256=CAPTURE_SHA,
        factory_calibration_sha256=_artifact_sha(factory.to_dict()),
        target_detection_sha256=_artifact_sha(detection.to_dict()),
        print_provenance=_print_provenance(),
        native_depth_evaluator=lambda _pose, _indices: {"status": "PASS"},
    )
    assert artifact.quality.passed
    assert artifact.aggregate["pose_policy"] == "uncertainty_validated"
    assert artifact.aggregate["observable_frame_ratio"] == pytest.approx(1.0)
    final = artifact.aggregate["final_pose_observability"]
    assert isinstance(final, dict) and final["passed"] is True
    assert final["effective_rank"] == 6
    assert final["candidate_ambiguity"]["valid_candidate_count"] == 2
    assert final["candidate_ambiguity"]["second_candidate_available"] is True
    assert all(
        "observability" in summary
        for summary in artifact.per_frame_pose_summary
        if summary["T_camera_from_target"] is not None
    )
    assert artifact.quality.metrics["checks"]["final_pose_observability"] is True  # type: ignore[index]
    policy = artifact.solver["reprojection_policy"]
    assert policy["release_state"] == "HOLD"  # type: ignore[index]
    assert policy["candidate_successor"]["preset"] == "uncertainty_validated_v2"  # type: ignore[index]
    assert policy["candidate_successor"]["production_eligible"] is False  # type: ignore[index]
    structured = artifact.aggregate["residual_diagnostics"]["final_shared_pose"]  # type: ignore[index]
    assert structured["eligible_corner_count"] == 24  # type: ignore[index]
    assert structured["structured_metrics"]["scope"] == "final"  # type: ignore[index]

    one_valid_candidate = dict(final)
    one_valid_candidate["candidate_ambiguity"] = {
        **final["candidate_ambiguity"],
        "valid_candidate_count": 1,
        "second_candidate_available": False,
        "second_candidate_index": None,
        "ambiguous": False,
    }
    aggregate = artifact.aggregate
    quality = evaluate_fixed_calibration_quality(
        thresholds=_config().solver,
        frame_count=6,
        accepted_frames=int(aggregate["accepted_frames"]),
        global_reprojection=aggregate["reprojection"]["global"],
        pose_repeatability=aggregate["pose_repeatability"],
        split_half=aggregate["split_half"],
        native_depth_sanity=aggregate["native_depth_sanity"],
        pose_policy="uncertainty_validated",
        final_pose_observability=one_valid_candidate,
        observable_frame_ratio=float(aggregate["observable_frame_ratio"]),
        ambiguous_frame_ratio=float(aggregate["ambiguous_frame_ratio"]),
    )
    assert quality.metrics["checks"]["final_pose_unambiguous"] is True  # type: ignore[index]

    no_valid_candidate = dict(one_valid_candidate)
    no_valid_candidate["candidate_ambiguity"] = {
        **one_valid_candidate["candidate_ambiguity"],
        "valid_candidate_count": 0,
    }
    no_candidate_quality = evaluate_fixed_calibration_quality(
        thresholds=_config().solver,
        frame_count=6,
        accepted_frames=int(aggregate["accepted_frames"]),
        global_reprojection=aggregate["reprojection"]["global"],
        pose_repeatability=aggregate["pose_repeatability"],
        split_half=aggregate["split_half"],
        native_depth_sanity=aggregate["native_depth_sanity"],
        pose_policy="uncertainty_validated",
        final_pose_observability=no_valid_candidate,
        observable_frame_ratio=float(aggregate["observable_frame_ratio"]),
        ambiguous_frame_ratio=float(aggregate["ambiguous_frame_ratio"]),
    )
    assert no_candidate_quality.metrics["checks"]["final_pose_unambiguous"] is False  # type: ignore[index]


def test_fixed_calibrator_native_depth_failure_is_fail_closed() -> None:
    artifact = _solve(native_depth_evaluator=lambda _pose, _indices: {"status": "FAIL"})

    assert not artifact.quality.passed
    assert not artifact.fixed_mount_calibration.quality.passed
    assert "native_depth_sanity" in artifact.quality.failure_reasons


def test_fixed_calibrator_rejects_input_identity_mismatch() -> None:
    with pytest.raises(ContractError, match="target SHA"):
        _solve(target_spec_sha256="e" * 64)
    with pytest.raises(ContractError, match="typed detection artifact"):
        _solve(target_detection_sha256="e" * 64)
    with pytest.raises(ContractError, match="typed factory artifact"):
        _solve(factory_calibration_sha256="e" * 64)


def test_fixed_calibrator_rejects_missing_factory_transform_chain() -> None:
    detection = _detection()
    factory = _factory(include_transform=False)
    with pytest.raises(ContractError, match="no transform path"):
        FixedCameraCalibrator().calibrate(
            _config(),
            detection,
            factory,
            target_spec_sha256=TARGET_SHA,
            capture_manifest_sha256=CAPTURE_SHA,
            factory_calibration_sha256=_artifact_sha(factory.to_dict()),
            target_detection_sha256=_artifact_sha(detection.to_dict()),
            print_provenance=_print_provenance(),
            native_depth_evaluator=lambda _pose, _indices: {"status": "PASS"},
        )


def test_fixed_calibrator_chain_tolerance_matches_rigid_transform_contract() -> None:
    detection = _detection()
    factory = _factory(near_tolerance_rotation=True)
    artifact = FixedCameraCalibrator().calibrate(
        _config(),
        detection,
        factory,
        target_spec_sha256=TARGET_SHA,
        capture_manifest_sha256=CAPTURE_SHA,
        factory_calibration_sha256=_artifact_sha(factory.to_dict()),
        target_detection_sha256=_artifact_sha(detection.to_dict()),
        print_provenance=_print_provenance(),
        native_depth_evaluator=lambda _pose, _indices: {"status": "PASS"},
    )

    assert artifact.quality.passed


def test_fixed_calibrator_failed_global_gate_cannot_validate_as_passed(tmp_path: Path) -> None:
    strict = replace(
        _config().solver,
        maximum_pose_translation_p95_mm=1e-8,
        maximum_pose_rotation_p95_deg=1e-8,
    )
    config = replace(_config(), solver=strict)
    detection = _detection()
    factory = _factory()
    artifact = FixedCameraCalibrator().calibrate(
        config,
        detection,
        factory,
        target_spec_sha256=TARGET_SHA,
        capture_manifest_sha256=CAPTURE_SHA,
        factory_calibration_sha256=_artifact_sha(factory.to_dict()),
        target_detection_sha256=_artifact_sha(detection.to_dict()),
        print_provenance=_print_provenance(),
        native_depth_evaluator=lambda _pose, _indices: {"status": "PASS"},
    )
    assert not artifact.quality.passed
    path = tmp_path / "failed.json"
    from camera_rig.artifacts.io import atomic_write_json

    atomic_write_json(path, artifact.to_dict())
    with pytest.raises(ArtifactError, match="quality is not passed"):
        load_and_validate_fixed_calibration(path)
