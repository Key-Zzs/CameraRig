from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from camera_rig.calibration.fixed.aggregation import (
    distribution,
    even_odd_partition,
    pose_delta,
    pose_inlier_indices,
    pose_medoid_index,
)
from camera_rig.calibration.fixed.artifact import (
    FixedCalibrationArtifact,
    load_and_validate_fixed_calibration,
    write_fixed_calibration,
)
from camera_rig.calibration.fixed.config import load_fixed_config
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform

REPOSITORY_ROOT = Path(__file__).parents[1]


def _pose(x_m: float) -> RigidTransform:
    matrix = np.eye(4)
    matrix[0, 3] = x_m
    matrix[2, 3] = 1.0
    return RigidTransform("target", "camera", matrix)


def test_fixed_config_example_is_strict_and_identity_workspace() -> None:
    config = load_fixed_config(REPOSITORY_ROOT / "configs/examples/fixed_calibration_contract.yaml")
    assert config.workspace_frame == "workspace"
    assert config.target_frame == "charuco_target"
    assert np.array_equal(config.T_workspace_from_target.matrix, np.eye(4))
    assert config.solver.minimum_accepted_frames == 50


def test_pose_aggregation_is_robust_and_frame_explicit() -> None:
    poses = [_pose(0.0), _pose(0.0005), _pose(-0.0004), _pose(0.050)]
    medoid = pose_medoid_index(poses)
    assert medoid in {0, 1, 2}
    assert pose_inlier_indices(
        poses,
        medoid,
        maximum_translation_mm=5.0,
        maximum_rotation_deg=0.5,
    ) == [0, 1, 2]
    assert pose_delta(poses[0], poses[1]).translation_mm == pytest.approx(0.5)
    assert distribution([1.0, 2.0, 3.0])["p95"] == pytest.approx(2.9)
    assert even_odd_partition([0, 1, 2, 3]) == ([0, 2], [1, 3])


def _artifact() -> FixedCalibrationArtifact:
    workspace_from_target = RigidTransform("charuco_target", "workspace", np.eye(4))
    detection_from_target_matrix = np.eye(4)
    detection_from_target_matrix[2, 3] = 1.0
    detection_from_target = RigidTransform(
        "charuco_target", "head/color_optical", detection_from_target_matrix
    )
    workspace_from_detection = workspace_from_target.compose(detection_from_target.inverse())
    detection_from_reference_matrix = np.eye(4)
    detection_from_reference_matrix[0, 3] = 0.02
    detection_from_reference = RigidTransform(
        "head/ir_left_optical", "head/color_optical", detection_from_reference_matrix
    )
    workspace_from_reference = workspace_from_detection.compose(detection_from_reference)
    quality = QualityReport(passed=True, metrics={"accepted_frames": 60})
    fixed = FixedMountCalibration(
        parent_frame="workspace",
        camera_reference_frame="head/ir_left_optical",
        T_parent_from_camera_reference=workspace_from_reference,
        quality=quality,
    )
    return FixedCalibrationArtifact(
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        workspace={
            "frame": "workspace",
            "target_frame": "charuco_target",
            "T_workspace_from_target": workspace_from_target.to_dict(),
        },
        camera={
            "detection_stream": "color",
            "detection_frame": "head/color_optical",
            "reference_stream": "ir_left",
            "reference_frame": "head/ir_left_optical",
        },
        target={
            "target_spec_sha256": "a" * 64,
            "print_provenance": {
                "horizontal_print_scale": 0.997,
                "vertical_print_scale": 0.997,
                "maximum_observed_print_scale_error": 0.003,
                "geometry_policy": "pose uses nominal persisted target geometry",
            },
        },
        inputs={
            "capture_manifest_sha256": "b" * 64,
            "factory_calibration_sha256": "c" * 64,
            "target_detection_sha256": "d" * 64,
        },
        solver={"method": "ippe", "refinement": "lm", "thresholds": {}},
        per_frame_pose_summary=(
            {"frame_index": 0, "corner_count": 20, "accepted": True, "failure_reasons": []},
        ),
        aggregate={
            "accepted_frames": 1,
            "accepted_ratio": 1.0,
            "reprojection": {},
            "pose_repeatability": {},
            "split_half": {},
            "native_depth_sanity": {"status": "SKIPPED_WITH_WARNING"},
        },
        T_detection_from_target=detection_from_target,
        T_workspace_from_detection=workspace_from_detection,
        T_detection_from_reference=detection_from_reference,
        T_workspace_from_reference=workspace_from_reference,
        fixed_mount_calibration=fixed,
        quality=quality,
        provenance={"overlays": "overlays"},
    )


def test_fixed_calibration_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "fixed.json"
    write_fixed_calibration(path, _artifact())
    restored = load_and_validate_fixed_calibration(path)
    assert restored.quality.passed
    assert restored.fixed_mount_calibration.camera_reference_frame == "head/ir_left_optical"


def test_fixed_calibration_rejects_wrong_chain() -> None:
    artifact = _artifact()
    wrong = RigidTransform(
        artifact.T_workspace_from_reference.source_frame,
        artifact.T_workspace_from_reference.target_frame,
        np.eye(4),
    )
    with pytest.raises(ContractError, match="chain is inconsistent"):
        replace(artifact, T_workspace_from_reference=wrong)


def test_fixed_calibration_loader_rejects_failed_quality(tmp_path: Path) -> None:
    artifact = _artifact()
    failed_quality = QualityReport(passed=False, failure_reasons=("gate failed",))
    failed_fixed = FixedMountCalibration(
        parent_frame="workspace",
        camera_reference_frame="head/ir_left_optical",
        T_parent_from_camera_reference=artifact.T_workspace_from_reference,
        quality=failed_quality,
    )
    failed = replace(
        artifact,
        fixed_mount_calibration=failed_fixed,
        quality=failed_quality,
    )
    path = tmp_path / "failed.json"
    from camera_rig.artifacts.io import atomic_write_json

    atomic_write_json(path, failed.to_dict())
    with pytest.raises(ArtifactError, match="quality is not passed"):
        load_and_validate_fixed_calibration(path)
