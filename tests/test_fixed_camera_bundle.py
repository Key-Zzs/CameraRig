from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.artifacts.io import deterministic_json_bytes, json_safe
from camera_rig.artifacts.models import CameraBundle
from camera_rig.artifacts.stream_validation import StreamValidationArtifact
from camera_rig.artifacts.target_detection import TargetDetectionArtifact, TargetDetectionFrame
from camera_rig.calibration.fixed.artifact import FixedCalibrationArtifact
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError, ContractError, TransformError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform
from camera_rig.provision.bundle import (
    build_fixed_camera_bundle,
    load_and_validate_fixed_camera_bundle,
    validate_fixed_camera_bundle_data,
    write_fixed_camera_bundle,
)
from camera_rig.targets.observation import TargetObservation

REPOSITORY_ROOT = Path(__file__).parents[1]
STREAMS = ("color", "depth", "ir_left", "ir_right")
TARGET_SHA = "a" * 64
CAPTURE_SHA = "b" * 64


def _digest(value: object) -> str:
    import hashlib

    return hashlib.sha256(deterministic_json_bytes(value)).hexdigest()


def _factory(*, connected: bool = True) -> FactoryCalibrationArtifact:
    reference = "head/ir_left_optical"
    profiles = {
        name: StreamProfile(
            name,
            640,
            480,
            30,
            {"color": "rgb8", "depth": "z16"}.get(name, "y8"),
        )
        for name in STREAMS
    }
    intrinsics = {
        name: CameraIntrinsics(
            f"head/{name}_optical",
            640,
            480,
            600.0,
            601.0,
            319.5,
            239.5,
            "none",
        )
        for name in STREAMS
    }
    transforms = []
    if connected:
        for index, name in enumerate(("color", "depth", "ir_right"), start=1):
            matrix = np.eye(4)
            matrix[0, 3] = index * 0.01
            transforms.append(RigidTransform(reference, intrinsics[name].frame, matrix))
    calibration = FactoryCalibration(
        device=CameraDeviceInfo("synthetic", "head", "D435i", "D435i", "synthetic"),
        stream_profiles=profiles,
        intrinsics=intrinsics,
        internal_transforms=tuple(transforms),
        depth_scale_m_per_unit=0.001,
    )
    return FactoryCalibrationArtifact(
        created_at="2026-08-25T00:00:00Z",
        calibration=calibration,
        quality=QualityReport(True, metrics={"synthetic": True}),
        provenance={"source": "unit-test"},
    )


def _stream_validation() -> StreamValidationArtifact:
    report: dict[str, object] = {
        "schema_version": "camera-rig.stream-validation.v1",
        "status": "PASS",
        "requested_frames": 3,
        "received_frames": 3,
        "duration_s": 0.1,
        "per_stream_observed_fps": {name: 30.0 for name in STREAMS},
        "per_stream_frame_number_discontinuities": {name: 0 for name in STREAMS},
        "per_stream_discontinuity_ratio": {name: 0.0 for name in STREAMS},
        "per_stream_timestamp_monotonicity": {name: True for name in STREAMS},
        "per_stream_timestamp_domain_counts": {name: {"hardware_clock": 3} for name in STREAMS},
        "ir_stereo_frame_match_ratio": 1.0,
        "comparable_timestamp_skew_ns": {"p50": 0.0, "p95": 0.0, "max": 0},
        "sync_valid_ratio": 1.0,
        "timeouts": 0,
        "missing_streams": {},
        "shape_consistency": {},
        "dtype_consistency": {},
        "depth_valid_ratio": 1.0,
        "rgb_variance": 1.0,
        "rgb_channel_variance": [1.0, 1.0, 1.0],
        "ir_variance": {"ir_left": 1.0, "ir_right": 1.0},
        "ir_distinct_ratio": 1.0,
        "failure_reasons": [],
    }
    return StreamValidationArtifact.from_accumulator_report(
        report, provenance={"source": "unit-test"}
    )


def _target_detection() -> TargetDetectionArtifact:
    observation = TargetObservation(
        plugin_name="synthetic-target",
        target_frame="charuco_target",
        point_ids=(0, 1, 2, 3),
        image_points_px=np.asarray([[100, 100], [200, 100], [100, 200], [200, 200]]),
        object_points_m=np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]]),
        image_size=(640, 480),
        quality=QualityReport(True),
        metadata={"target_spec_sha256": TARGET_SHA},
    )
    return TargetDetectionArtifact(
        target_spec_sha256=TARGET_SHA,
        capture_manifest_sha256=CAPTURE_SHA,
        stream="color",
        frame_count=1,
        per_frame=(TargetDetectionFrame(0, True, observation, "overlays/frame.png"),),
        aggregate={"success_ratio": 1.0},
        acceptance={"passed": True, "checks": {"synthetic": True}},
        selected_overlays={"best": 0},
        software={"camera_rig_version": "test", "opencv_version": "test"},
    )


def _fixed(
    factory: FactoryCalibrationArtifact,
    target_detection: TargetDetectionArtifact,
    *,
    distance_m: float = 1.0,
    printed_face_facing_camera: bool = True,
) -> FixedCalibrationArtifact:
    workspace_from_target = RigidTransform("charuco_target", "workspace", np.eye(4))
    detection_matrix = np.eye(4)
    if printed_face_facing_camera:
        detection_matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    detection_matrix[2, 3] = distance_m
    detection_from_target = RigidTransform("charuco_target", "head/color_optical", detection_matrix)
    workspace_from_detection = workspace_from_target.compose(detection_from_target.inverse())
    graph_transform = next(
        item
        for item in factory.calibration.internal_transforms
        if item.target_frame == "head/color_optical"
    )
    workspace_from_reference = workspace_from_detection.compose(graph_transform)
    quality = QualityReport(True, metrics={"accepted_frames": 60})
    mount = FixedMountCalibration(
        parent_frame="workspace",
        camera_reference_frame="head/ir_left_optical",
        T_parent_from_camera_reference=workspace_from_reference,
        quality=quality,
        provenance={"source": "fixed-calibration"},
    )
    return FixedCalibrationArtifact(
        created_at="2026-08-25T00:00:00Z",
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
        target={"target_spec_sha256": TARGET_SHA, "print_provenance": {}},
        inputs={
            "capture_manifest_sha256": CAPTURE_SHA,
            "factory_calibration_sha256": _digest(factory.to_dict()),
            "target_detection_sha256": _digest(target_detection.to_dict()),
        },
        solver={"method": "ippe", "refinement": "lm", "thresholds": {}},
        per_frame_pose_summary=({"frame_index": 0, "accepted": True, "failure_reasons": []},),
        aggregate={"accepted_frames": 60},
        T_detection_from_target=detection_from_target,
        T_workspace_from_detection=workspace_from_detection,
        T_detection_from_reference=graph_transform,
        T_workspace_from_reference=workspace_from_reference,
        fixed_mount_calibration=mount,
        quality=quality,
        provenance={"source": "unit-test"},
    )


def _bundle(*, distance_m: float = 1.0) -> CameraBundle:
    factory = _factory()
    detection = _target_detection()
    fixed = _fixed(factory, detection, distance_m=distance_m)
    return build_fixed_camera_bundle(
        bundle_id="synthetic-fixed-camera",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        factory=factory,
        stream_validation=_stream_validation(),
        target_detection=detection,
        fixed_calibration=fixed,
        provenance={"source": "unit-test", "artifacts": "artifacts"},
    )


def test_builder_aggregates_passing_evidence_and_required_checks() -> None:
    bundle = _bundle()
    assert bundle.schema_version == "camera-rig.bundle.v1"
    assert bundle.status == "passed"
    assert bundle.quality.passed
    checks = bundle.quality.metrics["checks"]
    assert isinstance(checks, dict)
    assert checks and all(checks.values())
    assert bundle.fixed_mount_calibration is not None
    assert bundle.fixed_mount_calibration.quality.passed
    evidence = bundle.provenance["evidence_sha256"]
    assert isinstance(evidence, dict)
    assert set(evidence) == {
        "factory_calibration",
        "stream_validation",
        "target_detection",
        "fixed_calibration",
    }


@pytest.mark.parametrize(
    "forged_policy",
    [
        {"release_state": "HOLD"},
        {"release_state": "RELEASED"},
        {
            "preset": "uncertainty_validated_v2",
            "release_state": "RELEASED",
            "release_manifest_sha256": "a" * 64,
            "structured_residual_decision": {"passed": True, "enforced": True},
        },
    ],
)
def test_uncertainty_publication_is_blocked_even_with_forged_release(
    forged_policy: dict[str, object],
) -> None:
    factory = _factory()
    detection = _target_detection()
    fixed = _fixed(factory, detection)
    forged = replace(
        fixed,
        solver={
            **fixed.solver,
            "pose_policy": "uncertainty_validated",
            "reprojection_policy": forged_policy,
        },
    )
    with pytest.raises(ContractError, match="not release-enabled"):
        build_fixed_camera_bundle(
            bundle_id="forged-release-must-not-publish",
            created_at="2026-09-04T00:00:00Z",
            factory=factory,
            stream_validation=_stream_validation(),
            target_detection=detection,
            fixed_calibration=forged,
            provenance={"source": "adversarial-unit-test"},
        )


@pytest.mark.parametrize(
    ("target_policy", "solver_policy"),
    [
        ("uncertainty_validated", None),
        (None, "uncertainty_validated"),
        ("legacy_strict", "uncertainty_validated"),
        ("uncertainty_validated", "legacy_strict"),
        ("unknown_policy", "unknown_policy"),
        ("legacy_strict", None),
        (None, "legacy_strict"),
    ],
)
def test_publication_rejects_missing_unknown_or_mismatched_policy_provenance(
    target_policy: str | None,
    solver_policy: str | None,
) -> None:
    factory = _factory()
    original = _target_detection()
    acceptance = dict(original.acceptance or {})
    if target_policy is not None:
        acceptance["policy"] = target_policy
    detection = replace(original, acceptance=acceptance)
    fixed = _fixed(factory, detection)
    solver = dict(fixed.solver)
    if solver_policy is not None:
        solver["pose_policy"] = solver_policy
    forged = replace(fixed, solver=solver)
    with pytest.raises(ContractError, match="not release-enabled"):
        build_fixed_camera_bundle(
            bundle_id="policy-provenance-must-fail-closed",
            created_at="2026-09-04T00:00:00Z",
            factory=factory,
            stream_validation=_stream_validation(),
            target_detection=detection,
            fixed_calibration=forged,
            provenance={"source": "adversarial-unit-test"},
        )


def test_write_reload_and_downstream_point_transform_smoke(tmp_path: Path) -> None:
    path = tmp_path / "fixed_camera_bundle.json"
    restored = write_fixed_camera_bundle(path, _bundle())
    assert restored.to_dict() == load_and_validate_fixed_camera_bundle(path).to_dict()
    fixed = restored.fixed_mount_calibration
    assert fixed is not None
    point_reference = np.asarray([0.04, -0.02, 0.30])
    point_workspace = fixed.T_parent_from_camera_reference.transform_points(point_reference)
    round_trip = fixed.T_parent_from_camera_reference.inverse().transform_points(point_workspace)
    np.testing.assert_allclose(round_trip, point_reference, atol=1e-12)


def test_status_is_passed_if_and_only_if_every_required_check_passes() -> None:
    data = copy.deepcopy(_bundle().to_dict())
    data["status"] = "failed"
    with pytest.raises(ArtifactError, match="if and only if"):
        validate_fixed_camera_bundle_data(json_safe(data))

    data = copy.deepcopy(_bundle().to_dict())
    quality = data["quality"]
    assert isinstance(quality, dict)
    metrics = quality["metrics"]
    assert isinstance(metrics, dict)
    checks = metrics["checks"]
    assert isinstance(checks, dict)
    checks["stream_validation_passed"] = False
    with pytest.raises(ArtifactError, match="if and only if"):
        validate_fixed_camera_bundle_data(json_safe(data))


@pytest.mark.parametrize("distance_m", [0.1, 0.05, 5.0, 6.0])
def test_builder_rejects_nonphysical_target_distance(distance_m: float) -> None:
    with pytest.raises(ContractError, match="strictly between"):
        _bundle(distance_m=distance_m)


def test_builder_rejects_negative_target_depth_and_reversed_printed_face() -> None:
    factory = _factory()
    detection = _target_detection()
    stream_validation = _stream_validation()
    with pytest.raises(ContractError, match="positive camera depth"):
        build_fixed_camera_bundle(
            bundle_id="invalid-physical-pose",
            created_at="2026-08-25T00:00:00Z",
            factory=factory,
            stream_validation=stream_validation,
            target_detection=detection,
            fixed_calibration=_fixed(factory, detection, distance_m=-1.0),
            provenance={},
        )
    with pytest.raises(ContractError, match="printed face"):
        build_fixed_camera_bundle(
            bundle_id="invalid-physical-pose",
            created_at="2026-08-25T00:00:00Z",
            factory=factory,
            stream_validation=stream_validation,
            target_detection=detection,
            fixed_calibration=_fixed(
                factory,
                detection,
                printed_face_facing_camera=False,
            ),
            provenance={},
        )


def test_semantic_reload_uses_rigid_transform_tolerance() -> None:
    data = copy.deepcopy(_bundle().to_dict())
    quality = data["quality"]
    assert isinstance(quality, dict)
    metrics = quality["metrics"]
    assert isinstance(metrics, dict)
    recorded = metrics["target_origin_distance_m"]
    assert isinstance(recorded, float)
    metrics["target_origin_distance_m"] = recorded + 5e-8
    assert validate_fixed_camera_bundle_data(json_safe(data)).status == "passed"


def test_factory_transform_binding_uses_rigid_transform_tolerance() -> None:
    factory = _factory()
    detection = _target_detection()
    fixed = _fixed(factory, detection)
    transforms = list(factory.calibration.internal_transforms)
    color_matrix = transforms[0].matrix.copy()
    color_matrix[0, 3] += 5e-8
    transforms[0] = RigidTransform(
        transforms[0].source_frame,
        transforms[0].target_frame,
        color_matrix,
    )
    calibration = replace(factory.calibration, internal_transforms=tuple(transforms))
    perturbed_factory = replace(factory, calibration=calibration)
    fixed = replace(
        fixed,
        inputs={
            **fixed.inputs,
            "factory_calibration_sha256": _digest(perturbed_factory.to_dict()),
        },
    )

    bundle = build_fixed_camera_bundle(
        bundle_id="factory-tolerance",
        created_at="2026-08-25T00:00:00Z",
        factory=perturbed_factory,
        stream_validation=_stream_validation(),
        target_detection=detection,
        fixed_calibration=fixed,
        provenance={},
    )
    assert bundle.status == "passed"


@pytest.mark.parametrize(
    "unsafe", ["/home/user/private.json", "C:/private/calibration.json", "file://private"]
)
def test_builder_rejects_absolute_provenance(unsafe: str) -> None:
    factory = _factory()
    detection = _target_detection()
    with pytest.raises(ContractError, match="absolute paths"):
        build_fixed_camera_bundle(
            bundle_id="synthetic-fixed-camera",
            created_at="2026-08-25T00:00:00Z",
            factory=factory,
            stream_validation=_stream_validation(),
            target_detection=detection,
            fixed_calibration=_fixed(factory, detection),
            provenance={"nested": {"path": unsafe}},
        )


def test_builder_rejects_failed_fixed_mount_and_disconnected_factory() -> None:
    factory = _factory()
    detection = _target_detection()
    fixed = _fixed(factory, detection)
    failed_quality = QualityReport(False, failure_reasons=("failed gate",))
    failed_mount = replace(fixed.fixed_mount_calibration, quality=failed_quality)
    failed_fixed = replace(
        fixed,
        fixed_mount_calibration=failed_mount,
        quality=failed_quality,
    )
    with pytest.raises(ContractError, match="fixed calibration evidence"):
        build_fixed_camera_bundle(
            bundle_id="failed",
            created_at="2026-08-25T00:00:00Z",
            factory=factory,
            stream_validation=_stream_validation(),
            target_detection=detection,
            fixed_calibration=failed_fixed,
            provenance={},
        )

    disconnected = _factory(connected=False)
    with pytest.raises(TransformError, match="no transform path"):
        build_fixed_camera_bundle(
            bundle_id="disconnected",
            created_at="2026-08-25T00:00:00Z",
            factory=disconnected,
            stream_validation=_stream_validation(),
            target_detection=detection,
            fixed_calibration=fixed,
            provenance={},
        )


def test_core_only_reload_import_blocks_optional_runtime_packages(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    write_fixed_camera_bundle(path, _bundle())
    script = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyrealsense2', 'cv2', 'PIL'}:
            raise RuntimeError(f'forbidden optional import: {fullname}')
        return None

sys.meta_path.insert(0, Blocker())
from camera_rig.provision.bundle import load_and_validate_fixed_camera_bundle
bundle = load_and_validate_fixed_camera_bundle(sys.argv[1])
fixed = bundle.fixed_mount_calibration
assert fixed is not None
point = fixed.T_parent_from_camera_reference.transform_points([0.0, 0.0, 0.0])
print(bundle.schema_version, bundle.status, round(float((point ** 2).sum() ** 0.5), 3))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("camera-rig.bundle.v1 passed ")
