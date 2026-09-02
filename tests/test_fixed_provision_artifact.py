from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from camera_rig.api import load_provisioned_camera_bundle
from camera_rig.artifacts.factory_calibration import (
    FactoryCalibrationArtifact,
    write_factory_calibration,
)
from camera_rig.artifacts.hashing import sha256_bytes, sha256_file
from camera_rig.artifacts.io import atomic_write_json, deterministic_json_bytes, load_json
from camera_rig.artifacts.stream_validation import write_stream_validation
from camera_rig.artifacts.target_detection import (
    TargetDetectionArtifact,
    TargetDetectionFrame,
    write_target_detection,
)
from camera_rig.calibration.fixed.artifact import (
    FixedCalibrationArtifact,
    write_fixed_calibration,
)
from camera_rig.capture.snapshot import write_snapshot
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.timestamps import SingleDeviceSyncReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.provision import artifact as artifact_module
from camera_rig.provision.artifact import (
    ARTIFACT_PATHS,
    DIAGNOSTIC_OVERLAY_ROOTS,
    FixedProvisionArtifactInputs,
    ProvisionOverlayInputs,
    write_fixed_provision_artifact,
)
from camera_rig.provision.bundle import build_fixed_camera_bundle, write_fixed_camera_bundle
from camera_rig.provision.validation import load_and_validate_fixed_provision
from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget
from camera_rig.targets.observation import TargetObservation

CREATED_AT = "2026-08-25T00:00:00Z"
ACQUISITION_ID = "33333333-3333-4333-8333-333333333333"
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _factory() -> FactoryCalibrationArtifact:
    names = ("color", "depth", "ir_left", "ir_right")
    profiles = {
        name: StreamProfile(
            name,
            3,
            2,
            30,
            {"color": "rgb8", "depth": "z16"}.get(name, "y8"),
            {"color": 0, "depth": 0, "ir_left": 1, "ir_right": 2}[name],
        )
        for name in names
    }
    intrinsics = {
        name: CameraIntrinsics(f"head/{name}_optical", 3, 2, 2.0, 2.0, 1.0, 0.5, "none")
        for name in names
    }
    transforms = []
    for index, target in enumerate(("color", "depth", "ir_right")):
        matrix = np.eye(4)
        matrix[0, 3] = index * 0.01
        transforms.append(RigidTransform("head/ir_left_optical", f"head/{target}_optical", matrix))
    calibration = FactoryCalibration(
        CameraDeviceInfo("synthetic", "head", "synthetic", "synthetic", "test-device"),
        profiles,
        intrinsics,
        tuple(transforms),
        0.001,
    )
    return FactoryCalibrationArtifact(
        CREATED_AT,
        calibration,
        QualityReport(True),
        {"source": "unit-test", "acquisition_id": ACQUISITION_ID},
    )


def _frames(count: int = 60) -> list[CameraFrame]:
    result = []
    for index in range(count):
        streams = {
            "color": StreamFrame(
                "color", np.arange(18, dtype=np.uint8).reshape(2, 3, 3), index, index, "test"
            ),
            "depth": StreamFrame(
                "depth", np.arange(6, dtype=np.uint16).reshape(2, 3), index, index, "test"
            ),
            "ir_left": StreamFrame(
                "ir_left", np.arange(6, dtype=np.uint8).reshape(2, 3), index, index, "test"
            ),
            "ir_right": StreamFrame(
                "ir_right",
                np.fliplr(np.arange(6, dtype=np.uint8).reshape(2, 3)),
                index,
                index,
                "test",
            ),
        }
        result.append(
            CameraFrame(
                "head",
                "test-device",
                streams,
                index,
                SingleDeviceSyncReport(
                    True,
                    ("color", "depth", "ir_left", "ir_right"),
                    0,
                    {name: 0 for name in streams},
                    True,
                ),
            )
        )
    return result


def _target(path: Path) -> ResolvedCharucoTarget:
    target = ResolvedCharucoTarget(
        target_name="charuco_test",
        target_frame="charuco_target",
        dictionary="DICT_5X5_100",
        squares_x=3,
        squares_y=3,
        square_length_m=0.03,
        marker_length_m=0.022,
        border_bits=1,
        legacy_pattern=False,
        board_width_m=0.09,
        board_height_m=0.09,
        corner_points=(
            (0, (0.03, 0.06, 0.0)),
            (1, (0.06, 0.06, 0.0)),
            (2, (0.03, 0.03, 0.0)),
            (3, (0.06, 0.03, 0.0)),
        ),
        marker_ids=(0, 1, 2, 3),
        camera_rig_version="0.3.0",
        opencv_version="test",
        source_config_sha256="1" * 64,
        board_png_sha256="2" * 64,
        print_pdf_sha256="3" * 64,
    )
    atomic_write_json(path, target.to_dict())
    return target.with_artifact_sha256(sha256_file(path))


def _detection(target: ResolvedCharucoTarget, capture_hash: str) -> TargetDetectionArtifact:
    target_hash = target.artifact_sha256
    observation = TargetObservation(
        plugin_name="charuco",
        target_frame=target.target_frame,
        point_ids=(0, 1, 2, 3),
        image_points_px=np.asarray(((0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5))),
        object_points_m=target.object_points_for((0, 1, 2, 3)),
        image_size=(3, 2),
        quality=QualityReport(True),
        metadata={"target_spec_sha256": target_hash},
    )
    overlay_indices = {"best": 0, "median_quality": 30, "worst_accepted": 59}
    overlays_by_index = {
        index: f"overlays/{label}_frame_{index:06d}.png" for label, index in overlay_indices.items()
    }
    frames = tuple(
        TargetDetectionFrame(index, True, observation, overlays_by_index.get(index))
        for index in range(60)
    )
    statistics = {"minimum": 4.0, "median": 4.0, "maximum": 4.0, "mean": 4.0}
    checks = {
        "frame_count_is_60": True,
        "success_ratio_at_least_0_95": True,
        "median_corners_at_least_20": True,
        "median_corner_fraction_at_least_0_80": True,
        "median_coverage_at_least_0_05": True,
        "median_jitter_at_most_0_5_px": True,
        "p95_jitter_at_most_1_0_px": True,
    }
    return TargetDetectionArtifact(
        target_spec_sha256=target_hash,
        capture_manifest_sha256=capture_hash,
        stream="color",
        frame_count=60,
        per_frame=frames,
        aggregate={
            "success_ratio": 1.0,
            "detected_marker_count": statistics,
            "detected_charuco_corner_count": statistics,
            "corner_fraction": {
                "minimum": 1.0,
                "median": 1.0,
                "maximum": 1.0,
                "mean": 1.0,
            },
            "coverage_ratio": {
                "minimum": 0.5,
                "median": 0.5,
                "maximum": 0.5,
                "mean": 0.5,
            },
            "temporal_jitter": {
                "minimum_occurrences": 48,
                "eligible_corner_count": 0,
                "median_radial_std_px": 0.0,
                "p95_radial_std_px": 0.0,
                "per_corner": [],
            },
        },
        acceptance={
            "passed": True,
            "thresholds": {
                "frame_count": 60,
                "success_ratio": 0.95,
                "median_charuco_corners": 20.0,
                "median_corner_fraction": 0.8,
                "median_coverage_ratio": 0.05,
                "median_jitter_px": 0.5,
                "p95_jitter_px": 1.0,
            },
            "checks": checks,
        },
        selected_overlays=overlay_indices,
        software={"camera_rig_version": "0.3.0", "opencv_version": "test"},
    )


def _stream_report() -> dict[str, object]:
    streams = ("color", "depth", "ir_left", "ir_right")
    return {
        "schema_version": "camera-rig.stream-validation.v1",
        "status": "PASS",
        "requested_frames": 300,
        "received_frames": 300,
        "duration_s": 10.0,
        "per_stream_observed_fps": {name: 30.0 for name in streams},
        "per_stream_frame_number_discontinuities": {name: 0 for name in streams},
        "per_stream_discontinuity_ratio": {name: 0.0 for name in streams},
        "per_stream_timestamp_monotonicity": {name: True for name in streams},
        "per_stream_timestamp_domain_counts": {name: {"test": 300} for name in streams},
        "ir_stereo_frame_match_ratio": 1.0,
        "comparable_timestamp_skew_ns": {"p50": 0.0, "p95": 0.0, "max": 0},
        "sync_valid_ratio": 1.0,
        "timeouts": 0,
        "missing_streams": {},
        "shape_consistency": {
            "color": [[2, 3, 3]],
            "depth": [[2, 3]],
            "ir_left": [[2, 3]],
            "ir_right": [[2, 3]],
        },
        "dtype_consistency": {
            "color": ["uint8"],
            "depth": ["uint16"],
            "ir_left": ["uint8"],
            "ir_right": ["uint8"],
        },
        "depth_valid_ratio": 1.0,
        "rgb_variance": 1.0,
        "rgb_channel_variance": [1.0, 1.0, 1.0],
        "ir_variance": {"ir_left": 1.0, "ir_right": 1.0},
        "ir_distinct_ratio": 1.0,
        "failure_reasons": [],
    }


def _inputs(root: Path) -> FixedProvisionArtifactInputs:
    root.mkdir()
    factory_artifact = _factory()
    factory_path = root / "factory.json"
    write_factory_calibration(factory_path, factory_artifact)
    factory_sha256 = sha256_file(factory_path)
    selected_indices = [(index * 299) // 59 for index in range(60)]
    selected_indices_sha256 = _digest(selected_indices)
    stream_path = root / "stream.json"
    stream_artifact = write_stream_validation(
        stream_path,
        _stream_report(),
        provenance={
            "source": "unit-test",
            "acquisition_id": ACQUISITION_ID,
            "active_profiles_sha256": _digest(
                {
                    name: profile.to_dict()
                    for name, profile in sorted(
                        factory_artifact.calibration.stream_profiles.items()
                    )
                }
            ),
            "device_identity_sha256": _digest(factory_artifact.calibration.device.to_dict()),
            "factory_calibration_sha256": factory_sha256,
            "selected_source_indices_sha256": selected_indices_sha256,
        },
    )
    stream_sha256 = sha256_file(stream_path)
    capture_root = root / "capture"
    write_snapshot(
        capture_root,
        _frames(),
        factory_artifact,
        {
            "copy_frames": True,
            "selected_source_indices": selected_indices,
            "source_frame_count": 300,
        },
        {
            "source": "unit-test",
            "acquisition_id": ACQUISITION_ID,
            "factory_calibration_sha256": factory_sha256,
            "selected_source_indices_sha256": selected_indices_sha256,
            "stream_validation_sha256": stream_sha256,
        },
        include_previews=False,
    )
    capture_hash = sha256_file(capture_root / "manifest.json")
    target_path = root / "target_spec.json"
    target = _target(target_path)
    detection_path = root / "detection.json"
    detection_artifact = _detection(target, capture_hash)
    write_target_detection(detection_path, detection_artifact)

    workspace_from_target = RigidTransform("charuco_target", "workspace", np.eye(4))
    detection_from_target_matrix = np.eye(4)
    detection_from_target_matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    detection_from_target_matrix[2, 3] = 1.0
    detection_from_target = RigidTransform(
        "charuco_target", "head/color_optical", detection_from_target_matrix
    )
    workspace_from_detection = workspace_from_target.compose(detection_from_target.inverse())
    detection_from_reference = factory_artifact.calibration.internal_transforms[0]
    workspace_from_reference = workspace_from_detection.compose(detection_from_reference)
    quality = QualityReport(True, metrics={"accepted_frames": 60})
    fixed_mount = FixedMountCalibration(
        "workspace", "head/ir_left_optical", workspace_from_reference, quality
    )
    fixed_artifact = FixedCalibrationArtifact(
        created_at=CREATED_AT,
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
            "target_spec_sha256": target.artifact_sha256,
            "print_provenance": {
                "horizontal_print_scale": 0.997,
                "vertical_print_scale": 0.997,
                "maximum_observed_print_scale_error": 0.003,
                "geometry_policy": "nominal persisted geometry",
            },
        },
        inputs={
            "capture_manifest_sha256": capture_hash,
            "factory_calibration_sha256": factory_sha256,
            "target_detection_sha256": sha256_file(detection_path),
        },
        solver={"method": "ippe", "refinement": "lm", "thresholds": {}},
        per_frame_pose_summary=tuple(
            {
                "frame_index": index,
                "corner_count": 4,
                "accepted": True,
                "failure_reasons": [],
            }
            for index in range(60)
        ),
        aggregate={
            "accepted_frames": 60,
            "accepted_ratio": 1.0,
            "reprojection": {},
            "pose_repeatability": {},
            "split_half": {},
            "native_depth_sanity": {"status": "PASS"},
        },
        T_detection_from_target=detection_from_target,
        T_workspace_from_detection=workspace_from_detection,
        T_detection_from_reference=detection_from_reference,
        T_workspace_from_reference=workspace_from_reference,
        fixed_mount_calibration=fixed_mount,
        quality=quality,
        provenance={
            "source": "unit-test",
            "overlay_files": [
                "diagnostics/overlays/fixed_calibration/best_frame_000000.png",
                "diagnostics/overlays/fixed_calibration/median_quality_frame_000030.png",
                "diagnostics/overlays/fixed_calibration/worst_accepted_frame_000059.png",
            ],
        },
    )
    fixed_path = root / "fixed.json"
    write_fixed_calibration(fixed_path, fixed_artifact)

    bundle = build_fixed_camera_bundle(
        bundle_id="unit-test-bundle",
        created_at=CREATED_AT,
        factory=factory_artifact,
        stream_validation=stream_artifact,
        target_detection=detection_artifact,
        fixed_calibration=fixed_artifact,
        provenance={"source": "unit-test"},
    )
    bundle_path = root / "bundle.json"
    write_fixed_camera_bundle(bundle_path, bundle)
    target_detection_overlays = _overlay_inputs(root / "target_detection_overlays")
    fixed_calibration_overlays = _overlay_inputs(root / "fixed_calibration_overlays")
    return FixedProvisionArtifactInputs(
        camera_bundle=bundle_path,
        factory_calibration=factory_path,
        capture_artifact=capture_root,
        target_spec=target_path,
        target_detection=detection_path,
        fixed_calibration=fixed_path,
        stream_validation=stream_path,
        target_detection_overlays=target_detection_overlays,
        fixed_calibration_overlays=fixed_calibration_overlays,
    )


def _overlay_inputs(root: Path) -> ProvisionOverlayInputs:
    root.mkdir()
    paths: dict[str, Path] = {}
    indices = {"best": 0, "median_quality": 30, "worst_accepted": 59}
    for label, index in indices.items():
        path = root / f"{label}_frame_{index:06d}.png"
        path.write_bytes(PNG_BYTES)
        paths[label] = path
    return ProvisionOverlayInputs(
        best=paths["best"],
        median_quality=paths["median_quality"],
        worst_accepted=paths["worst_accepted"],
    )


def _digest(value: object) -> str:
    return sha256_bytes(deterministic_json_bytes(value))


def _rebind_changed_stream(root: Path) -> None:
    stream_path = root / ARTIFACT_PATHS["stream_validation"]
    stream_sha256 = sha256_file(stream_path)
    bundle_path = root / ARTIFACT_PATHS["camera_bundle"]
    bundle = load_json(bundle_path)
    assert isinstance(bundle, dict)
    bundle_provenance = bundle["provenance"]
    assert isinstance(bundle_provenance, dict)
    evidence = bundle_provenance["evidence_sha256"]
    assert isinstance(evidence, dict)
    evidence["stream_validation"] = stream_sha256
    atomic_write_json(bundle_path, bundle)

    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    assert isinstance(manifest, dict)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    stream_reference = artifacts["stream_validation"]
    bundle_reference = artifacts["camera_bundle"]
    assert isinstance(stream_reference, dict) and isinstance(bundle_reference, dict)
    stream_reference["sha256"] = stream_sha256
    bundle_reference["sha256"] = sha256_file(bundle_path)
    atomic_write_json(manifest_path, manifest)
    artifact_module._write_checksums(root)


@pytest.fixture
def valid_inputs(tmp_path: Path) -> FixedProvisionArtifactInputs:
    return _inputs(tmp_path / "inputs")


def test_writer_publishes_exact_validated_layout(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    manifest = write_fixed_provision_artifact(
        output,
        valid_inputs,
        provenance={"config_sha256": "a" * 64},
        artifact_id="11111111-1111-4111-8111-111111111111",
        created_at=CREATED_AT,
    )
    assert manifest == load_and_validate_fixed_provision(output)
    public_bundle = load_provisioned_camera_bundle(output)
    assert public_bundle.status == "passed"
    assert public_bundle.fixed_mount_calibration is not None
    assert manifest.status == "passed" and manifest.quality.passed
    assert all((output / path).is_file() for path in ARTIFACT_PATHS.values())
    assert set(manifest.diagnostics) == set(DIAGNOSTIC_OVERLAY_ROOTS)
    assert all(
        (output / reference.path).is_file()
        for overlays in manifest.diagnostics.values()
        for reference in overlays.values()
    )
    assert manifest.diagnostics["target_detection"]["best"].path.endswith("/best_frame_000000.png")
    checksum_paths = {
        line.split("  ", maxsplit=1)[1]
        for line in (output / "checksums.sha256").read_text().splitlines()
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).as_posix() != "checksums.sha256"
    }
    assert checksum_paths == actual


def test_existing_output_is_replaced_by_default(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    first = write_fixed_provision_artifact(output, valid_inputs, provenance={"run": "first"})
    replacement = write_fixed_provision_artifact(output, valid_inputs, provenance={"run": "second"})
    assert replacement.artifact_id != first.artifact_id
    assert load_and_validate_fixed_provision(output).provenance == {"run": "second"}


def test_default_replacement_requires_a_fully_valid_new_artifact(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    write_fixed_provision_artifact(
        output,
        valid_inputs,
        provenance={"run": "first"},
        artifact_id="11111111-1111-4111-8111-111111111111",
    )
    replacement = write_fixed_provision_artifact(
        output,
        valid_inputs,
        provenance={"run": "second"},
        artifact_id="22222222-2222-4222-8222-222222222222",
    )
    assert replacement.artifact_id == "22222222-2222-4222-8222-222222222222"
    assert load_and_validate_fixed_provision(output).provenance == {"run": "second"}
    assert not list(tmp_path.glob(".fixed_camera.backup-*"))


def test_invalid_replacement_never_disturbs_existing_output(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    original = write_fixed_provision_artifact(output, valid_inputs, provenance={"run": "old"})
    invalid_fixed = tmp_path / "invalid_fixed.json"
    data = load_json(valid_inputs.fixed_calibration)
    assert isinstance(data, dict)
    inputs = data["inputs"]
    assert isinstance(inputs, dict)
    inputs["factory_calibration_sha256"] = "0" * 64
    atomic_write_json(invalid_fixed, data)
    invalid = replace(valid_inputs, fixed_calibration=invalid_fixed)
    with pytest.raises(ArtifactError, match=r"fixed_calibration|factory_calibration_sha256"):
        write_fixed_provision_artifact(output, invalid, provenance={"run": "invalid"})
    restored = load_and_validate_fixed_provision(output)
    assert restored.artifact_id == original.artifact_id
    assert restored.provenance == {"run": "old"}


def test_force_commit_preserves_both_directories_if_atomic_exchange_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "old").write_text("old")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "new").write_text("new")
    monkeypatch.setattr(
        "camera_rig.provision.validation.load_and_validate_fixed_provision",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        artifact_module,
        "_exchange_directories",
        lambda _source, _target: (_ for _ in ()).throw(OSError("injected publish failure")),
    )
    with pytest.raises(OSError, match="injected publish failure"):
        artifact_module._commit_directory(temporary, target, force=True)
    assert (target / "old").read_text() == "old"
    assert (temporary / "new").read_text() == "new"


def test_force_commit_remains_successful_if_backup_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "old").write_text("old")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "new").write_text("new")

    def injected_cleanup(_path: Path) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(
        "camera_rig.provision.validation.load_and_validate_fixed_provision",
        lambda _path: object(),
    )
    monkeypatch.setattr(artifact_module.shutil, "rmtree", injected_cleanup)
    artifact_module._commit_directory(temporary, target, force=True)
    assert (target / "new").read_text() == "new"
    assert not (target / "old").exists()
    assert (temporary / "old").read_text() == "old"


def test_checksum_corruption_and_unexpected_files_fail_closed(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    with (output / "camera_bundle.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_and_validate_fixed_provision(output)

    output = tmp_path / "fixed_camera_extra"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    (output / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ArtifactError, match=r"checksums\.sha256"):
        load_and_validate_fixed_provision(output)


def test_outer_validation_rejects_checksum_consistent_invalid_fixed_bundle(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    bundle_path = output / "camera_bundle.json"
    bundle = load_json(bundle_path)
    assert isinstance(bundle, dict)
    provenance = bundle["provenance"]
    assert isinstance(provenance, dict)
    provenance.pop("evidence_sha256")
    atomic_write_json(bundle_path, bundle)

    manifest_path = output / "manifest.json"
    manifest = load_json(manifest_path)
    assert isinstance(manifest, dict)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    bundle_reference = artifacts["camera_bundle"]
    assert isinstance(bundle_reference, dict)
    bundle_reference["sha256"] = sha256_file(bundle_path)
    atomic_write_json(manifest_path, manifest)
    artifact_module._write_checksums(output)

    with pytest.raises(ArtifactError, match="complete evidence SHA mapping"):
        load_and_validate_fixed_provision(output)


@pytest.mark.parametrize("frame_count", [1, 60, 299])
def test_outer_validation_requires_exactly_300_stream_frames(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs, frame_count: int
) -> None:
    output = tmp_path / "fixed_camera"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    stream_path = output / ARTIFACT_PATHS["stream_validation"]
    original = load_json(stream_path)
    assert isinstance(original, dict)
    provenance = original["provenance"]
    assert isinstance(provenance, dict)
    report = _stream_report()
    report["requested_frames"] = frame_count
    report["received_frames"] = frame_count
    write_stream_validation(stream_path, report, provenance=provenance)
    _rebind_changed_stream(output)

    with pytest.raises(ArtifactError, match="exactly 300"):
        load_and_validate_fixed_provision(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("acquisition_id", "44444444-4444-4444-8444-444444444444", "identities differ"),
        ("device_identity_sha256", "0" * 64, "device_identity_sha256"),
    ],
)
def test_outer_validation_rejects_wrong_stream_acquisition_binding(
    tmp_path: Path,
    valid_inputs: FixedProvisionArtifactInputs,
    field: str,
    value: str,
    message: str,
) -> None:
    output = tmp_path / "fixed_camera"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    stream_path = output / ARTIFACT_PATHS["stream_validation"]
    stream = load_json(stream_path)
    assert isinstance(stream, dict)
    provenance = stream["provenance"]
    assert isinstance(provenance, dict)
    provenance[field] = value
    atomic_write_json(stream_path, stream)
    _rebind_changed_stream(output)

    with pytest.raises(ArtifactError, match=message):
        load_and_validate_fixed_provision(output)


def test_missing_required_file_fails_closed(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    (output / "reports/stream_validation.json").unlink()
    with pytest.raises(ArtifactError, match=r"missing artifact file|checksum"):
        load_and_validate_fixed_provision(output)


def test_manifest_traversal_and_symlink_escape_are_rejected(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["camera_bundle"]["path"] = "../outside.json"
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(ArtifactError, match=r"camera_bundle|path"):
        load_and_validate_fixed_provision(output)

    output = tmp_path / "fixed_camera_symlink"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    external = tmp_path / "outside.json"
    external.write_text("{}")
    bundle = output / "camera_bundle.json"
    bundle.unlink()
    bundle.symlink_to(external)
    with pytest.raises(ArtifactError, match="symlink"):
        load_and_validate_fixed_provision(output)


def test_output_symlink_is_never_replaced(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    output = tmp_path / "fixed_camera"
    output.symlink_to(real, target_is_directory=True)
    with pytest.raises(ArtifactError, match="real directory"):
        write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    assert output.is_symlink()


def test_default_replacement_never_replaces_an_unowned_real_directory(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "unowned"
    output.mkdir()
    user_file = output / "user-file.txt"
    user_file.write_text("preserve")
    with pytest.raises(ArtifactError, match=r"manifest|fixed provision"):
        write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    assert user_file.read_text() == "preserve"


def test_failed_stage_never_publishes_a_passed_bundle(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    failed_stream = tmp_path / "failed_stream.json"
    report = _stream_report()
    report["status"] = "FAIL"
    report["received_frames"] = 299
    report["timeouts"] = 1
    report["failure_reasons"] = ["frame timeout"]
    write_stream_validation(failed_stream, report, provenance={"source": "failed-test"})
    invalid = replace(valid_inputs, stream_validation=failed_stream)
    output = tmp_path / "fixed_camera"
    with pytest.raises(ArtifactError, match="stream validation quality"):
        write_fixed_provision_artifact(output, invalid, provenance={"source": "failed-test"})
    assert not output.exists()
    assert not list(tmp_path.glob(".fixed_camera.tmp-*"))


def test_overlay_sources_are_explicit_real_png_files(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    missing = tmp_path / "best_missing.png"
    invalid = replace(
        valid_inputs,
        target_detection_overlays=replace(valid_inputs.target_detection_overlays, best=missing),
    )
    with pytest.raises(ArtifactError, match="real file"):
        write_fixed_provision_artifact(
            tmp_path / "missing_overlay", invalid, provenance={"source": "test"}
        )

    external = tmp_path / "external.png"
    external.write_bytes(PNG_BYTES)
    symlink = tmp_path / "worst_accepted_symlink.png"
    symlink.symlink_to(external)
    invalid = replace(
        valid_inputs,
        fixed_calibration_overlays=replace(
            valid_inputs.fixed_calibration_overlays, worst_accepted=symlink
        ),
    )
    with pytest.raises(ArtifactError, match="real file"):
        write_fixed_provision_artifact(
            tmp_path / "symlink_overlay", invalid, provenance={"source": "test"}
        )


def test_invalid_overlay_force_failure_preserves_existing_artifact(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    original = write_fixed_provision_artifact(output, valid_inputs, provenance={"run": "old"})
    corrupt = tmp_path / "median_quality_corrupt.png"
    corrupt.write_bytes(b"not a png")
    invalid = replace(
        valid_inputs,
        fixed_calibration_overlays=replace(
            valid_inputs.fixed_calibration_overlays, median_quality=corrupt
        ),
    )
    with pytest.raises(ArtifactError, match="not a valid PNG"):
        write_fixed_provision_artifact(output, invalid, provenance={"run": "invalid"})
    restored = load_and_validate_fixed_provision(output)
    assert restored.artifact_id == original.artifact_id
    assert restored.provenance == {"run": "old"}
    assert not list(tmp_path.glob(".fixed_camera.backup-*"))


def test_manifest_diagnostic_path_and_unexpected_overlay_fail_closed(
    tmp_path: Path, valid_inputs: FixedProvisionArtifactInputs
) -> None:
    output = tmp_path / "fixed_camera"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["diagnostics"]["target_detection"]["best"]["path"] = "../best.png"
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(ArtifactError, match=r"diagnostics|path"):
        load_and_validate_fixed_provision(output)

    output = tmp_path / "fixed_camera_extra_overlay"
    write_fixed_provision_artifact(output, valid_inputs, provenance={"source": "test"})
    extra = output / "diagnostics/overlays/target_detection/extra.png"
    extra.write_bytes(PNG_BYTES)
    with pytest.raises(ArtifactError, match=r"checksums\.sha256"):
        load_and_validate_fixed_provision(output)
