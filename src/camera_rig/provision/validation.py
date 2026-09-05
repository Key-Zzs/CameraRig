"""Fail-closed validation for complete fixed-camera provision directories."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from camera_rig.artifacts.capture_validation import validate_capture_artifact
from camera_rig.artifacts.factory_calibration import load_and_validate_factory_calibration
from camera_rig.artifacts.hashing import sha256_bytes, sha256_file
from camera_rig.artifacts.io import JsonValue, deterministic_json_bytes, load_json
from camera_rig.artifacts.stream_validation import load_and_validate_stream_validation
from camera_rig.artifacts.target_detection import load_and_validate_target_detection
from camera_rig.calibration.fixed.artifact import load_and_validate_fixed_calibration
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import ArtifactError, ContractError, SchemaValidationError
from camera_rig.provision.artifact import (
    ARTIFACT_PATHS_V2,
    CAPTURE_ROOT,
    DIAGNOSTIC_OVERLAY_ROOTS,
    FIXED_PROVISION_ARTIFACT_V2_SCHEMA_VERSION,
    OVERLAY_LABELS,
    FixedProvisionManifest,
    passed_provision_quality,
)
from camera_rig.provision.bootstrap_qualification import load_bootstrap_qualification
from camera_rig.provision.bundle import load_and_validate_fixed_camera_bundle
from camera_rig.targets.io import load_target
from camera_rig.targets.metrology import (
    load_target_metrology,
    load_target_scale_acceptance_policy,
)


def validate_fixed_provision_data(value: JsonValue) -> FixedProvisionManifest:
    """Schema-validate and reconstruct a fixed-provision manifest."""
    if not isinstance(value, dict):
        raise ArtifactError("fixed provision manifest must be a JSON object")
    try:
        schema_name = (
            "fixed_provision_artifact.v2.schema.json"
            if value.get("schema_version") == FIXED_PROVISION_ARTIFACT_V2_SCHEMA_VERSION
            else "fixed_provision_artifact.v1.schema.json"
        )
        validate_against_named_schema(value, schema_name)
        return FixedProvisionManifest.from_dict(dict(value))
    except SchemaValidationError as error:
        raise ArtifactError(str(error)) from error
    except (KeyError, TypeError, ValueError, ContractError) as error:
        raise ArtifactError(f"fixed provision manifest is invalid: {error}") from error


def load_and_validate_fixed_provision(root: str | Path) -> FixedProvisionManifest:
    """Validate exact files, hashes, sub-artifacts, identities, transforms, and quality."""
    artifact_root = Path(root)
    actual_files = _actual_files_without_symlinks(artifact_root)
    manifest = validate_fixed_provision_data(load_json(artifact_root / "manifest.json"))
    references = manifest.artifacts
    artifact_paths = {name: reference.path for name, reference in references.items()}
    for name, reference in references.items():
        path = _resolved_file(artifact_root, reference.path)
        if sha256_file(path) != reference.sha256:
            raise ArtifactError(f"manifest artifact checksum mismatch: {name}")
    diagnostic_references = {
        f"{group}.{label}": reference
        for group, overlays in manifest.diagnostics.items()
        for label, reference in overlays.items()
    }
    if len({reference.path for reference in diagnostic_references.values()}) != 6:
        raise ArtifactError("fixed provision must contain six distinct diagnostic overlays")
    for name, reference in diagnostic_references.items():
        path = _resolved_file(artifact_root, reference.path)
        if sha256_file(path) != reference.sha256:
            raise ArtifactError(f"manifest diagnostic checksum mismatch: {name}")
        _validate_png(path, name)

    checksum_entries = _load_checksum_file(artifact_root / "checksums.sha256")
    expected_checksum_paths = actual_files - {"checksums.sha256"}
    if set(checksum_entries) != expected_checksum_paths:
        raise ArtifactError("checksums.sha256 contains missing or unexpected paths")
    for relative, expected_hash in checksum_entries.items():
        if sha256_file(_resolved_file(artifact_root, relative)) != expected_hash:
            raise ArtifactError(f"checksum mismatch: {relative}")
    for reference in (*references.values(), *diagnostic_references.values()):
        if checksum_entries.get(reference.path) != reference.sha256:
            raise ArtifactError("manifest and checksums.sha256 digests differ")

    capture_prefix = f"{CAPTURE_ROOT}/"
    allowed_files = {
        "manifest.json",
        "checksums.sha256",
        *artifact_paths.values(),
        *(reference.path for reference in diagnostic_references.values()),
        *(path for path in actual_files if path.startswith(capture_prefix)),
    }
    if actual_files != allowed_files:
        raise ArtifactError("fixed provision artifact contains missing or unexpected files")

    bundle = load_and_validate_fixed_camera_bundle(
        _resolved_file(artifact_root, artifact_paths["camera_bundle"])
    )
    factory = load_and_validate_factory_calibration(
        _resolved_file(artifact_root, artifact_paths["factory_calibration"])
    )
    capture_root = artifact_root / CAPTURE_ROOT
    capture = validate_capture_artifact(capture_root)
    target = load_target(_resolved_file(artifact_root, artifact_paths["target_spec"]))
    detection = load_and_validate_target_detection(
        _resolved_file(artifact_root, artifact_paths["target_detection"])
    )
    fixed = load_and_validate_fixed_calibration(
        _resolved_file(artifact_root, artifact_paths["fixed_calibration"])
    )
    stream = load_and_validate_stream_validation(
        _resolved_file(artifact_root, artifact_paths["stream_validation"])
    )

    calibration = factory.calibration
    if not factory.quality.passed:
        raise ArtifactError("factory calibration quality is not passed")
    capture_quality = _object(capture["quality"], "capture.quality")
    if capture_quality.get("passed") is not True:
        raise ArtifactError("capture quality is not passed")
    if stream.status != "PASS" or not stream.quality.passed:
        raise ArtifactError("stream validation quality is not passed")
    if stream.requested_frames != 300 or stream.received_frames != 300:
        raise ArtifactError("fixed provisioning requires exactly 300 validated stream frames")
    if not detection.is_capture or detection.acceptance is None:
        raise ArtifactError("target detection must be a capture acceptance artifact")
    if detection.acceptance.get("passed") is not True:
        raise ArtifactError("target detection acceptance is not passed")
    if detection.selected_overlays is None or set(detection.selected_overlays) != set(
        OVERLAY_LABELS
    ):
        raise ArtifactError("target detection must select exactly three diagnostic overlays")
    detection_overlay_refs = manifest.diagnostics["target_detection"]
    for label, index in detection.selected_overlays.items():
        source_overlay = detection.per_frame[index].overlay
        if source_overlay is None:
            raise ArtifactError(f"target detection overlay is missing for {label}")
        expected = (
            f"{DIAGNOSTIC_OVERLAY_ROOTS['target_detection']}/{PurePosixPath(source_overlay).name}"
        )
        if detection_overlay_refs[label].path != expected:
            raise ArtifactError(f"target detection overlay identity differs: {label}")
    if detection.frame_count != 60 or capture.get("frame_count") != 60:
        raise ArtifactError("capture and target detection must contain exactly 60 frames")
    if bundle.status != "passed" or not bundle.quality.passed:
        raise ArtifactError("camera bundle status and quality must be passed")
    if bundle.fixed_mount_calibration is None or not bundle.fixed_mount_calibration.quality.passed:
        raise ArtifactError("camera bundle requires passed fixed mount calibration")

    if bundle.device.to_dict() != calibration.device.to_dict():
        raise ArtifactError("bundle and factory device identities differ")
    bundle_evidence = _object(bundle.provenance.get("evidence_sha256"), "bundle.evidence_sha256")
    for name in (
        "factory_calibration",
        "stream_validation",
        "target_detection",
        "fixed_calibration",
    ):
        if bundle_evidence.get(name) != references[name].sha256:
            raise ArtifactError(f"bundle evidence SHA differs from provision artifact: {name}")
    if (
        capture.get("camera") != calibration.device.camera_name
        or capture.get("serial") != calibration.device.serial
    ):
        raise ArtifactError("capture and factory device identities differ")
    _validate_acquisition_bindings(
        capture=capture,
        factory=factory,
        stream=stream,
        factory_sha256=references["factory_calibration"].sha256,
        stream_sha256=references["stream_validation"].sha256,
    )
    if not _factory_payload_matches_bundle(bundle.to_dict(), calibration.to_dict()):
        raise ArtifactError("bundle and factory internal calibration differ")
    if bundle.fixed_mount_calibration.to_dict() != fixed.fixed_mount_calibration.to_dict():
        raise ArtifactError("bundle and fixed-calibration mount records differ")

    target_hash = sha256_file(_resolved_file(artifact_root, artifact_paths["target_spec"]))
    capture_hash = sha256_file(capture_root / "manifest.json")
    factory_hash = references["factory_calibration"].sha256
    detection_hash = references["target_detection"].sha256
    if target.artifact_sha256 != target_hash:
        raise ArtifactError("resolved target hash reconstruction differs")
    if detection.target_spec_sha256 != target_hash:
        raise ArtifactError("target detection target identity differs")
    if detection.capture_manifest_sha256 != capture_hash:
        raise ArtifactError("target detection capture identity differs")
    fixed_target = _object(fixed.target, "fixed.target")
    fixed_inputs = _object(fixed.inputs, "fixed.inputs")
    expected_fixed_hashes = {
        "target_spec_sha256": target_hash,
        "capture_manifest_sha256": capture_hash,
        "factory_calibration_sha256": factory_hash,
        "target_detection_sha256": detection_hash,
    }
    for name, expected in expected_fixed_hashes.items():
        container = fixed_target if name == "target_spec_sha256" else fixed_inputs
        if container.get(name) != expected:
            raise ArtifactError(f"fixed calibration input identity differs: {name}")
    capture_factory = _string(capture["factory_calibration"], "capture.factory_calibration")
    if sha256_file(_resolved_file(capture_root, capture_factory)) != factory_hash:
        raise ArtifactError("capture and outer factory artifacts differ")

    fixed_camera = _object(fixed.camera, "fixed.camera")
    for stream_name, frame_name in (
        (_string(fixed_camera["detection_stream"], "detection_stream"), "detection_frame"),
        (_string(fixed_camera["reference_stream"], "reference_stream"), "reference_frame"),
    ):
        intrinsic = calibration.intrinsics.get(stream_name)
        if intrinsic is None or intrinsic.frame != fixed_camera.get(frame_name):
            raise ArtifactError("fixed calibration stream frame differs from factory intrinsics")
    if detection.stream != fixed_camera.get("detection_stream"):
        raise ArtifactError("target detection stream differs from fixed calibration")
    fixed_overlay_files = fixed.provenance.get("overlay_files")
    expected_fixed_overlay_files = {
        reference.path for reference in manifest.diagnostics["fixed_calibration"].values()
    }
    if (
        not isinstance(fixed_overlay_files, list)
        or not all(isinstance(value, str) for value in fixed_overlay_files)
        or len(fixed_overlay_files) != 3
        or set(fixed_overlay_files) != expected_fixed_overlay_files
    ):
        raise ArtifactError("fixed calibration overlay provenance differs from diagnostics")

    if manifest.quality.to_dict() != passed_provision_quality().to_dict():
        raise ArtifactError("outer provision quality is not fully passed")
    if manifest.schema_version == FIXED_PROVISION_ARTIFACT_V2_SCHEMA_VERSION:
        _validate_bootstrap_v2(
            artifact_root=artifact_root,
            references=references,
            bundle=bundle,
            factory=factory,
            stream=stream,
            detection=detection,
            fixed=fixed,
            target=target,
            target_hash=target_hash,
            factory_hash=factory_hash,
            capture_hash=capture_hash,
        )
    return manifest


def _validate_bootstrap_v2(
    *,
    artifact_root: Path,
    references: Mapping[str, object],
    bundle: object,
    factory: object,
    stream: object,
    detection: object,
    fixed: object,
    target: object,
    target_hash: str,
    factory_hash: str,
    capture_hash: str,
) -> None:
    from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
    from camera_rig.artifacts.models import CameraBundle
    from camera_rig.artifacts.stream_validation import StreamValidationArtifact
    from camera_rig.artifacts.target_detection import TargetDetectionArtifact
    from camera_rig.calibration.fixed.artifact import FixedCalibrationArtifact
    from camera_rig.calibration.fixed.depth_sanity import load_metric_depth_receipt
    from camera_rig.provision.artifact import ArtifactReference
    from camera_rig.provision.bundle import fixed_camera_bundle_fingerprint
    from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget

    if set(references) != set(ARTIFACT_PATHS_V2):
        raise ArtifactError("bootstrap provision v2 artifact set differs")
    typed_references = {
        name: reference
        for name, reference in references.items()
        if isinstance(reference, ArtifactReference)
    }
    if (
        len(typed_references) != len(references)
        or not isinstance(bundle, CameraBundle)
        or not isinstance(factory, FactoryCalibrationArtifact)
        or not isinstance(stream, StreamValidationArtifact)
        or not isinstance(detection, TargetDetectionArtifact)
        or not isinstance(fixed, FixedCalibrationArtifact)
        or not isinstance(target, ResolvedCharucoTarget)
    ):
        raise ArtifactError("bootstrap provision v2 inputs are invalid")
    policy_reference = typed_references["target_scale_acceptance_policy"]
    policy = load_target_scale_acceptance_policy(
        _resolved_file(artifact_root, policy_reference.path),
        expected_target_sha256=target_hash,
    )
    metrology = load_target_metrology(
        _resolved_file(artifact_root, typed_references["target_metrology"].path),
        expected_target=target,
        expected_acceptance_policy=policy,
        expected_acceptance_policy_sha256=policy_reference.sha256,
        require_pass=True,
    )
    qualification = load_bootstrap_qualification(
        _resolved_file(artifact_root, typed_references["bootstrap_qualification"].path),
        require_pass=True,
    )
    metric_value = load_metric_depth_receipt(
        _resolved_file(artifact_root, typed_references["metric_depth_receipt"].path),
        require_pass=True,
    )
    expected_camera_identity = _digest(factory.calibration.device.to_dict())
    expected_bundle_fingerprint = fixed_camera_bundle_fingerprint(
        factory=factory,
        stream_validation=stream,
        target_detection=detection,
        fixed_calibration=fixed,
    )
    expected_metric_bindings = {
        "camera_identity_sha256": expected_camera_identity,
        "target_identity_sha256": target_hash,
        "factory_calibration_sha256": factory_hash,
        "capture_manifest_sha256": capture_hash,
    }
    for name, expected in expected_metric_bindings.items():
        if metric_value.get(name) != expected:
            raise ArtifactError(f"metric-depth receipt binding differs: {name}")
    if metric_value.get("evaluation") != fixed.aggregate.get("native_depth_sanity"):
        raise ArtifactError("metric-depth receipt differs from fixed-calibration evaluation")
    recomputed_metric = _recompute_native_depth_evaluation(
        artifact_root=artifact_root,
        factory=factory,
        target=target,
        fixed=fixed,
    )
    if metric_value.get("evaluation") != recomputed_metric:
        raise ArtifactError("metric-depth receipt differs from raw capture replay")
    if (
        qualification.get("target_metrology_sha256") != typed_references["target_metrology"].sha256
        or qualification.get("metric_depth_receipt_sha256")
        != typed_references["metric_depth_receipt"].sha256
    ):
        raise ArtifactError("bootstrap qualification receipt binding differs")
    expected_qualification_bindings = {
        "camera_identity_sha256": expected_camera_identity,
        "camera_bundle_fingerprint": expected_bundle_fingerprint,
        "target_identity_sha256": target_hash,
    }
    for name, expected in expected_qualification_bindings.items():
        if qualification.get(name) != expected:
            raise ArtifactError(f"bootstrap qualification binding differs: {name}")
    if qualification.get("metric_depth_integrity") != metric_value.get("evaluation"):
        raise ArtifactError("bootstrap qualification metric-depth payload differs")
    print_provenance = _object(fixed.target.get("print_provenance"), "fixed.print_provenance")
    if (
        print_provenance.get("target_metrology_status") != "PASS"
        or print_provenance.get("target_metrology_sha256")
        != typed_references["target_metrology"].sha256
    ):
        raise ArtifactError("fixed calibration print provenance differs from metrology")
    authority = bundle.provenance.get("calibration_authority")
    if not isinstance(authority, dict):
        raise ArtifactError("bootstrap CameraBundle lacks calibration authority")
    expected_authority = {
        "schema_version": "camera-rig.calibration-authority.v1",
        "qualification_scope": "bootstrap_only",
        "production_authoritative": False,
        "qualification_state": "BOOTSTRAP_QUALIFIED",
        "qualification_fingerprint": qualification["qualification_fingerprint"],
        "target_metrology_sha256": typed_references["target_metrology"].sha256,
        "metric_depth_receipt_sha256": typed_references["metric_depth_receipt"].sha256,
    }
    if authority != expected_authority or metrology.status != "PASS":
        raise ArtifactError("bootstrap CameraBundle authority binding differs")


def _recompute_native_depth_evaluation(
    *, artifact_root: Path, factory: object, target: object, fixed: object
) -> dict[str, object]:
    from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
    from camera_rig.calibration.fixed.artifact import FixedCalibrationArtifact
    from camera_rig.calibration.fixed.depth_sanity import evaluate_native_depth_sanity
    from camera_rig.capture.replay import ReplayCameraSession
    from camera_rig.provision.config import (
        BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
        BOOTSTRAP_METRIC_DEPTH_THRESHOLDS,
        FIXED_PROVISION_CALIBRATION_FRAMES,
    )
    from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget

    if (
        not isinstance(factory, FactoryCalibrationArtifact)
        or not isinstance(target, ResolvedCharucoTarget)
        or not isinstance(fixed, FixedCalibrationArtifact)
    ):
        raise ArtifactError("metric-depth replay inputs are invalid")
    summary_indices: list[int] = []
    frame_indices: list[int] = []
    for summary in fixed.per_frame_pose_summary:
        index = summary.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ArtifactError("metric-depth replay frame index is invalid")
        summary_indices.append(index)
        if summary.get("pose_inlier") is True:
            frame_indices.append(index)
    capture_root = artifact_root / CAPTURE_ROOT
    with ReplayCameraSession.from_artifact(capture_root) as replay:
        if replay.frame_count != FIXED_PROVISION_CALIBRATION_FRAMES:
            raise ArtifactError("metric-depth replay capture must contain exactly 60 frames")
        frames = tuple(replay.capture() for _ in range(replay.frame_count))
    expected_indices = list(range(FIXED_PROVISION_CALIBRATION_FRAMES))
    if summary_indices != expected_indices:
        raise ArtifactError("metric-depth replay pose summary must cover frames 0..59 exactly once")
    if not frame_indices or len(set(frame_indices)) != len(frame_indices):
        raise ArtifactError("metric-depth replay pose-inlier frames must be unique")
    thresholds = BOOTSTRAP_METRIC_DEPTH_THRESHOLDS
    return evaluate_native_depth_sanity(
        target=target,
        calibration=factory.calibration,
        T_detection_from_target=fixed.T_detection_from_target,
        detection_stream=_string(fixed.camera.get("detection_stream"), "detection_stream"),
        frames=frames,
        frame_indices=tuple(frame_indices),
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


def _validate_png(path: Path, name: str) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(24)
    except OSError as error:
        raise ArtifactError(f"could not read diagnostic overlay {name}: {error}") from error
    if (
        len(header) < 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
        or int.from_bytes(header[16:20], "big") == 0
        or int.from_bytes(header[20:24], "big") == 0
    ):
        raise ArtifactError(f"diagnostic overlay is not a valid PNG: {name}")


def _factory_payload_matches_bundle(
    bundle: dict[str, object], calibration: dict[str, object]
) -> bool:
    pairs = (
        (bundle["stream_profiles"], calibration["stream_profiles"]),
        (bundle["intrinsics"], calibration["intrinsics"]),
        (bundle["internal_transforms"], calibration["internal_transforms"]),
        (bundle["depth_scale_m_per_unit"], calibration["depth_scale_m_per_unit"]),
    )
    return all(
        deterministic_json_bytes(left) == deterministic_json_bytes(right) for left, right in pairs
    )


def _validate_acquisition_bindings(
    *,
    capture: dict[str, object],
    factory: object,
    stream: object,
    factory_sha256: str,
    stream_sha256: str,
) -> None:
    from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
    from camera_rig.artifacts.stream_validation import StreamValidationArtifact

    if not isinstance(factory, FactoryCalibrationArtifact) or not isinstance(
        stream, StreamValidationArtifact
    ):
        raise ArtifactError("acquisition evidence types are invalid")
    factory_provenance = _object(factory.provenance, "factory.provenance")
    stream_provenance = _object(stream.provenance, "stream.provenance")
    capture_provenance = _object(capture["provenance"], "capture.provenance")
    acquisition_ids = {
        _string(factory_provenance.get("acquisition_id"), "factory.acquisition_id"),
        _string(stream_provenance.get("acquisition_id"), "stream.acquisition_id"),
        _string(capture_provenance.get("acquisition_id"), "capture.acquisition_id"),
    }
    if len(acquisition_ids) != 1:
        raise ArtifactError("factory, stream, and capture acquisition identities differ")

    calibration = factory.calibration
    expected_device_sha256 = _digest(calibration.device.to_dict())
    expected_profiles_sha256 = _digest(
        {name: profile.to_dict() for name, profile in sorted(calibration.stream_profiles.items())}
    )
    expected_indices = tuple((index * 299) // 59 for index in range(60))
    expected_indices_sha256 = _digest(list(expected_indices))
    expected_stream_bindings = {
        "device_identity_sha256": expected_device_sha256,
        "active_profiles_sha256": expected_profiles_sha256,
        "factory_calibration_sha256": factory_sha256,
        "selected_source_indices_sha256": expected_indices_sha256,
    }
    for name, expected in expected_stream_bindings.items():
        if stream_provenance.get(name) != expected:
            raise ArtifactError(f"stream validation acquisition binding differs: {name}")
    expected_capture_bindings = {
        "factory_calibration_sha256": factory_sha256,
        "stream_validation_sha256": stream_sha256,
        "selected_source_indices_sha256": expected_indices_sha256,
    }
    for name, expected in expected_capture_bindings.items():
        if capture_provenance.get(name) != expected:
            raise ArtifactError(f"capture acquisition binding differs: {name}")

    configuration = _object(capture["capture_configuration"], "capture.capture_configuration")
    indices_value = configuration.get("selected_source_indices")
    if not isinstance(indices_value, list) or tuple(indices_value) != expected_indices:
        raise ArtifactError("capture selected source indices differ from frozen 60-of-300 policy")
    if configuration.get("source_frame_count") != 300:
        raise ArtifactError("capture source frame count must be 300")


def _digest(value: object) -> str:
    return sha256_bytes(deterministic_json_bytes(value))


def _actual_files_without_symlinks(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError("fixed provision artifact root must be a real directory")
    result: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"fixed provision artifact must not contain symlinks: {path.name}")
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
    return result


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
        or value.casefold().startswith("file://")
    ):
        raise ArtifactError(f"unsafe artifact-relative path: {value!r}")
    return value


def _resolved_file(root: Path, relative: str) -> Path:
    safe = _safe_path(relative)
    path = root / safe
    current = root
    for part in PurePosixPath(safe).parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactError(f"symlink escape is not allowed: {relative}")
    if not path.is_file():
        raise ArtifactError(f"missing artifact file: {relative}")
    return path


def _load_checksum_file(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError("missing checksums.sha256")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ArtifactError(f"could not read checksums.sha256: {error}") from error
    entries: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ArtifactError("invalid checksums.sha256 line")
        digest, relative_value = parts
        relative = _safe_path(relative_value)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in entries
        ):
            raise ArtifactError("invalid checksums.sha256 entry")
        entries[relative] = digest
    return entries


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArtifactError(f"{name} must be an object with string keys")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{name} must be a non-empty string")
    return value
