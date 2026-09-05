"""Build and semantically validate final fixed-camera CameraBundle artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Final

import numpy as np

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.artifacts.hashing import sha256_bytes
from camera_rig.artifacts.io import (
    JsonValue,
    atomic_write_json,
    deterministic_json_bytes,
    load_json,
)
from camera_rig.artifacts.models import CameraBundle
from camera_rig.artifacts.stream_validation import StreamValidationArtifact
from camera_rig.artifacts.target_detection import TargetDetectionArtifact
from camera_rig.artifacts.validation import validate_bundle_data
from camera_rig.calibration.fixed.artifact import FixedCalibrationArtifact
from camera_rig.core.errors import ArtifactError, ContractError, TransformError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.core.transforms import RigidTransform
from camera_rig.provision.bootstrap_qualification import (
    validate_bootstrap_qualification_data,
)

MINIMUM_TARGET_DISTANCE_M: Final = 0.1
MAXIMUM_TARGET_DISTANCE_M: Final = 5.0
REQUIRED_FIXED_BUNDLE_CHECKS: Final = (
    "device_valid",
    "profiles_valid",
    "factory_valid",
    "stream_validation_passed",
    "target_detection_passed",
    "fixed_calibration_passed",
    "reference_frame_valid",
    "target_distance_physical",
    "bundle_self_validation_passed",
)
_EVIDENCE_NAMES: Final = (
    "factory_calibration",
    "stream_validation",
    "target_detection",
    "fixed_calibration",
)
_REQUIRED_STREAMS: Final = frozenset({"color", "depth", "ir_left", "ir_right"})


def build_fixed_camera_bundle(
    *,
    bundle_id: str,
    created_at: str,
    factory: FactoryCalibrationArtifact,
    stream_validation: StreamValidationArtifact,
    target_detection: TargetDetectionArtifact,
    fixed_calibration: FixedCalibrationArtifact,
    provenance: dict[str, object],
    bootstrap_qualification: dict[str, object] | None = None,
) -> CameraBundle:
    """Build one passing bundle from already-completed, mutually bound evidence."""
    _require_passing_evidence(
        factory,
        stream_validation,
        target_detection,
        fixed_calibration,
        bootstrap_qualification=bootstrap_qualification,
    )
    calibration = factory.calibration
    fixed_mount = fixed_calibration.fixed_mount_calibration
    reference_stream, distance_m = _validate_camera_semantics(
        calibration.stream_profiles,
        calibration.intrinsics,
        calibration.internal_transforms,
        fixed_mount.T_parent_from_camera_reference,
    )
    _validate_fixed_bindings(factory, target_detection, fixed_calibration, reference_stream)
    portable_provenance = dict(provenance)
    _reject_absolute_provenance(portable_provenance, "provenance")
    if "evidence_sha256" in portable_provenance:
        raise ContractError("provenance.evidence_sha256 is reserved for the bundle builder")
    portable_provenance["bundle_kind"] = "fixed-camera"
    portable_provenance["evidence_sha256"] = {
        "factory_calibration": _artifact_digest(factory.to_dict()),
        "stream_validation": _artifact_digest(stream_validation.to_dict()),
        "target_detection": _artifact_digest(target_detection.to_dict()),
        "fixed_calibration": _artifact_digest(fixed_calibration.to_dict()),
    }
    if bootstrap_qualification is not None:
        qualification = validate_bootstrap_qualification_data(bootstrap_qualification)
        expected_fingerprint = fixed_camera_bundle_fingerprint(
            factory=factory,
            stream_validation=stream_validation,
            target_detection=target_detection,
            fixed_calibration=fixed_calibration,
        )
        if qualification["camera_bundle_fingerprint"] != expected_fingerprint:
            raise ContractError("bootstrap qualification bundle fingerprint differs")
        waived = (
            qualification["qualification_state"] == "BOOTSTRAP_QUALIFIED_WITH_MANUAL_DEPTH_WAIVER"
        )
        authority = {
            "schema_version": (
                "camera-rig.calibration-authority.v2"
                if waived
                else "camera-rig.calibration-authority.v1"
            ),
            "qualification_scope": "bootstrap_only",
            "production_authoritative": False,
            "qualification_state": qualification["qualification_state"],
            "qualification_fingerprint": qualification["qualification_fingerprint"],
            "target_metrology_sha256": qualification["target_metrology_sha256"],
            "metric_depth_receipt_sha256": qualification["metric_depth_receipt_sha256"],
        }
        if waived:
            waiver = qualification["manual_waiver"]
            assert isinstance(waiver, dict)
            authority.update(
                {
                    "machine_status": "FAIL",
                    "waived_check": "metric_native_depth_integrity",
                    "waiver_fingerprint": waiver["waiver_fingerprint"],
                }
            )
        portable_provenance["calibration_authority"] = authority
    checks = {name: True for name in REQUIRED_FIXED_BUNDLE_CHECKS}
    quality = QualityReport(
        passed=True,
        metrics={
            "checks": checks,
            "reference_stream": reference_stream,
            "target_origin_distance_m": distance_m,
        },
        thresholds={
            "minimum_target_distance_m_exclusive": MINIMUM_TARGET_DISTANCE_M,
            "maximum_target_distance_m_exclusive": MAXIMUM_TARGET_DISTANCE_M,
        },
    )
    bundle = CameraBundle(
        status="passed",
        bundle_id=bundle_id,
        created_at=created_at,
        device=calibration.device,
        stream_profiles=calibration.stream_profiles,
        intrinsics=calibration.intrinsics,
        internal_transforms=calibration.internal_transforms,
        depth_scale_m_per_unit=calibration.depth_scale_m_per_unit,
        fixed_mount_calibration=fixed_mount,
        quality=quality,
        provenance=portable_provenance,
    )
    return validate_fixed_camera_bundle(bundle)


def validate_fixed_camera_bundle(bundle: CameraBundle) -> CameraBundle:
    """Apply final fixed-camera semantics beyond the generic CameraBundle schema."""
    fixed = bundle.fixed_mount_calibration
    if fixed is None:
        raise ContractError("fixed camera bundle requires fixed_mount_calibration")
    if not fixed.quality.passed:
        raise ContractError("fixed_mount_calibration quality must be passed")
    _reject_absolute_provenance(bundle.provenance, "provenance")
    _reject_absolute_provenance(fixed.provenance, "fixed_mount_calibration.provenance")
    evidence = bundle.provenance.get("evidence_sha256")
    if not isinstance(evidence, dict) or set(evidence) != set(_EVIDENCE_NAMES):
        raise ContractError("bundle provenance must contain the complete evidence SHA mapping")
    for name, digest in evidence.items():
        _require_digest(digest, f"provenance.evidence_sha256.{name}")
    authority = bundle.provenance.get("calibration_authority")
    if authority is not None:
        _validate_bootstrap_authority(authority)

    reference_stream, distance_m = _validate_camera_semantics(
        bundle.stream_profiles,
        bundle.intrinsics,
        bundle.internal_transforms,
        fixed.T_parent_from_camera_reference,
    )
    checks = bundle.quality.metrics.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_FIXED_BUNDLE_CHECKS):
        raise ContractError("bundle quality checks have missing or unknown fields")
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ContractError("bundle quality checks must be booleans")
    all_checks_passed = all(checks.values())
    if (bundle.status == "passed") != all_checks_passed:
        raise ContractError("bundle status must be passed if and only if every check passes")
    if bundle.status not in {"passed", "failed"}:
        raise ContractError("fixed camera bundle status must be passed or failed")
    if bundle.quality.passed != all_checks_passed:
        raise ContractError("bundle quality decision must equal all required checks")
    if not all_checks_passed:
        raise ContractError("final fixed camera bundle must pass every required check")
    if bundle.quality.failure_reasons:
        raise ContractError("passed fixed camera bundle cannot contain failure reasons")
    if checks["reference_frame_valid"] is not True:
        raise ContractError("reference-frame semantic check is not passed")
    if checks["target_distance_physical"] is not True:
        raise ContractError("target-distance semantic check is not passed")

    recorded_stream = bundle.quality.metrics.get("reference_stream")
    if recorded_stream != reference_stream:
        raise ContractError("recorded reference stream differs from fixed-mount frame membership")
    recorded_distance = _number(
        bundle.quality.metrics.get("target_origin_distance_m"), "target_origin_distance_m"
    )
    if not math.isclose(recorded_distance, distance_m, rel_tol=0.0, abs_tol=1e-7):
        raise ContractError("recorded target distance differs from transform-derived distance")
    expected_thresholds = {
        "minimum_target_distance_m_exclusive": MINIMUM_TARGET_DISTANCE_M,
        "maximum_target_distance_m_exclusive": MAXIMUM_TARGET_DISTANCE_M,
    }
    if bundle.quality.thresholds != expected_thresholds:
        raise ContractError("bundle target-distance thresholds differ from the frozen contract")
    return bundle


def validate_fixed_camera_bundle_data(value: JsonValue) -> CameraBundle:
    """Schema-reconstruct and then semantically validate a fixed-camera bundle."""
    try:
        return validate_fixed_camera_bundle(validate_bundle_data(value))
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, ContractError, TransformError) as error:
        raise ArtifactError(f"fixed camera bundle semantics are invalid: {error}") from error


def load_and_validate_fixed_camera_bundle(path: str | Path) -> CameraBundle:
    """Reload a persisted bundle without importing camera, vision, or preview runtimes."""
    return validate_fixed_camera_bundle_data(load_json(path))


def write_fixed_camera_bundle(path: str | Path, bundle: CameraBundle) -> CameraBundle:
    """Validate, atomically write, reload, and revalidate one final bundle."""
    validated = validate_fixed_camera_bundle(bundle)
    atomic_write_json(path, validated.to_dict())
    return load_and_validate_fixed_camera_bundle(path)


def _require_passing_evidence(
    factory: FactoryCalibrationArtifact,
    stream_validation: StreamValidationArtifact,
    target_detection: TargetDetectionArtifact,
    fixed_calibration: FixedCalibrationArtifact,
    *,
    bootstrap_qualification: dict[str, object] | None,
) -> None:
    if not factory.quality.passed:
        raise ContractError("factory calibration evidence must be passed")
    if stream_validation.status != "PASS" or not stream_validation.quality.passed:
        raise ContractError("stream validation evidence must be passed")
    acceptance = target_detection.acceptance
    if (
        not target_detection.is_capture
        or acceptance is None
        or acceptance.get("passed") is not True
    ):
        raise ContractError("capture target-detection evidence must be passed")
    if not fixed_calibration.quality.passed:
        raise ContractError("fixed calibration evidence must be passed")
    if not fixed_calibration.fixed_mount_calibration.quality.passed:
        raise ContractError("fixed_mount_calibration quality must be passed")
    if not solver_release_eligible(
        fixed_calibration.solver,
        acceptance,
        bootstrap_qualification=bootstrap_qualification,
    ):
        raise ContractError(
            "uncertainty_validated is not release-enabled; canonical provision is blocked"
        )


def solver_release_eligible(
    solver: Mapping[str, object],
    target_acceptance: Mapping[str, object],
    *,
    bootstrap_qualification: dict[str, object] | None = None,
) -> bool:
    """Validate policy agreement and fail closed for every uncertainty candidate."""
    known = {"legacy_strict", "pose_validated", "uncertainty_validated"}
    solver_policy = solver.get("pose_policy")
    target_policy = target_acceptance.get("policy")
    if solver_policy is None and target_policy is None:
        # Explicit backward-compatibility route for artifacts written before either
        # side serialized a policy. Those artifacts used the legacy strict policy.
        return True
    if not isinstance(solver_policy, str) or not isinstance(target_policy, str):
        return False
    if solver_policy not in known or target_policy not in known:
        return False
    if solver_policy != target_policy:
        return False
    if solver_policy != "uncertainty_validated":
        return True
    if bootstrap_qualification is None:
        return False
    try:
        report = validate_bootstrap_qualification_data(bootstrap_qualification)
    except ArtifactError:
        return False
    return report.get("status") == "PASS"


def fixed_camera_bundle_fingerprint(
    *,
    factory: FactoryCalibrationArtifact,
    stream_validation: StreamValidationArtifact,
    target_detection: TargetDetectionArtifact,
    fixed_calibration: FixedCalibrationArtifact,
) -> str:
    """Fingerprint stable bundle content without creating an authority cycle."""
    return _artifact_digest(
        {
            "device": factory.calibration.device.to_dict(),
            "stream_profiles": {
                name: value.to_dict()
                for name, value in sorted(factory.calibration.stream_profiles.items())
            },
            "intrinsics": {
                name: value.to_dict()
                for name, value in sorted(factory.calibration.intrinsics.items())
            },
            "internal_transforms": [
                value.to_dict() for value in factory.calibration.internal_transforms
            ],
            "depth_scale_m_per_unit": factory.calibration.depth_scale_m_per_unit,
            "fixed_mount_calibration": fixed_calibration.fixed_mount_calibration.to_dict(),
            "evidence_sha256": {
                "factory_calibration": _artifact_digest(factory.to_dict()),
                "stream_validation": _artifact_digest(stream_validation.to_dict()),
                "target_detection": _artifact_digest(target_detection.to_dict()),
                "fixed_calibration": _artifact_digest(fixed_calibration.to_dict()),
            },
        }
    )


def _validate_bootstrap_authority(value: object) -> None:
    common = {
        "schema_version",
        "qualification_scope",
        "production_authoritative",
        "qualification_state",
        "qualification_fingerprint",
        "target_metrology_sha256",
        "metric_depth_receipt_sha256",
    }
    if not isinstance(value, dict):
        raise ContractError("bundle calibration authority is incomplete")
    schema_version = value.get("schema_version")
    waived = schema_version == "camera-rig.calibration-authority.v2"
    expected_fields = (
        common | {"machine_status", "waived_check", "waiver_fingerprint"} if waived else common
    )
    if set(value) != expected_fields:
        raise ContractError("bundle calibration authority is incomplete")
    if (
        schema_version
        not in {"camera-rig.calibration-authority.v1", "camera-rig.calibration-authority.v2"}
        or value.get("qualification_scope") != "bootstrap_only"
        or value.get("production_authoritative") is not False
        or value.get("qualification_state")
        != ("BOOTSTRAP_QUALIFIED_WITH_MANUAL_DEPTH_WAIVER" if waived else "BOOTSTRAP_QUALIFIED")
    ):
        raise ContractError("bundle calibration authority semantics are invalid")
    digest_names = [
        "qualification_fingerprint",
        "target_metrology_sha256",
        "metric_depth_receipt_sha256",
    ]
    if waived:
        digest_names.append("waiver_fingerprint")
    for name in digest_names:
        _require_digest(value.get(name), f"calibration_authority.{name}")
    if waived and (
        value.get("machine_status") != "FAIL"
        or value.get("waived_check") != "metric_native_depth_integrity"
    ):
        raise ContractError("bundle calibration waiver authority semantics are invalid")


def _validate_fixed_bindings(
    factory: FactoryCalibrationArtifact,
    target_detection: TargetDetectionArtifact,
    fixed: FixedCalibrationArtifact,
    reference_stream: str,
) -> None:
    target_origin_detection = fixed.T_detection_from_target.transform_points(np.zeros(3))
    target_distance_detection = float(np.linalg.norm(target_origin_detection))
    if target_origin_detection[2] <= 0.0:
        raise ContractError("fixed calibration target origin must have positive camera depth")
    target_normal_detection = fixed.T_detection_from_target.matrix[:3, :3] @ np.asarray(
        [0.0, 0.0, 1.0]
    )
    if float(np.dot(target_normal_detection, -target_origin_detection)) <= 0.0:
        raise ContractError("fixed calibration printed face must point toward the camera")
    if not MINIMUM_TARGET_DISTANCE_M < target_distance_detection < MAXIMUM_TARGET_DISTANCE_M:
        raise ContractError(
            "detection-frame target distance must lie strictly between 0.1 m and 5 m"
        )
    target_sha = fixed.target.get("target_spec_sha256")
    if target_sha != target_detection.target_spec_sha256:
        raise ContractError("fixed calibration target identity differs from target detection")
    if fixed.inputs.get("capture_manifest_sha256") != target_detection.capture_manifest_sha256:
        raise ContractError("fixed calibration capture identity differs from target detection")
    if fixed.inputs.get("factory_calibration_sha256") != _artifact_digest(factory.to_dict()):
        raise ContractError("fixed calibration factory-calibration SHA binding is invalid")
    if fixed.inputs.get("target_detection_sha256") != _artifact_digest(target_detection.to_dict()):
        raise ContractError("fixed calibration target-detection SHA binding is invalid")
    camera = fixed.camera
    detection_stream = camera.get("detection_stream")
    detection_frame = camera.get("detection_frame")
    if (
        not isinstance(detection_stream, str)
        or detection_stream not in factory.calibration.intrinsics
    ):
        raise ContractError("fixed calibration detection stream is absent from factory intrinsics")
    if factory.calibration.intrinsics[detection_stream].frame != detection_frame:
        raise ContractError("fixed calibration detection frame differs from factory intrinsics")
    if camera.get("reference_stream") != reference_stream:
        raise ContractError("fixed calibration reference stream differs from bundle membership")
    if camera.get("reference_frame") != fixed.fixed_mount_calibration.camera_reference_frame:
        raise ContractError("fixed calibration reference frame differs from fixed mount")
    graph = _internal_graph(factory.calibration.internal_transforms)
    expected = graph.resolve(
        fixed.fixed_mount_calibration.camera_reference_frame,
        str(detection_frame),
    )
    if not np.allclose(
        expected.matrix,
        fixed.T_detection_from_reference.matrix,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ContractError("fixed calibration internal transform differs from factory graph")


def _validate_camera_semantics(
    stream_profiles: Mapping[str, StreamProfile],
    intrinsics: Mapping[str, CameraIntrinsics],
    internal_transforms: tuple[RigidTransform, ...],
    T_parent_from_reference: RigidTransform,
) -> tuple[str, float]:
    if set(stream_profiles) != _REQUIRED_STREAMS or set(intrinsics) != _REQUIRED_STREAMS:
        raise ContractError("fixed camera bundle requires all four raw stream profiles/intrinsics")
    reference_frame = T_parent_from_reference.source_frame
    matches = [name for name, intrinsic in intrinsics.items() if intrinsic.frame == reference_frame]
    if len(matches) != 1:
        raise ContractError(
            "fixed-mount reference frame must identify exactly one intrinsic stream"
        )
    graph = _internal_graph(internal_transforms)
    intrinsic_frames = [value.frame for value in intrinsics.values()]
    frame_set = set(intrinsic_frames)
    if len(frame_set) != len(intrinsic_frames):
        raise ContractError("each fixed-camera intrinsic stream must use a unique frame")
    for transform in internal_transforms:
        if transform.source_frame not in frame_set or transform.target_frame not in frame_set:
            raise ContractError("internal transform references a frame without intrinsics")
    for frame in intrinsic_frames:
        graph.resolve(reference_frame, frame)

    camera_origin = np.zeros(3, dtype=np.float64)
    target_origin_from_camera = T_parent_from_reference.transform_points(camera_origin)
    distance_m = float(np.linalg.norm(target_origin_from_camera))
    if not MINIMUM_TARGET_DISTANCE_M < distance_m < MAXIMUM_TARGET_DISTANCE_M:
        raise ContractError("target origin distance must lie strictly between 0.1 m and 5 m")
    round_trip = T_parent_from_reference.inverse().transform_points(target_origin_from_camera)
    if not np.allclose(round_trip, camera_origin, rtol=0.0, atol=1e-7):
        raise ContractError("fixed-mount point round trip is inconsistent")
    inverse_distance = float(
        np.linalg.norm(T_parent_from_reference.inverse().transform_points(np.zeros(3)))
    )
    if not math.isclose(inverse_distance, distance_m, rel_tol=0.0, abs_tol=1e-7):
        raise ContractError("forward and inverse target distances are inconsistent")
    return matches[0], distance_m


def _internal_graph(transforms: tuple[RigidTransform, ...]) -> TransformGraph:
    graph = TransformGraph()
    for transform in transforms:
        graph.add(transform)
    return graph


def _artifact_digest(value: object) -> str:
    return sha256_bytes(deterministic_json_bytes(value))


def _reject_absolute_provenance(value: object, path: str) -> None:
    if isinstance(value, str):
        windows = PureWindowsPath(value)
        if (
            Path(value).is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or value.casefold().startswith("file://")
        ):
            raise ContractError(f"{path} must not contain absolute paths or file URIs")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_provenance(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_absolute_provenance(item, f"{path}[{index}]")


def _require_digest(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ContractError(f"{name} must be finite")
    return numeric
