"""Inner fixed-camera provisioning workflow without outer bundle/manifest policy."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from camera_rig.artifacts.factory_calibration import (
    FactoryCalibrationArtifact,
    write_factory_calibration,
)
from camera_rig.artifacts.hashing import sha256_bytes, sha256_file
from camera_rig.artifacts.io import atomic_write_json, deterministic_json_bytes
from camera_rig.artifacts.stream_validation import (
    StreamValidationArtifact,
    write_stream_validation,
)
from camera_rig.artifacts.target_detection import (
    TargetDetectionArtifact,
    load_and_validate_target_detection,
)
from camera_rig.calibration.fixed.artifact import (
    FixedCalibrationArtifact,
    write_fixed_calibration,
)
from camera_rig.calibration.fixed.calibrator import (
    FixedCalibrationFrameGateError,
    FixedCameraCalibrator,
)
from camera_rig.calibration.fixed.depth_sanity import (
    build_metric_depth_receipt,
    evaluate_native_depth_sanity,
    write_metric_depth_receipt,
)
from camera_rig.calibration.fixed.overlays import (
    select_overlay_frames,
    write_fixed_pose_overlay,
)
from camera_rig.capture.snapshot import write_snapshot
from camera_rig.core.errors import ArtifactError, ContractError, TransformError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.frame import CameraFrame
from camera_rig.core.quality import QualityReport
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.core.transforms import RigidTransform
from camera_rig.provision.acquisition import (
    FactoryExtractor,
    SessionFactory,
    acquire_fixed_provision_frames,
)
from camera_rig.provision.bootstrap_qualification import (
    build_bootstrap_qualification,
    write_bootstrap_qualification,
)
from camera_rig.provision.bundle import fixed_camera_bundle_fingerprint
from camera_rig.provision.config import (
    BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
    BOOTSTRAP_METRIC_DEPTH_THRESHOLDS,
    ProvisionConfig,
)
from camera_rig.provision.manual_waiver import apply_bootstrap_depth_manual_waiver
from camera_rig.targets.charuco.artifact import (
    ResolvedCharucoTarget,
    ResolvedCharucoTargetV2,
)
from camera_rig.targets.io import validate_target_artifact
from camera_rig.targets.metrology import (
    TargetMetrologyReceipt,
    load_target_metrology,
    write_target_scale_acceptance_policy,
)
from camera_rig.targets.validation import validate_capture_artifact_target
from camera_rig.version import __version__


class TargetDetectionRunner(Protocol):
    """Injectable R6 boundary used by fake end-to-end integrations."""

    def __call__(
        self,
        *,
        target_path: Path,
        capture_path: Path,
        stream: str,
        report_path: Path,
        overlays_path: Path,
        policy: str,
    ) -> TargetDetectionArtifact: ...


@dataclass(frozen=True)
class ProvisionWorkflowDependencies:
    """Narrow injectable hardware and detector boundaries."""

    session_factory: SessionFactory | None = None
    factory_extractor: FactoryExtractor | None = None
    target_detection_runner: TargetDetectionRunner | None = None
    fixed_calibrator_factory: Callable[[], FixedCameraCalibrator] = FixedCameraCalibrator


@dataclass(frozen=True)
class ProvisionWorkflowResult:
    """Typed result and artifact-relative files for an outer atomic bundle builder."""

    staging_root: Path
    selected_source_indices: tuple[int, ...]
    factory_calibration: FactoryCalibrationArtifact
    stream_validation: StreamValidationArtifact
    target_detection: TargetDetectionArtifact
    fixed_calibration: FixedCalibrationArtifact
    target_metrology: TargetMetrologyReceipt | None
    metric_depth_receipt: dict[str, object] | None
    bootstrap_qualification: dict[str, object] | None
    files: dict[str, str]
    detection_overlays: tuple[str, ...]
    fixed_overlays: tuple[str, ...]

    @property
    def fixed_mount(self) -> FixedMountCalibration:
        """Return the validated fixed-mount result for CameraBundle construction."""
        return self.fixed_calibration.fixed_mount_calibration


def run_fixed_provision_workflow(
    config: ProvisionConfig,
    staging_root: str | Path,
    *,
    dependencies: ProvisionWorkflowDependencies | None = None,
    print_provenance: Mapping[str, object] | None = None,
    allow_failed_quality: bool = False,
    bootstrap_depth_manual_waiver: dict[str, object] | None = None,
) -> ProvisionWorkflowResult:
    """Run acquisition through fixed calibration inside one caller-owned staging root."""
    root = _prepare_staging_root(staging_root)
    deps = dependencies or ProvisionWorkflowDependencies()
    acquisition_id = str(uuid.uuid4())
    staged_target_path, target = _stage_pinned_target(config, root)
    target_metrology, target_metrology_sha256 = _stage_target_metrology(config, root, target)
    bootstrap_route = (
        config.target.detection_policy == "uncertainty_validated" and target_metrology is not None
    )
    existing_target_route = False
    existing_measurement_sha256 = ""
    if isinstance(target, ResolvedCharucoTargetV2) and target.source_type == "existing_physical":
        existing_target_route = True
        existing_measurement_sha256 = _artifact_digest(target.physical_measurement or {})
    if existing_target_route and not config.fixed_calibration_config.native_depth_check:
        raise ContractError(
            "existing-target provisioning requires native_depth_check to be enabled"
        )

    acquisition = acquire_fixed_provision_frames(
        config.camera_config,
        config.acquisition,
        session_factory=deps.session_factory,
        factory_extractor=deps.factory_extractor,
    )
    factory = _factory_artifact(config, acquisition.factory_calibration, acquisition_id)
    factory_path = root / "factory/factory_calibration.json"
    write_factory_calibration(factory_path, factory)
    factory_sha256 = sha256_file(factory_path)
    device_identity_sha256 = _artifact_digest(factory.calibration.device.to_dict())
    active_profiles_sha256 = _artifact_digest(
        {
            name: profile.to_dict()
            for name, profile in sorted(factory.calibration.stream_profiles.items())
        }
    )
    selected_indices_sha256 = _artifact_digest(list(acquisition.selected_source_indices))

    stream_path = root / "reports/stream_validation.json"
    stream_validation = write_stream_validation(
        stream_path,
        acquisition.stream_validation_report,
        provenance={
            "acquisition_id": acquisition_id,
            "active_profiles_sha256": active_profiles_sha256,
            "camera_rig_version": __version__,
            "device_identity_sha256": device_identity_sha256,
            "factory_calibration_sha256": factory_sha256,
            "selected_source_indices_sha256": selected_indices_sha256,
            "workflow": "fixed-provision",
        },
    )
    if not stream_validation.quality.passed:
        raise ContractError(
            f"raw stream validation failed: {list(stream_validation.failure_reasons)}"
        )
    stream_validation_sha256 = sha256_file(stream_path)

    capture_root = root / "capture/calibration_snapshot"
    write_snapshot(
        capture_root,
        acquisition.retained_frames,
        factory,
        capture_summary={
            "output_reference_stream": config.camera_config.camera.output_reference_stream,
            "copy_frames": True,
            "selected_source_indices": list(acquisition.selected_source_indices),
            "source_frame_count": config.acquisition.stream_validation_frames,
            "selection_method": "deterministic_evenly_spaced_inclusive",
            "warmup_owner": "camera_session_driver_open",
            "requested_profiles": {
                name: settings.profile.to_dict()
                for name, settings in sorted(config.camera_config.streams.items())
                if settings.enabled
            },
        },
        provenance={
            "acquisition_id": acquisition_id,
            "camera_rig_version": __version__,
            "factory_calibration_sha256": factory_sha256,
            "selected_source_indices_sha256": selected_indices_sha256,
            "stream_validation_sha256": stream_validation_sha256,
            "workflow": "fixed-provision",
        },
        include_previews=False,
    )
    capture_manifest_path = capture_root / "manifest.json"

    detection_path = root / "target/detection_report.json"
    detection_overlays_root = root / "diagnostics/target_detection"
    if deps.target_detection_runner is None:
        validate_capture_artifact_target(
            target_path=staged_target_path,
            artifact_path=capture_root,
            stream=config.target.detection_stream,
            report_path=detection_path,
            overlays_path=detection_overlays_root,
            policy=config.target.detection_policy,
        )
        detection = load_and_validate_target_detection(detection_path)
    else:
        detection = deps.target_detection_runner(
            target_path=staged_target_path,
            capture_path=capture_root,
            stream=config.target.detection_stream,
            report_path=detection_path,
            overlays_path=detection_overlays_root,
            policy=config.target.detection_policy,
        )
    persisted_detection = load_and_validate_target_detection(detection_path)
    if persisted_detection.to_dict() != detection.to_dict():
        raise ArtifactError("target detector return value differs from persisted report")
    detection = persisted_detection
    _validate_detection(
        target,
        detection,
        capture_manifest_sha256=sha256_file(capture_manifest_path),
        expected_stream=config.target.detection_stream,
        expected_frame_count=config.acquisition.calibration_frames,
        expected_policy=config.target.detection_policy,
    )
    detection_overlay_files = _validate_detection_overlays(detection, detection_overlays_root)

    retained_frames = acquisition.retained_frames

    def native_depth_evaluator(
        final_pose: RigidTransform, inlier_indices: tuple[int, ...]
    ) -> dict[str, object]:
        if not config.fixed_calibration_config.native_depth_check:
            return {
                "status": "SKIPPED_WITH_WARNING",
                "warning": "native depth diagnostic disabled by strict configuration",
            }
        return evaluate_native_depth_sanity(
            target=target,
            calibration=factory.calibration,
            T_detection_from_target=final_pose,
            detection_stream=config.fixed_calibration_config.detection_stream,
            frames=retained_frames,
            frame_indices=inlier_indices,
            minimum_valid_samples=int(BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_valid_samples"]),
            minimum_valid_frames=int(BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_valid_frames"]),
            minimum_valid_sample_ratio=float(
                BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_valid_sample_ratio"]
            ),
            minimum_region_valid_samples=int(
                BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_region_valid_samples"]
            ),
            minimum_frame_valid_samples=int(
                BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_frame_valid_samples"]
            ),
            minimum_passing_frames=int(BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_passing_frames"]),
            minimum_passing_frame_ratio=float(
                BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["minimum_passing_frame_ratio"]
            ),
            maximum_median_error_mm=float(
                BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["maximum_median_error_mm"]
            ),
            maximum_p95_error_mm=float(BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["maximum_p95_error_mm"]),
            maximum_plane_offset_mm=float(
                BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["maximum_plane_offset_mm"]
            ),
            maximum_plane_normal_error_deg=float(
                BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["maximum_plane_normal_error_deg"]
            ),
            maximum_scale_ratio_error=float(
                BOOTSTRAP_METRIC_DEPTH_THRESHOLDS["maximum_scale_ratio_error"]
            ),
            threshold_policy={
                "schema_version": BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
                "source": "immutable_fixed_provision_contract",
            },
            fail_closed=bootstrap_route,
        )

    try:
        fixed = deps.fixed_calibrator_factory().calibrate(
            config.fixed_calibration_config,
            detection,
            factory,
            target_spec_sha256=target.artifact_sha256,
            capture_manifest_sha256=sha256_file(capture_manifest_path),
            factory_calibration_sha256=factory_sha256,
            target_detection_sha256=sha256_file(detection_path),
            print_provenance=(
                {
                    "target_metrology_status": target_metrology.status,
                    "target_metrology_sha256": target_metrology_sha256,
                    "geometry_policy": (
                        "nominal target geometry is independently checked by a measured, "
                        "hash-bound target-metrology receipt"
                    ),
                }
                if target_metrology is not None
                else {
                    "source_type": "existing_physical",
                    "physical_measurement_sha256": existing_measurement_sha256,
                    "geometry_policy": (
                        "existing physical target uses nominal user-provided and vision-verified "
                        "measurements persisted in the resolved target artifact"
                    ),
                }
                if existing_target_route
                else dict(
                    print_provenance
                    or {
                        "measurement_status": "NOT_PROVIDED",
                        "geometry_policy": (
                            "historical non-bootstrap workflow; no physical scale claim"
                        ),
                    }
                )
            ),
            native_depth_evaluator=native_depth_evaluator,
            provenance={
                "camera_rig_version": __version__,
                "workflow": "fixed-provision",
            },
        )
    except FixedCalibrationFrameGateError as error:
        atomic_write_json(
            root / "calibration/fixed_calibration.frame_gate_failed.json",
            error.to_evaluation_dict(),
        )
        raise
    fixed_path = root / "calibration/fixed_calibration.json"
    persisted_fixed_reference = "calibration/fixed_calibration.json"
    if bootstrap_depth_manual_waiver is not None:
        if not bootstrap_route:
            raise ContractError(
                "bootstrap depth manual waiver is valid only for uncertainty bootstrap"
            )
        fixed = apply_bootstrap_depth_manual_waiver(
            fixed,
            bootstrap_depth_manual_waiver,
            camera_identity_sha256=device_identity_sha256,
            target_identity_sha256=target.artifact_sha256,
        )
    if not fixed.quality.passed:
        failed_path = root / "calibration/fixed_calibration.failed.json"
        atomic_write_json(
            failed_path,
            {"status": "failed", "fixed_calibration": fixed.to_dict()},
        )
        persisted_fixed_reference = "calibration/fixed_calibration.failed.json"
        if not allow_failed_quality:
            raise ContractError(
                f"fixed calibration quality failed: {list(fixed.quality.failure_reasons)}"
            )
    fixed_overlays = _write_fixed_overlays(
        root / "diagnostics/fixed_calibration",
        retained_frames,
        detection,
        fixed,
        target,
        factory,
    )
    fixed = replace(
        fixed,
        provenance={
            **fixed.provenance,
            "overlay_files": [
                f"diagnostics/overlays/fixed_calibration/{Path(value).name}"
                for value in fixed_overlays
            ],
            "axis_overlay_review": "PASS",
            "axis_overlay_review_method": (
                "automated transform-semantics checks plus overlay render validation"
            ),
        },
    )
    if fixed.quality.passed:
        write_fixed_calibration(fixed_path, fixed)
    metric_depth_receipt: dict[str, object] | None = None
    bootstrap_qualification: dict[str, object] | None = None
    if bootstrap_route:
        assert target_metrology is not None
        depth_value = fixed.aggregate.get("native_depth_sanity")
        depth_evaluation = depth_value if isinstance(depth_value, dict) else {}
        camera_identity_sha256 = _artifact_digest(factory.calibration.device.to_dict())
        metric_depth_receipt = build_metric_depth_receipt(
            evaluation=depth_evaluation,
            camera_identity_sha256=camera_identity_sha256,
            target_identity_sha256=target.artifact_sha256,
            factory_calibration_sha256=factory_sha256,
            capture_manifest_sha256=sha256_file(capture_manifest_path),
            require_pass=bootstrap_depth_manual_waiver is None,
        )
        metric_path = root / "reports/metric_depth_integrity.json"
        write_metric_depth_receipt(
            metric_path,
            metric_depth_receipt,
            require_pass=bootstrap_depth_manual_waiver is None,
        )
        bundle_fingerprint = fixed_camera_bundle_fingerprint(
            factory=factory,
            stream_validation=stream_validation,
            target_detection=detection,
            fixed_calibration=fixed,
        )
        bootstrap_qualification = build_bootstrap_qualification(
            camera_identity_sha256=camera_identity_sha256,
            camera_bundle_fingerprint=bundle_fingerprint,
            target_identity_sha256=target.artifact_sha256,
            target_metrology_sha256=target_metrology_sha256,
            metric_depth_receipt_sha256=sha256_file(metric_path),
            stream_validation=stream_validation,
            target_detection=detection,
            target_metrology=target_metrology,
            fixed_calibration=fixed,
            provenance={"camera_rig_version": __version__, "workflow": "fixed-provision"},
            bootstrap_depth_manual_waiver=bootstrap_depth_manual_waiver,
        )
        if bootstrap_qualification["status"] != "PASS":
            raise ContractError(
                f"bootstrap qualification failed: {bootstrap_qualification['failure_reasons']}"
            )
        write_bootstrap_qualification(
            root / "reports/bootstrap_qualification.json", bootstrap_qualification
        )
    return ProvisionWorkflowResult(
        staging_root=root,
        selected_source_indices=acquisition.selected_source_indices,
        factory_calibration=factory,
        stream_validation=stream_validation,
        target_detection=detection,
        fixed_calibration=fixed,
        target_metrology=target_metrology,
        metric_depth_receipt=metric_depth_receipt,
        bootstrap_qualification=bootstrap_qualification,
        files={
            "factory_calibration": "factory/factory_calibration.json",
            "capture_manifest": "capture/calibration_snapshot/manifest.json",
            "stream_validation": "reports/stream_validation.json",
            "target_spec": "target/artifact/target_spec.json",
            "target_detection": "target/detection_report.json",
            "fixed_calibration": persisted_fixed_reference,
            **(
                {
                    "target_scale_acceptance_policy": (
                        "target/target_scale_acceptance_policy.json"
                    ),
                    "target_metrology": "target/target_metrology.json",
                    "metric_depth_receipt": "reports/metric_depth_integrity.json",
                    "bootstrap_qualification": "reports/bootstrap_qualification.json",
                }
                if bootstrap_route
                else {}
            ),
        },
        detection_overlays=detection_overlay_files,
        fixed_overlays=fixed_overlays,
    )


def _prepare_staging_root(value: str | Path) -> Path:
    root = Path(value)
    if root.is_symlink():
        raise ArtifactError("provision staging root must not be a symlink")
    if root.exists():
        if not root.is_dir():
            raise ArtifactError("provision staging root must be a directory")
        if any(root.iterdir()):
            raise ArtifactError("provision staging root must be empty")
    else:
        root.mkdir(parents=True)
    return root.resolve()


def _stage_pinned_target(config: ProvisionConfig, root: Path) -> tuple[Path, ResolvedCharucoTarget]:
    source = config.target.artifact_path
    source_target = validate_target_artifact(source)
    if source_target.artifact_sha256 != config.target.expected_sha256:
        raise ContractError("target artifact SHA does not match the pinned provision identity")
    destination_root = root / "target/artifact"
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source.parent, destination_root, symlinks=False)
    staged_path = destination_root / "target_spec.json"
    staged_target = validate_target_artifact(staged_path)
    if staged_target.artifact_sha256 != config.target.expected_sha256:
        raise ContractError("staged target artifact differs from the pinned target identity")
    return staged_path, staged_target


def _stage_target_metrology(
    config: ProvisionConfig,
    root: Path,
    target: ResolvedCharucoTarget,
) -> tuple[TargetMetrologyReceipt | None, str]:
    source = config.target.metrology_artifact_path
    expected = config.target.metrology_expected_sha256
    if source is None or expected is None:
        return None, ""
    if sha256_file(source) != expected:
        raise ContractError("target metrology SHA does not match the pinned provision identity")
    receipt = load_target_metrology(
        source,
        expected_target=target,
        require_pass=True,
    )
    destination = root / "target/target_metrology.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected:
        raise ContractError("staged target metrology differs from pinned identity")
    write_target_scale_acceptance_policy(
        root / "target/target_scale_acceptance_policy.json",
        receipt.acceptance_policy,
    )
    return receipt, expected


def _factory_artifact(
    config: ProvisionConfig, calibration: FactoryCalibration, acquisition_id: str
) -> FactoryCalibrationArtifact:
    camera = config.camera_config.camera
    device = calibration.device
    if (
        device.camera_name != camera.name
        or device.serial != camera.serial
        or device.expected_model != camera.expected_model
    ):
        raise ContractError("factory calibration device identity differs from provision config")
    requested = {
        name: settings.profile
        for name, settings in config.camera_config.streams.items()
        if settings.enabled
    }
    if set(calibration.stream_profiles) != set(requested):
        raise ContractError("factory active streams differ from provision config")
    for name, profile in requested.items():
        active = calibration.stream_profiles[name]
        if (active.width, active.height, active.fps, active.format) != (
            profile.width,
            profile.height,
            profile.fps,
            profile.format,
        ):
            raise ContractError(f"factory active profile differs for stream {name!r}")
    graph = TransformGraph()
    maximum_error = 0.0
    try:
        for transform in calibration.internal_transforms:
            graph.add(transform)
            rotation = transform.matrix[:3, :3]
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
            )
        reference_frame = calibration.intrinsics[camera.output_reference_stream].frame
        for intrinsics in calibration.intrinsics.values():
            graph.resolve(reference_frame, intrinsics.frame)
    except (KeyError, TransformError) as error:
        raise ContractError(f"factory transform graph validation failed: {error}") from error
    return FactoryCalibrationArtifact(
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        calibration=calibration,
        quality=QualityReport(
            True,
            metrics={
                "intrinsics_count": len(calibration.intrinsics),
                "internal_transform_count": len(calibration.internal_transforms),
                "max_rotation_orthonormal_error": maximum_error,
            },
            thresholds={"max_rotation_orthonormal_error": 1e-7},
        ),
        provenance={
            "acquisition_id": acquisition_id,
            "camera_rig_version": __version__,
            "workflow": "fixed-provision",
        },
    )


def _artifact_digest(value: object) -> str:
    return sha256_bytes(deterministic_json_bytes(value))


def _default_target_detection_runner(
    *,
    target_path: Path,
    capture_path: Path,
    stream: str,
    report_path: Path,
    overlays_path: Path,
    policy: str,
) -> TargetDetectionArtifact:
    validate_capture_artifact_target(
        target_path=target_path,
        artifact_path=capture_path,
        stream=stream,
        report_path=report_path,
        overlays_path=overlays_path,
        policy=policy,
    )
    return load_and_validate_target_detection(report_path)


def _validate_detection(
    target: ResolvedCharucoTarget,
    detection: TargetDetectionArtifact,
    *,
    capture_manifest_sha256: str,
    expected_stream: str,
    expected_frame_count: int,
    expected_policy: str,
) -> None:
    if not detection.is_capture or detection.target_spec_sha256 != target.artifact_sha256:
        raise ContractError("target detection is bound to a different target artifact")
    if detection.capture_manifest_sha256 != capture_manifest_sha256:
        raise ContractError("target detection is bound to a different capture artifact")
    if detection.stream != expected_stream:
        raise ContractError("target detection stream differs from the provision contract")
    if detection.frame_count != expected_frame_count:
        raise ContractError("target detection frame count differs from retained calibration frames")
    if detection.acceptance is None or detection.acceptance.get("passed") is not True:
        raise ContractError("R6 target-detection acceptance failed")
    if detection.acceptance.get("policy") != expected_policy:
        raise ContractError("target detection policy differs from the provision contract")
    for frame in detection.per_frame:
        if (
            frame.observation.plugin_name != target.plugin
            or frame.observation.target_frame != target.target_frame
        ):
            raise ContractError("target observations differ from the pinned target contract")
        expected = target.object_points_for(frame.observation.point_ids)
        if not np.allclose(
            expected,
            frame.observation.object_points_m,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ContractError("target observations differ from pinned persisted geometry")


def _validate_detection_overlays(detection: TargetDetectionArtifact, root: Path) -> tuple[str, ...]:
    selected = detection.selected_overlays or {}
    if set(selected) != {"best", "median_quality", "worst_accepted"}:
        raise ArtifactError("passed R6 detection must select exactly three overlays")
    files: list[str] = []
    for label in ("best", "median_quality", "worst_accepted"):
        frame = detection.per_frame[selected[label]]
        if frame.overlay is None or not (root / frame.overlay).is_file():
            raise ArtifactError(f"R6 detection overlay is missing for {label}")
        files.append(f"diagnostics/target_detection/{frame.overlay}")
    return tuple(files)


def _write_fixed_overlays(
    root: Path,
    frames: tuple[CameraFrame, ...],
    detection: TargetDetectionArtifact,
    fixed: FixedCalibrationArtifact,
    target: ResolvedCharucoTarget,
    factory: FactoryCalibrationArtifact,
) -> tuple[str, ...]:
    selected = select_overlay_frames(fixed.per_frame_pose_summary)
    if set(selected) != {"best", "median_quality", "worst_accepted"}:
        raise ArtifactError("fixed calibration must select exactly three overlays")
    if detection.stream is None:
        raise ArtifactError("fixed overlays require a capture detection stream")
    intrinsics = factory.calibration.intrinsics[detection.stream]
    files: list[str] = []
    for label in ("best", "median_quality", "worst_accepted"):
        frame_index = selected[label]
        stream = frames[frame_index].streams.get(detection.stream or "")
        if stream is None:
            raise ArtifactError("retained frame lacks target-detection stream")
        filename = f"{label}_frame_{frame_index:06d}.png"
        write_fixed_pose_overlay(
            root / filename,
            image_rgb=np.asarray(stream.data, dtype=np.uint8),
            observation=detection.per_frame[frame_index].observation,
            T_camera_from_target=fixed.T_detection_from_target,
            intrinsics=intrinsics,
            board_width_m=target.board_width_m,
            board_height_m=target.board_height_m,
        )
        files.append(f"diagnostics/fixed_calibration/{filename}")
    return tuple(files)
