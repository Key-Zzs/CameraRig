from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import load_json
from camera_rig.artifacts.target_detection import (
    TargetDetectionArtifact,
    TargetDetectionFrame,
    write_target_detection,
)
from camera_rig.calibration.fixed.calibrator import FixedCameraCalibrator
from camera_rig.calibration.fixed.config import (
    FixedCalibrationConfig,
    FixedSolverThresholds,
)
from camera_rig.calibration.pose import (
    PlanarPoseEstimator,
    UncertaintyValidatedThresholds,
    project_points_px,
)
from camera_rig.config.models import (
    CameraConfig,
    CameraSettings,
    CaptureSettings,
    StreamSettings,
)
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ArtifactError, ContractError, SchemaValidationError
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.timestamps import SingleDeviceSyncReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.provision.config import (
    ProvisionAcquisitionSettings,
    ProvisionConfig,
    ProvisionTargetSettings,
)
from camera_rig.provision.preflight import (
    _validate_evidence_destination,
    run_fixed_provision_preflight,
)
from camera_rig.provision.workflow import (
    ProvisionWorkflowDependencies,
    run_fixed_provision_workflow,
)
from camera_rig.targets.io import load_target, validate_target_artifact
from camera_rig.targets.observation import TargetObservation

pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")
pytestmark = pytest.mark.charuco


def test_preflight_evidence_requires_private_nonoverlapping_local_path(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match=r"private \.local"):
        _validate_evidence_destination(
            tmp_path / "tracked-evidence",
            tmp_path / "report.json",
            tmp_path / "overlays",
        )
    evidence = tmp_path / ".local" / "evidence"
    with pytest.raises(ArtifactError, match="must not overlap"):
        _validate_evidence_destination(
            evidence,
            evidence / "report.json",
            tmp_path / "overlays",
        )


_WIDTH = 64
_HEIGHT = 48
_SERIAL = "synthetic-fixed-001"
_STREAMS = ("color", "depth", "ir_left", "ir_right")


def _profile(name: str) -> StreamProfile:
    formats = {"color": "rgb8", "depth": "z16", "ir_left": "y8", "ir_right": "y8"}
    return StreamProfile(name, _WIDTH, _HEIGHT, 30, formats[name])


def _intrinsics(name: str) -> CameraIntrinsics:
    return CameraIntrinsics(
        frame=f"head/{name}_optical",
        width=_WIDTH,
        height=_HEIGHT,
        fx=80.0,
        fy=81.0,
        cx=31.5,
        cy=23.5,
        distortion_model="none",
    )


def _factory(*, connected: bool = True) -> FactoryCalibration:
    transforms: list[RigidTransform] = []
    targets = ("color", "depth", "ir_right") if connected else ("depth", "ir_right")
    for name in targets:
        transforms.append(
            RigidTransform(
                "head/ir_left_optical",
                f"head/{name}_optical",
                np.eye(4, dtype=np.float64),
            )
        )
    return FactoryCalibration(
        device=CameraDeviceInfo(
            driver="synthetic",
            camera_name="head",
            expected_model="D435i",
            reported_model="D435i",
            serial=_SERIAL,
        ),
        stream_profiles={name: _profile(name) for name in _STREAMS},
        intrinsics={name: _intrinsics(name) for name in _STREAMS},
        internal_transforms=tuple(transforms),
        depth_scale_m_per_unit=0.001,
    )


def _config(target_root: Path) -> ProvisionConfig:
    target_path = target_root / "target_spec.json"
    target = validate_target_artifact(target_path)
    camera = CameraConfig(
        camera=CameraSettings(
            name="head",
            driver="synthetic",
            expected_model="D435i",
            serial=_SERIAL,
            output_reference_stream="ir_left",
        ),
        streams={name: StreamSettings(True, _profile(name)) for name in _STREAMS},
        capture=CaptureSettings(
            warmup_frames=7,
            timeout_ms=100,
            copy_frames=True,
            required_streams=_STREAMS,
        ),
    )
    fixed = FixedCalibrationConfig(
        workspace_frame="workspace",
        target_frame=target.target_frame,
        T_workspace_from_target=RigidTransform(
            target.target_frame, "workspace", np.eye(4, dtype=np.float64)
        ),
        detection_stream="color",
        reference_stream="ir_left",
        solver=FixedSolverThresholds(
            method="ippe",
            refinement="lm",
            minimum_corners_per_frame=12,
            minimum_accepted_frames=50,
            minimum_accepted_ratio=0.9,
            maximum_frame_rmse_px=0.5,
            maximum_frame_p95_px=1.0,
            maximum_pose_translation_p95_mm=3.0,
            maximum_pose_rotation_p95_deg=0.3,
            maximum_split_translation_delta_mm=2.0,
            maximum_split_rotation_delta_deg=0.2,
            pose_outlier_translation_mm=5.0,
            pose_outlier_rotation_deg=0.5,
        ),
        native_depth_check=True,
    )
    return ProvisionConfig(
        camera_config=camera,
        fixed_calibration_config=fixed,
        acquisition=ProvisionAcquisitionSettings(300, 60),
        target=ProvisionTargetSettings(
            artifact_reference="target/target_spec.json",
            artifact_path=target_path,
            expected_sha256=target.artifact_sha256,
            detection_stream="color",
        ),
        source_path=target_root / "provision.yaml",
    )


@dataclass
class _FakeSession:
    config: CameraConfig
    invalid_stream_quality: bool = False
    enter_count: int = 0
    exit_count: int = 0
    capture_count: int = 0

    def __enter__(self) -> _FakeSession:
        self.enter_count += 1
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exit_count += 1

    def capture(self) -> CameraFrame:
        index = self.capture_count
        self.capture_count += 1
        x = np.arange(_WIDTH, dtype=np.uint16)[None, :]
        y = np.arange(_HEIGHT, dtype=np.uint16)[:, None]
        gray = ((x + 2 * y + index) % 251).astype(np.uint8)
        color = np.stack((gray, np.roll(gray, 1, axis=1), np.roll(gray, 1, axis=0)), axis=2)
        depth_value = 0 if self.invalid_stream_quality else 720
        arrays = {
            "color": color,
            "depth": np.full((_HEIGHT, _WIDTH), depth_value, dtype=np.uint16),
            "ir_left": gray,
            "ir_right": ((3 * x + y + index + 7) % 251).astype(np.uint8),
        }
        timestamp = index * 33_333_333
        streams = {
            name: StreamFrame(
                name,
                arrays[name],
                index,
                sensor_timestamp_ns=timestamp,
                timestamp_domain="synthetic_hardware_clock",
                original_timestamp=timestamp / 1_000_000.0,
            )
            for name in _STREAMS
        }
        return CameraFrame(
            camera_name=self.config.camera.name,
            serial=self.config.camera.serial,
            streams=streams,
            host_receive_timestamp_ns=timestamp,
            sync_report=SingleDeviceSyncReport(
                valid=True,
                comparable_streams=_STREAMS,
                max_skew_ns=0,
                per_stream_skew_ns={name: 0 for name in _STREAMS},
                frame_number_match=True,
            ),
        )


def _known_pose(target_frame: str, *, x_offset_m: float = 0.0) -> RigidTransform:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [-0.105 + x_offset_m, 0.075, 0.72]
    return RigidTransform(target_frame, "head/color_optical", matrix)


def _statistics(value: float) -> dict[str, float]:
    return {"minimum": value, "median": value, "maximum": value, "mean": value}


@dataclass
class _FakeDetector:
    mode: str = "success"
    call_count: int = 0
    target_path: Path | None = None
    policy: str | None = None

    def __call__(
        self,
        *,
        target_path: Path,
        capture_path: Path,
        stream: str,
        report_path: Path,
        overlays_path: Path,
        policy: str,
    ) -> TargetDetectionArtifact:
        self.call_count += 1
        self.target_path = target_path
        self.policy = policy
        target = load_target(target_path)
        target_sha = "e" * 64 if self.mode == "wrong_detection" else target.artifact_sha256
        point_count = 3 if self.mode == "pnp" else len(target.corner_points)
        point_ids = tuple(point_id for point_id, _point in target.corner_points[:point_count])
        object_points = target.object_points_for(point_ids)
        frames: list[TargetDetectionFrame] = []
        overlay_by_index = {0: "best.png", 30: "median.png", 59: "worst.png"}
        for index in range(60):
            x_offset = 0.012 if self.mode == "repeatability" and index % 2 else 0.0
            pixels = project_points_px(
                object_points,
                _known_pose(target.target_frame, x_offset_m=x_offset),
                _intrinsics(stream),
            )
            if self.mode == "reprojection" and index % 5 == 0:
                pixels = pixels + np.random.default_rng(4000 + index).normal(
                    0.0, 0.65, pixels.shape
                )
            observation = TargetObservation(
                plugin_name=target.plugin,
                target_frame=target.target_frame,
                point_ids=point_ids,
                image_points_px=pixels,
                object_points_m=object_points,
                image_size=(_WIDTH, _HEIGHT),
                quality=QualityReport(True),
                metadata={"target_spec_sha256": target_sha},
            )
            frames.append(
                TargetDetectionFrame(
                    frame_index=index,
                    success=True,
                    observation=observation,
                    overlay=overlay_by_index.get(index),
                )
            )
        passed = self.mode != "r6_quality"
        checks = {
            "frame_count_is_60": passed,
            "success_ratio_at_least_0_95": True,
            "median_corners_at_least_20": True,
            "median_corner_fraction_at_least_0_80": True,
            "median_coverage_at_least_0_05": True,
            "median_jitter_at_most_0_5_px": True,
            "p95_jitter_at_most_1_0_px": True,
        }
        artifact = TargetDetectionArtifact(
            target_spec_sha256=target_sha,
            capture_manifest_sha256=sha256_file(capture_path / "manifest.json"),
            stream=stream,
            frame_count=60,
            per_frame=tuple(frames),
            aggregate={
                "success_ratio": 1.0,
                "detected_marker_count": _statistics(17.0),
                "detected_charuco_corner_count": _statistics(float(point_count)),
                "corner_fraction": _statistics(1.0),
                "coverage_ratio": _statistics(0.1),
                "temporal_jitter": {
                    "minimum_occurrences": 2,
                    "eligible_corner_count": point_count,
                    "median_radial_std_px": 0.01,
                    "p95_radial_std_px": 0.02,
                    "per_corner": [],
                },
            },
            acceptance={
                "passed": passed,
                "policy": "pose_validated" if self.mode == "wrong_policy" else policy,
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
            selected_overlays={"best": 0, "median_quality": 30, "worst_accepted": 59},
            software={"camera_rig_version": "0.3.0", "opencv_version": "synthetic"},
        )
        overlays_path.mkdir(parents=True)
        for filename in overlay_by_index.values():
            Image.new("RGB", (_WIDTH, _HEIGHT), color=(32, 64, 96)).save(
                overlays_path / filename,
                format="PNG",
            )
        write_target_detection(report_path, artifact)
        return artifact


def _dependencies(
    config: ProvisionConfig,
    *,
    mode: str = "success",
) -> tuple[ProvisionWorkflowDependencies, _FakeSession, _FakeDetector]:
    session = _FakeSession(
        config.camera_config,
        invalid_stream_quality=mode == "stream_quality",
    )
    detector = _FakeDetector(mode)
    dependencies = ProvisionWorkflowDependencies(
        session_factory=lambda _supplied: session,
        factory_extractor=lambda _supplied: _factory(connected=mode != "chain"),
        target_detection_runner=detector,
    )
    return dependencies, session, detector


def test_fixed_provision_workflow_uses_one_session_and_stages_modular_artifacts(
    generated_charuco_target: Path,
    tmp_path: Path,
) -> None:
    config = _config(generated_charuco_target)
    dependencies, session, detector = _dependencies(config)
    staging = tmp_path / "staging"

    result = run_fixed_provision_workflow(config, staging, dependencies=dependencies)

    assert (session.enter_count, session.exit_count, session.capture_count) == (1, 1, 300)
    assert len(result.selected_source_indices) == 60
    assert result.selected_source_indices[0] == 0
    assert result.selected_source_indices[-1] == 299
    assert len(set(result.selected_source_indices)) == 60
    assert detector.call_count == 1
    assert detector.target_path == staging / "target/artifact/target_spec.json"
    assert result.stream_validation.requested_frames == 300
    assert result.stream_validation.received_frames == 300
    assert result.fixed_calibration.quality.passed
    assert result.fixed_mount.quality.passed
    np.testing.assert_allclose(
        result.fixed_calibration.T_detection_from_target.matrix,
        _known_pose(config.fixed_calibration_config.target_frame).matrix,
        atol=8e-4,
    )
    for relative in result.files.values():
        assert (staging / relative).is_file()
    assert all((staging / relative).is_file() for relative in result.detection_overlays)
    assert all((staging / relative).is_file() for relative in result.fixed_overlays)
    assert len(result.detection_overlays) == len(result.fixed_overlays) == 3
    source_names = {path.name for path in generated_charuco_target.iterdir() if path.is_file()}
    staged_target = staging / "target/artifact"
    assert {path.name for path in staged_target.iterdir() if path.is_file()} == source_names
    for name in source_names:
        assert sha256_file(staged_target / name) == sha256_file(generated_charuco_target / name)
    manifest = load_json(staging / result.files["capture_manifest"])
    assert isinstance(manifest, dict)
    assert manifest["frame_count"] == 60
    capture_configuration = manifest["capture_configuration"]
    assert isinstance(capture_configuration, dict)
    assert capture_configuration["selected_source_indices"] == list(result.selected_source_indices)
    assert capture_configuration["warmup_owner"] == "camera_session_driver_open"
    capture_provenance = manifest["provenance"]
    assert isinstance(capture_provenance, dict)
    acquisition_id = result.factory_calibration.provenance["acquisition_id"]
    assert result.stream_validation.provenance["acquisition_id"] == acquisition_id
    assert capture_provenance["acquisition_id"] == acquisition_id
    assert result.stream_validation.provenance["factory_calibration_sha256"] == sha256_file(
        staging / result.files["factory_calibration"]
    )
    assert capture_provenance["stream_validation_sha256"] == sha256_file(
        staging / result.files["stream_validation"]
    )


def test_uncertainty_hold_preflight_retains_evidence_and_blocks_publication(
    generated_charuco_target: Path,
    tmp_path: Path,
) -> None:
    base = _config(generated_charuco_target)
    config = replace(
        base,
        target=replace(base.target, detection_policy="uncertainty_validated"),
    )
    dependencies, _session, _detector = _dependencies(config)
    permissive = UncertaintyValidatedThresholds(
        maximum_frame_translation_worst_std_mm=1000.0,
        maximum_frame_rotation_worst_std_deg=1000.0,
        maximum_final_translation_worst_std_mm=1000.0,
        maximum_final_rotation_worst_std_deg=1000.0,
        maximum_scaled_condition_number=1e9,
        ambiguity_material_translation_mm=1e9,
        ambiguity_material_rotation_deg=1e9,
    )
    dependencies = replace(
        dependencies,
        fixed_calibrator_factory=lambda: FixedCameraCalibrator(
            PlanarPoseEstimator(permissive), permissive
        ),
    )
    evidence_root = tmp_path / ".local" / "private-evidence"
    preflight = run_fixed_provision_preflight(
        config,
        report=tmp_path / "candidate.json",
        overlays=tmp_path / "candidate-overlays",
        evidence_root=evidence_root,
        dependencies=dependencies,
    )
    assert preflight["status"] == "FAIL"
    assert preflight["would_publish_fixed_provision"] is False
    assert "UNCERTAINTY_VALIDATED_PRESET_NOT_RELEASED" in preflight["failure_reasons"]
    assert preflight["candidate_numerical_decision"] == "NOT_RUN"
    assert preflight["publication_eligibility"] == {
        "eligible": False,
        "release_state": "HOLD",
        "reason": "UNCERTAINTY_VALIDATED_PRESET_NOT_RELEASED",
    }
    assert (evidence_root / "capture/calibration_snapshot/manifest.json").is_file()
    assert (evidence_root / "target/detection_report.json").is_file()


@pytest.mark.parametrize(
    ("mode", "message", "detector_calls"),
    [
        ("stream_quality", "raw stream validation failed", 0),
        ("wrong_detection", "different target artifact", 1),
        ("wrong_policy", "policy differs", 1),
        ("r6_quality", "R6 target-detection acceptance failed", 1),
        ("pnp", "no frame passing the pose frame gates", 1),
        ("repeatability", "fixed calibration quality failed", 1),
        ("chain", "factory transform graph validation failed", 0),
    ],
)
def test_fixed_provision_workflow_fails_closed(
    generated_charuco_target: Path,
    tmp_path: Path,
    mode: str,
    message: str,
    detector_calls: int,
) -> None:
    config = _config(generated_charuco_target)
    dependencies, session, detector = _dependencies(config, mode=mode)
    staging = tmp_path / mode

    with pytest.raises(ContractError, match=message):
        run_fixed_provision_workflow(config, staging, dependencies=dependencies)

    assert (session.enter_count, session.exit_count, session.capture_count) == (1, 1, 300)
    assert detector.call_count == detector_calls
    assert not (staging / "calibration/fixed_calibration.json").exists()
    if mode == "repeatability":
        assert (staging / "calibration/fixed_calibration.failed.json").is_file()


def test_live_preflight_reuses_workflow_and_never_publishes(
    generated_charuco_target: Path,
    tmp_path: Path,
) -> None:
    config = _config(generated_charuco_target)
    dependencies, session, detector = _dependencies(config)
    report = tmp_path / "preflight.json"
    overlays = tmp_path / "preflight-overlays"

    value = run_fixed_provision_preflight(
        config,
        report=report,
        overlays=overlays,
        dependencies=dependencies,
    )

    assert value["schema_version"] == "camera-rig.fixed-provision-preflight.v1"
    assert value["status"] == "PASS"
    assert value["would_publish_fixed_provision"] is True
    assert value["evaluation_core"] == "run_fixed_provision_workflow"
    assert value["publication"] == {
        "camera_bundle_written": False,
        "fixed_provision_written": False,
        "canonical_output_modified": False,
    }
    assert value["raw_stream"]["status"] == "PASS"  # type: ignore[index]
    assert value["fixed_pose_frames"]["frame_gate_accepted"] == 60  # type: ignore[index]
    assert len(value["per_frame"]) == 60  # type: ignore[arg-type]
    assert report.is_file()
    assert (overlays / "fixed_calibration").is_dir()
    assert not any(path.name == "camera_bundle.json" for path in tmp_path.rglob("*"))
    assert (session.enter_count, session.exit_count, session.capture_count) == (1, 1, 300)
    assert detector.call_count == 1
    invalid = deepcopy(value)
    invalid["pose_policy"] = "uncertainty_validated"
    invalid["observability"] = {"status": "NOT_EVALUATED"}
    invalid["final"]["final_pose_observability"] = None  # type: ignore[index]
    with pytest.raises(SchemaValidationError):
        validate_against_named_schema(
            invalid,  # type: ignore[arg-type]
            "fixed_provision_preflight.v1.schema.json",
        )


def test_live_preflight_rolls_back_overlays_when_report_publication_fails(
    generated_charuco_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(generated_charuco_target)
    dependencies, _session, _detector = _dependencies(config)
    report = tmp_path / "preflight.json"
    overlays = tmp_path / "preflight-overlays"
    monkeypatch.setattr(
        "camera_rig.provision.preflight.atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic report failure")),
    )
    with pytest.raises(OSError, match="synthetic report failure"):
        run_fixed_provision_preflight(
            config,
            report=report,
            overlays=overlays,
            dependencies=dependencies,
        )
    assert not report.exists()
    assert not overlays.exists()


def test_live_preflight_and_actual_workflow_have_offline_deterministic_parity(
    generated_charuco_target: Path,
    tmp_path: Path,
) -> None:
    config = _config(generated_charuco_target)
    actual_dependencies, _actual_session, _actual_detector = _dependencies(config)
    actual = run_fixed_provision_workflow(
        config,
        tmp_path / "actual-evaluation",
        dependencies=actual_dependencies,
    )
    preflight_dependencies, _preflight_session, _preflight_detector = _dependencies(config)
    preflight = run_fixed_provision_preflight(
        config,
        report=tmp_path / "preflight-parity.json",
        overlays=tmp_path / "preflight-parity-overlays",
        dependencies=preflight_dependencies,
    )

    actual_decisions = [
        (
            item["frame_index"],
            item["accepted"],
            item["failure_reasons"],
            item["reprojection_decision"],
        )
        for item in actual.fixed_calibration.per_frame_pose_summary
    ]
    preflight_decisions = [
        (
            item["frame_index"],
            item["accepted"],
            item["failure_reasons"],
            item["reprojection_decision"],
        )
        for item in preflight["per_frame"]  # type: ignore[union-attr]
    ]
    assert preflight_decisions == actual_decisions
    assert preflight["final"]["decision"] == "WOULD_PASS"  # type: ignore[index]
    assert actual.fixed_calibration.quality.passed is True


@pytest.mark.parametrize("mode", ["reprojection", "repeatability"])
def test_live_preflight_and_actual_workflow_have_failed_final_decision_parity(
    generated_charuco_target: Path,
    tmp_path: Path,
    mode: str,
) -> None:
    config = _config(generated_charuco_target)
    actual_dependencies, _actual_session, _actual_detector = _dependencies(config, mode=mode)
    actual = run_fixed_provision_workflow(
        config,
        tmp_path / f"{mode}-actual",
        dependencies=actual_dependencies,
        allow_failed_quality=True,
    )
    preflight_dependencies, _preflight_session, _preflight_detector = _dependencies(
        config, mode=mode
    )
    preflight = run_fixed_provision_preflight(
        config,
        report=tmp_path / f"{mode}-parity.json",
        overlays=tmp_path / f"{mode}-parity-overlays",
        dependencies=preflight_dependencies,
    )

    actual_decisions = [
        (
            item["frame_index"],
            item["accepted"],
            item["failure_reasons"],
            item["reprojection_decision"],
        )
        for item in actual.fixed_calibration.per_frame_pose_summary
    ]
    preflight_decisions = [
        (
            item["frame_index"],
            item["accepted"],
            item["failure_reasons"],
            item["reprojection_decision"],
        )
        for item in preflight["per_frame"]  # type: ignore[union-attr]
    ]
    assert preflight_decisions == actual_decisions
    assert preflight["failure_reasons"] == list(actual.fixed_calibration.quality.failure_reasons)
    assert preflight["final"]["decision"] == "WOULD_FAIL"  # type: ignore[index]
    assert actual.fixed_calibration.quality.passed is False


def test_existing_physical_native_depth_skip_fails_actual_and_preflight_equally(
    generated_charuco_target: Path,
    tmp_path: Path,
) -> None:
    config = _config(generated_charuco_target)

    class ExistingPhysicalCalibrator:
        def calibrate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["print_provenance"] = {
                "source_type": "existing_physical",
                "physical_measurement_sha256": "d" * 64,
                "geometry_policy": "registered physical target",
            }
            kwargs["native_depth_evaluator"] = lambda _pose, _indices: {
                "status": "SKIPPED_WITH_WARNING",
                "warning": "synthetic unsupported projection",
            }
            return FixedCameraCalibrator().calibrate(*args, **kwargs)

    actual_dependencies, _session, _detector = _dependencies(config)
    actual_dependencies = replace(
        actual_dependencies,
        fixed_calibrator_factory=ExistingPhysicalCalibrator,
    )
    actual = run_fixed_provision_workflow(
        config,
        tmp_path / "existing-actual",
        dependencies=actual_dependencies,
        allow_failed_quality=True,
    )
    assert actual.fixed_calibration.quality.passed is False
    assert "native_depth_sanity" in actual.fixed_calibration.quality.failure_reasons

    preflight_dependencies, _session, _detector = _dependencies(config)
    preflight_dependencies = replace(
        preflight_dependencies,
        fixed_calibrator_factory=ExistingPhysicalCalibrator,
    )
    preflight = run_fixed_provision_preflight(
        config,
        report=tmp_path / "existing-preflight.json",
        overlays=tmp_path / "existing-preflight-overlays",
        dependencies=preflight_dependencies,
    )
    assert preflight["status"] == "FAIL"
    assert preflight["would_publish_fixed_provision"] is False
    assert preflight["failure_reasons"] == list(actual.fixed_calibration.quality.failure_reasons)


def test_live_preflight_and_actual_workflow_have_raw_failure_parity(
    generated_charuco_target: Path,
    tmp_path: Path,
) -> None:
    config = _config(generated_charuco_target)
    actual_staging = tmp_path / "stream-actual"
    actual_dependencies, _actual_session, _actual_detector = _dependencies(
        config, mode="stream_quality"
    )
    with pytest.raises(ContractError, match="raw stream validation failed"):
        run_fixed_provision_workflow(
            config,
            actual_staging,
            dependencies=actual_dependencies,
        )
    actual_raw = load_json(actual_staging / "reports/stream_validation.json")
    assert isinstance(actual_raw, dict)
    actual_quality = actual_raw["quality"]
    assert isinstance(actual_quality, dict)

    preflight_dependencies, _preflight_session, _preflight_detector = _dependencies(
        config, mode="stream_quality"
    )
    preflight = run_fixed_provision_preflight(
        config,
        report=tmp_path / "stream-parity.json",
        overlays=tmp_path / "stream-parity-overlays",
        dependencies=preflight_dependencies,
    )
    assert preflight["raw_stream"] == {
        "status": actual_raw["status"],
        "metrics": actual_quality["metrics"],
        "failure_reasons": actual_quality["failure_reasons"],
    }


@pytest.mark.parametrize("mode", ["stream_quality", "reprojection", "repeatability"])
def test_live_preflight_persists_fail_closed_diagnostics(
    generated_charuco_target: Path,
    tmp_path: Path,
    mode: str,
) -> None:
    config = _config(generated_charuco_target)
    dependencies, _session, _detector = _dependencies(config, mode=mode)
    value = run_fixed_provision_preflight(
        config,
        report=tmp_path / f"{mode}.json",
        overlays=tmp_path / f"{mode}-overlays",
        dependencies=dependencies,
    )
    assert value["status"] == "FAIL"
    assert value["would_publish_fixed_provision"] is False
    assert value["publication"]["fixed_provision_written"] is False  # type: ignore[index]
    if mode == "stream_quality":
        assert value["raw_stream"]["status"] == "FAIL"  # type: ignore[index]
        assert value["target"]["status"] == "NOT_EVALUATED"  # type: ignore[index]
        assert value["fixed_pose_frames"]["status"] == "NOT_EVALUATED"  # type: ignore[index]
    else:
        assert value["raw_stream"]["status"] == "PASS"  # type: ignore[index]
        assert value["fixed_pose_frames"]["status"] == "EVALUATED"  # type: ignore[index]
        assert value["final"]["decision"] == "WOULD_FAIL"  # type: ignore[index]
        if mode == "reprojection":
            counts = value["fixed_pose_frames"]["failure_reason_counts"]  # type: ignore[index]
            assert counts["frame_reprojection_rmse_exceeded"] > 0  # type: ignore[index]


@pytest.mark.parametrize(
    ("failure_mode", "expected_reason"),
    [
        ("observability", "POSE_CONDITION_NUMBER_EXCEEDED"),
        ("ambiguity", "POSE_AMBIGUOUS"),
    ],
)
def test_live_preflight_reports_uncertainty_frame_gate_failures(
    generated_charuco_target: Path,
    tmp_path: Path,
    failure_mode: str,
    expected_reason: str,
) -> None:
    base = _config(generated_charuco_target)
    config = replace(
        base,
        target=replace(base.target, detection_policy="uncertainty_validated"),
    )
    dependencies, _session, _detector = _dependencies(config)
    thresholds = UncertaintyValidatedThresholds()
    if failure_mode == "observability":
        thresholds = replace(thresholds, maximum_scaled_condition_number=1.0)
        pose_estimator = PlanarPoseEstimator(thresholds)
    else:

        class ForcedAmbiguousEstimator(PlanarPoseEstimator):
            def estimate(self, observation, intrinsics):  # type: ignore[no-untyped-def]
                estimate = super().estimate(observation, intrinsics)
                ambiguity = replace(
                    estimate.observability.candidate_ambiguity,
                    materially_distinct=True,
                    statistically_competitive=True,
                    ambiguous=True,
                )
                observability = replace(
                    estimate.observability,
                    candidate_ambiguity=ambiguity,
                    passed=False,
                    failure_reasons=("POSE_AMBIGUOUS",),
                )
                return replace(estimate, observability=observability)

        pose_estimator = ForcedAmbiguousEstimator()
    calibrator = FixedCameraCalibrator(pose_estimator)
    assert calibrator.uncertainty_thresholds == pose_estimator.observability_thresholds
    dependencies = replace(
        dependencies,
        fixed_calibrator_factory=lambda: calibrator,
    )
    actual_staging = tmp_path / f"{failure_mode}-actual"
    with pytest.raises(ContractError, match="no frame passing the pose frame gates"):
        run_fixed_provision_workflow(config, actual_staging, dependencies=dependencies)
    actual = load_json(actual_staging / "calibration/fixed_calibration.frame_gate_failed.json")
    assert isinstance(actual, dict)
    actual_per_frame = actual["per_frame_pose_summary"]

    preflight_dependencies, _session, _detector = _dependencies(config)
    preflight_dependencies = replace(
        preflight_dependencies,
        fixed_calibrator_factory=lambda: calibrator,
    )
    value = run_fixed_provision_preflight(
        config,
        report=tmp_path / f"{failure_mode}.json",
        overlays=tmp_path / f"{failure_mode}-overlays",
        dependencies=preflight_dependencies,
    )
    assert value["status"] == "FAIL"
    assert value["final"]["status"] == "NOT_EVALUATED"  # type: ignore[index]
    counts = value["fixed_pose_frames"]["failure_reason_counts"]  # type: ignore[index]
    assert counts.get(expected_reason) == 60, counts  # type: ignore[union-attr]
    assert value["per_frame"] == actual_per_frame
