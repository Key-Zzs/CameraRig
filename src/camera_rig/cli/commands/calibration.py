"""Factory calibration export and validation commands."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from camera_rig.artifacts.capture_validation import validate_capture_artifact
from camera_rig.artifacts.factory_calibration import (
    FactoryCalibrationArtifact,
    load_and_validate_factory_calibration,
    write_factory_calibration,
)
from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.target_detection import load_and_validate_target_detection
from camera_rig.calibration.fixed.artifact import (
    load_and_validate_fixed_calibration,
    write_fixed_calibration,
)
from camera_rig.calibration.fixed.calibrator import FixedCameraCalibrator
from camera_rig.calibration.fixed.config import load_fixed_config
from camera_rig.calibration.fixed.counterfactuals import evaluate_model_counterfactuals
from camera_rig.calibration.fixed.depth_sanity import evaluate_native_depth_sanity
from camera_rig.calibration.fixed.overlays import select_overlay_frames, write_fixed_pose_overlay
from camera_rig.capture.replay import ReplayCameraSession
from camera_rig.config.loader import load_config
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.frame import CameraFrame
from camera_rig.core.quality import QualityReport
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.drivers.realsense.driver import RealSenseDriver
from camera_rig.drivers.realsense.factory_calibration import extract_factory_calibration
from camera_rig.targets.io import validate_target_artifact
from camera_rig.version import __version__


def add_calibration_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = commands.add_parser("calibration", help="camera calibration operations")
    groups = parser.add_subparsers(dest="calibration_group", required=True)
    factory = groups.add_parser("factory", help="read factory calibration from active streams")
    factory_commands = factory.add_subparsers(dest="factory_command", required=True)

    export = factory_commands.add_parser("export", help="export active factory calibration")
    export.add_argument("--config", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(handler=_export_factory)

    validate = factory_commands.add_parser("validate", help="validate a factory artifact")
    validate.add_argument("--input", type=Path, required=True)
    validate.set_defaults(handler=_validate_factory)

    fixed = groups.add_parser("fixed", help="fixed-camera extrinsic calibration")
    fixed_commands = fixed.add_subparsers(dest="fixed_command", required=True)
    solve = fixed_commands.add_parser("solve", help="solve from a validated replay artifact")
    solve.add_argument("--config", type=Path, required=True)
    solve.add_argument("--capture", type=Path, required=True)
    solve.add_argument("--target", type=Path, required=True)
    solve.add_argument("--detection-report", type=Path, required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--overlays", type=Path, required=True)
    solve.set_defaults(handler=_solve_fixed)

    fixed_validate = fixed_commands.add_parser(
        "validate", help="validate a fixed-camera calibration artifact"
    )
    fixed_validate.add_argument("--input", type=Path, required=True)
    fixed_validate.set_defaults(handler=_validate_fixed)

    counterfactuals = groups.add_parser(
        "evaluate-model-counterfactuals",
        help="evaluate retained detections under wrong camera/target assumptions",
    )
    counterfactuals.add_argument("--detection-report", type=Path, required=True)
    counterfactuals.add_argument("--factory-calibration", type=Path, required=True)
    counterfactuals.add_argument("--output", type=Path, required=True)
    counterfactuals.set_defaults(handler=_evaluate_model_counterfactuals)


def _export_factory(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    driver = RealSenseDriver(config)
    with driver:
        calibration = extract_factory_calibration(driver)
    graph = TransformGraph()
    max_orthonormal_error = 0.0
    for transform in calibration.internal_transforms:
        graph.add(transform)
        rotation = transform.matrix[:3, :3]
        error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        max_orthonormal_error = max(max_orthonormal_error, error)
    reference = f"{config.camera.name}/{config.camera.output_reference_stream}_optical"
    for intrinsic in calibration.intrinsics.values():
        graph.resolve(reference, intrinsic.frame)
    quality = QualityReport(
        passed=True,
        metrics={
            "intrinsics_count": len(calibration.intrinsics),
            "internal_transform_count": len(calibration.internal_transforms),
            "max_rotation_orthonormal_error": max_orthonormal_error,
        },
        thresholds={"max_rotation_orthonormal_error": 1e-7},
    )
    artifact = FactoryCalibrationArtifact(
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        calibration=calibration,
        quality=quality,
        provenance={
            "camera_rig_version": __version__,
            "git_commit": _command_value(["git", "rev-parse", "HEAD"]),
            "system_realsense_cli_version": _command_value(["rs-enumerate-devices", "--version"]),
            "pyrealsense2_package_version": calibration.device.sdk_version or "unknown",
            "firmware_version": calibration.device.firmware_version or "unknown",
            "product_id": calibration.device.product_id or "unknown",
            "config_sha256": sha256_file(arguments.config),
        },
    )
    write_factory_calibration(arguments.output, artifact)
    print(
        "factory calibration: PASS "
        f"({len(calibration.intrinsics)} intrinsics, "
        f"{len(calibration.internal_transforms)} transforms)"
    )
    return 0


def _validate_factory(arguments: argparse.Namespace) -> int:
    artifact = load_and_validate_factory_calibration(arguments.input)
    calibration = artifact.calibration
    print(
        f"valid {artifact.schema_version}: {len(calibration.intrinsics)} intrinsics, "
        f"{len(calibration.internal_transforms)} transforms"
    )
    return 0


def _solve_fixed(arguments: argparse.Namespace) -> int:
    config = load_fixed_config(arguments.config)
    target = validate_target_artifact(arguments.target)
    manifest = validate_capture_artifact(arguments.capture)
    manifest_path = arguments.capture / "manifest.json"
    factory_relative = manifest.get("factory_calibration")
    if not isinstance(factory_relative, str):
        raise ArtifactError("capture manifest factory path must be a string")
    factory_path = arguments.capture / factory_relative
    factory = load_and_validate_factory_calibration(factory_path)
    detection = load_and_validate_target_detection(arguments.detection_report)
    _validate_target_observations(target, detection)
    frames = _read_replay_frames(arguments.capture)

    def depth_evaluator(final_pose: object, inlier_indices: tuple[int, ...]) -> dict[str, object]:
        from camera_rig.core.transforms import RigidTransform

        if not isinstance(final_pose, RigidTransform):
            raise ContractError("native depth evaluator received an invalid pose")
        if not config.native_depth_check:
            return {
                "status": "SKIPPED_WITH_WARNING",
                "warning": "native depth diagnostic disabled by strict configuration",
            }
        return evaluate_native_depth_sanity(
            target=target,
            calibration=factory.calibration,
            T_detection_from_target=final_pose,
            detection_stream=config.detection_stream,
            frames=frames,
            frame_indices=inlier_indices,
        )

    artifact = FixedCameraCalibrator().calibrate(
        config,
        detection,
        factory,
        target_spec_sha256=target.artifact_sha256,
        capture_manifest_sha256=sha256_file(manifest_path),
        factory_calibration_sha256=sha256_file(factory_path),
        target_detection_sha256=sha256_file(arguments.detection_report),
        print_provenance={
            "horizontal_print_scale": 0.997,
            "vertical_print_scale": 0.997,
            "maximum_observed_print_scale_error": 0.003,
            "geometry_policy": (
                "pose uses nominal persisted target geometry; print measurement is "
                "provenance and systematic-scale information"
            ),
        },
        native_depth_evaluator=depth_evaluator,
        provenance={
            "camera_rig_version": __version__,
            "git_commit": _command_value(["git", "rev-parse", "HEAD"]),
        },
    )
    if not artifact.quality.passed:
        failed_path = arguments.output.with_suffix(arguments.output.suffix + ".failed.json")
        from camera_rig.artifacts.io import atomic_write_json

        atomic_write_json(
            failed_path,
            {"status": "failed", "fixed_calibration": artifact.to_dict()},
        )
        raise ContractError(
            f"fixed calibration quality gates failed: {list(artifact.quality.failure_reasons)}"
        )
    overlay_files = _write_fixed_overlays(
        arguments.overlays,
        frames,
        detection,
        artifact,
        factory.calibration.intrinsics[config.detection_stream],
        target.board_width_m,
        target.board_height_m,
    )
    artifact = replace(
        artifact,
        provenance={
            **artifact.provenance,
            "overlay_files": overlay_files,
            "axis_overlay_review": "READY_FOR_AUTOMATED_REVIEW",
        },
    )
    write_fixed_calibration(arguments.output, artifact)
    global_reprojection = artifact.aggregate["reprojection"]
    if not isinstance(global_reprojection, dict) or not isinstance(
        global_reprojection.get("global"), dict
    ):
        raise ArtifactError("fixed calibration reprojection aggregate is invalid")
    metrics = global_reprojection["global"]
    release_label = (
        "NUMERICAL_PASS RELEASE_HOLD"
        if artifact.solver.get("pose_policy") == "uncertainty_validated"
        else "PASS"
    )
    print(
        f"fixed calibration: {release_label} "
        f"({artifact.aggregate['accepted_frames']}/{detection.frame_count} frames, "
        f"RMSE={metrics['rmse_px']:.4f}px)"
    )
    return 0


def _validate_fixed(arguments: argparse.Namespace) -> int:
    artifact = load_and_validate_fixed_calibration(arguments.input)
    release_label = (
        "numerical_quality=passed, release=HOLD"
        if artifact.solver.get("pose_policy") == "uncertainty_validated"
        else "quality=passed"
    )
    print(
        f"valid {artifact.schema_version}: {release_label}, "
        f"reference_frame={artifact.fixed_mount_calibration.camera_reference_frame!r}"
    )
    return 0


def _evaluate_model_counterfactuals(arguments: argparse.Namespace) -> int:
    from camera_rig.artifacts.io import atomic_write_json

    detection = load_and_validate_target_detection(arguments.detection_report)
    factory = load_and_validate_factory_calibration(arguments.factory_calibration)
    report = evaluate_model_counterfactuals(
        detection,
        factory,
        detection_report_sha256=sha256_file(arguments.detection_report),
        factory_calibration_sha256=sha256_file(arguments.factory_calibration),
    )
    atomic_write_json(arguments.output, report)
    counterfactual_values = report.get("counterfactuals")
    if not isinstance(counterfactual_values, list):
        raise ArtifactError("model counterfactual evaluation returned invalid variants")
    print(
        "model counterfactual evaluation: ANALYSIS_ONLY_NO_GROUND_TRUTH "
        f"({len(counterfactual_values)} variants)"
    )
    return 0


def _read_replay_frames(capture: Path) -> list[CameraFrame]:
    frames: list[CameraFrame] = []
    with ReplayCameraSession.from_artifact(capture) as replay:
        while True:
            frame = replay.poll_frame()
            if frame is None:
                return frames
            frames.append(frame)


def _validate_target_observations(target: object, detection: object) -> None:
    from camera_rig.artifacts.target_detection import TargetDetectionArtifact
    from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget

    if not isinstance(target, ResolvedCharucoTarget) or not isinstance(
        detection, TargetDetectionArtifact
    ):
        raise ContractError("fixed calibration target inputs are invalid")
    if detection.target_spec_sha256 != target.artifact_sha256:
        raise ContractError("target detection report is bound to a different target artifact")
    for frame in detection.per_frame:
        expected = target.object_points_for(frame.observation.point_ids)
        if not np.allclose(
            expected,
            frame.observation.object_points_m,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ContractError("target observation object points differ from persisted geometry")


def _write_fixed_overlays(
    root: Path,
    frames: list[CameraFrame],
    detection: object,
    artifact: object,
    intrinsics: object,
    board_width_m: float,
    board_height_m: float,
) -> list[str]:
    from camera_rig.artifacts.target_detection import TargetDetectionArtifact
    from camera_rig.calibration.fixed.artifact import FixedCalibrationArtifact
    from camera_rig.core.intrinsics import CameraIntrinsics

    if not isinstance(detection, TargetDetectionArtifact):
        raise ContractError("invalid target-detection artifact for overlays")
    if not isinstance(artifact, FixedCalibrationArtifact) or not isinstance(
        intrinsics, CameraIntrinsics
    ):
        raise ContractError("invalid fixed-calibration overlay inputs")
    selected = select_overlay_frames(artifact.per_frame_pose_summary)
    files: list[str] = []
    for label, frame_index in selected.items():
        filename = f"{label}_frame_{frame_index:06d}.png"
        stream = frames[frame_index].streams.get(detection.stream or "")
        if stream is None:
            raise ArtifactError("capture frame lacks the target-detection stream")
        write_fixed_pose_overlay(
            root / filename,
            image_rgb=np.asarray(stream.data, dtype=np.uint8),
            observation=detection.per_frame[frame_index].observation,
            T_camera_from_target=artifact.T_detection_from_target,
            intrinsics=intrinsics,
            board_width_m=board_width_m,
            board_height_m=board_height_m,
        )
        files.append(filename)
    if len(files) != 3:
        raise ArtifactError("fixed calibration requires best, median, and worst overlays")
    return files


def _command_value(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return "unknown"
    value = " ".join(result.stdout.split())
    return value or "unknown"
