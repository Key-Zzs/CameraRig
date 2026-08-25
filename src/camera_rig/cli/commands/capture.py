"""Live raw-stream capture and validation commands."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from camera_rig.artifacts.factory_calibration import FactoryCalibrationArtifact
from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import atomic_write_json
from camera_rig.capture.session import CameraSession
from camera_rig.capture.snapshot import write_snapshot
from camera_rig.capture.validation import StreamValidationAccumulator
from camera_rig.config.loader import load_config
from camera_rig.core.errors import ContractError, FrameTimeoutError
from camera_rig.core.quality import QualityReport
from camera_rig.drivers.realsense.factory_calibration import extract_factory_calibration
from camera_rig.version import __version__


def add_capture_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser = commands.add_parser("capture", help="single-camera raw capture operations")
    subcommands = parser.add_subparsers(dest="capture_command", required=True)
    validate = subcommands.add_parser("validate-streams", help="validate raw active streams")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--frames", type=int, default=300)
    validate.add_argument("--report", type=Path, required=True)
    validate.set_defaults(handler=_validate_streams)

    snapshot = subcommands.add_parser("snapshot", help="persist raw frames for replay")
    snapshot.add_argument("--config", type=Path, required=True)
    snapshot.add_argument("--frames", type=int, default=1)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--force", action="store_true")
    snapshot.set_defaults(handler=_snapshot)
    return parser


def _validate_streams(arguments: argparse.Namespace) -> int:
    if arguments.frames < 2:
        raise ContractError("stream validation requires at least two frames")
    config = load_config(arguments.config)
    accumulator = StreamValidationAccumulator(config, arguments.frames)
    timeouts = 0
    with CameraSession.from_config(config) as camera:
        for _ in range(arguments.frames):
            try:
                accumulator.add(camera.capture())
            except FrameTimeoutError:
                timeouts += 1
                break
    report = accumulator.report(timeouts)
    atomic_write_json(arguments.report, report)
    print(
        f"stream validation: {report['status']} "
        f"({report['received_frames']}/{report['requested_frames']} frames)"
    )
    if report["status"] != "PASS":
        reasons = report["failure_reasons"]
        raise ContractError(f"raw stream validation failed: {reasons}")
    return 0


def _snapshot(arguments: argparse.Namespace) -> int:
    if arguments.frames < 1:
        raise ContractError("snapshot requires at least one frame")
    config = load_config(arguments.config)
    with CameraSession.from_config(config) as camera:
        calibration = extract_factory_calibration(camera.driver)
        frames = [camera.capture() for _ in range(arguments.frames)]
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    git_commit = _command_value(["git", "rev-parse", "HEAD"])
    config_hash = sha256_file(arguments.config)
    factory = FactoryCalibrationArtifact(
        created_at=created_at,
        calibration=calibration,
        quality=QualityReport(
            passed=True,
            metrics={
                "intrinsics_count": len(calibration.intrinsics),
                "internal_transform_count": len(calibration.internal_transforms),
            },
        ),
        provenance={
            "camera_rig_version": __version__,
            "git_commit": git_commit,
            "pyrealsense2_package_version": calibration.device.sdk_version or "unknown",
            "firmware_version": calibration.device.firmware_version or "unknown",
            "product_id": calibration.device.product_id or "unknown",
            "config_sha256": config_hash,
        },
    )
    manifest = write_snapshot(
        arguments.output,
        frames,
        factory,
        capture_summary={
            "output_reference_stream": config.camera.output_reference_stream,
            "copy_frames": config.capture.copy_frames,
            "requested_profiles": {
                name: settings.profile.to_dict()
                for name, settings in sorted(config.streams.items())
                if settings.enabled
            },
        },
        provenance={
            "camera_rig_version": __version__,
            "git_commit": git_commit,
            "config_sha256": config_hash,
        },
        include_previews=True,
        force=arguments.force,
    )
    print(f"snapshot: PASS ({manifest['frame_count']} frames)")
    return 0


def _command_value(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return "unknown"
    return " ".join(result.stdout.split()) or "unknown"
