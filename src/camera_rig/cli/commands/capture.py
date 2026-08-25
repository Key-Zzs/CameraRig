"""Live raw-stream capture and validation commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_rig.artifacts.io import atomic_write_json
from camera_rig.capture.session import CameraSession
from camera_rig.capture.validation import StreamValidationAccumulator
from camera_rig.config.loader import load_config
from camera_rig.core.errors import ContractError, FrameTimeoutError


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
