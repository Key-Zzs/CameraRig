"""CameraRig command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from camera_rig.artifacts.validation import load_and_validate_bundle
from camera_rig.cli.commands.device import add_device_commands
from camera_rig.config.loader import load_config
from camera_rig.core.errors import CameraRigError
from camera_rig.version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="camera-rig",
        description="Single-camera acquisition, calibration, validation, and replay toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="show tracebacks for expected errors")
    commands = parser.add_subparsers(dest="command")

    add_device_commands(commands)

    config_parser = commands.add_parser("config", help="configuration operations")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    config_validate = config_commands.add_parser("validate", help="validate strict YAML")
    config_validate.add_argument("--config", type=Path, required=True, help="YAML config path")
    config_validate.set_defaults(handler=_validate_config)

    artifact_parser = commands.add_parser("artifact", help="artifact operations")
    artifact_commands = artifact_parser.add_subparsers(dest="artifact_command", required=True)
    artifact_validate = artifact_commands.add_parser("validate", help="validate CameraBundle JSON")
    artifact_validate.add_argument("--bundle", type=Path, required=True, help="bundle JSON path")
    artifact_validate.set_defaults(handler=_validate_artifact)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CameraRig command-line interface."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler = getattr(arguments, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(arguments))
    except CameraRigError as error:
        if arguments.debug:
            raise
        print(f"error: {error}", file=sys.stderr)
        return 2


def _validate_config(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    enabled = sum(settings.enabled for settings in config.streams.values())
    print(
        f"valid {config.schema_version}: camera={config.camera.name!r}, "
        f"serial={config.camera.serial!r}, enabled_streams={enabled}"
    )
    return 0


def _validate_artifact(arguments: argparse.Namespace) -> int:
    bundle = load_and_validate_bundle(arguments.bundle)
    print(
        f"valid {bundle.schema_version}: bundle_id={bundle.bundle_id!r}, status={bundle.status!r}"
    )
    return 0
