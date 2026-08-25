"""Inspect and validate SDK-independent capture replay artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_rig.artifacts.capture_validation import validate_capture_artifact
from camera_rig.capture.replay import ReplayCameraSession


def add_replay_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = commands.add_parser("replay", help="offline raw capture replay")
    subcommands = parser.add_subparsers(dest="replay_command", required=True)
    inspect = subcommands.add_parser("inspect", help="inspect a capture manifest")
    inspect.add_argument("--artifact", type=Path, required=True)
    inspect.set_defaults(handler=_inspect)
    validate = subcommands.add_parser("validate", help="validate and replay all frames")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.set_defaults(handler=_validate)


def _inspect(arguments: argparse.Namespace) -> int:
    manifest = validate_capture_artifact(arguments.artifact)
    print(
        f"valid {manifest['schema_version']}: artifact_id={manifest['artifact_id']}, "
        f"frames={manifest['frame_count']}"
    )
    return 0


def _validate(arguments: argparse.Namespace) -> int:
    with ReplayCameraSession.from_artifact(arguments.artifact) as replay:
        count = replay.frame_count
        for _ in range(count):
            replay.capture()
    print(f"replay validation: PASS ({count} frames)")
    return 0
