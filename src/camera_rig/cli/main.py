"""CameraRig command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from camera_rig.version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="camera-rig",
        description="Single-camera acquisition, calibration, validation, and replay toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CameraRig command-line interface."""
    build_parser().parse_args(argv)
    return 0
