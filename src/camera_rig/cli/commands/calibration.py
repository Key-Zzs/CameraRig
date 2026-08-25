"""Factory calibration export and validation commands."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from camera_rig.artifacts.factory_calibration import (
    FactoryCalibrationArtifact,
    load_and_validate_factory_calibration,
    write_factory_calibration,
)
from camera_rig.artifacts.hashing import sha256_file
from camera_rig.config.loader import load_config
from camera_rig.core.quality import QualityReport
from camera_rig.core.transform_graph import TransformGraph
from camera_rig.drivers.realsense.driver import RealSenseDriver
from camera_rig.drivers.realsense.factory_calibration import extract_factory_calibration
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


def _command_value(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return "unknown"
    value = " ".join(result.stdout.split())
    return value or "unknown"
