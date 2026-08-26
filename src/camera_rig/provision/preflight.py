"""Read-only preflight for one-command fixed-camera provisioning."""

from __future__ import annotations

import importlib
from pathlib import Path

from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.core.errors import ArtifactError, MissingOptionalDependencyError
from camera_rig.provision.config import ProvisionConfig
from camera_rig.targets.io import validate_target_artifact


def preflight_fixed_provision(
    config: ProvisionConfig,
    *,
    output: str | Path,
    force: bool = False,
) -> dict[str, object]:
    """Validate every non-hardware input without opening the camera or writing output."""
    target = validate_target_artifact(config.target.artifact_path)
    if target.artifact_sha256 != config.target.expected_sha256:
        raise ArtifactError(
            "resolved target SHA does not match target.expected_sha256; "
            "do not regenerate or substitute the physical-board artifact"
        )
    if target.target_frame != config.fixed_calibration_config.target_frame:
        raise ArtifactError("target coordinate frame differs from the workspace contract")
    destination = Path(output)
    if destination.exists() and not force:
        raise ArtifactError(f"provision output already exists: {destination}")
    if destination.is_symlink():
        raise ArtifactError("provision output must not be a symlink")
    if destination.parent.exists() and destination.parent.is_symlink():
        raise ArtifactError("provision output parent must not be a symlink")
    if destination.exists() and force:
        from camera_rig.provision.validation import load_and_validate_fixed_provision

        try:
            load_and_validate_fixed_provision(destination)
        except ArtifactError as error:
            raise ArtifactError(
                "--force may replace only an existing validated fixed-provision artifact"
            ) from error
    _require_runtime_dependencies()
    return {
        "schema_version": config.schema_version,
        "mode": "dry-run-safe-preflight",
        "camera_driver": config.camera_config.camera.driver,
        "expected_model": config.camera_config.camera.expected_model,
        "enabled_streams": sorted(
            name for name, settings in config.camera_config.streams.items() if settings.enabled
        ),
        "stream_validation_frames": config.acquisition.stream_validation_frames,
        "calibration_frames": config.acquisition.calibration_frames,
        "selected_frame_policy": "deterministic_evenly_spaced",
        "target_artifact": config.target.artifact_reference,
        "target_sha256": target.artifact_sha256,
        "target_frame": target.target_frame,
        "workspace_frame": config.fixed_calibration_config.workspace_frame,
        "detection_stream": config.fixed_calibration_config.detection_stream,
        "target_detection_policy": config.target.detection_policy,
        "reference_stream": config.fixed_calibration_config.reference_stream,
        "output_exists": destination.exists(),
        "force": force,
        "camera_will_open": False,
        "final_artifact_will_be_created": False,
        "optional_dependencies": {
            "realsense": "available",
            "charuco": "available",
            "viz": "available",
        },
    }


def _require_runtime_dependencies() -> None:
    cv2_module()
    try:
        importlib.import_module("PIL")
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'fixed provisioning requires: pip install "camera-rig[viz]"'
        ) from error
    try:
        importlib.import_module("pyrealsense2")
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'fixed provisioning requires: pip install "camera-rig[realsense]"'
        ) from error
