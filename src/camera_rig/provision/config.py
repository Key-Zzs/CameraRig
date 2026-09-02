"""Strict one-YAML fixed-camera provisioning configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Final

import numpy as np
import yaml

from camera_rig.artifacts.hashing import sha256_bytes
from camera_rig.artifacts.io import JsonValue, json_safe
from camera_rig.calibration.fixed.config import (
    FIXED_CALIBRATION_CONFIG_SCHEMA_VERSION,
    FixedCalibrationConfig,
)
from camera_rig.config.loader import validate_camera_config_data
from camera_rig.config.models import CONFIG_SCHEMA_VERSION, CameraConfig
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import ArtifactError, ConfigurationError, ContractError, TransformError

FIXED_PROVISION_CONFIG_SCHEMA_VERSION: Final = "camera-rig.fixed-provision.v1"
FIXED_PROVISION_STREAM_VALIDATION_FRAMES: Final = 300
FIXED_PROVISION_CALIBRATION_FRAMES: Final = 60
REQUIRED_FIXED_STREAMS: Final = frozenset({"color", "depth", "ir_left", "ir_right"})


@dataclass(frozen=True)
class ProvisionAcquisitionSettings:
    """Frame counts for the single live acquisition."""

    stream_validation_frames: int
    calibration_frames: int

    def __post_init__(self) -> None:
        if self.calibration_frames != FIXED_PROVISION_CALIBRATION_FRAMES:
            raise ContractError(f"calibration_frames must be {FIXED_PROVISION_CALIBRATION_FRAMES}")
        if self.stream_validation_frames != FIXED_PROVISION_STREAM_VALIDATION_FRAMES:
            raise ContractError(
                f"stream_validation_frames must be {FIXED_PROVISION_STREAM_VALIDATION_FRAMES}"
            )


@dataclass(frozen=True)
class ProvisionTargetSettings:
    """Pinned target identity and YAML-relative runtime path."""

    artifact_reference: str
    artifact_path: Path
    expected_sha256: str
    detection_stream: str
    detection_policy: str = "legacy_strict"

    def __post_init__(self) -> None:
        if self.artifact_path.name != "target_spec.json":
            raise ContractError("target.artifact must name target_spec.json")
        if len(self.expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.expected_sha256
        ):
            raise ContractError("target.expected_sha256 must be a lowercase SHA-256 digest")
        if self.detection_policy not in {
            "legacy_strict",
            "pose_validated",
            "uncertainty_validated",
        }:
            raise ContractError("target.detection_policy is unsupported")


@dataclass(frozen=True)
class ProvisionConfig:
    """Typed composition of existing camera and fixed-calibration contracts."""

    camera_config: CameraConfig
    fixed_calibration_config: FixedCalibrationConfig
    acquisition: ProvisionAcquisitionSettings
    target: ProvisionTargetSettings
    source_path: Path
    schema_version: str = FIXED_PROVISION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FIXED_PROVISION_CONFIG_SCHEMA_VERSION:
            raise ContractError("unsupported fixed provision config schema")
        enabled = {
            name for name, settings in self.camera_config.streams.items() if settings.enabled
        }
        fixed = self.fixed_calibration_config
        if fixed.detection_stream not in enabled:
            raise ContractError("target detection stream must be enabled")
        if fixed.reference_stream not in enabled:
            raise ContractError("fixed calibration reference stream must be enabled")
        if fixed.native_depth_check and "depth" not in enabled:
            raise ContractError("native depth sanity requires the depth stream to be enabled")
        if enabled != REQUIRED_FIXED_STREAMS:
            raise ContractError("fixed provisioning requires all four raw streams to be enabled")
        required = set(self.camera_config.capture.required_streams) or enabled
        if required != REQUIRED_FIXED_STREAMS:
            raise ContractError("fixed provisioning must capture all four raw streams")
        if not self.camera_config.capture.copy_frames:
            raise ContractError("fixed provisioning requires copy_frames to be true")
        if self.camera_config.camera.output_reference_stream != fixed.reference_stream:
            raise ContractError(
                "camera output_reference_stream must equal fixed calibration reference_stream"
            )
        if fixed.solver.minimum_accepted_frames > self.acquisition.calibration_frames:
            raise ContractError("minimum_accepted_frames exceeds calibration_frames")
        if not np.array_equal(fixed.T_workspace_from_target.matrix, np.eye(4, dtype=np.float64)):
            raise ContractError(
                "fixed provisioning requires T_workspace_from_target to be identity"
            )

    def extract_camera_config(self) -> CameraConfig:
        """Return the canonical existing CameraConfig contract."""
        return self.camera_config

    def extract_fixed_calibration_config(self) -> FixedCalibrationConfig:
        """Return the canonical existing FixedCalibrationConfig contract."""
        return self.fixed_calibration_config

    @classmethod
    def from_dict(cls, data: dict[str, object], *, source_path: Path) -> ProvisionConfig:
        camera_config = validate_camera_config_data(
            json_safe(
                {
                    "schema_version": CONFIG_SCHEMA_VERSION,
                    "camera": data["camera"],
                    "streams": data["streams"],
                    "capture": data["capture"],
                }
            )
        )
        provision = _object(data["provision"], "provision")
        target = _object(data["target"], "target")
        fixed = _object(data["fixed_calibration"], "fixed_calibration")
        target_reference = _relative_artifact_reference(
            _string(target["artifact"], "target.artifact")
        )
        fixed_config = FixedCalibrationConfig.from_dict(
            {
                "schema_version": FIXED_CALIBRATION_CONFIG_SCHEMA_VERSION,
                "workspace": data["workspace"],
                "camera": {
                    "detection_stream": target["detection_stream"],
                    "reference_stream": fixed["reference_stream"],
                },
                "solver": {
                    key: value
                    for key, value in fixed.items()
                    if key not in {"reference_stream", "native_depth_check"}
                },
                "diagnostics": {"native_depth_check": fixed["native_depth_check"]},
            }
        )
        acquisition = ProvisionAcquisitionSettings(
            stream_validation_frames=_int(
                provision["stream_validation_frames"], "provision.stream_validation_frames"
            ),
            calibration_frames=_int(
                provision["calibration_frames"], "provision.calibration_frames"
            ),
        )
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            camera_config=camera_config,
            fixed_calibration_config=fixed_config,
            acquisition=acquisition,
            target=ProvisionTargetSettings(
                artifact_reference=target_reference,
                artifact_path=(source_path.parent / target_reference).resolve(strict=False),
                expected_sha256=_string(target["expected_sha256"], "target.expected_sha256"),
                detection_stream=_string(target["detection_stream"], "target.detection_stream"),
                detection_policy=_string(
                    target.get("detection_policy", "legacy_strict"),
                    "target.detection_policy",
                ),
            ),
            source_path=source_path.resolve(strict=False),
        )


def load_provision_config(path: str | Path) -> ProvisionConfig:
    """Load the one-YAML contract without touching target files or camera hardware."""
    return load_provision_config_with_sha256(path)[0]


def load_provision_config_with_sha256(path: str | Path) -> tuple[ProvisionConfig, str]:
    """Load one immutable YAML byte snapshot and return its exact SHA-256 identity."""
    source = Path(path)
    try:
        raw = source.read_bytes()
        decoded: object = yaml.safe_load(raw.decode("utf-8"))
        value = json_safe(decoded)
    except (OSError, UnicodeError, yaml.YAMLError, ArtifactError) as error:
        raise ConfigurationError(f"could not load fixed provision config: {error}") from error
    return validate_provision_config_data(value, source_path=source), sha256_bytes(raw)


def validate_provision_config_data(value: JsonValue, *, source_path: str | Path) -> ProvisionConfig:
    """Strictly validate decoded provision data and reconstruct composed contracts."""
    validate_against_named_schema(value, "fixed_provision.v1.schema.json")
    if not isinstance(value, dict):
        raise ConfigurationError("fixed provision config root must be a mapping")
    try:
        return ProvisionConfig.from_dict(dict(value), source_path=Path(source_path))
    except (KeyError, TypeError, ValueError, ContractError, TransformError) as error:
        raise ConfigurationError(f"fixed provision config is invalid: {error}") from error


def _relative_artifact_reference(value: str) -> str:
    windows = PureWindowsPath(value)
    if (
        Path(value).is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in value
        or value.casefold().startswith("file://")
        or "://" in value
    ):
        raise ContractError("target.artifact must be a portable YAML-relative path")
    return value


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value
