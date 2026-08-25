"""Versioned RealSense factory-calibration artifact and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

from camera_rig.artifacts.io import JsonValue, atomic_write_json, json_safe, load_json
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core._validation import require_non_empty, string_keyed_copy
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import (
    ArtifactError,
    ContractError,
    SchemaValidationError,
    TransformError,
)
from camera_rig.core.factory_calibration import FactoryCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform

FACTORY_CALIBRATION_SCHEMA_VERSION: Final = "camera-rig.factory-calibration.v1"


@dataclass(frozen=True)
class FactoryCalibrationArtifact:
    """Persisted active-profile factory calibration with portable provenance."""

    created_at: str
    calibration: FactoryCalibration
    quality: QualityReport
    provenance: dict[str, object]
    schema_version: str = field(default=FACTORY_CALIBRATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        require_non_empty(self.created_at, "created_at")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ContractError("created_at must be an ISO-8601 date-time") from error
        provenance = string_keyed_copy(self.provenance, "provenance")
        _reject_absolute_values(provenance, "provenance")
        object.__setattr__(self, "provenance", provenance)
        json_safe(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        calibration = self.calibration
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "device": calibration.device.to_dict(),
            "active_stream_profiles": {
                name: profile.to_dict()
                for name, profile in sorted(calibration.stream_profiles.items())
            },
            "intrinsics": {
                name: value.to_dict() for name, value in sorted(calibration.intrinsics.items())
            },
            "internal_transforms": [
                transform.to_dict() for transform in calibration.internal_transforms
            ],
            "depth_scale_m_per_unit": calibration.depth_scale_m_per_unit,
            "quality": self.quality.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FactoryCalibrationArtifact:
        if data.get("schema_version") != FACTORY_CALIBRATION_SCHEMA_VERSION:
            raise ContractError(f"schema_version must be {FACTORY_CALIBRATION_SCHEMA_VERSION!r}")
        profiles = _object(data["active_stream_profiles"], "active_stream_profiles")
        intrinsics = _object(data["intrinsics"], "intrinsics")
        transforms = _array(data["internal_transforms"], "internal_transforms")
        calibration = FactoryCalibration(
            device=CameraDeviceInfo.from_dict(_object(data["device"], "device")),
            stream_profiles={
                name: StreamProfile.from_dict(_object(value, f"active_stream_profiles.{name}"))
                for name, value in profiles.items()
            },
            intrinsics={
                name: CameraIntrinsics.from_dict(_object(value, f"intrinsics.{name}"))
                for name, value in intrinsics.items()
            },
            internal_transforms=tuple(
                RigidTransform.from_dict(_object(value, "internal_transforms[]"))
                for value in transforms
            ),
            depth_scale_m_per_unit=_number(
                data["depth_scale_m_per_unit"], "depth_scale_m_per_unit"
            ),
        )
        return cls(
            created_at=_string(data["created_at"], "created_at"),
            calibration=calibration,
            quality=QualityReport.from_dict(_object(data["quality"], "quality")),
            provenance=_object(data["provenance"], "provenance"),
        )


def write_factory_calibration(path: str | Path, artifact: FactoryCalibrationArtifact) -> None:
    """Atomically persist and immediately revalidate the factory artifact."""
    atomic_write_json(path, artifact.to_dict())
    load_and_validate_factory_calibration(path)


def validate_factory_calibration_data(value: JsonValue) -> FactoryCalibrationArtifact:
    """Schema-validate decoded JSON and reconstruct all typed contracts."""
    if not isinstance(value, dict):
        raise ArtifactError("factory calibration root must be a JSON object")
    try:
        validate_against_named_schema(value, "factory_calibration.v1.schema.json")
        return FactoryCalibrationArtifact.from_dict(dict(value))
    except SchemaValidationError as error:
        raise ArtifactError(str(error)) from error
    except (KeyError, TypeError, ValueError, ContractError, TransformError) as error:
        raise ArtifactError(f"factory calibration contract is invalid: {error}") from error


def load_and_validate_factory_calibration(path: str | Path) -> FactoryCalibrationArtifact:
    return validate_factory_calibration_data(load_json(path))


def _reject_absolute_values(value: object, path: str) -> None:
    if isinstance(value, str):
        if value.startswith("/") or value.casefold().startswith("file://"):
            raise ContractError(f"{path} must not contain absolute paths")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_values(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_absolute_values(item, f"{path}[{index}]")


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be an object with string keys")
    return value


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be a number")
    return float(value)
