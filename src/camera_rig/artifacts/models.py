"""Stable top-level CameraBundle artifact contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from camera_rig.artifacts.io import json_safe
from camera_rig.core._validation import (
    require_non_empty,
    require_positive_finite,
    string_keyed_copy,
)
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ContractError
from camera_rig.core.fixed_mount import FixedMountCalibration
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform

BUNDLE_SCHEMA_VERSION: Final = "camera-rig.bundle.v1"
COORDINATE_CONVENTION: Final[dict[str, str]] = {
    "vector": "column",
    "handedness": "right-handed",
    "length_unit": "meter",
    "time_unit": "nanosecond",
    "transform_naming": "T_target_from_source",
}


@dataclass(frozen=True)
class CameraBundle:
    """Versioned calibration and provenance artifact for one physical camera."""

    status: str
    bundle_id: str
    created_at: str
    device: CameraDeviceInfo
    stream_profiles: dict[str, StreamProfile]
    intrinsics: dict[str, CameraIntrinsics]
    internal_transforms: tuple[RigidTransform, ...]
    depth_scale_m_per_unit: float
    quality: QualityReport
    provenance: dict[str, object]
    fixed_mount_calibration: FixedMountCalibration | None = None
    coordinate_convention: dict[str, str] = field(
        default_factory=lambda: dict(COORDINATE_CONVENTION)
    )
    schema_version: str = field(default=BUNDLE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        require_non_empty(self.status, "status")
        require_non_empty(self.bundle_id, "bundle_id")
        require_non_empty(self.created_at, "created_at")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ContractError("created_at must be an ISO-8601 date-time") from error
        if self.coordinate_convention != COORDINATE_CONVENTION:
            raise ContractError("coordinate_convention must match the CameraRig v1 convention")
        profiles = string_keyed_copy(self.stream_profiles, "stream_profiles")
        intrinsics = string_keyed_copy(self.intrinsics, "intrinsics")
        for name, profile in profiles.items():
            if profile.stream_name != name:
                raise ContractError(f"stream profile key {name!r} does not match profile name")
        for name, intrinsic in intrinsics.items():
            if name not in profiles:
                raise ContractError(f"intrinsics key {name!r} has no stream profile")
            profile = profiles[name]
            if (intrinsic.width, intrinsic.height) != (profile.width, profile.height):
                raise ContractError(f"intrinsics dimensions do not match stream {name!r}")
        object.__setattr__(self, "stream_profiles", profiles)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "internal_transforms", tuple(self.internal_transforms))
        object.__setattr__(
            self,
            "depth_scale_m_per_unit",
            require_positive_finite(self.depth_scale_m_per_unit, "depth_scale_m_per_unit"),
        )
        object.__setattr__(self, "provenance", string_keyed_copy(self.provenance, "provenance"))
        json_safe(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the canonical persisted representation."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "coordinate_convention": dict(self.coordinate_convention),
            "device": self.device.to_dict(),
            "stream_profiles": {
                name: value.to_dict() for name, value in sorted(self.stream_profiles.items())
            },
            "intrinsics": {
                name: value.to_dict() for name, value in sorted(self.intrinsics.items())
            },
            "internal_transforms": [value.to_dict() for value in self.internal_transforms],
            "depth_scale_m_per_unit": self.depth_scale_m_per_unit,
            "fixed_mount_calibration": (
                None
                if self.fixed_mount_calibration is None
                else self.fixed_mount_calibration.to_dict()
            ),
            "quality": self.quality.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CameraBundle:
        """Reconstruct a bundle after schema validation and re-run contract checks."""
        if data.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ContractError(f"schema_version must be {BUNDLE_SCHEMA_VERSION!r}")
        device = _object(data["device"], "device")
        profiles = _object(data["stream_profiles"], "stream_profiles")
        intrinsics = _object(data["intrinsics"], "intrinsics")
        convention = _object(data["coordinate_convention"], "coordinate_convention")
        quality = _object(data["quality"], "quality")
        provenance = _object(data["provenance"], "provenance")
        transforms = _array(data["internal_transforms"], "internal_transforms")
        fixed_data = data.get("fixed_mount_calibration")
        return cls(
            status=_string(data["status"], "status"),
            bundle_id=_string(data["bundle_id"], "bundle_id"),
            created_at=_string(data["created_at"], "created_at"),
            coordinate_convention={
                str(key): _string(value, f"coordinate_convention.{key}")
                for key, value in convention.items()
            },
            device=CameraDeviceInfo.from_dict(device),
            stream_profiles={
                str(name): StreamProfile.from_dict(_object(value, f"stream_profiles.{name}"))
                for name, value in profiles.items()
            },
            intrinsics={
                str(name): CameraIntrinsics.from_dict(_object(value, f"intrinsics.{name}"))
                for name, value in intrinsics.items()
            },
            internal_transforms=tuple(
                RigidTransform.from_dict(_object(value, "internal_transforms[]"))
                for value in transforms
            ),
            depth_scale_m_per_unit=_number(
                data["depth_scale_m_per_unit"], "depth_scale_m_per_unit"
            ),
            fixed_mount_calibration=(
                None
                if fixed_data is None
                else FixedMountCalibration.from_dict(_object(fixed_data, "fixed_mount_calibration"))
            ),
            quality=QualityReport.from_dict(quality),
            provenance=provenance,
        )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} keys must be strings")
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
