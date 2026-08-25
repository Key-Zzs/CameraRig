"""Factory calibration artifact contract."""

from __future__ import annotations

from dataclasses import dataclass

from camera_rig.core._validation import (
    decoded_float,
    require_positive_finite,
    string_keyed_copy,
)
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.stream import StreamProfile
from camera_rig.core.transforms import RigidTransform


@dataclass(frozen=True)
class FactoryCalibration:
    """Factory parameters for streams within one physical camera device."""

    device: CameraDeviceInfo
    stream_profiles: dict[str, StreamProfile]
    intrinsics: dict[str, CameraIntrinsics]
    internal_transforms: tuple[RigidTransform, ...]
    depth_scale_m_per_unit: float

    def __post_init__(self) -> None:
        profiles = string_keyed_copy(self.stream_profiles, "stream_profiles")
        intrinsics = string_keyed_copy(self.intrinsics, "intrinsics")
        for name, profile in profiles.items():
            if name != profile.stream_name:
                raise ContractError(
                    f"stream profile key {name!r} does not match {profile.stream_name!r}"
                )
        for name, intrinsic in intrinsics.items():
            if name not in profiles:
                raise ContractError(f"intrinsics key {name!r} has no stream profile")
            if intrinsic.width != profiles[name].width or intrinsic.height != profiles[name].height:
                raise ContractError(f"intrinsics dimensions do not match stream profile {name!r}")
        object.__setattr__(self, "stream_profiles", profiles)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "internal_transforms", tuple(self.internal_transforms))
        object.__setattr__(
            self,
            "depth_scale_m_per_unit",
            require_positive_finite(self.depth_scale_m_per_unit, "depth_scale_m_per_unit"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe-compatible representation."""
        return {
            "device": self.device.to_dict(),
            "stream_profiles": {
                name: profile.to_dict() for name, profile in sorted(self.stream_profiles.items())
            },
            "intrinsics": {
                name: value.to_dict() for name, value in sorted(self.intrinsics.items())
            },
            "internal_transforms": [value.to_dict() for value in self.internal_transforms],
            "depth_scale_m_per_unit": self.depth_scale_m_per_unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FactoryCalibration:
        """Reconstruct factory calibration from decoded JSON data."""
        device = _dict_value(data, "device")
        profiles = _dict_value(data, "stream_profiles")
        intrinsics = _dict_value(data, "intrinsics")
        transforms = data["internal_transforms"]
        if not isinstance(transforms, list):
            raise TypeError("internal_transforms must be an array")
        return cls(
            device=CameraDeviceInfo.from_dict(device),
            stream_profiles={
                str(name): StreamProfile.from_dict(_ensure_dict(value))
                for name, value in profiles.items()
            },
            intrinsics={
                str(name): CameraIntrinsics.from_dict(_ensure_dict(value))
                for name, value in intrinsics.items()
            },
            internal_transforms=tuple(
                RigidTransform.from_dict(_ensure_dict(value)) for value in transforms
            ),
            depth_scale_m_per_unit=decoded_float(
                data["depth_scale_m_per_unit"], "depth_scale_m_per_unit"
            ),
        )


def _dict_value(data: dict[str, object], key: str) -> dict[str, object]:
    return _ensure_dict(data[key])


def _ensure_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return {str(key): item for key, item in value.items()}
