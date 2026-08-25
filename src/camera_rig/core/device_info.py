"""Physical camera identity contract."""

from __future__ import annotations

from dataclasses import dataclass, field

from camera_rig.core._validation import (
    decoded_optional_string,
    decoded_string,
    require_non_empty,
    string_keyed_copy,
)


@dataclass(frozen=True)
class CameraDeviceInfo:
    """Identity reported for one physical camera device.

    ``serial`` is deliberately and permanently a string, even when it contains only
    decimal digits.
    """

    driver: str
    camera_name: str
    expected_model: str
    reported_model: str
    serial: str
    firmware_version: str | None = None
    sdk_version: str | None = None
    usb_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("driver", "camera_name", "expected_model", "reported_model", "serial"):
            require_non_empty(getattr(self, name), name)
        for name in ("firmware_version", "sdk_version", "usb_type"):
            value = getattr(self, name)
            if value is not None:
                require_non_empty(value, name)
        object.__setattr__(self, "metadata", string_keyed_copy(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe-compatible representation."""
        return {
            "driver": self.driver,
            "camera_name": self.camera_name,
            "expected_model": self.expected_model,
            "reported_model": self.reported_model,
            "serial": self.serial,
            "firmware_version": self.firmware_version,
            "sdk_version": self.sdk_version,
            "usb_type": self.usb_type,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CameraDeviceInfo:
        """Reconstruct a device contract from decoded JSON data."""
        return cls(
            driver=decoded_string(data["driver"], "driver"),
            camera_name=decoded_string(data["camera_name"], "camera_name"),
            expected_model=decoded_string(data["expected_model"], "expected_model"),
            reported_model=decoded_string(data["reported_model"], "reported_model"),
            serial=decoded_string(data["serial"], "serial"),
            firmware_version=decoded_optional_string(
                data.get("firmware_version"), "firmware_version"
            ),
            sdk_version=decoded_optional_string(data.get("sdk_version"), "sdk_version"),
            usb_type=decoded_optional_string(data.get("usb_type"), "usb_type"),
            metadata=_object_mapping(data.get("metadata", {})),
        )


def _object_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be an object")
    return {str(key): item for key, item in value.items()}
