"""Exact RealSense device selection and read-only inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from camera_rig.config.models import CameraConfig
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import DeviceNotFoundError, ProfileNotSupportedError
from camera_rig.core.stream import StreamProfile
from camera_rig.drivers.profiles import canonical_d435i, validate_model
from camera_rig.drivers.realsense.sdk_adapter import SDKAdapter


@dataclass(frozen=True)
class DiscoveredDevice:
    """Selected raw SDK handle plus stable public descriptions."""

    handle: object
    info: CameraDeviceInfo
    profiles: tuple[StreamProfile, ...]


def list_devices(adapter: SDKAdapter) -> tuple[CameraDeviceInfo, ...]:
    """List devices without selecting or opening a pipeline."""
    try:
        devices = adapter.query_devices()
        return tuple(_listed_device_info(adapter, device) for device in devices)
    except RuntimeError as error:
        raise DeviceNotFoundError(f"RealSense device enumeration failed: {error}") from error


def discover(config: CameraConfig, adapter: SDKAdapter) -> DiscoveredDevice:
    """Select only the configured serial and validate its exact D435i identity."""
    selected: object | None = None
    selected_fields: dict[str, object] | None = None
    visible_serials: list[str] = []
    try:
        for device in adapter.query_devices():
            fields = adapter.read_device_fields(device)
            serial = _required_string(fields, "serial")
            visible_serials.append(serial)
            if serial == config.camera.serial:
                selected = device
                selected_fields = fields
    except RuntimeError as error:
        raise DeviceNotFoundError(f"RealSense device enumeration failed: {error}") from error
    if selected is None or selected_fields is None:
        raise DeviceNotFoundError(
            "configured RealSense serial was not found; "
            f"visible device count={len(visible_serials)}"
        )
    info = _device_info_from_fields(config, adapter.package_version, selected_fields)
    try:
        profiles = adapter.supported_profiles(selected)
    except RuntimeError as error:
        raise ProfileNotSupportedError(
            f"could not enumerate RealSense stream profiles: {error}"
        ) from error
    return DiscoveredDevice(selected, info, profiles)


def _listed_device_info(adapter: SDKAdapter, device: object) -> CameraDeviceInfo:
    fields = adapter.read_device_fields(device)
    reported = _required_string(fields, "reported_model")
    product_id = _optional_string(fields.get("product_id"))
    known = {
        "reported_model",
        "serial",
        "firmware_version",
        "product_id",
        "product_line",
        "usb_type",
        "physical_port",
    }
    return CameraDeviceInfo(
        driver="realsense",
        camera_name="unconfigured",
        expected_model=reported,
        reported_model=reported,
        serial=_required_string(fields, "serial"),
        canonical_model=canonical_d435i(reported, product_id),
        product_id=product_id,
        product_line=_optional_string(fields.get("product_line")),
        physical_port=_optional_string(fields.get("physical_port")),
        firmware_version=_optional_string(fields.get("firmware_version")),
        sdk_version=adapter.package_version,
        usb_type=_optional_string(fields.get("usb_type")),
        metadata={key: value for key, value in fields.items() if key not in known},
    )


def _device_info_from_fields(
    config: CameraConfig, sdk_version: str, fields: dict[str, object]
) -> CameraDeviceInfo:
    return _device_info_from_identity(config.camera, sdk_version, fields)


def _device_info_from_identity(
    identity: _IdentityLike, sdk_version: str, fields: dict[str, object]
) -> CameraDeviceInfo:
    reported = _required_string(fields, "reported_model")
    product_id = _optional_string(fields.get("product_id"))
    expected = identity.expected_model
    canonical = validate_model(expected, reported, product_id)
    known = {
        "reported_model",
        "serial",
        "firmware_version",
        "product_id",
        "product_line",
        "usb_type",
        "physical_port",
    }
    metadata = {key: value for key, value in fields.items() if key not in known}
    return CameraDeviceInfo(
        driver="realsense",
        camera_name=identity.name,
        expected_model=expected,
        reported_model=reported,
        serial=_required_string(fields, "serial"),
        canonical_model=canonical,
        product_id=product_id,
        product_line=_optional_string(fields.get("product_line")),
        physical_port=_optional_string(fields.get("physical_port")),
        firmware_version=_optional_string(fields.get("firmware_version")),
        sdk_version=sdk_version,
        usb_type=_optional_string(fields.get("usb_type")),
        metadata=metadata,
    )


class _IdentityLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def expected_model(self) -> str: ...


def _required_string(fields: dict[str, object], name: str) -> str:
    value = _optional_string(fields.get(name))
    if value is None:
        raise DeviceNotFoundError(f"RealSense device did not report required field {name!r}")
    return value


def _optional_string(value: object) -> str | None:
    return None if value is None or not str(value).strip() else str(value)
