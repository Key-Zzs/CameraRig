"""Narrow, injectable boundary around the optional RealSense C extension."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Mapping
from typing import Any, Protocol

from camera_rig.core.errors import MissingOptionalDependencyError
from camera_rig.core.stream import StreamProfile
from camera_rig.drivers.profiles import STREAM_LAYOUT


class SDKAdapter(Protocol):
    """High-level operations consumed by discovery and lifecycle code."""

    @property
    def package_version(self) -> str: ...

    def query_devices(self) -> tuple[object, ...]: ...

    def read_device_fields(self, device: object) -> dict[str, object]: ...

    def supported_profiles(self, device: object) -> tuple[StreamProfile, ...]: ...

    def create_pipeline(self) -> object: ...

    def create_config(self) -> object: ...

    def configure(
        self, config: object, serial: str, profiles: tuple[StreamProfile, ...]
    ) -> None: ...

    def resolve(self, pipeline: object, config: object) -> object: ...

    def start(self, pipeline: object, config: object) -> object: ...

    def active_profiles(self, pipeline_profile: object) -> tuple[StreamProfile, ...]: ...

    def active_profile_handles(self, pipeline_profile: object) -> dict[str, object]: ...

    def intrinsics_data(self, profile: object) -> dict[str, object]: ...

    def extrinsics_data(
        self, source: object, target: object
    ) -> tuple[tuple[float, ...], tuple[float, ...]]: ...

    def depth_scale(self, pipeline_profile: object) -> float: ...

    def wait_for_frames(self, pipeline: object, timeout_ms: int) -> object: ...

    def stop(self, pipeline: object) -> None: ...


class RealSenseSDKAdapter:
    """Real SDK implementation; importing this module does not import the extension."""

    def __init__(self, sdk: Any | None = None) -> None:
        self._sdk = sdk

    @property
    def rs(self) -> Any:
        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("pyrealsense2")
            except ImportError as error:
                raise MissingOptionalDependencyError(
                    'RealSense support requires: pip install "camera-rig[realsense]"'
                ) from error
        return self._sdk

    @property
    def package_version(self) -> str:
        try:
            return importlib.metadata.version("pyrealsense2")
        except importlib.metadata.PackageNotFoundError:
            value = getattr(self.rs, "__version__", None)
            return "unknown" if value is None else str(value)

    def query_devices(self) -> tuple[object, ...]:
        return tuple(self.rs.context().query_devices())

    def read_device_fields(self, device: object) -> dict[str, object]:
        keys = {
            "reported_model": "name",
            "serial": "serial_number",
            "firmware_version": "firmware_version",
            "product_id": "product_id",
            "product_line": "product_line",
            "usb_type": "usb_type_descriptor",
            "physical_port": "physical_port",
            "asic_serial": "asic_serial_number",
            "firmware_update_id": "firmware_update_id",
            "camera_locked": "camera_locked",
            "imu_type": "imu_type",
        }
        result: dict[str, object] = {}
        for output_name, enum_name in keys.items():
            enum_value = getattr(self.rs.camera_info, enum_name, None)
            if enum_value is None:
                continue
            try:
                if device.supports(enum_value):  # type: ignore[attr-defined]
                    result[output_name] = str(device.get_info(enum_value))  # type: ignore[attr-defined]
            except RuntimeError:
                continue
        try:
            advanced = self.rs.rs400_advanced_mode(device)
            result["advanced_mode_supported"] = True
            result["advanced_mode_enabled"] = bool(advanced.is_enabled())
        except RuntimeError:
            result["advanced_mode_supported"] = False
        return result

    def supported_profiles(self, device: object) -> tuple[StreamProfile, ...]:
        result: list[StreamProfile] = []
        for sensor_number, sensor in enumerate(device.query_sensors()):  # type: ignore[attr-defined]
            sensor_name = self._sensor_name(sensor, sensor_number)
            for raw in sensor.get_stream_profiles():
                converted = self._profile(raw, sensor_name)
                if converted is not None:
                    result.append(converted)
        return tuple(result)

    def create_pipeline(self) -> object:
        return self.rs.pipeline()

    def create_config(self) -> object:
        return self.rs.config()

    def configure(self, config: object, serial: str, profiles: tuple[StreamProfile, ...]) -> None:
        config.enable_device(serial)  # type: ignore[attr-defined]
        for profile in profiles:
            sdk_stream_name, index, _, _ = STREAM_LAYOUT[profile.stream_name]
            sdk_stream = getattr(self.rs.stream, sdk_stream_name)
            sdk_format = getattr(self.rs.format, profile.format)
            config.enable_stream(  # type: ignore[attr-defined]
                sdk_stream,
                index,
                profile.width,
                profile.height,
                sdk_format,
                profile.fps,
            )

    def resolve(self, pipeline: object, config: object) -> object:
        return config.resolve(self.rs.pipeline_wrapper(pipeline))  # type: ignore[attr-defined]

    def start(self, pipeline: object, config: object) -> object:
        return pipeline.start(config)  # type: ignore[attr-defined]

    def active_profiles(self, pipeline_profile: object) -> tuple[StreamProfile, ...]:
        result: list[StreamProfile] = []
        for profile in pipeline_profile.get_streams():  # type: ignore[attr-defined]
            converted = self._profile(profile, None)
            if converted is not None:
                result.append(converted)
        return tuple(result)

    def active_profile_handles(self, pipeline_profile: object) -> dict[str, object]:
        result: dict[str, object] = {}
        for profile in pipeline_profile.get_streams():  # type: ignore[attr-defined]
            sdk_stream = _enum_suffix(profile.stream_type())
            index = int(profile.stream_index())
            stream_name = _stream_name(sdk_stream, index)
            if stream_name is not None:
                result[stream_name] = profile
        return result

    def intrinsics_data(self, profile: object) -> dict[str, object]:
        video = profile.as_video_stream_profile()  # type: ignore[attr-defined]
        value = video.get_intrinsics()
        return {
            "width": int(value.width),
            "height": int(value.height),
            "fx": float(value.fx),
            "fy": float(value.fy),
            "cx": float(value.ppx),
            "cy": float(value.ppy),
            "distortion_model": _enum_suffix(value.model),
            "distortion_coeffs": tuple(float(item) for item in value.coeffs),
        }

    def extrinsics_data(
        self, source: object, target: object
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        value = source.get_extrinsics_to(target)  # type: ignore[attr-defined]
        return (
            tuple(float(item) for item in value.rotation),
            tuple(float(item) for item in value.translation),
        )

    def depth_scale(self, pipeline_profile: object) -> float:
        device = pipeline_profile.get_device()  # type: ignore[attr-defined]
        return float(device.first_depth_sensor().get_depth_scale())

    def wait_for_frames(self, pipeline: object, timeout_ms: int) -> object:
        return pipeline.wait_for_frames(timeout_ms)  # type: ignore[attr-defined]

    def stop(self, pipeline: object) -> None:
        pipeline.stop()  # type: ignore[attr-defined]

    def _sensor_name(self, sensor: object, sensor_number: int) -> str:
        name = self.rs.camera_info.name
        try:
            if sensor.supports(name):  # type: ignore[attr-defined]
                return str(sensor.get_info(name))  # type: ignore[attr-defined]
        except RuntimeError:
            pass
        return f"sensor-{sensor_number}"

    def _profile(self, raw: object, sensor_name: str | None) -> StreamProfile | None:
        try:
            video = raw.as_video_stream_profile()  # type: ignore[attr-defined]
            sdk_stream = _enum_suffix(raw.stream_type())  # type: ignore[attr-defined]
            index = int(raw.stream_index())  # type: ignore[attr-defined]
            stream_name = _stream_name(sdk_stream, index)
            if stream_name is None:
                return None
            return StreamProfile(
                stream_name=stream_name,
                width=int(video.width()),
                height=int(video.height()),
                fps=int(raw.fps()),  # type: ignore[attr-defined]
                format=_enum_suffix(raw.format()),  # type: ignore[attr-defined]
                index=index,
                sensor_identifier=sensor_name,
            )
        except RuntimeError:
            return None


def _enum_suffix(value: object) -> str:
    return str(value).rsplit(".", maxsplit=1)[-1].casefold()


def _stream_name(sdk_stream: str, index: int) -> str | None:
    mapping: Mapping[tuple[str, int], str] = {
        ("color", 0): "color",
        ("depth", 0): "depth",
        ("infrared", 1): "ir_left",
        ("infrared", 2): "ir_right",
    }
    return mapping.get((sdk_stream, index))
