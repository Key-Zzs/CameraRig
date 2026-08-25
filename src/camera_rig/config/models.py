"""Typed strict-configuration models."""

from __future__ import annotations

from dataclasses import dataclass

from camera_rig.core._validation import require_non_empty, require_non_negative_int
from camera_rig.core.errors import ContractError
from camera_rig.core.stream import StreamProfile

CONFIG_SCHEMA_VERSION = "camera-rig.config.v1"


@dataclass(frozen=True)
class CameraSettings:
    """Identity selection for one physical camera."""

    name: str
    driver: str
    expected_model: str
    serial: str
    output_reference_stream: str

    def __post_init__(self) -> None:
        for field_name in ("name", "driver", "expected_model", "serial", "output_reference_stream"):
            require_non_empty(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class StreamSettings:
    """Enable flag plus one stream profile."""

    enabled: bool
    profile: StreamProfile


@dataclass(frozen=True)
class SyncSettings:
    """Validation thresholds for streams inside one device frameset."""

    max_comparable_stream_skew_ms: float = 5.0
    require_stereo_frame_number_match: bool = True

    def __post_init__(self) -> None:
        if self.max_comparable_stream_skew_ms < 0:
            raise ContractError("max_comparable_stream_skew_ms must be non-negative")


@dataclass(frozen=True)
class CaptureSettings:
    """Device-local capture behavior configuration contract."""

    warmup_frames: int
    timeout_ms: int
    copy_frames: bool
    required_streams: tuple[str, ...] = ()
    sync: SyncSettings = SyncSettings()

    def __post_init__(self) -> None:
        require_non_negative_int(self.warmup_frames, "warmup_frames")
        require_non_negative_int(self.timeout_ms, "timeout_ms")
        if self.timeout_ms == 0:
            raise ContractError("timeout_ms must be greater than zero")
        object.__setattr__(self, "required_streams", tuple(self.required_streams))
        if len(set(self.required_streams)) != len(self.required_streams):
            raise ContractError("required_streams must not contain duplicates")


@dataclass(frozen=True)
class CameraConfig:
    """Versioned configuration for exactly one physical camera."""

    camera: CameraSettings
    streams: dict[str, StreamSettings]
    capture: CaptureSettings
    schema_version: str = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ContractError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
        streams = dict(self.streams)
        if not streams:
            raise ContractError("at least one stream configuration is required")
        reference = streams.get(self.camera.output_reference_stream)
        if reference is None:
            raise ContractError("output_reference_stream must name a configured stream")
        if not reference.enabled:
            raise ContractError("output_reference_stream must be enabled")
        object.__setattr__(self, "streams", streams)
        required = self.capture.required_streams or tuple(
            name for name, settings in streams.items() if settings.enabled
        )
        for name in required:
            settings = streams.get(name)
            if settings is None or not settings.enabled:
                raise ContractError(f"required stream {name!r} must be configured and enabled")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CameraConfig:
        """Construct a typed configuration after JSON Schema validation."""
        camera_data = _object(data["camera"], "camera")
        streams_data = _object(data["streams"], "streams")
        capture_data = _object(data["capture"], "capture")
        sync_data = _object(capture_data.get("sync", {}), "capture.sync")
        required_data = capture_data.get("required_streams", [])
        if not isinstance(required_data, list):
            raise ContractError("capture.required_streams must be an array")
        streams: dict[str, StreamSettings] = {}
        for stream_name, value in streams_data.items():
            stream = _object(value, f"streams.{stream_name}")
            streams[stream_name] = StreamSettings(
                enabled=_boolean(stream["enabled"], f"streams.{stream_name}.enabled"),
                profile=StreamProfile(
                    stream_name=stream_name,
                    width=_integer(stream["width"], f"streams.{stream_name}.width"),
                    height=_integer(stream["height"], f"streams.{stream_name}.height"),
                    fps=_integer(stream["fps"], f"streams.{stream_name}.fps"),
                    format=_string(stream["format"], f"streams.{stream_name}.format"),
                ),
            )
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            camera=CameraSettings(
                name=_string(camera_data["name"], "camera.name"),
                driver=_string(camera_data["driver"], "camera.driver"),
                expected_model=_string(camera_data["expected_model"], "camera.expected_model"),
                serial=_string(camera_data["serial"], "camera.serial"),
                output_reference_stream=_string(
                    camera_data["output_reference_stream"], "camera.output_reference_stream"
                ),
            ),
            streams=streams,
            capture=CaptureSettings(
                warmup_frames=_integer(capture_data["warmup_frames"], "capture.warmup_frames"),
                timeout_ms=_integer(capture_data["timeout_ms"], "capture.timeout_ms"),
                copy_frames=_boolean(capture_data["copy_frames"], "capture.copy_frames"),
                required_streams=tuple(
                    _string(value, "capture.required_streams[]") for value in required_data
                ),
                sync=SyncSettings(
                    max_comparable_stream_skew_ms=_number(
                        sync_data.get("max_comparable_stream_skew_ms", 5.0),
                        "capture.sync.max_comparable_stream_skew_ms",
                    ),
                    require_stereo_frame_number_match=_boolean(
                        sync_data.get("require_stereo_frame_number_match", True),
                        "capture.sync.require_stereo_frame_number_match",
                    ),
                ),
            ),
        )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} keys must be strings")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a boolean")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be a number")
    return float(value)
