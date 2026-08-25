"""Single-device stream profile contract."""

from __future__ import annotations

from dataclasses import dataclass

from camera_rig.core._validation import (
    decoded_int,
    decoded_optional_string,
    decoded_string,
    require_non_empty,
    require_positive_int,
)
from camera_rig.core.errors import ContractError


@dataclass(frozen=True)
class StreamProfile:
    """Requested or reported profile for one stream inside a camera device."""

    stream_name: str
    width: int
    height: int
    fps: int
    format: str
    index: int | None = None
    sensor_identifier: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.stream_name, "stream_name")
        require_positive_int(self.width, "width")
        require_positive_int(self.height, "height")
        require_positive_int(self.fps, "fps")
        require_non_empty(self.format, "format")
        if self.index is not None and (isinstance(self.index, bool) or self.index < 0):
            raise ContractError("index must be a non-negative integer when provided")
        if self.sensor_identifier is not None:
            require_non_empty(self.sensor_identifier, "sensor_identifier")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe-compatible representation."""
        return {
            "stream_name": self.stream_name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "format": self.format,
            "index": self.index,
            "sensor_identifier": self.sensor_identifier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> StreamProfile:
        """Reconstruct a stream profile from decoded JSON data."""
        index = data.get("index")
        sensor_identifier = data.get("sensor_identifier")
        return cls(
            stream_name=decoded_string(data["stream_name"], "stream_name"),
            width=decoded_int(data["width"], "width"),
            height=decoded_int(data["height"], "height"),
            fps=decoded_int(data["fps"], "fps"),
            format=decoded_string(data["format"], "format"),
            index=None if index is None else decoded_int(index, "index"),
            sensor_identifier=decoded_optional_string(sensor_identifier, "sensor_identifier"),
        )
