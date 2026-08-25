"""SDK-independent frame data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from camera_rig.core._validation import (
    require_non_empty,
    require_non_negative_int,
    string_keyed_copy,
)
from camera_rig.core.errors import ContractError
from camera_rig.core.timestamps import SingleDeviceSyncReport


@dataclass(frozen=True)
class StreamFrame:
    """One SDK-independent stream buffer and its original timing metadata."""

    stream_name: str
    data: npt.NDArray[np.generic]
    frame_number: int
    sensor_timestamp_ns: int | None = None
    timestamp_domain: str | None = None
    original_timestamp: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.stream_name, "stream_name")
        if not isinstance(self.data, np.ndarray):
            raise ContractError("data must be a NumPy array")
        require_non_negative_int(self.frame_number, "frame_number")
        if self.sensor_timestamp_ns is not None:
            require_non_negative_int(self.sensor_timestamp_ns, "sensor_timestamp_ns")
        if self.timestamp_domain is not None:
            require_non_empty(self.timestamp_domain, "timestamp_domain")
        if self.original_timestamp is not None and not np.isfinite(self.original_timestamp):
            raise ContractError("original_timestamp must be finite when provided")
        object.__setattr__(self, "metadata", string_keyed_copy(self.metadata, "metadata"))


@dataclass(frozen=True)
class CameraFrame:
    """A frameset from streams within exactly one physical camera."""

    camera_name: str
    serial: str
    streams: dict[str, StreamFrame]
    host_receive_timestamp_ns: int
    sync_report: SingleDeviceSyncReport | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.camera_name, "camera_name")
        require_non_empty(self.serial, "serial")
        require_non_negative_int(self.host_receive_timestamp_ns, "host_receive_timestamp_ns")
        streams = string_keyed_copy(self.streams, "streams")
        for key, frame in streams.items():
            if key != frame.stream_name:
                raise ContractError(
                    f"stream mapping key {key!r} does not match frame name {frame.stream_name!r}"
                )
        object.__setattr__(self, "streams", streams)

    def _stream(self, name: str) -> StreamFrame | None:
        return self.streams.get(name)

    @property
    def rgb(self) -> StreamFrame | None:
        """Return the color stream, or ``None`` when it is absent."""
        return self._stream("color")

    @property
    def color(self) -> StreamFrame | None:
        """Return the RGB color stream, or ``None`` when it is absent."""
        return self._stream("color")

    @property
    def depth(self) -> StreamFrame | None:
        """Return the depth stream, or ``None`` when it is absent."""
        return self._stream("depth")

    @property
    def ir_left(self) -> StreamFrame | None:
        """Return the left IR stream, or ``None`` when it is absent."""
        return self._stream("ir_left")

    @property
    def ir_right(self) -> StreamFrame | None:
        """Return the right IR stream, or ``None`` when it is absent."""
        return self._stream("ir_right")
