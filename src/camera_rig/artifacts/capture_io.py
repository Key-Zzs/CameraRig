"""Capture metadata conversion shared by snapshot and replay."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from camera_rig.core.errors import ArtifactError
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.timestamps import SingleDeviceSyncReport


def camera_frame_metadata(frame: CameraFrame) -> dict[str, object]:
    """Serialize one CameraFrame without embedding its arrays."""
    return {
        "camera_name": frame.camera_name,
        "serial": frame.serial,
        "host_receive_timestamp_ns": frame.host_receive_timestamp_ns,
        "streams": {
            name: {
                "stream_name": stream.stream_name,
                "frame_number": stream.frame_number,
                "sensor_timestamp_ns": stream.sensor_timestamp_ns,
                "timestamp_domain": stream.timestamp_domain,
                "original_timestamp": stream.original_timestamp,
                "metadata": dict(stream.metadata),
            }
            for name, stream in sorted(frame.streams.items())
        },
        "sync_report": None if frame.sync_report is None else frame.sync_report.to_dict(),
    }


def restore_camera_frame(
    metadata: dict[str, object], arrays: dict[str, npt.NDArray[np.generic]]
) -> CameraFrame:
    """Reconstruct typed frame contracts from validated metadata and raw arrays."""
    streams_data = _object(metadata.get("streams"), "streams")
    streams: dict[str, StreamFrame] = {}
    if set(streams_data) != set(arrays):
        raise ArtifactError("frame metadata stream names differ from NPZ array names")
    for name, value in streams_data.items():
        stream = _object(value, f"streams.{name}")
        streams[name] = StreamFrame(
            stream_name=_string(stream.get("stream_name"), "stream_name"),
            data=np.asarray(arrays[name]).copy(),
            frame_number=_int(stream.get("frame_number"), "frame_number"),
            sensor_timestamp_ns=_optional_int(
                stream.get("sensor_timestamp_ns"), "sensor_timestamp_ns"
            ),
            timestamp_domain=_optional_string(stream.get("timestamp_domain"), "timestamp_domain"),
            original_timestamp=_optional_float(
                stream.get("original_timestamp"), "original_timestamp"
            ),
            metadata=_object(stream.get("metadata", {}), "metadata"),
        )
    sync_data = metadata.get("sync_report")
    return CameraFrame(
        camera_name=_string(metadata.get("camera_name"), "camera_name"),
        serial=_string(metadata.get("serial"), "serial"),
        streams=streams,
        host_receive_timestamp_ns=_int(
            metadata.get("host_receive_timestamp_ns"), "host_receive_timestamp_ns"
        ),
        sync_report=(
            None
            if sync_data is None
            else SingleDeviceSyncReport.from_dict(_object(sync_data, "sync_report"))
        ),
    )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArtifactError(f"{name} must be an object with string keys")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ArtifactError(f"{name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{name} must be an integer")
    return value


def _optional_int(value: object, name: str) -> int | None:
    return None if value is None else _int(value, name)


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactError(f"{name} must be a number")
    return float(value)
