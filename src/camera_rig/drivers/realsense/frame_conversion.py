"""Copy one SDK frameset into stable CameraFrame NumPy ownership."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from camera_rig.capture.synchronization import build_sync_report
from camera_rig.config.models import CameraConfig
from camera_rig.core.errors import ContractError
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.drivers.realsense.sdk_adapter import SDKAdapter

EXPECTED_DTYPES = {
    "color": np.dtype(np.uint8),
    "depth": np.dtype(np.uint16),
    "ir_left": np.dtype(np.uint8),
    "ir_right": np.dtype(np.uint8),
}


def convert_frameset(config: CameraConfig, adapter: SDKAdapter, frameset: object) -> CameraFrame:
    """Deep-copy required raw streams and preserve original SDK timing metadata."""
    handles = adapter.frameset_frames(frameset)
    required = config.capture.required_streams or tuple(
        name for name, settings in config.streams.items() if settings.enabled
    )
    missing = sorted(set(required) - set(handles))
    if missing:
        raise ContractError(f"RealSense frameset is missing required streams: {missing}")
    streams: dict[str, StreamFrame] = {}
    for name in required:
        handle = handles[name]
        data = np.asarray(adapter.frame_array(handle)).copy()
        _validate_array(config, name, data)
        original_timestamp = adapter.frame_timestamp(handle)
        streams[name] = StreamFrame(
            stream_name=name,
            data=data,
            frame_number=adapter.frame_number(handle),
            sensor_timestamp_ns=round(original_timestamp * 1_000_000),
            timestamp_domain=adapter.frame_timestamp_domain(handle),
            original_timestamp=original_timestamp,
            metadata=adapter.frame_metadata(handle),
        )
    host_receive_timestamp_ns = time.monotonic_ns()
    return CameraFrame(
        camera_name=config.camera.name,
        serial=config.camera.serial,
        streams=streams,
        host_receive_timestamp_ns=host_receive_timestamp_ns,
        sync_report=build_sync_report(config, streams),
    )


def _validate_array(config: CameraConfig, name: str, data: npt.NDArray[np.generic]) -> None:
    profile = config.streams[name].profile
    expected_shape = (
        (profile.height, profile.width, 3) if name == "color" else (profile.height, profile.width)
    )
    if data.shape != expected_shape:
        raise ContractError(
            f"stream {name!r} shape mismatch: expected {expected_shape}, got {data.shape}"
        )
    expected_dtype = EXPECTED_DTYPES[name]
    if data.dtype != expected_dtype:
        raise ContractError(
            f"stream {name!r} dtype mismatch: expected {expected_dtype}, got {data.dtype}"
        )
