"""Stable single-camera driver lifecycle contract."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.stream import StreamProfile


class CameraLifecycleState(str, Enum):
    """Explicit lifecycle states for one physical camera driver."""

    CREATED = "created"
    OPENING = "opening"
    STREAMING = "streaming"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@runtime_checkable
class CameraDriver(Protocol):
    """Operations common to a driver for exactly one physical camera."""

    @property
    def state(self) -> CameraLifecycleState: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def get_device_info(self) -> CameraDeviceInfo: ...

    def get_supported_profiles(self) -> tuple[StreamProfile, ...]: ...

    def __enter__(self) -> CameraDriver: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
