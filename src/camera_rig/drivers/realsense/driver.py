"""Strict single-device RealSense discovery and lifecycle driver."""

from __future__ import annotations

from contextlib import suppress

from camera_rig.config.models import CameraConfig
from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import (
    CameraRigError,
    DeviceBusyError,
    DeviceDisconnectedError,
    FrameTimeoutError,
    LifecycleError,
)
from camera_rig.core.stream import StreamProfile
from camera_rig.drivers.base import CameraLifecycleState
from camera_rig.drivers.profiles import (
    requested_profiles,
    validate_active,
    validate_supported,
)
from camera_rig.drivers.realsense.discovery import DiscoveredDevice, discover
from camera_rig.drivers.realsense.sdk_adapter import RealSenseSDKAdapter, SDKAdapter


class RealSenseDriver:
    """One configured RealSense pipeline with explicit ownership and cleanup."""

    def __init__(self, config: CameraConfig, adapter: SDKAdapter | None = None) -> None:
        self.config = config
        self.adapter = adapter or RealSenseSDKAdapter()
        self._state = CameraLifecycleState.CREATED
        self._discovered: DiscoveredDevice | None = None
        self._pipeline: object | None = None
        self._active_profiles: tuple[StreamProfile, ...] = ()
        self._pipeline_profile: object | None = None
        self._started = False

    @classmethod
    def from_config(cls, config: CameraConfig) -> RealSenseDriver:
        return cls(config)

    @property
    def state(self) -> CameraLifecycleState:
        return self._state

    @property
    def active_profiles(self) -> tuple[StreamProfile, ...]:
        return self._active_profiles

    @property
    def pipeline_profile(self) -> object:
        if self._pipeline_profile is None or not self._started:
            raise LifecycleError("RealSense active pipeline profile is unavailable")
        return self._pipeline_profile

    def get_device_info(self) -> CameraDeviceInfo:
        return self._ensure_discovered().info

    def get_supported_profiles(self) -> tuple[StreamProfile, ...]:
        return self._ensure_discovered().profiles

    def open(self) -> None:
        if self._state not in {CameraLifecycleState.CREATED, CameraLifecycleState.CLOSED}:
            raise LifecycleError(f"cannot open RealSense driver from state {self._state.value}")
        self._state = CameraLifecycleState.OPENING
        pipeline: object | None = None
        try:
            discovered = self._ensure_discovered()
            requested = requested_profiles(self.config)
            validate_supported(requested, discovered.profiles)
            pipeline = self.adapter.create_pipeline()
            sdk_config = self.adapter.create_config()
            self.adapter.configure(sdk_config, self.config.camera.serial, requested)
            resolved = self.adapter.resolve(pipeline, sdk_config)
            validate_active(requested, self.adapter.active_profiles(resolved))
            active = self.adapter.start(pipeline, sdk_config)
            self._pipeline = pipeline
            self._started = True
            self._pipeline_profile = active
            self._active_profiles = self.adapter.active_profiles(active)
            validate_active(requested, self._active_profiles)
            for _ in range(self.config.capture.warmup_frames):
                self.wait_for_frames()
            self._state = CameraLifecycleState.STREAMING
        except Exception as error:
            self._state = CameraLifecycleState.FAILED
            if pipeline is not None:
                with suppress(Exception):
                    self.adapter.stop(pipeline)
            self._pipeline = None
            self._started = False
            self._active_profiles = ()
            self._pipeline_profile = None
            raise _mapped_error(error) from error

    def wait_for_frames(self) -> object:
        if self._pipeline is None or not self._started:
            raise LifecycleError("RealSense driver is not streaming")
        try:
            return self.adapter.wait_for_frames(self._pipeline, self.config.capture.timeout_ms)
        except Exception as error:
            raise _mapped_error(error) from error

    def close(self) -> None:
        if self._state in {CameraLifecycleState.CREATED, CameraLifecycleState.CLOSED}:
            self._state = CameraLifecycleState.CLOSED
            return
        self._state = CameraLifecycleState.CLOSING
        try:
            if self._pipeline is not None and self._started:
                self.adapter.stop(self._pipeline)
        except Exception as error:
            self._state = CameraLifecycleState.FAILED
            raise _mapped_error(error) from error
        finally:
            self._pipeline = None
            self._started = False
            self._active_profiles = ()
            self._pipeline_profile = None
        self._state = CameraLifecycleState.CLOSED

    def __enter__(self) -> RealSenseDriver:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _ensure_discovered(self) -> DiscoveredDevice:
        if self._discovered is None:
            self._discovered = discover(self.config, self.adapter)
        return self._discovered


def _mapped_error(error: Exception) -> CameraRigError:
    if isinstance(error, CameraRigError):
        return error
    message = str(error)
    lowered = message.casefold()
    if "timeout" in lowered or "didn't arrive" in lowered:
        return FrameTimeoutError(f"RealSense frame timeout: {message}")
    if "busy" in lowered or "resource" in lowered:
        return DeviceBusyError(f"RealSense device is busy: {message}")
    if "disconnect" in lowered or "no device" in lowered:
        return DeviceDisconnectedError(f"RealSense device disconnected: {message}")
    return LifecycleError(f"RealSense lifecycle operation failed: {message}")
