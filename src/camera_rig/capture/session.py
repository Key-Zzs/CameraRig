"""Context-managed single-camera capture session."""

from __future__ import annotations

from camera_rig.config.models import CameraConfig
from camera_rig.core.errors import ConfigurationError, LifecycleError
from camera_rig.core.frame import CameraFrame
from camera_rig.drivers.base import CameraLifecycleState
from camera_rig.drivers.realsense.driver import RealSenseDriver
from camera_rig.drivers.realsense.frame_conversion import convert_frameset


class CameraSession:
    """Open one physical camera once, capture many framesets, and close once."""

    def __init__(self, config: CameraConfig, driver: RealSenseDriver | None = None) -> None:
        if not config.capture.copy_frames:
            raise ConfigurationError(
                "zero-copy RealSense capture is not supported by this contract"
            )
        self.config = config
        self.driver = driver or RealSenseDriver(config)

    @classmethod
    def from_config(cls, config: CameraConfig) -> CameraSession:
        if config.camera.driver.casefold() != "realsense":
            raise ConfigurationError(
                f"CameraSession does not support driver {config.camera.driver!r}"
            )
        return cls(config)

    @property
    def state(self) -> CameraLifecycleState:
        return self.driver.state

    def open(self) -> None:
        self.driver.open()

    def close(self) -> None:
        self.driver.close()

    def capture(self) -> CameraFrame:
        return self.wait_for_frame()

    def wait_for_frame(self) -> CameraFrame:
        try:
            frameset = self.driver.wait_for_frames()
            return convert_frameset(self.config, self.driver.adapter, frameset)
        except Exception:
            self.close()
            raise

    def poll_frame(self) -> CameraFrame | None:
        if self.driver.state is not CameraLifecycleState.STREAMING:
            raise LifecycleError("CameraSession is not streaming")
        try:
            frameset = self.driver.poll_for_frames()
            if frameset is None:
                return None
            return convert_frameset(self.config, self.driver.adapter, frameset)
        except Exception:
            self.close()
            raise

    def __enter__(self) -> CameraSession:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
