"""Small explicit driver registry."""

from __future__ import annotations

from collections.abc import Callable

from camera_rig.config.models import CameraConfig
from camera_rig.core.errors import UnsupportedDriverError
from camera_rig.drivers.base import CameraDriver

DriverFactory = Callable[[CameraConfig], CameraDriver]
_FACTORIES: dict[str, DriverFactory] = {}


def register_driver(name: str, factory: DriverFactory) -> None:
    """Register a driver factory under a stable lower-case name."""
    _FACTORIES[name.casefold()] = factory


def create_driver(config: CameraConfig) -> CameraDriver:
    """Create the configured driver without opening hardware."""
    name = config.camera.driver.casefold()
    if name == "realsense" and name not in _FACTORIES:
        from camera_rig.drivers.realsense.driver import RealSenseDriver

        register_driver(name, RealSenseDriver.from_config)
    try:
        factory = _FACTORIES[name]
    except KeyError as error:
        raise UnsupportedDriverError(
            f"unsupported camera driver: {config.camera.driver!r}"
        ) from error
    return factory(config)
