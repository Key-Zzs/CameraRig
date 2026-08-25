from __future__ import annotations

import os
from pathlib import Path

import pytest

from camera_rig.config.loader import load_config
from camera_rig.drivers.realsense.driver import RealSenseDriver


def _hardware_config() -> Path:
    value = os.environ.get("CAMERA_RIG_HARDWARE_CONFIG")
    if value is None:
        pytest.skip("CAMERA_RIG_HARDWARE_CONFIG is not set")
    return Path(value)


@pytest.mark.hardware
def test_d435i_discovery_profiles_and_lifecycle() -> None:
    config = load_config(_hardware_config())
    driver = RealSenseDriver(config)
    info = driver.get_device_info()
    assert info.serial == config.camera.serial
    assert info.canonical_model == "D435i"
    assert info.product_id == "0B3A"
    assert info.usb_type == "3.2"
    with driver:
        assert {profile.stream_name for profile in driver.active_profiles} == {
            "color",
            "depth",
            "ir_left",
            "ir_right",
        }
