from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from camera_rig.capture.session import CameraSession
from camera_rig.config.loader import load_config
from camera_rig.drivers.realsense.driver import RealSenseDriver
from camera_rig.drivers.realsense.factory_calibration import extract_factory_calibration


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


@pytest.mark.hardware
def test_d435i_active_factory_calibration() -> None:
    config = load_config(_hardware_config())
    with RealSenseDriver(config) as driver:
        calibration = extract_factory_calibration(driver)
    assert len(calibration.intrinsics) == 4
    assert len(calibration.internal_transforms) == 3
    assert calibration.depth_scale_m_per_unit > 0
    transforms = {value.target_frame: value for value in calibration.internal_transforms}
    baseline = transforms["head/ir_right_optical"].matrix[:3, 3]
    assert float((baseline**2).sum() ** 0.5) > 0


@pytest.mark.hardware
def test_d435i_raw_capture_buffer_ownership_and_reopen() -> None:
    config = load_config(_hardware_config())
    with CameraSession.from_config(config) as camera:
        first = camera.capture()
        saved = {name: frame.data.copy() for name, frame in first.streams.items()}
        for _ in range(5):
            camera.capture()
        for name, expected in saved.items():
            np.testing.assert_array_equal(first.streams[name].data, expected)
        assert first.color is not None and first.color.data.shape == (480, 640, 3)
        assert first.color.data.dtype == np.uint8
        assert first.depth is not None and first.depth.data.dtype == np.uint16
        assert first.ir_left is not None and first.ir_left.data.dtype == np.uint8
        assert first.ir_right is not None and first.ir_right.data.dtype == np.uint8
        assert first.sync_report is not None
    with CameraSession.from_config(config) as camera:
        assert camera.capture().color is not None
