from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from camera_rig.config.loader import load_config
from camera_rig.core.errors import (
    DeviceMismatchError,
    DeviceNotFoundError,
    LifecycleError,
    MissingOptionalDependencyError,
    ProfileNotSupportedError,
    UnsupportedDriverError,
)
from camera_rig.core.stream import StreamProfile
from camera_rig.drivers.base import CameraLifecycleState
from camera_rig.drivers.profiles import (
    canonical_d435i,
    requested_profiles,
    validate_supported,
)
from camera_rig.drivers.realsense.discovery import discover, list_devices
from camera_rig.drivers.realsense.driver import RealSenseDriver
from camera_rig.drivers.realsense.sdk_adapter import RealSenseSDKAdapter
from camera_rig.drivers.registry import create_driver

ROOT = Path(__file__).parents[1]
CONFIG = load_config(ROOT / "configs/examples/single_camera_contract.yaml")


def _profiles() -> tuple[StreamProfile, ...]:
    return requested_profiles(CONFIG)


class FakeAdapter:
    package_version = "2.58.test"

    def __init__(self, *, devices: list[dict[str, object]] | None = None) -> None:
        self.devices = devices if devices is not None else [_device(CONFIG.camera.serial)]
        self.profiles = _profiles()
        self.configured_serial: str | None = None
        self.configured_profiles: tuple[StreamProfile, ...] = ()
        self.stop_calls = 0
        self.wait_calls = 0
        self.fail_start = False
        self.fail_wait = False

    def query_devices(self) -> tuple[object, ...]:
        return tuple(self.devices)

    def read_device_fields(self, device: object) -> dict[str, object]:
        assert isinstance(device, dict)
        return device

    def supported_profiles(self, device: object) -> tuple[StreamProfile, ...]:
        return self.profiles

    def create_pipeline(self) -> object:
        return {"pipeline": True}

    def create_config(self) -> object:
        return {"config": True}

    def configure(self, config: object, serial: str, profiles: tuple[StreamProfile, ...]) -> None:
        self.configured_serial = serial
        self.configured_profiles = profiles

    def resolve(self, pipeline: object, config: object) -> object:
        return self.configured_profiles

    def start(self, pipeline: object, config: object) -> object:
        if self.fail_start:
            raise RuntimeError("backend start failure")
        return self.configured_profiles

    def active_profiles(self, pipeline_profile: object) -> tuple[StreamProfile, ...]:
        assert isinstance(pipeline_profile, tuple)
        return pipeline_profile

    def wait_for_frames(self, pipeline: object, timeout_ms: int) -> object:
        self.wait_calls += 1
        if self.fail_wait:
            raise RuntimeError("Frame didn't arrive within timeout")
        return {"frameset": self.wait_calls}

    def stop(self, pipeline: object) -> None:
        self.stop_calls += 1


def _device(serial: str, **updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "reported_model": "Intel RealSense D435I",
        "serial": serial,
        "firmware_version": "5.15.1.55",
        "product_id": "0B3A",
        "product_line": "D400",
        "usb_type": "3.2",
        "physical_port": "test-port",
        "camera_locked": "YES",
    }
    result.update(updates)
    return result


@pytest.mark.parametrize(
    ("model", "product_id", "expected"),
    [
        ("D435i", "0B3A", "D435i"),
        ("RealSense D435I", "0B3A", "D435i"),
        ("Intel RealSense D435I", "0b3a", "D435i"),
        ("D435", "0B07", None),
        ("D415", "0AD3", None),
        ("D455", "0B5C", None),
    ],
)
def test_canonical_model_is_explicit(model: str, product_id: str, expected: str | None) -> None:
    assert canonical_d435i(model, product_id) == expected


def test_discovery_requires_exact_serial_with_multiple_devices() -> None:
    adapter = FakeAdapter(devices=[_device("other"), _device(CONFIG.camera.serial)])
    selected = discover(CONFIG, adapter)
    assert selected.info.serial == CONFIG.camera.serial
    assert selected.info.canonical_model == "D435i"
    assert selected.info.product_id == "0B3A"


def test_zero_devices_and_wrong_serial_fail_closed() -> None:
    with pytest.raises(DeviceNotFoundError, match="visible device count=0"):
        discover(CONFIG, FakeAdapter(devices=[]))
    with pytest.raises(DeviceNotFoundError, match="visible device count=1"):
        discover(CONFIG, FakeAdapter(devices=[_device("wrong")]))


def test_list_includes_other_realsense_models_without_fuzzy_canonicalization() -> None:
    adapter = FakeAdapter(
        devices=[_device("other", reported_model="Intel RealSense D455", product_id="0B5C")]
    )
    listed = list_devices(adapter)
    assert listed[0].reported_model == "Intel RealSense D455"
    assert listed[0].canonical_model is None


@pytest.mark.parametrize(
    "updates",
    [
        {"reported_model": "Intel RealSense D435", "product_id": "0B07"},
        {"reported_model": "Intel RealSense D435I", "product_id": "0B07"},
    ],
)
def test_wrong_model_or_product_id_fails(updates: dict[str, object]) -> None:
    with pytest.raises(DeviceMismatchError, match="model mismatch"):
        discover(CONFIG, FakeAdapter(devices=[_device(CONFIG.camera.serial, **updates)]))


def test_ir_stream_indices_and_sensor_are_distinct() -> None:
    profiles = {profile.stream_name: profile for profile in _profiles()}
    assert profiles["ir_left"].index == 1
    assert profiles["ir_right"].index == 2
    assert profiles["ir_left"].sensor_identifier == "Stereo Module"


def test_unsupported_profile_fails() -> None:
    supported = tuple(value for value in _profiles() if value.stream_name != "ir_right")
    with pytest.raises(ProfileNotSupportedError, match="ir_right"):
        validate_supported(_profiles(), supported)


def test_lifecycle_open_warmup_close_and_reopen() -> None:
    adapter = FakeAdapter()
    driver = RealSenseDriver(CONFIG, adapter)
    driver.open()
    assert driver.state is CameraLifecycleState.STREAMING
    assert adapter.configured_serial == CONFIG.camera.serial
    assert adapter.wait_calls == CONFIG.capture.warmup_frames
    with pytest.raises(LifecycleError, match="cannot open"):
        driver.open()
    driver.close()
    driver.close()
    assert driver.state is CameraLifecycleState.CLOSED
    driver.open()
    driver.close()
    assert adapter.stop_calls == 2


def test_close_before_open_is_idempotent() -> None:
    driver = RealSenseDriver(CONFIG, FakeAdapter())
    driver.close()
    driver.close()
    assert driver.state is CameraLifecycleState.CLOSED


def test_start_failure_attempts_cleanup() -> None:
    adapter = FakeAdapter()
    adapter.fail_start = True
    driver = RealSenseDriver(CONFIG, adapter)
    with pytest.raises(LifecycleError, match="start failure"):
        driver.open()
    assert driver.state is CameraLifecycleState.FAILED
    assert adapter.stop_calls == 1


def test_context_manager_cleans_up_after_exception() -> None:
    adapter = FakeAdapter()
    with pytest.raises(RuntimeError, match="body"), RealSenseDriver(CONFIG, adapter):
        raise RuntimeError("body")
    assert adapter.stop_calls == 1


def test_missing_optional_dependency_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fail_import)
    adapter = RealSenseSDKAdapter()
    with pytest.raises(MissingOptionalDependencyError, match=r"camera-rig\[realsense\]"):
        _ = adapter.rs


def test_unknown_driver_registry_fails() -> None:
    unknown = replace(CONFIG, camera=replace(CONFIG.camera, driver="unknown"))
    with pytest.raises(UnsupportedDriverError, match="unknown"):
        create_driver(unknown)
