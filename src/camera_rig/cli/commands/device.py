"""Read-only RealSense inspection and lifecycle smoke commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from camera_rig.artifacts.io import atomic_write_json, json_safe
from camera_rig.config.loader import load_config
from camera_rig.core.errors import ConfigurationError, UnsupportedDriverError
from camera_rig.drivers.realsense.discovery import list_devices
from camera_rig.drivers.realsense.driver import RealSenseDriver
from camera_rig.drivers.realsense.sdk_adapter import RealSenseSDKAdapter


def add_device_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the device command group."""
    parser = commands.add_parser("device", help="physical device discovery and lifecycle")
    subcommands = parser.add_subparsers(dest="device_command", required=True)

    list_parser = subcommands.add_parser("list", help="list visible devices without opening them")
    list_parser.add_argument("--driver", default="realsense")
    list_parser.add_argument("--json", action="store_true", dest="as_json")
    list_parser.set_defaults(handler=_list)

    inspect_parser = subcommands.add_parser("inspect", help="inspect one configured device")
    inspect_parser.add_argument("--config", type=Path, required=True)
    inspect_parser.add_argument("--show-profiles", action="store_true")
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")
    inspect_parser.set_defaults(handler=_inspect)

    smoke_parser = subcommands.add_parser("smoke", help="repeat open, warmup, and close")
    smoke_parser.add_argument("--config", type=Path, required=True)
    smoke_parser.add_argument("--cycles", type=int, default=3)
    smoke_parser.add_argument("--report", type=Path, required=True)
    smoke_parser.set_defaults(handler=_smoke)


def _list(arguments: argparse.Namespace) -> int:
    _require_realsense(arguments.driver)
    devices = list_devices(RealSenseSDKAdapter())
    payload = [device.to_dict() for device in devices]
    if arguments.as_json:
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    else:
        if not devices:
            print("No RealSense devices found.")
        for device in devices:
            print(
                f"{device.canonical_model or device.reported_model} "
                f"serial={device.serial} product_id={device.product_id or 'unknown'} "
                f"usb={device.usb_type or 'unknown'}"
            )
    return 0


def _inspect(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    _require_realsense(config.camera.driver)
    driver = RealSenseDriver(config)
    device = driver.get_device_info()
    payload: dict[str, object] = {"device": device.to_dict()}
    if arguments.show_profiles:
        payload["supported_profiles"] = [
            profile.to_dict() for profile in driver.get_supported_profiles()
        ]
    if arguments.as_json:
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    else:
        print(
            f"model={device.reported_model} canonical={device.canonical_model} "
            f"serial={device.serial} product_id={device.product_id} usb={device.usb_type}"
        )
        if arguments.show_profiles:
            for profile in driver.get_supported_profiles():
                print(
                    f"{profile.stream_name}[{profile.index}] {profile.width}x{profile.height}@"
                    f"{profile.fps} {profile.format} sensor={profile.sensor_identifier}"
                )
    return 0


def _smoke(arguments: argparse.Namespace) -> int:
    if arguments.cycles < 1:
        raise ConfigurationError("cycles must be greater than zero")
    config = load_config(arguments.config)
    _require_realsense(config.camera.driver)
    cycles: list[dict[str, object]] = []
    started = time.monotonic()
    for index in range(arguments.cycles):
        cycle_started = time.monotonic()
        driver = RealSenseDriver(config)
        try:
            driver.open()
            info = driver.get_device_info()
            active = [value.to_dict() for value in driver.active_profiles]
        finally:
            driver.close()
        cycles.append(
            {
                "cycle": index + 1,
                "duration_s": time.monotonic() - cycle_started,
                "closed_state": driver.state.value,
                "active_profiles": active,
            }
        )
        print(f"cycle {index + 1}/{arguments.cycles}: PASS")
    report = {
        "schema_version": "camera-rig.device-smoke.v1",
        "status": "PASS",
        "device": info.to_dict(),
        "requested_cycles": arguments.cycles,
        "completed_cycles": len(cycles),
        "duration_s": time.monotonic() - started,
        "cycles": cycles,
    }
    atomic_write_json(arguments.report, report)
    print(f"device smoke: PASS ({len(cycles)} cycles)")
    return 0


def _require_realsense(driver: str) -> None:
    if driver.casefold() != "realsense":
        raise UnsupportedDriverError(f"unsupported camera driver: {driver!r}")
