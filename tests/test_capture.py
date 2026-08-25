from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from camera_rig.capture.session import CameraSession
from camera_rig.capture.synchronization import build_sync_report
from camera_rig.capture.validation import StreamValidationAccumulator
from camera_rig.config.loader import load_config
from camera_rig.core.errors import ConfigurationError, ContractError
from camera_rig.core.frame import StreamFrame
from camera_rig.drivers.realsense.frame_conversion import convert_frameset

ROOT = Path(__file__).parents[1]
CONFIG = load_config(ROOT / "configs/examples/single_camera_contract.yaml")


class FrameAdapter:
    def frameset_frames(self, frameset: object) -> dict[str, object]:
        assert isinstance(frameset, dict)
        return frameset

    def frame_array(self, frame: object) -> np.ndarray:
        assert isinstance(frame, dict)
        value = frame["data"]
        assert isinstance(value, np.ndarray)
        return value

    def frame_number(self, frame: object) -> int:
        assert isinstance(frame, dict)
        return int(frame["number"])

    def frame_timestamp(self, frame: object) -> float:
        assert isinstance(frame, dict)
        return float(frame["timestamp"])

    def frame_timestamp_domain(self, frame: object) -> str:
        assert isinstance(frame, dict)
        return str(frame["domain"])

    def frame_metadata(self, frame: object) -> dict[str, object]:
        assert isinstance(frame, dict)
        return {"frame_counter": frame["number"]}


def _frameset(number: int = 7) -> dict[str, object]:
    return {
        "color": {
            "data": np.arange(640 * 480 * 3, dtype=np.uint8).reshape(480, 640, 3),
            "number": number,
            "timestamp": 1000.125,
            "domain": "hardware_clock",
        },
        "depth": {
            "data": np.ones((480, 640), dtype=np.uint16),
            "number": number + 20,
            "timestamp": 1000.5,
            "domain": "hardware_clock",
        },
        "ir_left": {
            "data": np.arange(640 * 480, dtype=np.uint8).reshape(480, 640),
            "number": number,
            "timestamp": 1000.0,
            "domain": "hardware_clock",
        },
        "ir_right": {
            "data": np.flipud(np.arange(640 * 480, dtype=np.uint8).reshape(480, 640)),
            "number": number,
            "timestamp": 1000.25,
            "domain": "hardware_clock",
        },
    }


def test_frameset_conversion_copies_raw_arrays_and_timestamps() -> None:
    raw = _frameset()
    frame = convert_frameset(CONFIG, FrameAdapter(), raw)  # type: ignore[arg-type]
    assert frame.color is not None and frame.color.data.dtype == np.uint8
    assert frame.color.data.shape == (480, 640, 3)
    assert frame.depth is not None and frame.depth.data.dtype == np.uint16
    assert frame.ir_left is not None and frame.ir_right is not None
    assert frame.ir_left.sensor_timestamp_ns == 1_000_000_000
    assert frame.color.sensor_timestamp_ns == 1_000_125_000
    assert frame.color.original_timestamp == 1000.125
    original = frame.color.data.copy()
    raw_color = raw["color"]
    assert isinstance(raw_color, dict)
    raw_array = raw_color["data"]
    assert isinstance(raw_array, np.ndarray)
    raw_array.fill(0)
    np.testing.assert_array_equal(frame.color.data, original)
    assert frame.sync_report is not None and frame.sync_report.valid


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("color", np.zeros((480, 640, 3), dtype=np.float32), "dtype mismatch"),
        ("depth", np.zeros((479, 640), dtype=np.uint16), "shape mismatch"),
    ],
)
def test_conversion_rejects_wrong_shape_or_dtype(name: str, data: np.ndarray, message: str) -> None:
    raw = _frameset()
    stream = raw[name]
    assert isinstance(stream, dict)
    stream["data"] = data
    with pytest.raises(ContractError, match=message):
        convert_frameset(CONFIG, FrameAdapter(), raw)  # type: ignore[arg-type]


def test_conversion_rejects_missing_required_stream() -> None:
    raw = _frameset()
    del raw["ir_right"]
    with pytest.raises(ContractError, match="missing required"):
        convert_frameset(CONFIG, FrameAdapter(), raw)  # type: ignore[arg-type]


def test_sync_report_does_not_subtract_different_domains() -> None:
    streams = {
        "ir_left": StreamFrame(
            "ir_left", np.zeros((1, 1), dtype=np.uint8), 1, 100, "hardware_clock"
        ),
        "ir_right": StreamFrame(
            "ir_right", np.zeros((1, 1), dtype=np.uint8), 1, 10_000, "system_time"
        ),
    }
    report = build_sync_report(CONFIG, streams)
    assert report.per_stream_skew_ns == {"ir_left": 0}
    assert "ir_right" not in report.comparable_streams
    assert any("differs" in warning for warning in report.warnings)


def test_zero_copy_configuration_fails_fast() -> None:
    config = replace(CONFIG, capture=replace(CONFIG.capture, copy_frames=False))
    with pytest.raises(ConfigurationError, match="zero-copy"):
        CameraSession.from_config(config)


def test_stream_validation_statistics_pass_for_contiguous_30_fps() -> None:
    accumulator = StreamValidationAccumulator(CONFIG, requested_frames=3)
    adapter = FrameAdapter()
    for index in range(3):
        raw = _frameset(number=10 + index)
        for stream in raw.values():
            assert isinstance(stream, dict)
            stream["timestamp"] = float(stream["timestamp"]) + index * (1000 / 30)
        accumulator.add(convert_frameset(CONFIG, adapter, raw))  # type: ignore[arg-type]
    report = accumulator.report()
    assert report["status"] == "PASS"
    assert report["per_stream_frame_number_discontinuities"] == {
        "color": 0,
        "depth": 0,
        "ir_left": 0,
        "ir_right": 0,
    }
    observed = report["per_stream_observed_fps"]
    assert isinstance(observed, dict)
    assert observed["color"] == pytest.approx(30.0, rel=1e-6)


def test_stream_validation_statistics_reject_discontinuity() -> None:
    accumulator = StreamValidationAccumulator(CONFIG, requested_frames=2)
    adapter = FrameAdapter()
    for index, number in enumerate((10, 12)):
        raw = _frameset(number=number)
        for stream in raw.values():
            assert isinstance(stream, dict)
            stream["timestamp"] = float(stream["timestamp"]) + index * (1000 / 30)
        accumulator.add(convert_frameset(CONFIG, adapter, raw))  # type: ignore[arg-type]
    report = accumulator.report()
    assert report["status"] == "FAIL"
    assert any("discontinuity ratio" in reason for reason in report["failure_reasons"])
