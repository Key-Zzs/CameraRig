from __future__ import annotations

import numpy as np
import pytest

from camera_rig.core.device_info import CameraDeviceInfo
from camera_rig.core.errors import ContractError
from camera_rig.core.frame import CameraFrame, StreamFrame
from camera_rig.core.stream import StreamProfile
from camera_rig.core.timestamps import SingleDeviceSyncReport


def test_device_serial_remains_string() -> None:
    info = CameraDeviceInfo("synthetic", "head", "test", "test", "000123")
    assert info.serial == "000123"


@pytest.mark.parametrize(("field", "value"), [("width", 0), ("height", -1), ("fps", 0)])
def test_stream_profile_validation(field: str, value: int) -> None:
    values = {
        "stream_name": "color",
        "width": 640,
        "height": 480,
        "fps": 30,
        "format": "rgb8",
    }
    values[field] = value
    with pytest.raises(ContractError, match=field):
        StreamProfile(**values)  # type: ignore[arg-type]


def test_camera_frame_accessors_return_none_for_missing_streams() -> None:
    color = StreamFrame("color", np.zeros((2, 2, 3), dtype=np.uint8), frame_number=1)
    frame = CameraFrame("head", "001", {"color": color}, 100)
    assert frame.rgb is color
    assert frame.depth is None
    assert frame.ir_left is None
    assert frame.ir_right is None


def test_camera_frame_rejects_mismatched_stream_key() -> None:
    color = StreamFrame("color", np.zeros((2, 2, 3), dtype=np.uint8), frame_number=1)
    with pytest.raises(ContractError, match="does not match"):
        CameraFrame("head", "001", {"depth": color}, 100)


def test_single_device_sync_report_is_data_only() -> None:
    report = SingleDeviceSyncReport(
        valid=True,
        comparable_streams=("color", "depth"),
        max_skew_ns=100,
        per_stream_skew_ns={"color": 0, "depth": 100},
        frame_number_match=True,
    )
    assert report.to_dict()["max_skew_ns"] == 100
