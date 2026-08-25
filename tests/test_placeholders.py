from __future__ import annotations

import pytest

from camera_rig.calibration.moving.eye_in_hand import (
    EyeInHandCalibrator,
    calibrate_eye_in_hand,
)
from camera_rig.calibration.moving.robot_world_hand_eye import (
    RobotWorldHandEyeCalibrator,
    calibrate_robot_world_hand_eye,
)
from camera_rig.core.errors import FeatureNotAvailableError


@pytest.mark.parametrize(
    "call",
    [calibrate_eye_in_hand, EyeInHandCalibrator().calibrate],
)
def test_eye_in_hand_fails_explicitly(call: object) -> None:
    assert callable(call)
    with pytest.raises(
        FeatureNotAvailableError,
        match="Eye-in-hand calibration is reserved but not implemented",
    ):
        call()


@pytest.mark.parametrize(
    "call",
    [calibrate_robot_world_hand_eye, RobotWorldHandEyeCalibrator().calibrate],
)
def test_robot_world_hand_eye_fails_explicitly(call: object) -> None:
    assert callable(call)
    with pytest.raises(
        FeatureNotAvailableError,
        match="Robot-world/hand-eye calibration is reserved but not implemented",
    ):
        call()
