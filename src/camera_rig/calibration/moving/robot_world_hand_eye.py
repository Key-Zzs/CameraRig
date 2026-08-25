"""Reserved robot-world/hand-eye calibration interface."""

from __future__ import annotations

from typing import NoReturn

from camera_rig.core.errors import FeatureNotAvailableError

_MESSAGE = "Robot-world/hand-eye calibration is reserved but not implemented."


class RobotWorldHandEyeCalibrator:
    """Reserved interface that fails explicitly until a later implementation."""

    def calibrate(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Fail explicitly; no placeholder result is fabricated."""
        raise FeatureNotAvailableError(_MESSAGE)


def calibrate_robot_world_hand_eye(*_args: object, **_kwargs: object) -> NoReturn:
    """Fail explicitly; robot-world/hand-eye calibration is not available."""
    raise FeatureNotAvailableError(_MESSAGE)
