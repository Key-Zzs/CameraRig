"""Reserved eye-in-hand calibration interface."""

from __future__ import annotations

from typing import NoReturn

from camera_rig.core.errors import FeatureNotAvailableError

_MESSAGE = "Eye-in-hand calibration is reserved but not implemented."


class EyeInHandCalibrator:
    """Reserved interface that fails explicitly until a later implementation."""

    def calibrate(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Fail explicitly; no placeholder result is fabricated."""
        raise FeatureNotAvailableError(_MESSAGE)


def calibrate_eye_in_hand(*_args: object, **_kwargs: object) -> NoReturn:
    """Fail explicitly; eye-in-hand calibration is not available."""
    raise FeatureNotAvailableError(_MESSAGE)
