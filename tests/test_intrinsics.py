from __future__ import annotations

import math

import pytest

from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics


def _intrinsics(**changes: object) -> CameraIntrinsics:
    values: dict[str, object] = {
        "frame": "head/color_optical",
        "width": 640,
        "height": 480,
        "fx": 600.0,
        "fy": 601.0,
        "cx": 319.5,
        "cy": 239.5,
        "distortion_model": "none",
        "distortion_coeffs": (),
    }
    values.update(changes)
    return CameraIntrinsics(**values)  # type: ignore[arg-type]


def test_valid_intrinsics() -> None:
    assert _intrinsics().fx == 600.0


@pytest.mark.parametrize(("field", "value"), [("fx", -1.0), ("fy", 0.0)])
def test_non_positive_focal_length_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ContractError, match="fx and fy"):
        _intrinsics(**{field: value})


@pytest.mark.parametrize(("field", "value"), [("cx", 640.0), ("cx", -0.1), ("cy", 480.0)])
def test_principal_point_outside_image_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ContractError, match=field):
        _intrinsics(**{field: value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(value: float) -> None:
    with pytest.raises(ContractError, match="finite"):
        _intrinsics(fx=value)


@pytest.mark.parametrize(("field", "value"), [("width", 0), ("height", -1)])
def test_invalid_dimensions_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ContractError, match=field):
        _intrinsics(**{field: value})
