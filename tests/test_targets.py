from __future__ import annotations

import numpy as np
import pytest

from camera_rig.core.errors import ContractError
from camera_rig.core.quality import QualityReport
from camera_rig.targets.base import TargetDetector
from camera_rig.targets.observation import TargetObservation


def _observation(**changes: object) -> TargetObservation:
    values: dict[str, object] = {
        "plugin_name": "synthetic-target",
        "target_frame": "target",
        "point_ids": (1, 2),
        "image_points_px": np.array([[10.0, 20.0], [30.0, 40.0]]),
        "object_points_m": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        "image_size": (640, 480),
        "quality": QualityReport(True),
        "metadata": {},
    }
    values.update(changes)
    return TargetObservation(**values)  # type: ignore[arg-type]


def test_valid_target_observation() -> None:
    observation = _observation()
    assert observation.image_points_px.shape == (2, 2)
    assert not observation.image_points_px.flags.writeable


@pytest.mark.parametrize(
    "changes",
    [
        {"point_ids": (1,)},
        {"image_points_px": np.zeros((2, 3))},
        {"object_points_m": np.zeros((2, 2))},
    ],
)
def test_shape_or_count_mismatch_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        _observation(**changes)


@pytest.mark.parametrize("field", ["image_points_px", "object_points_m"])
def test_non_finite_target_points_are_rejected(field: str) -> None:
    points = np.zeros((2, 2 if field == "image_points_px" else 3))
    points[0, 0] = np.nan
    with pytest.raises(ContractError, match="finite"):
        _observation(**{field: points})


def test_target_detector_protocol_is_importable() -> None:
    assert TargetDetector.__name__ == "TargetDetector"
