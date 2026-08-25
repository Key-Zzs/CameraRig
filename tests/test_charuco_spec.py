from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from camera_rig.core.errors import ConfigurationError
from camera_rig.targets.charuco.spec import load_charuco_target_spec

REPOSITORY_ROOT = Path(__file__).parents[1]
STANDARD_CONFIG = REPOSITORY_ROOT / "configs/targets/charuco_a4_v1.yaml"
pytestmark = pytest.mark.charuco


def test_standard_charuco_spec_is_strict_and_physical() -> None:
    spec = load_charuco_target_spec(STANDARD_CONFIG)
    assert spec.plugin == "charuco"
    assert spec.dictionary == "DICT_5X5_100"
    assert (spec.squares_x, spec.squares_y) == (7, 5)
    assert spec.charuco_corner_count == 24
    assert spec.board_width_m == pytest.approx(0.210)
    assert spec.board_height_m == pytest.approx(0.150)
    assert spec.legacy_pattern is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data["target"].__setitem__("unknown", 1), "Additional properties"),
        (lambda data: data["target"].__setitem__("dictionary", "DICT_UNKNOWN"), "not one of"),
        (lambda data: data["target"].__setitem__("squares_x", 2), "minimum"),
        (lambda data: data["target"].__setitem__("legacy_pattern", None), "boolean"),
        (lambda data: data["print"].__setitem__("page_size", "Letter"), "A4"),
        (lambda data: data["coordinate_frame"].__setitem__("y_axis", "board_down"), "board_up"),
    ],
)
def test_invalid_source_specs_fail_closed(tmp_path: Path, mutator: object, message: str) -> None:
    data = yaml.safe_load(STANDARD_CONFIG.read_text(encoding="utf-8"))
    assert callable(mutator)
    mutator(data)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_charuco_target_spec(path)


def test_marker_must_be_smaller_than_square(tmp_path: Path) -> None:
    data = copy.deepcopy(yaml.safe_load(STANDARD_CONFIG.read_text(encoding="utf-8")))
    data["target"]["marker_length_m"] = data["target"]["square_length_m"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="smaller"):
        load_charuco_target_spec(path)
