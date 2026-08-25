from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from camera_rig.cli.main import main
from camera_rig.config.loader import load_config
from camera_rig.core.errors import ConfigurationError, SchemaValidationError

REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLE = REPOSITORY_ROOT / "configs/examples/single_camera_contract.yaml"


def _example_data() -> dict[str, object]:
    loaded = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_yaml(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_valid_yaml() -> None:
    config = load_config(EXAMPLE)
    assert config.camera.serial == "344522070241"
    assert config.camera.output_reference_stream == "ir_left"
    assert set(config.streams) == {"color", "depth", "ir_left", "ir_right"}


@pytest.mark.parametrize("content", ["", "- not\n- a\n- mapping\n"])
def test_empty_or_non_mapping_yaml_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SchemaValidationError, match=r"^\$"):
        load_config(path)


def test_unknown_root_field_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    data["unknown"] = True
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        load_config(_write_yaml(tmp_path, data))


def test_plural_cameras_root_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    data["cameras"] = [data.pop("camera")]
    with pytest.raises(SchemaValidationError, match="camera"):
        load_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize("serial", [344522070241, "", "   "])
def test_invalid_serial_is_rejected(tmp_path: Path, serial: object) -> None:
    data = _example_data()
    camera = data["camera"]
    assert isinstance(camera, dict)
    camera["serial"] = serial
    with pytest.raises(SchemaValidationError, match=r"\$\.camera\.serial"):
        load_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize(("field", "value"), [("width", 0), ("height", -1), ("fps", 0)])
def test_invalid_stream_numbers_are_rejected(tmp_path: Path, field: str, value: int) -> None:
    data = _example_data()
    streams = data["streams"]
    assert isinstance(streams, dict)
    color = streams["color"]
    assert isinstance(color, dict)
    color[field] = value
    with pytest.raises(SchemaValidationError, match=rf"\$\.streams\.color\.{field}"):
        load_config(_write_yaml(tmp_path, data))


def test_unknown_camera_field_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    camera = data["camera"]
    assert isinstance(camera, dict)
    camera["device_index"] = 0
    with pytest.raises(SchemaValidationError, match=r"\$\.camera"):
        load_config(_write_yaml(tmp_path, data))


def test_unknown_stream_field_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    streams = data["streams"]
    assert isinstance(streams, dict)
    color = streams["color"]
    assert isinstance(color, dict)
    color["align"] = True
    with pytest.raises(SchemaValidationError, match=r"\$\.streams\.color"):
        load_config(_write_yaml(tmp_path, data))


def test_unknown_stream_name_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    streams = data["streams"]
    assert isinstance(streams, dict)
    streams["fisheye"] = copy.deepcopy(streams["color"])
    with pytest.raises(SchemaValidationError, match=r"\$\.streams"):
        load_config(_write_yaml(tmp_path, data))


def test_schema_version_mismatch_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    data["schema_version"] = "camera-rig.config.v2"
    with pytest.raises(SchemaValidationError, match=r"\$\.schema_version"):
        load_config(_write_yaml(tmp_path, data))


def test_disabled_reference_stream_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    streams = data["streams"]
    assert isinstance(streams, dict)
    ir_left = streams["ir_left"]
    assert isinstance(ir_left, dict)
    ir_left["enabled"] = False
    with pytest.raises(ConfigurationError, match="output_reference_stream must be enabled"):
        load_config(_write_yaml(tmp_path, data))


def test_unavailable_mount_configuration_is_rejected(tmp_path: Path) -> None:
    data = _example_data()
    data["mount"] = {"type": "eye_in_hand"}
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        load_config(_write_yaml(tmp_path, data))


def test_config_cli_success(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "validate", "--config", str(EXAMPLE)]) == 0
    output = capsys.readouterr()
    assert "valid camera-rig.config.v1" in output.out
    assert output.err == ""


def test_config_cli_failure_is_concise(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("cameras: []\n", encoding="utf-8")
    assert main(["config", "validate", "--config", str(path)]) == 2
    output = capsys.readouterr()
    assert "error:" in output.err
    assert "Traceback" not in output.err
