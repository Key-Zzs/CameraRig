"""Strict source specification for printable ChArUco targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import json_safe
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import ArtifactError, ConfigurationError, SchemaValidationError


@dataclass(frozen=True)
class CharucoTargetSpec:
    """Validated source parameters used to generate one immutable board artifact."""

    target_name: str
    target_frame: str
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    border_bits: int
    legacy_pattern: bool
    page_size: str
    orientation: str
    dpi: int
    horizontal_check_length_mm: float
    vertical_check_length_mm: float
    source_config_sha256: str
    schema_version: str = "camera-rig.target.charuco.v1"
    print_mode: str = "generated"
    page_type: str = "A4"
    page_width_mm: float = 297.0
    page_height_mm: float = 210.0
    board_x_mm: float = 15.0
    board_y_mm: float = 30.0
    board_only: bool = False
    separate_scale_check: bool = False
    plugin: str = "charuco"

    @property
    def board_width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def board_height_m(self) -> float:
        return self.squares_y * self.square_length_m

    @property
    def charuco_corner_count(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)


def load_charuco_target_spec(path: str | Path) -> CharucoTargetSpec:
    """Load strict YAML and enforce relationships not expressible in the JSON Schema."""
    source = Path(path)
    try:
        decoded: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigurationError(
            f"could not load ChArUco target config {source}: {error}"
        ) from error
    try:
        value = json_safe(decoded)
        if not isinstance(value, dict):
            raise ConfigurationError("ChArUco target config root must be an object")
        schema_version_value = value.get("schema_version")
        if not isinstance(schema_version_value, str):
            raise ConfigurationError("ChArUco schema_version must be a string")
        schema_version = schema_version_value
        schema_name = {
            "camera-rig.target.charuco.v1": "charuco_target.v1.schema.json",
            "camera-rig.target.charuco.v2": "charuco_target.v2.schema.json",
        }.get(schema_version)
        if schema_name is None:
            raise ConfigurationError(f"unsupported ChArUco target schema: {schema_version!r}")
        validate_against_named_schema(value, schema_name)
    except (ArtifactError, SchemaValidationError) as error:
        raise ConfigurationError(f"invalid ChArUco target config: {error}") from error
    target = _object(value["target"], "target")
    printing = _object(value["print"], "print")
    squares_x = _int(target["squares_x"], "target.squares_x")
    squares_y = _int(target["squares_y"], "target.squares_y")
    square_length = _number(target["square_length_m"], "target.square_length_m")
    marker_length = _number(target["marker_length_m"], "target.marker_length_m")
    if marker_length >= square_length:
        raise ConfigurationError("target.marker_length_m must be smaller than square_length_m")
    if schema_version == "camera-rig.target.charuco.v1":
        print_values: dict[str, object] = {
            "page_size": _string(printing["page_size"], "print.page_size"),
            "orientation": _string(printing["orientation"], "print.orientation"),
            "page_type": "A4",
            "page_width_mm": 297.0,
            "page_height_mm": 210.0,
            "board_x_mm": 15.0,
            "board_y_mm": 30.0,
            "board_only": False,
            "separate_scale_check": False,
            "horizontal_check_length_mm": _number(
                printing["horizontal_check_length_mm"], "print.horizontal_check_length_mm"
            ),
            "vertical_check_length_mm": _number(
                printing["vertical_check_length_mm"], "print.vertical_check_length_mm"
            ),
        }
    else:
        page = _object(printing["page"], "print.page")
        layout = _object(printing["layout"], "print.layout")
        scale = _object(printing["scale_check"], "print.scale_check")
        page_type = _string(page["type"], "print.page.type")
        if page_type == "custom":
            page_width_mm = _number(page["width_mm"], "print.page.width_mm")
            page_height_mm = _number(page["height_mm"], "print.page.height_mm")
        elif page_type == "A4":
            page_width_mm, page_height_mm = 210.0, 297.0
        else:
            page_width_mm, page_height_mm = 297.0, 420.0
        print_values = {
            "page_size": page_type,
            "orientation": ("landscape" if page_width_mm >= page_height_mm else "portrait"),
            "page_type": page_type,
            "page_width_mm": page_width_mm,
            "page_height_mm": page_height_mm,
            "board_x_mm": _number(layout["board_x_mm"], "print.layout.board_x_mm"),
            "board_y_mm": _number(layout["board_y_mm"], "print.layout.board_y_mm"),
            "board_only": _bool(printing["board_only"], "print.board_only"),
            "separate_scale_check": _bool(
                scale["separate_page"], "print.scale_check.separate_page"
            ),
            "horizontal_check_length_mm": _number(
                scale["horizontal_length_mm"], "print.scale_check.horizontal_length_mm"
            ),
            "vertical_check_length_mm": _number(
                scale["vertical_length_mm"], "print.scale_check.vertical_length_mm"
            ),
        }
        if printing["mode"] != "generated":
            raise ConfigurationError("print.mode must be generated")
        if scale["enabled"] is not True:
            raise ConfigurationError("print.scale_check.enabled must be true")
        if (
            _number(print_values["board_x_mm"], "print.layout.board_x_mm")
            + squares_x * square_length * 1000.0
            > page_width_mm + 1e-6
            or _number(print_values["board_y_mm"], "print.layout.board_y_mm")
            + squares_y * square_length * 1000.0
            > page_height_mm + 1e-6
        ):
            raise ConfigurationError("generated board does not fit within the configured page")
    return CharucoTargetSpec(
        target_name=_string(target["name"], "target.name"),
        target_frame=_string(target["target_frame"], "target.target_frame"),
        dictionary=_string(target["dictionary"], "target.dictionary"),
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_length,
        marker_length_m=marker_length,
        border_bits=_int(target["border_bits"], "target.border_bits"),
        legacy_pattern=_bool(target["legacy_pattern"], "target.legacy_pattern"),
        page_size=str(print_values["page_size"]),
        orientation=str(print_values["orientation"]),
        dpi=_int(printing["dpi"], "print.dpi"),
        horizontal_check_length_mm=_number(
            print_values["horizontal_check_length_mm"], "horizontal_check_length_mm"
        ),
        vertical_check_length_mm=_number(
            print_values["vertical_check_length_mm"], "vertical_check_length_mm"
        ),
        source_config_sha256=sha256_file(source),
        schema_version=str(schema_version),
        print_mode="generated",
        page_type=str(print_values["page_type"]),
        page_width_mm=_number(print_values["page_width_mm"], "page_width_mm"),
        page_height_mm=_number(print_values["page_height_mm"], "page_height_mm"),
        board_x_mm=_number(print_values["board_x_mm"], "board_x_mm"),
        board_y_mm=_number(print_values["board_y_mm"], "board_y_mm"),
        board_only=_bool(print_values["board_only"], "board_only"),
        separate_scale_check=_bool(print_values["separate_scale_check"], "separate_scale_check"),
    )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a string")
    return value


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be a number")
    return float(value)
