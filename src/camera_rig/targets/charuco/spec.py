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
        validate_against_named_schema(value, "charuco_target.v1.schema.json")
    except (ArtifactError, SchemaValidationError) as error:
        raise ConfigurationError(f"invalid ChArUco target config: {error}") from error
    if not isinstance(value, dict):
        raise ConfigurationError("ChArUco target config root must be an object")
    target = _object(value["target"], "target")
    printing = _object(value["print"], "print")
    square_length = _number(target["square_length_m"], "target.square_length_m")
    marker_length = _number(target["marker_length_m"], "target.marker_length_m")
    if marker_length >= square_length:
        raise ConfigurationError("target.marker_length_m must be smaller than square_length_m")
    return CharucoTargetSpec(
        target_name=_string(target["name"], "target.name"),
        target_frame=_string(target["target_frame"], "target.target_frame"),
        dictionary=_string(target["dictionary"], "target.dictionary"),
        squares_x=_int(target["squares_x"], "target.squares_x"),
        squares_y=_int(target["squares_y"], "target.squares_y"),
        square_length_m=square_length,
        marker_length_m=marker_length,
        border_bits=_int(target["border_bits"], "target.border_bits"),
        legacy_pattern=_bool(target["legacy_pattern"], "target.legacy_pattern"),
        page_size=_string(printing["page_size"], "print.page_size"),
        orientation=_string(printing["orientation"], "print.orientation"),
        dpi=_int(printing["dpi"], "print.dpi"),
        horizontal_check_length_mm=_number(
            printing["horizontal_check_length_mm"], "print.horizontal_check_length_mm"
        ),
        vertical_check_length_mm=_number(
            printing["vertical_check_length_mm"], "print.vertical_check_length_mm"
        ),
        source_config_sha256=sha256_file(source),
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
