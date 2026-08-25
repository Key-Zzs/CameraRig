"""Strict YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from camera_rig.artifacts.io import json_safe
from camera_rig.config.models import CameraConfig
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import ArtifactError, ConfigurationError, ContractError


def load_config(path: str | Path) -> CameraConfig:
    """Load, strictly validate, and reconstruct a versioned single-camera config."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConfigurationError(f"could not read configuration {source}: {error}") from error
    try:
        decoded: object = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"could not parse YAML configuration {source}: {error}") from error
    try:
        value = json_safe(decoded)
    except ArtifactError as error:
        raise ConfigurationError(
            f"configuration contains unsupported YAML values: {error}"
        ) from error
    validate_against_named_schema(value, "camera_config.v1.schema.json")
    if not isinstance(value, dict):
        raise ConfigurationError("configuration root must be a mapping")
    try:
        typed_value: dict[str, object] = dict(value)
        return CameraConfig.from_dict(typed_value)
    except (KeyError, TypeError, ValueError, ContractError) as error:
        raise ConfigurationError(f"configuration contract is invalid: {error}") from error
