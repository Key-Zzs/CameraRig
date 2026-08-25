"""CameraBundle JSON Schema validation."""

from __future__ import annotations

from pathlib import Path

from camera_rig.artifacts.io import JsonValue, load_json
from camera_rig.artifacts.models import CameraBundle
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import ArtifactError, ContractError, SchemaValidationError


def validate_bundle_data(value: JsonValue) -> CameraBundle:
    """Validate decoded JSON against schema and reconstruct the typed bundle."""
    if not isinstance(value, dict):
        raise ArtifactError("camera bundle root must be a JSON object")
    try:
        validate_against_named_schema(value, "camera_bundle.v1.schema.json")
    except SchemaValidationError as error:
        raise ArtifactError(str(error)) from error
    try:
        typed_value: dict[str, object] = dict(value)
        return CameraBundle.from_dict(typed_value)
    except (KeyError, TypeError, ValueError, ContractError) as error:
        raise ArtifactError(f"camera bundle contract is invalid: {error}") from error


def load_and_validate_bundle(path: str | Path) -> CameraBundle:
    """Load, schema-validate, and reconstruct a CameraBundle."""
    return validate_bundle_data(load_json(path))
