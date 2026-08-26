"""Strict JSON Schema loading and error reporting."""

from __future__ import annotations

import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource

from camera_rig.artifacts.io import JsonValue, load_json
from camera_rig.core.errors import ArtifactError, SchemaValidationError


def schema_path(name: str) -> Path:
    """Resolve a repository or installed data-file schema path."""
    repository_candidate = Path(__file__).resolve().parents[3] / "schemas" / name
    if repository_candidate.is_file():
        return repository_candidate
    installed_candidate = (
        Path(sysconfig.get_path("data")) / "share" / "camera-rig" / "schemas" / name
    )
    if installed_candidate.is_file():
        return installed_candidate
    raise SchemaValidationError(f"schema file is unavailable: {name}")


def validate_against_named_schema(instance: JsonValue, schema_name: str) -> None:
    """Validate a JSON-compatible value and report the most relevant precise path."""
    try:
        schema = load_json(schema_path(schema_name))
    except ArtifactError as error:
        raise SchemaValidationError(str(error)) from error
    if not isinstance(schema, dict):
        raise SchemaValidationError(f"schema {schema_name} must be a JSON object")
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, registry=_local_schema_registry(schema_name))
    except SchemaError as error:
        raise SchemaValidationError(
            f"invalid bundled schema {schema_name}: {error.message}"
        ) from error
    errors = sorted(validator.iter_errors(instance), key=_validation_error_key)
    if errors:
        first_error = errors[0]
        raise SchemaValidationError(
            f"{_format_path(first_error.absolute_path)}: {first_error.message}"
        )


def _local_schema_registry(
    primary_schema_name: str,
) -> Registry[bool | Mapping[str, Any]]:
    """Build an offline registry for references between bundled schemas."""
    directory = schema_path(primary_schema_name).parent
    registry: Registry[bool | Mapping[str, Any]] = Registry()
    for candidate in sorted(directory.glob("*.json")):
        try:
            value = load_json(candidate)
        except ArtifactError as error:
            raise SchemaValidationError(str(error)) from error
        if not isinstance(value, dict):
            continue
        identifier = value.get("$id")
        if isinstance(identifier, str):
            registry = registry.with_resource(identifier, Resource.from_contents(value))
    return registry


def _validation_error_key(error: ValidationError) -> tuple[int, str, str]:
    return (-len(error.absolute_path), _format_path(error.absolute_path), error.message)


def _format_path(parts: Sequence[object]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path
