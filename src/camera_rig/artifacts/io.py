"""Deterministic and atomic JSON artifact I/O."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from camera_rig.core.errors import ArtifactError

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def json_safe(value: object) -> JsonValue:
    """Recursively convert supported Python and NumPy values to JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactError("JSON artifacts cannot contain NaN or infinity")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if not math.isfinite(converted):
            raise ArtifactError("JSON artifacts cannot contain NaN or infinity")
        return converted
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactError("JSON object keys must be strings")
            result[key] = json_safe(item)
        return result
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: json_safe(getattr(value, item.name)) for item in fields(value)}
    raise ArtifactError(f"value of type {type(value).__name__} is not JSON-safe")


def deterministic_json_bytes(value: object) -> bytes:
    """Encode a value using the stable CameraRig JSON formatting contract."""
    try:
        text = json.dumps(
            json_safe(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"could not encode JSON artifact: {error}") from error
    return f"{text}\n".encode()


def atomic_write_json(path: str | Path, value: object) -> None:
    """Atomically replace ``path`` with deterministic UTF-8 JSON.

    A temporary sibling is flushed and synced before ``os.replace``. Any failure removes
    the temporary file and leaves an existing target untouched.
    """
    target = Path(path)
    payload = deterministic_json_bytes(value)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
    except OSError as error:
        raise ArtifactError(f"could not prepare atomic write for {target}: {error}") from error
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except OSError as error:
        raise ArtifactError(f"could not atomically write {target}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: str | Path) -> JsonValue:
    """Load UTF-8 JSON, reporting file and parse failures as ``ArtifactError``."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            value: object = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"could not load JSON artifact {source}: {error}") from error
    return json_safe(value)
