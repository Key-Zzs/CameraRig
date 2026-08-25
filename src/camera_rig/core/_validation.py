"""Small validation primitives shared by core contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeVar

from camera_rig.core.errors import ContractError

T = TypeVar("T")


def require_non_empty(value: str, field_name: str) -> str:
    """Return a stripped non-empty string or raise a contract error."""
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value


def require_positive_int(value: int, field_name: str) -> int:
    """Require a positive integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field_name} must be a positive integer")
    return value


def require_non_negative_int(value: int, field_name: str) -> int:
    """Require a non-negative integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field_name} must be a non-negative integer")
    return value


def require_positive_finite(value: float, field_name: str) -> float:
    """Require a finite, strictly positive number."""
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ContractError(f"{field_name} must be finite and greater than zero")
    return numeric


def string_keyed_copy(values: Mapping[str, T], field_name: str) -> dict[str, T]:
    """Copy a mapping after ensuring all keys are non-empty strings."""
    copied: dict[str, T] = {}
    for key, value in values.items():
        require_non_empty(key, f"{field_name} key")
        copied[key] = value
    return copied


def decoded_string(value: object, field_name: str) -> str:
    """Require a decoded JSON string without coercion."""
    if not isinstance(value, str):
        raise ContractError(f"{field_name} must be a string")
    return value


def decoded_optional_string(value: object, field_name: str) -> str | None:
    """Require a decoded JSON string or null without coercion."""
    if value is None:
        return None
    return decoded_string(value, field_name)


def decoded_int(value: object, field_name: str) -> int:
    """Require a decoded JSON integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{field_name} must be an integer")
    return value


def decoded_float(value: object, field_name: str) -> float:
    """Require a decoded JSON number, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{field_name} must be a number")
    return float(value)


def decoded_bool(value: object, field_name: str) -> bool:
    """Require a decoded JSON boolean without coercion."""
    if not isinstance(value, bool):
        raise ContractError(f"{field_name} must be a boolean")
    return value
