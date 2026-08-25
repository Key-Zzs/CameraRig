"""Extensible calibration and artifact quality report."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from camera_rig.core._validation import decoded_bool, decoded_string, string_keyed_copy
from camera_rig.core.errors import ContractError


@dataclass(frozen=True)
class QualityReport:
    """Generic quality decision with JSON-safe metric and threshold mappings."""

    passed: bool
    metrics: dict[str, object] = field(default_factory=dict)
    thresholds: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", string_keyed_copy(self.metrics, "metrics"))
        object.__setattr__(self, "thresholds", string_keyed_copy(self.thresholds, "thresholds"))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))
        if self.passed and self.failure_reasons:
            raise ContractError("a passed quality report cannot contain failure reasons")
        _require_json_safe(self.metrics, "metrics")
        _require_json_safe(self.thresholds, "thresholds")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe-compatible representation."""
        return {
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "warnings": list(self.warnings),
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> QualityReport:
        """Reconstruct a quality report from decoded JSON data."""
        return cls(
            passed=decoded_bool(data["passed"], "passed"),
            metrics=_mapping(data.get("metrics", {}), "metrics"),
            thresholds=_mapping(data.get("thresholds", {}), "thresholds"),
            warnings=_string_tuple(data.get("warnings", []), "warnings"),
            failure_reasons=_string_tuple(data.get("failure_reasons", []), "failure_reasons"),
        )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return {decoded_string(key, f"{name} key"): item for key, item in value.items()}


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return tuple(decoded_string(item, f"{name}[]") for item in value)


def _require_json_safe(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float | np.floating):
        if not math.isfinite(float(value)):
            raise ContractError(f"{path} contains a non-finite number")
        return
    if isinstance(value, np.integer | np.bool_):
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ContractError(f"{path} contains a non-finite array value")
        return
    if isinstance(value, tuple | list):
        for index, item in enumerate(value):
            _require_json_safe(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} contains a non-string object key")
            _require_json_safe(item, f"{path}.{key}")
        return
    raise ContractError(f"{path} contains non-JSON-safe type {type(value).__name__}")
