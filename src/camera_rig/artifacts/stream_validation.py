"""Strict raw-stream validation artifact contract and I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Final

from camera_rig.artifacts.io import JsonValue, atomic_write_json, json_safe, load_json
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core._validation import string_keyed_copy
from camera_rig.core.errors import ArtifactError, ContractError, SchemaValidationError
from camera_rig.core.quality import QualityReport

STREAM_VALIDATION_SCHEMA_VERSION: Final = "camera-rig.stream-validation.v1"

_STATISTIC_FIELDS: Final = frozenset(
    {
        "per_stream_observed_fps",
        "per_stream_frame_number_discontinuities",
        "per_stream_discontinuity_ratio",
        "per_stream_timestamp_monotonicity",
        "per_stream_timestamp_domain_counts",
        "ir_stereo_frame_match_ratio",
        "comparable_timestamp_skew_ns",
        "sync_valid_ratio",
        "timeouts",
        "missing_streams",
        "shape_consistency",
        "dtype_consistency",
        "depth_valid_ratio",
        "rgb_variance",
        "rgb_channel_variance",
        "ir_variance",
        "ir_distinct_ratio",
    }
)
_ACCUMULATOR_FIELDS: Final = _STATISTIC_FIELDS | {
    "schema_version",
    "status",
    "requested_frames",
    "received_frames",
    "duration_s",
    "failure_reasons",
}


@dataclass(frozen=True)
class StreamValidationArtifact:
    """Portable decision artifact reconstructed from accumulator statistics."""

    status: str
    requested_frames: int
    received_frames: int
    duration_s: float
    statistics: dict[str, object]
    failure_reasons: tuple[str, ...]
    quality: QualityReport
    provenance: dict[str, object]
    schema_version: str = field(default=STREAM_VALIDATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL"}:
            raise ContractError("stream validation status must be PASS or FAIL")
        if (
            isinstance(self.requested_frames, bool)
            or not isinstance(self.requested_frames, int)
            or self.requested_frames < 1
        ):
            raise ContractError("requested_frames must be a positive integer")
        if (
            isinstance(self.received_frames, bool)
            or not isinstance(self.received_frames, int)
            or not 0 <= self.received_frames <= self.requested_frames
        ):
            raise ContractError("received_frames must lie between zero and requested_frames")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ContractError("duration_s must be finite and non-negative")
        statistics = string_keyed_copy(self.statistics, "statistics")
        if set(statistics) != _STATISTIC_FIELDS:
            raise ContractError("stream validation statistics have missing or unknown fields")
        failures = tuple(self.failure_reasons)
        if any(not isinstance(reason, str) or not reason.strip() for reason in failures):
            raise ContractError("failure_reasons must contain non-empty strings")
        passed = self.status == "PASS"
        if passed != (not failures):
            raise ContractError("status is inconsistent with failure_reasons")
        if passed and self.received_frames != self.requested_frames:
            raise ContractError("passed validation must receive every requested frame")
        expected_quality = _quality_for(
            passed=passed,
            requested_frames=self.requested_frames,
            received_frames=self.received_frames,
            statistics=statistics,
            failure_reasons=failures,
        )
        if self.quality.to_dict() != expected_quality.to_dict():
            raise ContractError("quality is inconsistent with stream validation statistics")
        provenance = string_keyed_copy(self.provenance, "provenance")
        _reject_nonportable_values(provenance, "provenance")
        object.__setattr__(self, "statistics", statistics)
        object.__setattr__(self, "failure_reasons", failures)
        object.__setattr__(self, "provenance", provenance)
        json_safe(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "requested_frames": self.requested_frames,
            "received_frames": self.received_frames,
            "duration_s": self.duration_s,
            **self.statistics,
            "failure_reasons": list(self.failure_reasons),
            "quality": self.quality.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_accumulator_report(
        cls, report: dict[str, object], *, provenance: dict[str, object]
    ) -> StreamValidationArtifact:
        """Wrap the existing accumulator report without changing its measurements."""
        if set(report) != _ACCUMULATOR_FIELDS:
            raise ContractError("accumulator report has missing or unknown fields")
        if report.get("schema_version") != STREAM_VALIDATION_SCHEMA_VERSION:
            raise ContractError("unsupported stream validation schema version")
        status = _string(report["status"], "status")
        requested = _int(report["requested_frames"], "requested_frames")
        received = _int(report["received_frames"], "received_frames")
        duration = _float(report["duration_s"], "duration_s")
        statistics = {name: report[name] for name in _STATISTIC_FIELDS}
        failures = _string_tuple(report["failure_reasons"], "failure_reasons")
        return cls(
            status=status,
            requested_frames=requested,
            received_frames=received,
            duration_s=duration,
            statistics=statistics,
            failure_reasons=failures,
            quality=_quality_for(
                passed=status == "PASS",
                requested_frames=requested,
                received_frames=received,
                statistics=statistics,
                failure_reasons=failures,
            ),
            provenance=provenance,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> StreamValidationArtifact:
        return cls(
            status=_string(data["status"], "status"),
            requested_frames=_int(data["requested_frames"], "requested_frames"),
            received_frames=_int(data["received_frames"], "received_frames"),
            duration_s=_float(data["duration_s"], "duration_s"),
            statistics={name: data[name] for name in _STATISTIC_FIELDS},
            failure_reasons=_string_tuple(data["failure_reasons"], "failure_reasons"),
            quality=QualityReport.from_dict(_object(data["quality"], "quality")),
            provenance=_object(data["provenance"], "provenance"),
        )


def write_stream_validation(
    path: str | Path,
    report: dict[str, object],
    *,
    provenance: dict[str, object],
) -> StreamValidationArtifact:
    """Wrap, atomically write, reload, and validate an accumulator report."""
    artifact = StreamValidationArtifact.from_accumulator_report(report, provenance=provenance)
    atomic_write_json(path, artifact.to_dict())
    return load_and_validate_stream_validation(path)


def validate_stream_validation_data(value: JsonValue) -> StreamValidationArtifact:
    """Schema-validate decoded JSON and reconstruct the typed artifact."""
    if not isinstance(value, dict):
        raise ArtifactError("stream validation root must be a JSON object")
    try:
        validate_against_named_schema(value, "stream_validation.v1.schema.json")
        return StreamValidationArtifact.from_dict(dict(value))
    except SchemaValidationError as error:
        raise ArtifactError(str(error)) from error
    except (KeyError, TypeError, ValueError, ContractError) as error:
        raise ArtifactError(f"stream validation contract is invalid: {error}") from error


def load_and_validate_stream_validation(path: str | Path) -> StreamValidationArtifact:
    """Load and strictly validate one persisted stream-validation artifact."""
    return validate_stream_validation_data(load_json(path))


def _quality_for(
    *,
    passed: bool,
    requested_frames: int,
    received_frames: int,
    statistics: dict[str, object],
    failure_reasons: tuple[str, ...],
) -> QualityReport:
    return QualityReport(
        passed=passed,
        metrics={
            "requested_frames": requested_frames,
            "received_frames": received_frames,
            "sync_valid_ratio": _float(statistics["sync_valid_ratio"], "sync_valid_ratio"),
            "timeouts": _int(statistics["timeouts"], "timeouts"),
        },
        thresholds={"minimum_sync_valid_ratio": 1.0, "maximum_timeouts": 0},
        failure_reasons=failure_reasons,
    )


def _reject_nonportable_values(value: object, path: str) -> None:
    if isinstance(value, str):
        windows = PureWindowsPath(value)
        if Path(value).is_absolute() or windows.is_absolute() or windows.drive:
            raise ContractError(f"{path} must not contain absolute paths")
        if value.casefold().startswith("file://"):
            raise ContractError(f"{path} must not contain file URIs")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonportable_values(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_nonportable_values(item, f"{path}[{index}]")


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be a number")
    return float(value)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an array")
    return tuple(_string(item, f"{name}[]") for item in value)
