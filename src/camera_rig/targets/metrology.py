"""Measured physical-target metrology receipts and scale acceptance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Final

from camera_rig.artifacts.hashing import sha256_bytes
from camera_rig.artifacts.io import (
    JsonValue,
    atomic_write_json,
    deterministic_json_bytes,
    load_json,
)
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.targets.charuco.artifact import ResolvedCharucoTarget

TARGET_METROLOGY_SCHEMA_VERSION: Final = "camera-rig.target-metrology.v1"
TARGET_SCALE_ACCEPTANCE_POLICY_SCHEMA_VERSION: Final = (
    "camera-rig.target-scale-acceptance-policy.v1"
)
_METROLOGY_CHECKS: Final = {
    "minimum_three_horizontal_repeats",
    "minimum_three_vertical_repeats",
    "measurement_resolution_within_uncertainty",
    "positive_scale_budget_after_uncertainty",
    "horizontal_scale_within_budget",
    "vertical_scale_within_budget",
    "scale_anisotropy_within_limit",
}


@dataclass(frozen=True)
class TargetScaleAcceptance:
    """Preregistered downstream-error budget used to judge printed scale."""

    allowed_translation_error_mm: float
    maximum_working_distance_mm: float

    def __post_init__(self) -> None:
        values = (
            self.allowed_translation_error_mm,
            self.maximum_working_distance_mm,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ContractError("target-scale acceptance values must be finite and positive")
        if self.allowed_translation_error_mm >= self.maximum_working_distance_mm:
            raise ContractError("translation error budget must be smaller than working distance")

    @property
    def downstream_scale_budget(self) -> float:
        return self.allowed_translation_error_mm / self.maximum_working_distance_mm

    def to_dict(self, *, measurement_relative_uncertainty: float) -> dict[str, object]:
        usable = max(0.0, self.downstream_scale_budget - measurement_relative_uncertainty)
        return {
            "allowed_translation_error_mm": self.allowed_translation_error_mm,
            "maximum_working_distance_mm": self.maximum_working_distance_mm,
            "downstream_scale_budget": self.downstream_scale_budget,
            "measurement_relative_uncertainty": measurement_relative_uncertainty,
            "maximum_absolute_scale_error": usable,
            "maximum_scale_anisotropy": usable,
            "derivation": (
                "allowed_translation_error_mm / maximum_working_distance_mm "
                "minus measurement_relative_uncertainty"
            ),
        }


def build_target_scale_acceptance_policy(
    *,
    created_at: str,
    target: ResolvedCharucoTarget,
    acceptance: TargetScaleAcceptance,
    provenance: dict[str, object],
) -> dict[str, object]:
    """Freeze target-scale acceptance before any physical readings are supplied."""

    _timestamp(created_at)
    policy: dict[str, object] = {
        "schema_version": TARGET_SCALE_ACCEPTANCE_POLICY_SCHEMA_VERSION,
        "state": "FROZEN_BEFORE_MEASUREMENT",
        "created_at": created_at,
        "target_identity_sha256": target.artifact_sha256,
        "allowed_translation_error_mm": acceptance.allowed_translation_error_mm,
        "maximum_working_distance_mm": acceptance.maximum_working_distance_mm,
        "downstream_scale_budget": acceptance.downstream_scale_budget,
        "derivation": "allowed_translation_error_mm / maximum_working_distance_mm",
        "provenance": dict(provenance),
    }
    policy["policy_fingerprint"] = _payload_fingerprint(policy, "policy_fingerprint")
    return policy


def validate_target_scale_acceptance_policy(
    value: object, *, expected_target_sha256: str | None = None
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "state",
        "created_at",
        "target_identity_sha256",
        "allowed_translation_error_mm",
        "maximum_working_distance_mm",
        "downstream_scale_budget",
        "derivation",
        "provenance",
        "policy_fingerprint",
    }:
        raise ArtifactError("target-scale acceptance policy fields are incomplete")
    if value.get("schema_version") != TARGET_SCALE_ACCEPTANCE_POLICY_SCHEMA_VERSION:
        raise ArtifactError("target-scale acceptance policy schema is unsupported")
    if value.get("state") != "FROZEN_BEFORE_MEASUREMENT":
        raise ArtifactError("target-scale acceptance policy is not preregistered")
    provenance = value.get("provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"authority", "measurement_values_available"}
        or not isinstance(provenance.get("authority"), str)
        or not provenance["authority"].strip()
        or provenance.get("measurement_values_available") is not False
    ):
        raise ArtifactError(
            "target-scale acceptance policy must identify its authority and predate readings"
        )
    try:
        _timestamp(_string(value.get("created_at"), "created_at"))
        target_sha = _string(value.get("target_identity_sha256"), "target_identity_sha256")
        _digest(target_sha, "target_identity_sha256")
        acceptance = TargetScaleAcceptance(
            _positive_number(value.get("allowed_translation_error_mm"), "allowed translation"),
            _positive_number(value.get("maximum_working_distance_mm"), "working distance"),
        )
    except ContractError as error:
        raise ArtifactError(str(error)) from error
    if expected_target_sha256 is not None and target_sha != expected_target_sha256:
        raise ArtifactError("target-scale policy target fingerprint differs")
    if value.get("derivation") != (
        "allowed_translation_error_mm / maximum_working_distance_mm"
    ) or not _same_number(value.get("downstream_scale_budget"), acceptance.downstream_scale_budget):
        raise ArtifactError("target-scale acceptance policy derivation differs")
    if value.get("policy_fingerprint") != _payload_fingerprint(value, "policy_fingerprint"):
        raise ArtifactError("target-scale acceptance policy fingerprint differs")
    return dict(value)


def write_target_scale_acceptance_policy(path: str | Path, policy: dict[str, object]) -> None:
    validate_target_scale_acceptance_policy(policy)
    output = Path(path)
    if output.exists():
        raise ArtifactError("target-scale acceptance policy is immutable and already exists")
    atomic_write_json(output, policy)


def load_target_scale_acceptance_policy(
    path: str | Path, *, expected_target_sha256: str | None = None
) -> dict[str, object]:
    return validate_target_scale_acceptance_policy(
        load_json(path), expected_target_sha256=expected_target_sha256
    )


@dataclass(frozen=True)
class TargetMetrologyReceipt:
    """Tamper-evident-by-binding receipt for genuine physical measurements."""

    created_at: str
    target_identity_sha256: str
    nominal: dict[str, object]
    measurement: dict[str, object]
    results: dict[str, object]
    acceptance: dict[str, object]
    acceptance_policy: dict[str, object]
    provenance: dict[str, object]
    status: str
    schema_version: str = TARGET_METROLOGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_METROLOGY_SCHEMA_VERSION:
            raise ContractError("unsupported target metrology schema")
        _timestamp(self.created_at)
        _digest(self.target_identity_sha256, "target_identity_sha256")
        if self.status not in {"PASS", "FAIL"}:
            raise ContractError("target metrology status must be PASS or FAIL")
        for name in (
            "nominal",
            "measurement",
            "results",
            "acceptance",
            "acceptance_policy",
            "provenance",
        ):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ContractError(f"target metrology {name} must be an object")
        checks = self.results.get("checks")
        if (
            not isinstance(checks, dict)
            or not checks
            or not all(isinstance(value, bool) for value in checks.values())
        ):
            raise ContractError("target metrology results.checks must contain booleans")
        if (self.status == "PASS") != all(checks.values()):
            raise ContractError("target metrology status differs from its checks")
        _validate_receipt_semantics(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "target_identity_sha256": self.target_identity_sha256,
            "status": self.status,
            "nominal": dict(self.nominal),
            "measurement": dict(self.measurement),
            "results": dict(self.results),
            "acceptance": dict(self.acceptance),
            "acceptance_policy": dict(self.acceptance_policy),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TargetMetrologyReceipt:
        required = {
            "schema_version",
            "created_at",
            "target_identity_sha256",
            "status",
            "nominal",
            "measurement",
            "results",
            "acceptance",
            "acceptance_policy",
            "provenance",
        }
        if set(value) != required:
            raise ContractError("target metrology has missing or unknown fields")
        return cls(
            schema_version=_string(value["schema_version"], "schema_version"),
            created_at=_string(value["created_at"], "created_at"),
            target_identity_sha256=_string(
                value["target_identity_sha256"], "target_identity_sha256"
            ),
            status=_string(value["status"], "status"),
            nominal=_object(value["nominal"], "nominal"),
            measurement=_object(value["measurement"], "measurement"),
            results=_object(value["results"], "results"),
            acceptance=_object(value["acceptance"], "acceptance"),
            acceptance_policy=_object(value["acceptance_policy"], "acceptance_policy"),
            provenance=_object(value["provenance"], "provenance"),
        )


def evaluate_target_metrology(
    *,
    created_at: str,
    target: ResolvedCharucoTarget,
    horizontal_square_count: int,
    vertical_square_count: int,
    horizontal_measurements_mm: tuple[float, ...],
    vertical_measurements_mm: tuple[float, ...],
    measurement_method: str,
    instrument: str,
    instrument_resolution_mm: float,
    measurement_uncertainty_mm: float,
    acceptance_policy: dict[str, object],
    provenance: dict[str, object],
    outer_board_measurements_mm: dict[str, tuple[float, ...]] | None = None,
    flatness_evidence: dict[str, object] | None = None,
) -> TargetMetrologyReceipt:
    """Evaluate repeated long-baseline measurements against a preregistered budget."""
    _timestamp(created_at)
    _digest(target.artifact_sha256, "target_identity_sha256")
    policy = validate_target_scale_acceptance_policy(
        acceptance_policy, expected_target_sha256=target.artifact_sha256
    )
    policy_created_at = _parsed_timestamp(_string(policy["created_at"], "policy.created_at"))
    measurement_created_at = _parsed_timestamp(created_at)
    if policy_created_at >= measurement_created_at:
        raise ContractError("target-scale acceptance policy must predate measurement receipt")
    acceptance = TargetScaleAcceptance(
        _positive_number(policy["allowed_translation_error_mm"], "allowed translation"),
        _positive_number(policy["maximum_working_distance_mm"], "working distance"),
    )
    acceptance_policy_sha256 = sha256_bytes(deterministic_json_bytes(policy))
    if not 1 <= horizontal_square_count <= target.squares_x:
        raise ContractError("horizontal_square_count is outside target geometry")
    if not 1 <= vertical_square_count <= target.squares_y:
        raise ContractError("vertical_square_count is outside target geometry")
    nominal_horizontal = horizontal_square_count * target.square_length_m * 1000.0
    nominal_vertical = vertical_square_count * target.square_length_m * 1000.0
    horizontal = _measurements(horizontal_measurements_mm, "horizontal_measurements_mm")
    vertical = _measurements(vertical_measurements_mm, "vertical_measurements_mm")
    resolution = _positive(instrument_resolution_mm, "instrument_resolution_mm")
    uncertainty = _positive(measurement_uncertainty_mm, "measurement_uncertainty_mm")
    if not measurement_method.strip() or not instrument.strip():
        raise ContractError("measurement method and instrument must be non-empty")
    horizontal_mean = fmean(horizontal)
    vertical_mean = fmean(vertical)
    horizontal_scale = horizontal_mean / nominal_horizontal
    vertical_scale = vertical_mean / nominal_vertical
    anisotropy = abs(horizontal_scale - vertical_scale)
    horizontal_type_a = 2.0 * stdev(horizontal) / math.sqrt(len(horizontal))
    vertical_type_a = 2.0 * stdev(vertical) / math.sqrt(len(vertical))
    horizontal_expanded = math.hypot(uncertainty, horizontal_type_a)
    vertical_expanded = math.hypot(uncertainty, vertical_type_a)
    relative_uncertainty = max(
        horizontal_expanded / nominal_horizontal,
        vertical_expanded / nominal_vertical,
    )
    thresholds = acceptance.to_dict(measurement_relative_uncertainty=relative_uncertainty)
    thresholds["acceptance_policy_sha256"] = acceptance_policy_sha256
    anisotropy_uncertainty = (
        horizontal_expanded / nominal_horizontal + vertical_expanded / nominal_vertical
    )
    thresholds["maximum_scale_anisotropy"] = max(
        0.0, acceptance.downstream_scale_budget - anisotropy_uncertainty
    )
    thresholds["anisotropy_relative_uncertainty"] = anisotropy_uncertainty
    thresholds["anisotropy_derivation"] = (
        "downstream_scale_budget minus summed horizontal and vertical relative uncertainty"
    )
    maximum_scale_error = _finite_number(
        thresholds["maximum_absolute_scale_error"], "maximum_absolute_scale_error"
    )
    if maximum_scale_error < 0.0:
        raise ContractError("maximum_absolute_scale_error must be finite and non-negative")
    checks = {
        "minimum_three_horizontal_repeats": len(horizontal) >= 3,
        "minimum_three_vertical_repeats": len(vertical) >= 3,
        "measurement_resolution_within_uncertainty": resolution <= uncertainty,
        "positive_scale_budget_after_uncertainty": maximum_scale_error > 0.0,
        "horizontal_scale_within_budget": abs(horizontal_scale - 1.0) <= maximum_scale_error,
        "vertical_scale_within_budget": abs(vertical_scale - 1.0) <= maximum_scale_error,
        "scale_anisotropy_within_limit": anisotropy
        <= _finite_number(thresholds["maximum_scale_anisotropy"], "anisotropy limit"),
    }
    return TargetMetrologyReceipt(
        created_at=created_at,
        target_identity_sha256=target.artifact_sha256,
        status="PASS" if all(checks.values()) else "FAIL",
        nominal={
            "horizontal_baseline_mm": nominal_horizontal,
            "vertical_baseline_mm": nominal_vertical,
            "horizontal_square_count": horizontal_square_count,
            "vertical_square_count": vertical_square_count,
            "square_length_mm": target.square_length_m * 1000.0,
            "board_width_mm": target.board_width_m * 1000.0,
            "board_height_mm": target.board_height_m * 1000.0,
            "geometry_source": "target_spec",
        },
        measurement={
            "method": measurement_method,
            "units": "mm",
            "instrument": instrument,
            "instrument_resolution_mm": resolution,
            "uncertainty_mm": uncertainty,
            "uncertainty_confidence": "expanded_k2_combining_declared_and_type_a_repeatability",
            "horizontal_type_a_expanded_uncertainty_mm": horizontal_type_a,
            "vertical_type_a_expanded_uncertainty_mm": vertical_type_a,
            "horizontal_combined_expanded_uncertainty_mm": horizontal_expanded,
            "vertical_combined_expanded_uncertainty_mm": vertical_expanded,
            "horizontal_measurements_mm": list(horizontal),
            "vertical_measurements_mm": list(vertical),
            "repeat_count_horizontal": len(horizontal),
            "repeat_count_vertical": len(vertical),
            "outer_board_measurements_mm": {
                key: list(values) for key, values in (outer_board_measurements_mm or {}).items()
            },
            "flatness_evidence": dict(flatness_evidence or {}),
        },
        results={
            "horizontal_mean_mm": horizontal_mean,
            "vertical_mean_mm": vertical_mean,
            "horizontal_scale": horizontal_scale,
            "vertical_scale": vertical_scale,
            "horizontal_relative_scale_error": horizontal_scale - 1.0,
            "vertical_relative_scale_error": vertical_scale - 1.0,
            "scale_anisotropy": anisotropy,
            "checks": checks,
        },
        acceptance=thresholds,
        acceptance_policy=policy,
        provenance=dict(provenance),
    )


def write_target_metrology(path: str | Path, receipt: TargetMetrologyReceipt) -> None:
    output = Path(path)
    if output.exists():
        raise ArtifactError("target metrology receipt is immutable and already exists")
    atomic_write_json(output, receipt.to_dict())
    load_target_metrology(path, expected_target_sha256=receipt.target_identity_sha256)


def load_target_metrology(
    path: str | Path,
    *,
    expected_target_sha256: str | None = None,
    expected_target: ResolvedCharucoTarget | None = None,
    expected_acceptance_policy: dict[str, object] | None = None,
    expected_acceptance_policy_sha256: str | None = None,
    require_pass: bool = False,
) -> TargetMetrologyReceipt:
    value: JsonValue = load_json(path)
    if not isinstance(value, dict):
        raise ArtifactError("target metrology root must be an object")
    try:
        receipt = TargetMetrologyReceipt.from_dict(dict(value))
    except (KeyError, TypeError, ValueError, ContractError) as error:
        raise ArtifactError(f"target metrology contract is invalid: {error}") from error
    target_sha = (
        expected_target.artifact_sha256 if expected_target is not None else expected_target_sha256
    )
    if target_sha is not None and receipt.target_identity_sha256 != target_sha:
        raise ArtifactError("target metrology target fingerprint differs")
    if expected_target is not None:
        nominal = receipt.nominal
        expected_values = {
            "square_length_mm": expected_target.square_length_m * 1000.0,
            "board_width_mm": expected_target.board_width_m * 1000.0,
            "board_height_mm": expected_target.board_height_m * 1000.0,
        }
        if any(
            not _same_number(nominal.get(name), value) for name, value in expected_values.items()
        ):
            raise ArtifactError("target metrology nominal geometry differs from target artifact")
        if (
            _positive_integer(nominal.get("horizontal_square_count"), "h count")
            > expected_target.squares_x
            or _positive_integer(nominal.get("vertical_square_count"), "v count")
            > expected_target.squares_y
        ):
            raise ArtifactError("target metrology square count exceeds target geometry")
    embedded_policy = validate_target_scale_acceptance_policy(
        receipt.acceptance_policy,
        expected_target_sha256=receipt.target_identity_sha256,
    )
    embedded_policy_sha256 = sha256_bytes(deterministic_json_bytes(embedded_policy))
    if expected_acceptance_policy is not None:
        validated_expected_policy = validate_target_scale_acceptance_policy(
            expected_acceptance_policy,
            expected_target_sha256=receipt.target_identity_sha256,
        )
        if embedded_policy != validated_expected_policy:
            raise ArtifactError("target metrology embedded acceptance policy differs")
    if (
        expected_acceptance_policy_sha256 is not None
        and embedded_policy_sha256 != expected_acceptance_policy_sha256
    ):
        raise ArtifactError("target metrology acceptance-policy digest differs")
    if require_pass and receipt.status != "PASS":
        raise ArtifactError("target metrology is not passed")
    return receipt


def _measurements(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) < 3:
        raise ContractError(f"{name} must contain at least three repeats")
    return tuple(_positive(value, f"{name}[]") for value in values)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ContractError(f"{name} must be finite and positive")
    return float(value)


def _parsed_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ContractError("created_at must be an ISO-8601 date-time") from error
    if parsed.tzinfo is None:
        raise ContractError("created_at must include a timezone")
    return parsed


def _timestamp(value: str) -> None:
    _parsed_timestamp(value)


def _digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be an object with string keys")
    return value


def _validate_receipt_semantics(receipt: TargetMetrologyReceipt) -> None:
    policy = validate_target_scale_acceptance_policy(
        receipt.acceptance_policy,
        expected_target_sha256=receipt.target_identity_sha256,
    )
    if _parsed_timestamp(_string(policy["created_at"], "policy.created_at")) >= _parsed_timestamp(
        receipt.created_at
    ):
        raise ContractError("target-scale acceptance policy must predate measurement receipt")
    nominal_required = {
        "horizontal_baseline_mm",
        "vertical_baseline_mm",
        "horizontal_square_count",
        "vertical_square_count",
        "square_length_mm",
        "board_width_mm",
        "board_height_mm",
        "geometry_source",
    }
    measurement_required = {
        "method",
        "units",
        "instrument",
        "instrument_resolution_mm",
        "uncertainty_mm",
        "uncertainty_confidence",
        "horizontal_type_a_expanded_uncertainty_mm",
        "vertical_type_a_expanded_uncertainty_mm",
        "horizontal_combined_expanded_uncertainty_mm",
        "vertical_combined_expanded_uncertainty_mm",
        "horizontal_measurements_mm",
        "vertical_measurements_mm",
        "repeat_count_horizontal",
        "repeat_count_vertical",
        "outer_board_measurements_mm",
        "flatness_evidence",
    }
    result_required = {
        "horizontal_mean_mm",
        "vertical_mean_mm",
        "horizontal_scale",
        "vertical_scale",
        "horizontal_relative_scale_error",
        "vertical_relative_scale_error",
        "scale_anisotropy",
        "checks",
    }
    acceptance_required = {
        "acceptance_policy_sha256",
        "anisotropy_derivation",
        "anisotropy_relative_uncertainty",
        "allowed_translation_error_mm",
        "maximum_working_distance_mm",
        "downstream_scale_budget",
        "measurement_relative_uncertainty",
        "maximum_absolute_scale_error",
        "maximum_scale_anisotropy",
        "derivation",
    }
    if set(receipt.nominal) != nominal_required:
        raise ContractError("target metrology nominal fields are incomplete")
    if set(receipt.measurement) != measurement_required:
        raise ContractError("target metrology measurement fields are incomplete")
    if set(receipt.results) != result_required:
        raise ContractError("target metrology result fields are incomplete")
    if set(receipt.acceptance) != acceptance_required:
        raise ContractError("target metrology acceptance fields are incomplete")
    if receipt.measurement.get("units") != "mm":
        raise ContractError("target metrology units must be mm")
    if receipt.measurement.get("uncertainty_confidence") != (
        "expanded_k2_combining_declared_and_type_a_repeatability"
    ):
        raise ContractError("target metrology uncertainty confidence is invalid")
    if receipt.nominal.get("geometry_source") != "target_spec":
        raise ContractError("target metrology geometry source is invalid")
    policy_sha256 = _string(receipt.acceptance.get("acceptance_policy_sha256"), "acceptance policy")
    _digest(policy_sha256, "acceptance_policy_sha256")
    if policy_sha256 != sha256_bytes(deterministic_json_bytes(policy)):
        raise ContractError("target metrology acceptance-policy digest differs")
    horizontal = _number_list(
        receipt.measurement.get("horizontal_measurements_mm"), "horizontal_measurements_mm"
    )
    vertical = _number_list(
        receipt.measurement.get("vertical_measurements_mm"), "vertical_measurements_mm"
    )
    if len(horizontal) < 3 or len(vertical) < 3:
        raise ContractError("target metrology requires at least three measurements per axis")
    if receipt.measurement.get("repeat_count_horizontal") != len(
        horizontal
    ) or receipt.measurement.get("repeat_count_vertical") != len(vertical):
        raise ContractError("target metrology repeat counts differ from measurements")
    h_count = _positive_integer(receipt.nominal.get("horizontal_square_count"), "h count")
    v_count = _positive_integer(receipt.nominal.get("vertical_square_count"), "v count")
    square = _positive_number(receipt.nominal.get("square_length_mm"), "square length")
    nominal_h = _positive_number(receipt.nominal.get("horizontal_baseline_mm"), "nominal h")
    nominal_v = _positive_number(receipt.nominal.get("vertical_baseline_mm"), "nominal v")
    if not _same_number(nominal_h, h_count * square) or not _same_number(
        nominal_v, v_count * square
    ):
        raise ContractError("target metrology nominal baselines differ from square geometry")
    _positive_number(receipt.nominal.get("board_width_mm"), "board width")
    _positive_number(receipt.nominal.get("board_height_mm"), "board height")
    resolution = _positive_number(
        receipt.measurement.get("instrument_resolution_mm"), "instrument resolution"
    )
    declared_uncertainty = _positive_number(
        receipt.measurement.get("uncertainty_mm"), "measurement uncertainty"
    )
    h_type_a = 2.0 * stdev(horizontal) / math.sqrt(len(horizontal))
    v_type_a = 2.0 * stdev(vertical) / math.sqrt(len(vertical))
    h_combined = math.hypot(declared_uncertainty, h_type_a)
    v_combined = math.hypot(declared_uncertainty, v_type_a)
    derived_measurement = {
        "horizontal_type_a_expanded_uncertainty_mm": h_type_a,
        "vertical_type_a_expanded_uncertainty_mm": v_type_a,
        "horizontal_combined_expanded_uncertainty_mm": h_combined,
        "vertical_combined_expanded_uncertainty_mm": v_combined,
    }
    for name, expected_value in derived_measurement.items():
        if not _same_number(receipt.measurement.get(name), expected_value):
            raise ContractError(f"target metrology derived uncertainty differs: {name}")
    h_scale = fmean(horizontal) / nominal_h
    v_scale = fmean(vertical) / nominal_v
    expected = {
        "horizontal_mean_mm": fmean(horizontal),
        "vertical_mean_mm": fmean(vertical),
        "horizontal_scale": h_scale,
        "vertical_scale": v_scale,
        "horizontal_relative_scale_error": h_scale - 1.0,
        "vertical_relative_scale_error": v_scale - 1.0,
        "scale_anisotropy": abs(h_scale - v_scale),
    }
    for name, value in expected.items():
        stored = receipt.results.get(name)
        if (
            isinstance(stored, bool)
            or not isinstance(stored, int | float)
            or not math.isclose(float(stored), value, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ContractError(f"target metrology derived result differs: {name}")
    allowed = _positive_number(
        receipt.acceptance.get("allowed_translation_error_mm"), "allowed translation"
    )
    distance = _positive_number(
        receipt.acceptance.get("maximum_working_distance_mm"), "working distance"
    )
    if not _same_number(policy.get("allowed_translation_error_mm"), allowed) or not _same_number(
        policy.get("maximum_working_distance_mm"), distance
    ):
        raise ContractError("target metrology acceptance differs from frozen policy")
    uncertainty = max(h_combined / nominal_h, v_combined / nominal_v)
    budget = allowed / distance
    maximum = max(0.0, budget - uncertainty)
    anisotropy_uncertainty = h_combined / nominal_h + v_combined / nominal_v
    maximum_anisotropy = max(0.0, budget - anisotropy_uncertainty)
    if (
        not math.isclose(
            _finite_number(
                receipt.acceptance.get("downstream_scale_budget"), "downstream scale budget"
            ),
            budget,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_number(
                receipt.acceptance.get("maximum_absolute_scale_error"), "maximum scale error"
            ),
            maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not _same_number(receipt.acceptance.get("measurement_relative_uncertainty"), uncertainty)
        or not _same_number(receipt.acceptance.get("maximum_scale_anisotropy"), maximum_anisotropy)
        or not _same_number(
            receipt.acceptance.get("anisotropy_relative_uncertainty"),
            anisotropy_uncertainty,
        )
        or receipt.acceptance.get("anisotropy_derivation")
        != "downstream_scale_budget minus summed horizontal and vertical relative uncertainty"
    ):
        raise ContractError("target metrology acceptance derivation differs")
    checks = receipt.results.get("checks")
    if not isinstance(checks, dict) or set(checks) != _METROLOGY_CHECKS:
        raise ContractError("target metrology check set differs")
    expected_checks = {
        "minimum_three_horizontal_repeats": len(horizontal) >= 3,
        "minimum_three_vertical_repeats": len(vertical) >= 3,
        "measurement_resolution_within_uncertainty": resolution <= declared_uncertainty,
        "positive_scale_budget_after_uncertainty": maximum > 0.0,
        "horizontal_scale_within_budget": abs(h_scale - 1.0) <= maximum,
        "vertical_scale_within_budget": abs(v_scale - 1.0) <= maximum,
        "scale_anisotropy_within_limit": abs(h_scale - v_scale) <= maximum_anisotropy,
    }
    if checks != expected_checks:
        raise ContractError("target metrology checks differ from recomputed decision")


def _number_list(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an array")
    return tuple(_positive_number(item, f"{name}[]") for item in value)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be numeric")
    return _positive(float(value), name)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ContractError(f"{name} must be finite")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _same_number(value: object, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    )


def _payload_fingerprint(value: dict[str, object], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return sha256_bytes(deterministic_json_bytes(payload))
