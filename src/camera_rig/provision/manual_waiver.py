"""Explicit human waiver for bootstrap-only native-depth fluctuations."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Final, cast

from camera_rig.artifacts.hashing import sha256_bytes
from camera_rig.artifacts.io import (
    atomic_write_json,
    deterministic_json_bytes,
    json_safe,
    load_json,
)
from camera_rig.calibration.fixed.artifact import FixedCalibrationArtifact
from camera_rig.calibration.fixed.depth_sanity import validate_native_depth_evaluation
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.quality import QualityReport

BOOTSTRAP_DEPTH_MANUAL_WAIVER_SCHEMA_VERSION: Final = "camera-rig.bootstrap-depth-manual-waiver.v1"
WAIVED_CHECK: Final = "metric_native_depth_integrity"
_DOES_NOT_WAIVE: Final = [
    "raw_stream_quality",
    "target_identity",
    "target_metrology",
    "target_detection",
    "physical_pnp",
    "catastrophic_reprojection",
    "pose_uncertainty",
    "observability",
    "ippe_ambiguity",
    "temporal_repeatability",
    "split_half_stability",
    "multipose_holdout",
    "intrinsic_health",
    "physical_3d_acceptance",
]


def build_bootstrap_depth_manual_waiver(
    *,
    authorized_at: str,
    authorization_statement: str,
    camera_identity_sha256: str,
    target_identity_sha256: str,
    provenance: dict[str, object],
) -> dict[str, object]:
    """Build a narrow user-authorized exception without changing machine thresholds."""
    try:
        datetime.fromisoformat(authorized_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError("authorized_at must be an ISO-8601 date-time") from error
    if not authorization_statement.strip():
        raise ContractError("authorization_statement must be non-empty")
    _digest(camera_identity_sha256, "camera_identity_sha256")
    _digest(target_identity_sha256, "target_identity_sha256")
    report: dict[str, object] = {
        "schema_version": BOOTSTRAP_DEPTH_MANUAL_WAIVER_SCHEMA_VERSION,
        "status": "PASS",
        "waiver_scope": "bootstrap_metric_native_depth_integrity_only",
        "qualification_scope": "bootstrap_only",
        "production_authoritative": False,
        "authority": "human_user",
        "authorized_at": authorized_at,
        "authorization_statement": authorization_statement,
        "camera_identity_sha256": camera_identity_sha256,
        "target_identity_sha256": target_identity_sha256,
        "waived_check": WAIVED_CHECK,
        "reason": "user_authorized_occasional_depth_fluctuation_exception",
        "does_not_waive": list(_DOES_NOT_WAIVE),
        "provenance": dict(provenance),
    }
    report["waiver_fingerprint"] = waiver_fingerprint(report)
    return validate_bootstrap_depth_manual_waiver_data(report)


def waiver_fingerprint(report: dict[str, object]) -> str:
    payload = {key: value for key, value in report.items() if key != "waiver_fingerprint"}
    return sha256_bytes(deterministic_json_bytes(json_safe(payload)))


def write_bootstrap_depth_manual_waiver(path: str | Path, report: dict[str, object]) -> None:
    atomic_write_json(path, json_safe(validate_bootstrap_depth_manual_waiver_data(report)))


def load_bootstrap_depth_manual_waiver(path: str | Path) -> dict[str, object]:
    return validate_bootstrap_depth_manual_waiver_data(load_json(path))


def validate_bootstrap_depth_manual_waiver_data(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactError("bootstrap depth manual waiver root must be an object")
    required = {
        "schema_version",
        "status",
        "waiver_scope",
        "qualification_scope",
        "production_authoritative",
        "authority",
        "authorized_at",
        "authorization_statement",
        "camera_identity_sha256",
        "target_identity_sha256",
        "waived_check",
        "reason",
        "does_not_waive",
        "provenance",
        "waiver_fingerprint",
    }
    if set(value) != required:
        raise ArtifactError("bootstrap depth manual waiver has missing or unknown fields")
    expected = {
        "schema_version": BOOTSTRAP_DEPTH_MANUAL_WAIVER_SCHEMA_VERSION,
        "status": "PASS",
        "waiver_scope": "bootstrap_metric_native_depth_integrity_only",
        "qualification_scope": "bootstrap_only",
        "production_authoritative": False,
        "authority": "human_user",
        "waived_check": WAIVED_CHECK,
        "reason": "user_authorized_occasional_depth_fluctuation_exception",
        "does_not_waive": _DOES_NOT_WAIVE,
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise ArtifactError("bootstrap depth manual waiver semantics are invalid")
    statement = value.get("authorization_statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ArtifactError("bootstrap depth manual waiver authorization is empty")
    authorized_at = value.get("authorized_at")
    if not isinstance(authorized_at, str):
        raise ArtifactError("bootstrap depth manual waiver timestamp is invalid")
    try:
        datetime.fromisoformat(authorized_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ArtifactError("bootstrap depth manual waiver timestamp is invalid") from error
    for name in ("camera_identity_sha256", "target_identity_sha256", "waiver_fingerprint"):
        candidate = value.get(name)
        if not isinstance(candidate, str):
            raise ArtifactError(f"bootstrap depth manual waiver {name} must be a digest")
        try:
            _digest(candidate, name)
        except ContractError as error:
            raise ArtifactError(str(error)) from error
    if not isinstance(value.get("provenance"), dict):
        raise ArtifactError("bootstrap depth manual waiver provenance must be an object")
    report = cast(dict[str, object], dict(value))
    if report["waiver_fingerprint"] != waiver_fingerprint(report):
        raise ArtifactError("bootstrap depth manual waiver fingerprint differs")
    return report


def apply_bootstrap_depth_manual_waiver(
    fixed: FixedCalibrationArtifact,
    waiver: dict[str, object],
    *,
    camera_identity_sha256: str,
    target_identity_sha256: str,
) -> FixedCalibrationArtifact:
    """Apply an effective bootstrap decision while retaining the machine failure verbatim."""
    receipt = validate_bootstrap_depth_manual_waiver_data(waiver)
    if receipt["camera_identity_sha256"] != camera_identity_sha256:
        raise ContractError("bootstrap depth manual waiver camera identity differs")
    if receipt["target_identity_sha256"] != target_identity_sha256:
        raise ContractError("bootstrap depth manual waiver target identity differs")
    if fixed.quality.passed or set(fixed.quality.failure_reasons) != {"native_depth_sanity"}:
        raise ContractError(
            "bootstrap depth manual waiver requires native_depth_sanity to be the only failure"
        )
    if fixed.fixed_mount_calibration.quality.to_dict() != fixed.quality.to_dict():
        raise ContractError("fixed calibration and fixed-mount machine quality differ")
    depth = fixed.aggregate.get("native_depth_sanity")
    if not isinstance(depth, dict) or depth.get("status") != "FAIL":
        raise ContractError("bootstrap depth manual waiver requires a machine depth FAIL")
    validate_native_depth_evaluation(depth, require_pass=False, require_fixed_bootstrap_policy=True)
    checks = fixed.quality.metrics.get("checks")
    if not isinstance(checks, dict) or checks.get("native_depth_sanity") is not False:
        raise ContractError("fixed calibration machine depth check is not failed")
    effective_checks = dict(checks)
    effective_checks["native_depth_sanity"] = True
    machine_quality = fixed.quality.to_dict()
    warning = (
        "native_depth_sanity machine FAIL accepted only for bootstrap initialization by an "
        "explicit human waiver; production gates remain required"
    )
    effective_quality = QualityReport(
        passed=True,
        metrics={
            **fixed.quality.metrics,
            "checks": effective_checks,
            "machine_quality_without_waiver": machine_quality,
            "bootstrap_depth_manual_waiver": {
                "status": "APPLIED",
                "waived_check": WAIVED_CHECK,
                "waiver_fingerprint": receipt["waiver_fingerprint"],
                "production_authoritative": False,
            },
        },
        thresholds=fixed.quality.thresholds,
        warnings=(*fixed.quality.warnings, warning),
    )
    fixed_mount = replace(
        fixed.fixed_mount_calibration,
        quality=effective_quality,
        provenance={
            **fixed.fixed_mount_calibration.provenance,
            "bootstrap_depth_manual_waiver_fingerprint": receipt["waiver_fingerprint"],
            "production_authoritative": False,
        },
    )
    return replace(
        fixed,
        fixed_mount_calibration=fixed_mount,
        quality=effective_quality,
        provenance={
            **fixed.provenance,
            "bootstrap_depth_manual_waiver_fingerprint": receipt["waiver_fingerprint"],
            "machine_quality_status": "FAIL",
            "effective_bootstrap_quality_status": "PASS_WITH_MANUAL_DEPTH_WAIVER",
        },
    )


def _digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
