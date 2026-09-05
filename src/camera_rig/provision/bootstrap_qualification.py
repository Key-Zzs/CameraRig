"""Composed bootstrap-only qualification contract for fixed-camera provisioning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from camera_rig.artifacts.hashing import sha256_bytes
from camera_rig.artifacts.io import (
    JsonValue,
    atomic_write_json,
    deterministic_json_bytes,
    json_safe,
    load_json,
)
from camera_rig.artifacts.stream_validation import StreamValidationArtifact
from camera_rig.artifacts.target_detection import TargetDetectionArtifact
from camera_rig.calibration.fixed.artifact import FixedCalibrationArtifact
from camera_rig.calibration.fixed.depth_sanity import validate_native_depth_evaluation
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.provision.manual_waiver import (
    validate_bootstrap_depth_manual_waiver_data,
)
from camera_rig.targets.metrology import TargetMetrologyReceipt

BOOTSTRAP_QUALIFICATION_SCHEMA_VERSION: Final = "camera-rig.fixed-bootstrap-qualification.v1"
BOOTSTRAP_QUALIFICATION_V2_SCHEMA_VERSION: Final = "camera-rig.fixed-bootstrap-qualification.v2"
STRUCTURED_RESIDUAL_PRODUCTION_GATE: Final = "NOT_SUPPORTED_DUE_TO_PLANAR_IDENTIFIABILITY_LIMIT"
_CATASTROPHIC_ROLE: Final = "gross_invalid_projection_or_pnp_ceiling_only"
_CATASTROPHIC_DOES_NOT_PROVE: Final = [
    "factory_intrinsics_correctness",
    "target_scale_correctness",
    "production_metric_pose_correctness",
]
_CATASTROPHIC_MAXIMUM_FINAL_RMSE_PX: Final = 1.5
_CATASTROPHIC_MAXIMUM_FINAL_P95_PX: Final = 2.0
_CHECK_NAMES: Final = {
    "raw_stream_quality",
    "target_identity",
    "target_metrology",
    "target_detection",
    "physical_pnp",
    "catastrophic_reprojection_rmse",
    "catastrophic_reprojection_p95",
    "pose_uncertainty",
    "observability",
    "ippe_ambiguity",
    "temporal_repeatability_translation",
    "temporal_repeatability_rotation",
    "split_half_translation",
    "split_half_rotation",
    "metric_native_depth_integrity",
}


def build_bootstrap_qualification(
    *,
    camera_identity_sha256: str,
    camera_bundle_fingerprint: str,
    target_identity_sha256: str,
    target_metrology_sha256: str,
    metric_depth_receipt_sha256: str,
    stream_validation: StreamValidationArtifact,
    target_detection: TargetDetectionArtifact,
    target_metrology: TargetMetrologyReceipt,
    fixed_calibration: FixedCalibrationArtifact,
    provenance: dict[str, object],
    bootstrap_depth_manual_waiver: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a white-listed bootstrap decision; structured residuals never affect it."""
    for name, digest in (
        ("camera_identity_sha256", camera_identity_sha256),
        ("camera_bundle_fingerprint", camera_bundle_fingerprint),
        ("target_identity_sha256", target_identity_sha256),
        ("target_metrology_sha256", target_metrology_sha256),
        ("metric_depth_receipt_sha256", metric_depth_receipt_sha256),
    ):
        _digest(digest, name)
    if target_metrology.target_identity_sha256 != target_identity_sha256:
        raise ContractError("metrology target identity differs from bootstrap target")
    quality_metrics = fixed_calibration.quality.metrics
    checks_value = quality_metrics.get("checks")
    quality_checks = checks_value if isinstance(checks_value, dict) else {}
    acceptance = target_detection.acceptance or {}
    depth_value = fixed_calibration.aggregate.get("native_depth_sanity")
    native_depth = depth_value if isinstance(depth_value, dict) else {}
    validate_native_depth_evaluation(
        native_depth,
        require_pass=native_depth.get("status") == "PASS",
        require_fixed_bootstrap_policy=True,
    )
    uncertainty_route = fixed_calibration.solver.get("pose_policy") == "uncertainty_validated"
    checks = {
        "raw_stream_quality": (
            stream_validation.status == "PASS" and stream_validation.quality.passed
        ),
        "target_identity": target_detection.target_spec_sha256 == target_identity_sha256,
        "target_metrology": target_metrology.status == "PASS",
        "target_detection": acceptance.get("passed") is True,
        "physical_pnp": _positive_integer(fixed_calibration.aggregate.get("accepted_frames")),
        "catastrophic_reprojection_rmse": (
            quality_checks.get("gross_global_reprojection_rmse") is True
            if uncertainty_route
            else quality_checks.get("global_reprojection_rmse") is True
        ),
        "catastrophic_reprojection_p95": (
            quality_checks.get("gross_global_reprojection_p95") is True
            if uncertainty_route
            else quality_checks.get("global_reprojection_p95") is True
        ),
        "pose_uncertainty": (
            quality_checks.get("final_pose_translation_uncertainty") is True
            and quality_checks.get("final_pose_rotation_uncertainty") is True
        ),
        "observability": (
            quality_checks.get("final_pose_observability") is True
            and quality_checks.get("final_pose_full_rank") is True
            and quality_checks.get("final_pose_condition_number") is True
        ),
        "ippe_ambiguity": quality_checks.get("final_pose_unambiguous") is True,
        "temporal_repeatability_translation": quality_checks.get("pose_translation_p95") is True,
        "temporal_repeatability_rotation": quality_checks.get("pose_rotation_p95") is True,
        "split_half_translation": quality_checks.get("split_translation_delta") is True,
        "split_half_rotation": quality_checks.get("split_rotation_delta") is True,
        "metric_native_depth_integrity": native_depth.get("status") == "PASS",
    }
    failures = [name for name, passed in checks.items() if not passed]
    waiver = None
    if bootstrap_depth_manual_waiver is not None:
        waiver = validate_bootstrap_depth_manual_waiver_data(bootstrap_depth_manual_waiver)
        if waiver["camera_identity_sha256"] != camera_identity_sha256:
            raise ContractError("bootstrap depth manual waiver camera identity differs")
        if waiver["target_identity_sha256"] != target_identity_sha256:
            raise ContractError("bootstrap depth manual waiver target identity differs")
        if failures != ["metric_native_depth_integrity"]:
            raise ContractError(
                "bootstrap depth manual waiver requires metric depth to be the only failure"
            )
    effective_checks = dict(checks)
    if waiver is not None:
        effective_checks["metric_native_depth_integrity"] = True
    effective_failures = [name for name, passed in effective_checks.items() if not passed]
    structured_value = quality_metrics.get("final_structured_residual")
    structured = structured_value if isinstance(structured_value, dict) else {}
    report: dict[str, object] = {
        "schema_version": (
            BOOTSTRAP_QUALIFICATION_V2_SCHEMA_VERSION
            if waiver is not None
            else BOOTSTRAP_QUALIFICATION_SCHEMA_VERSION
        ),
        "status": "PASS" if not effective_failures else "FAIL",
        "qualification_state": (
            "BOOTSTRAP_QUALIFIED_WITH_MANUAL_DEPTH_WAIVER"
            if waiver is not None and not effective_failures
            else "BOOTSTRAP_QUALIFIED"
            if not effective_failures
            else "BOOTSTRAP_NOT_QUALIFIED"
        ),
        "qualification_scope": "bootstrap_only",
        "production_authoritative": False,
        "camera_identity_sha256": camera_identity_sha256,
        "camera_bundle_fingerprint": camera_bundle_fingerprint,
        "target_identity_sha256": target_identity_sha256,
        "target_metrology_sha256": target_metrology_sha256,
        "metric_depth_receipt_sha256": metric_depth_receipt_sha256,
        "checks": checks,
        "failure_reasons": failures,
        "catastrophic_reprojection": {
            "role": _CATASTROPHIC_ROLE,
            "does_not_prove": _CATASTROPHIC_DOES_NOT_PROVE,
            "decision": quality_metrics.get("reprojection_decision"),
        },
        "metric_depth_integrity": native_depth,
        "structured_residual": {
            "role": "diagnostic_only",
            "available": bool(structured),
            "enforced": False,
            "production_gate": STRUCTURED_RESIDUAL_PRODUCTION_GATE,
            "metrics": structured,
        },
        "provenance": dict(provenance),
    }
    if waiver is not None:
        report.update(
            {
                "machine_status": "PASS" if not failures else "FAIL",
                "effective_checks": effective_checks,
                "manual_waiver": waiver,
            }
        )
    report["qualification_fingerprint"] = qualification_fingerprint(report)
    return report


def qualification_fingerprint(report: Mapping[str, object]) -> str:
    payload = {key: value for key, value in report.items() if key != "qualification_fingerprint"}
    return sha256_bytes(deterministic_json_bytes(json_safe(payload)))


def write_bootstrap_qualification(path: str | Path, report: dict[str, object]) -> None:
    validate_bootstrap_qualification_data(report)
    atomic_write_json(path, json_safe(report))


def load_bootstrap_qualification(
    path: str | Path, *, require_pass: bool = False
) -> dict[str, object]:
    value: JsonValue = load_json(path)
    report = validate_bootstrap_qualification_data(value)
    if require_pass and report["status"] != "PASS":
        raise ArtifactError("bootstrap qualification is not passed")
    return report


def validate_bootstrap_qualification_data(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactError("bootstrap qualification root must be an object")
    required = {
        "schema_version",
        "status",
        "qualification_state",
        "qualification_scope",
        "production_authoritative",
        "camera_identity_sha256",
        "camera_bundle_fingerprint",
        "target_identity_sha256",
        "target_metrology_sha256",
        "metric_depth_receipt_sha256",
        "checks",
        "failure_reasons",
        "catastrophic_reprojection",
        "metric_depth_integrity",
        "structured_residual",
        "provenance",
        "qualification_fingerprint",
    }
    schema_version = value.get("schema_version")
    is_v2 = schema_version == BOOTSTRAP_QUALIFICATION_V2_SCHEMA_VERSION
    if is_v2:
        required.update({"machine_status", "effective_checks", "manual_waiver"})
    if set(value) != required:
        raise ArtifactError("bootstrap qualification has missing or unknown fields")
    if schema_version not in {
        BOOTSTRAP_QUALIFICATION_SCHEMA_VERSION,
        BOOTSTRAP_QUALIFICATION_V2_SCHEMA_VERSION,
    }:
        raise ArtifactError("bootstrap qualification schema is unsupported")
    if value.get("qualification_scope") != "bootstrap_only":
        raise ArtifactError("bootstrap qualification scope must be bootstrap_only")
    if value.get("production_authoritative") is not False:
        raise ArtifactError("bootstrap qualification cannot be production-authoritative")
    checks = value.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != _CHECK_NAMES
        or not all(isinstance(item, bool) for item in checks.values())
    ):
        raise ArtifactError("bootstrap qualification checks are incomplete")
    machine_status = "PASS" if all(checks.values()) else "FAIL"
    effective_checks = checks
    if is_v2:
        waiver = validate_bootstrap_depth_manual_waiver_data(value.get("manual_waiver"))
        if waiver["camera_identity_sha256"] != value.get("camera_identity_sha256"):
            raise ArtifactError("bootstrap depth manual waiver camera identity differs")
        if waiver["target_identity_sha256"] != value.get("target_identity_sha256"):
            raise ArtifactError("bootstrap depth manual waiver target identity differs")
        if value.get("machine_status") != machine_status or machine_status != "FAIL":
            raise ArtifactError("bootstrap qualification machine status is invalid")
        candidate_effective = value.get("effective_checks")
        if (
            not isinstance(candidate_effective, dict)
            or set(candidate_effective) != _CHECK_NAMES
            or not all(isinstance(item, bool) for item in candidate_effective.values())
        ):
            raise ArtifactError("bootstrap qualification effective checks are incomplete")
        expected_effective = dict(checks)
        expected_effective["metric_native_depth_integrity"] = True
        if candidate_effective != expected_effective:
            raise ArtifactError("bootstrap qualification effective checks are invalid")
        if [name for name, passed in checks.items() if not passed] != [
            "metric_native_depth_integrity"
        ]:
            raise ArtifactError("bootstrap depth manual waiver covers an invalid failure set")
        effective_checks = candidate_effective
    expected_status = "PASS" if all(effective_checks.values()) else "FAIL"
    if value.get("status") != expected_status:
        raise ArtifactError("bootstrap qualification status differs from checks")
    expected_state = (
        "BOOTSTRAP_QUALIFIED_WITH_MANUAL_DEPTH_WAIVER"
        if is_v2 and expected_status == "PASS"
        else "BOOTSTRAP_QUALIFIED"
        if expected_status == "PASS"
        else "BOOTSTRAP_NOT_QUALIFIED"
    )
    if value.get("qualification_state") != expected_state:
        raise ArtifactError("bootstrap qualification state differs from checks")
    failures = value.get("failure_reasons")
    expected_failures = [name for name, passed in checks.items() if not passed]
    if failures != expected_failures:
        raise ArtifactError("bootstrap qualification failure reasons differ from checks")
    structured = value.get("structured_residual")
    if not isinstance(structured, dict) or structured.get("role") != "diagnostic_only":
        raise ArtifactError("structured residual must remain diagnostic-only")
    if (
        structured.get("enforced") is not False
        or structured.get("production_gate") != STRUCTURED_RESIDUAL_PRODUCTION_GATE
    ):
        raise ArtifactError("structured residual production-gate disposition is invalid")
    catastrophic = value.get("catastrophic_reprojection")
    if not isinstance(catastrophic, dict) or set(catastrophic) != {
        "role",
        "does_not_prove",
        "decision",
    }:
        raise ArtifactError("catastrophic reprojection disclaimer is incomplete")
    if (
        catastrophic.get("role") != _CATASTROPHIC_ROLE
        or catastrophic.get("does_not_prove") != _CATASTROPHIC_DOES_NOT_PROVE
    ):
        raise ArtifactError("catastrophic reprojection disclaimer is invalid")
    _validate_catastrophic_reprojection_decision(
        catastrophic.get("decision"),
        expected_rmse_check=checks["catastrophic_reprojection_rmse"],
        expected_p95_check=checks["catastrophic_reprojection_p95"],
    )
    metric = value.get("metric_depth_integrity")
    if not isinstance(metric, dict) or (
        (metric.get("status") == "PASS") != checks["metric_native_depth_integrity"]
    ):
        raise ArtifactError("metric-depth status differs from its bootstrap hard check")
    try:
        validate_native_depth_evaluation(
            metric,
            require_pass=checks["metric_native_depth_integrity"],
            require_fixed_bootstrap_policy=True,
        )
    except ContractError as error:
        raise ArtifactError(f"bootstrap metric-depth integrity is invalid: {error}") from error
    for name in (
        "camera_identity_sha256",
        "camera_bundle_fingerprint",
        "target_identity_sha256",
        "target_metrology_sha256",
        "metric_depth_receipt_sha256",
        "qualification_fingerprint",
    ):
        candidate = value.get(name)
        if not isinstance(candidate, str):
            raise ArtifactError(f"bootstrap qualification {name} must be a digest")
        try:
            _digest(candidate, name)
        except ContractError as error:
            raise ArtifactError(str(error)) from error
    report = cast(dict[str, object], dict(value))
    if report["qualification_fingerprint"] != qualification_fingerprint(report):
        raise ArtifactError("bootstrap qualification fingerprint differs")
    return report


def _validate_catastrophic_reprojection_decision(
    value: object, *, expected_rmse_check: bool, expected_p95_check: bool
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "policy",
        "passed",
        "checks",
        "metrics",
        "applied_thresholds",
        "legacy_precision_thresholds",
    }:
        raise ArtifactError("catastrophic reprojection decision is incomplete")
    if value.get("policy") != "uncertainty_gross_model_consistency":
        raise ArtifactError("catastrophic reprojection policy is invalid")
    metrics = value.get("metrics")
    applied = value.get("applied_thresholds")
    decision_checks = value.get("checks")
    legacy = value.get("legacy_precision_thresholds")
    if (
        not isinstance(metrics, dict)
        or set(metrics) != {"rmse_px", "p95_px"}
        or not isinstance(applied, dict)
        or set(applied) != {"maximum_final_rmse_px", "maximum_final_p95_px"}
        or not isinstance(decision_checks, dict)
        or set(decision_checks) != {"rmse_within_applied_threshold", "p95_within_applied_threshold"}
        or not isinstance(legacy, dict)
        or set(legacy) != {"maximum_frame_rmse_px", "maximum_frame_p95_px"}
    ):
        raise ArtifactError("catastrophic reprojection decision fields are invalid")
    rmse = _finite_number(metrics.get("rmse_px"), "catastrophic rmse")
    p95 = _finite_number(metrics.get("p95_px"), "catastrophic p95")
    if (
        applied.get("maximum_final_rmse_px") != _CATASTROPHIC_MAXIMUM_FINAL_RMSE_PX
        or applied.get("maximum_final_p95_px") != _CATASTROPHIC_MAXIMUM_FINAL_P95_PX
    ):
        raise ArtifactError("catastrophic reprojection thresholds are not frozen")
    expected_checks = {
        "rmse_within_applied_threshold": rmse <= _CATASTROPHIC_MAXIMUM_FINAL_RMSE_PX,
        "p95_within_applied_threshold": p95 <= _CATASTROPHIC_MAXIMUM_FINAL_P95_PX,
    }
    if decision_checks != expected_checks:
        raise ArtifactError("catastrophic reprojection checks differ from metrics")
    if value.get("passed") is not all(expected_checks.values()):
        raise ArtifactError("catastrophic reprojection status differs from checks")
    if (
        expected_checks["rmse_within_applied_threshold"] != expected_rmse_check
        or expected_checks["p95_within_applied_threshold"] != expected_p95_check
    ):
        raise ArtifactError("catastrophic reprojection differs from bootstrap hard checks")
    for name, candidate in legacy.items():
        if _finite_number(candidate, f"legacy {name}") <= 0.0:
            raise ArtifactError("legacy reprojection thresholds must be positive")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactError(f"{name} must be numeric")
    candidate = float(value)
    if candidate != candidate or candidate in {float("inf"), float("-inf")}:
        raise ArtifactError(f"{name} must be finite")
    return candidate


def _digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
