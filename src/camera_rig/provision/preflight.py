"""Read-only preflight for one-command fixed-camera provisioning."""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import cast

from camera_rig.artifacts.io import JsonValue, atomic_write_json, load_json
from camera_rig.calibration.fixed.aggregation import distribution
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.config.validation import validate_against_named_schema
from camera_rig.core.errors import ArtifactError, CameraRigError, MissingOptionalDependencyError
from camera_rig.provision.config import ProvisionConfig
from camera_rig.targets.io import validate_target_artifact

PROVISION_PREFLIGHT_SCHEMA_VERSION = "camera-rig.fixed-provision-preflight.v1"


def preflight_fixed_provision(
    config: ProvisionConfig,
    *,
    output: str | Path,
    force: bool = True,
) -> dict[str, object]:
    """Validate every non-hardware input without opening the camera or writing output."""
    target = validate_target_artifact(config.target.artifact_path)
    if target.artifact_sha256 != config.target.expected_sha256:
        raise ArtifactError(
            "resolved target SHA does not match target.expected_sha256; "
            "do not regenerate or substitute the physical-board artifact"
        )
    if target.target_frame != config.fixed_calibration_config.target_frame:
        raise ArtifactError("target coordinate frame differs from the workspace contract")
    destination = Path(output)
    if destination.exists() and not force:
        raise ArtifactError(f"provision output already exists: {destination}")
    if destination.is_symlink():
        raise ArtifactError("provision output must not be a symlink")
    if destination.parent.exists() and destination.parent.is_symlink():
        raise ArtifactError("provision output parent must not be a symlink")
    if destination.exists() and force:
        from camera_rig.provision.validation import load_and_validate_fixed_provision

        try:
            load_and_validate_fixed_provision(destination)
        except ArtifactError as error:
            raise ArtifactError(
                "default replacement requires an existing validated fixed-provision artifact"
            ) from error
    _require_runtime_dependencies()
    return {
        "schema_version": config.schema_version,
        "mode": "dry-run-safe-preflight",
        "camera_driver": config.camera_config.camera.driver,
        "expected_model": config.camera_config.camera.expected_model,
        "enabled_streams": sorted(
            name for name, settings in config.camera_config.streams.items() if settings.enabled
        ),
        "stream_validation_frames": config.acquisition.stream_validation_frames,
        "calibration_frames": config.acquisition.calibration_frames,
        "selected_frame_policy": "deterministic_evenly_spaced",
        "target_artifact": config.target.artifact_reference,
        "target_sha256": target.artifact_sha256,
        "target_frame": target.target_frame,
        "workspace_frame": config.fixed_calibration_config.workspace_frame,
        "detection_stream": config.fixed_calibration_config.detection_stream,
        "target_detection_policy": config.target.detection_policy,
        "reference_stream": config.fixed_calibration_config.reference_stream,
        "output_exists": destination.exists(),
        "overwrite_existing": force,
        "camera_will_open": False,
        "final_artifact_will_be_created": False,
        "optional_dependencies": {
            "realsense": "available",
            "charuco": "available",
            "viz": "available",
        },
    }


def _require_runtime_dependencies() -> None:
    cv2_module()
    try:
        importlib.import_module("PIL")
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'fixed provisioning requires: pip install "camera-rig[viz]"'
        ) from error
    try:
        importlib.import_module("pyrealsense2")
    except ImportError as error:
        raise MissingOptionalDependencyError(
            'fixed provisioning requires: pip install "camera-rig[realsense]"'
        ) from error


def run_fixed_provision_preflight(
    config: ProvisionConfig,
    *,
    report: str | Path,
    overlays: str | Path,
    dependencies: object | None = None,
) -> dict[str, object]:
    """Run the production acquisition/evaluation core without publishing a provision."""
    from camera_rig.provision.workflow import (
        ProvisionWorkflowDependencies,
        run_fixed_provision_workflow,
    )

    report_path = Path(report)
    overlays_path = Path(overlays)
    _validate_diagnostic_destinations(report_path, overlays_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=report_path.parent, prefix=f".{report_path.stem}.evaluation-")
    )
    failure: str | None = None
    result = None
    try:
        try:
            result = run_fixed_provision_workflow(
                config,
                staging,
                dependencies=(
                    dependencies
                    if isinstance(dependencies, ProvisionWorkflowDependencies)
                    else None
                ),
                allow_failed_quality=True,
            )
        except CameraRigError as error:
            failure = str(error)
        value = _build_preflight_report(config, staging, result=result, failure=failure)
        validate_against_named_schema(
            cast(JsonValue, value), "fixed_provision_preflight.v1.schema.json"
        )
        overlays_preexisted = overlays_path.exists()
        _publish_diagnostic_overlays(staging, overlays_path)
        try:
            atomic_write_json(report_path, value)
        except Exception:
            shutil.rmtree(overlays_path, ignore_errors=True)
            if overlays_preexisted:
                overlays_path.mkdir(parents=True, exist_ok=True)
            raise
        return value
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _build_preflight_report(
    config: ProvisionConfig,
    staging: Path,
    *,
    result: object | None,
    failure: str | None,
) -> dict[str, object]:
    raw = _optional_json(staging / "reports/stream_validation.json")
    detection = _optional_json(staging / "target/detection_report.json")
    fixed = _fixed_evaluation_json(staging, result)
    raw_status = raw.get("status") if raw else "FAIL"
    raw_reasons = _string_list(raw.get("failure_reasons")) if raw else _failure_list(failure)
    target_evaluated = raw_status == "PASS" and bool(detection)
    fixed_evaluated = target_evaluated and bool(fixed)
    quality = _mapping(fixed.get("quality")) if fixed else {}
    would_pass = fixed_evaluated and quality.get("passed") is True
    per_frame = _mapping_list(fixed.get("per_frame_pose_summary")) if fixed else []
    solved = [item for item in per_frame if item.get("T_camera_from_target") is not None]
    frame_gate = [item for item in per_frame if item.get("frame_gate_accepted") is True]
    accepted = [item for item in per_frame if item.get("accepted") is True]
    reason_counts = Counter(
        reason for item in per_frame for reason in _string_list(item.get("failure_reasons"))
    )
    reprojection = _reprojection_summary(config, solved, fixed)
    observability = _observability_summary(solved, fixed)
    final = _final_summary(fixed, quality, would_pass)
    target_acceptance = _mapping(detection.get("acceptance")) if detection else {}
    target_aggregate = _mapping(detection.get("aggregate")) if detection else {}
    final_failures = _string_list(quality.get("failure_reasons"))
    if not fixed_evaluated and failure:
        final_failures = [failure]
    return {
        "schema_version": PROVISION_PREFLIGHT_SCHEMA_VERSION,
        "attempt_id": str(uuid.uuid4()),
        "status": "PASS" if would_pass else "FAIL",
        "would_publish_fixed_provision": would_pass,
        "camera": {"logical_name": config.camera_config.camera.name},
        "target_fingerprint": config.target.expected_sha256,
        "pose_policy": config.target.detection_policy,
        "evaluation_core": "run_fixed_provision_workflow",
        "raw_stream": {
            "status": raw_status,
            "metrics": _mapping(raw.get("quality")).get("metrics", {}) if raw else {},
            "failure_reasons": raw_reasons,
        },
        "target": (
            {
                "status": "PASS" if target_acceptance.get("passed") is True else "FAIL",
                "detection_success_ratio": target_aggregate.get("success_ratio"),
                "corner_statistics": target_aggregate.get("detected_charuco_corner_count"),
                "coverage_advisory": target_aggregate.get("coverage_ratio"),
                "acceptance": target_acceptance,
            }
            if target_evaluated
            else {"status": "NOT_EVALUATED", "failure_reasons": raw_reasons}
        ),
        "fixed_pose_frames": (
            {
                "status": "EVALUATED",
                "total": len(per_frame),
                "solved": len(solved),
                "frame_gate_accepted": len(frame_gate),
                "pose_inlier_accepted": len(accepted),
                "required_frames": config.fixed_calibration_config.solver.minimum_accepted_frames,
                "required_ratio": config.fixed_calibration_config.solver.minimum_accepted_ratio,
                "failure_reason_counts": dict(sorted(reason_counts.items())),
            }
            if fixed_evaluated
            else {"status": "NOT_EVALUATED", "failure_reasons": final_failures}
        ),
        "reprojection": reprojection,
        "observability": observability,
        "final": final,
        "per_frame": per_frame,
        "failure_reasons": final_failures,
        "publication": {
            "camera_bundle_written": False,
            "fixed_provision_written": False,
            "canonical_output_modified": False,
        },
    }


def _reprojection_summary(
    config: ProvisionConfig,
    solved: list[dict[str, object]],
    fixed: dict[str, object],
) -> dict[str, object]:
    if not fixed:
        return {"status": "NOT_EVALUATED"}
    rmse = _numbers(solved, "reprojection_rmse_px")
    p95 = _numbers(solved, "reprojection_p95_px")
    solver = _mapping(fixed.get("solver"))
    policy = _mapping(solver.get("reprojection_policy"))
    first_decision = _mapping(solved[0].get("reprojection_decision")) if solved else {}
    quality = _mapping(fixed.get("quality"))
    quality_metrics = _mapping(quality.get("metrics"))
    final_decision = _mapping(quality_metrics.get("reprojection_decision"))
    return {
        "status": "EVALUATED",
        "policy_actually_used": policy.get("name"),
        "frame_rmse_px": distribution(rmse) if rmse else None,
        "frame_p95_px": distribution(p95) if p95 else None,
        "legacy_thresholds": {
            "maximum_frame_rmse_px": (config.fixed_calibration_config.solver.maximum_frame_rmse_px),
            "maximum_frame_p95_px": (config.fixed_calibration_config.solver.maximum_frame_p95_px),
        },
        "applied_frame_thresholds": first_decision.get("applied_thresholds"),
        "applied_final_thresholds": final_decision.get("applied_thresholds"),
        "applied_policy": policy,
    }


def _observability_summary(
    solved: list[dict[str, object]], fixed: dict[str, object]
) -> dict[str, object]:
    if not fixed:
        return {"status": "NOT_EVALUATED"}
    observability = [
        _mapping(item.get("observability")) for item in solved if item.get("observability")
    ]
    return {
        "status": "EVALUATED" if observability else "NOT_APPLICABLE_FOR_POLICY",
        "translation_worst_std_mm": _optional_distribution(
            observability, "translation_worst_axis_std_mm"
        ),
        "rotation_worst_std_deg": _optional_distribution(
            observability, "rotation_worst_axis_std_deg"
        ),
        "scaled_condition_number": _optional_distribution(observability, "scaled_condition_number"),
        "ambiguous_frames": sum(
            _mapping(item.get("candidate_ambiguity")).get("ambiguous") is True
            for item in observability
        ),
    }


def _final_summary(
    fixed: dict[str, object], quality: dict[str, object], would_pass: bool
) -> dict[str, object]:
    if not fixed or fixed.get("status") == "failed_before_shared_pose":
        return {
            "status": "NOT_EVALUATED",
            "decision": "WOULD_FAIL",
            "failure_reasons": _string_list(quality.get("failure_reasons")),
        }
    aggregate = _mapping(fixed.get("aggregate"))
    reprojection = _mapping(aggregate.get("reprojection"))
    metrics = _mapping(quality.get("metrics"))
    return {
        "status": "EVALUATED",
        "global_reprojection": reprojection.get("global"),
        "reprojection_decision": metrics.get("reprojection_decision"),
        "final_pose_observability": aggregate.get("final_pose_observability"),
        "pose_repeatability": aggregate.get("pose_repeatability"),
        "split_half": aggregate.get("split_half"),
        "native_depth": aggregate.get("native_depth_sanity"),
        "decision": "WOULD_PASS" if would_pass else "WOULD_FAIL",
        "failure_reasons": _string_list(quality.get("failure_reasons")),
    }


def _validate_diagnostic_destinations(report: Path, overlays: Path) -> None:
    if report.is_symlink() or overlays.is_symlink():
        raise ArtifactError("provision preflight destinations must not be symlinks")
    if report.exists():
        raise ArtifactError("provision preflight report destination must not already exist")
    if overlays.exists() and (not overlays.is_dir() or any(overlays.iterdir())):
        raise ArtifactError("provision preflight overlays destination must be empty")


def _publish_diagnostic_overlays(staging: Path, destination: Path) -> None:
    source = staging / "diagnostics"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.publish-")
    )
    candidate = temporary_root / "overlays"
    try:
        if source.exists():
            shutil.copytree(source, candidate)
        else:
            candidate.mkdir()
        if destination.exists():
            destination.rmdir()
        os.replace(candidate, destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _fixed_evaluation_json(staging: Path, result: object | None) -> dict[str, object]:
    fixed = getattr(result, "fixed_calibration", None)
    if fixed is not None and hasattr(fixed, "to_dict"):
        value = fixed.to_dict()
        return value if isinstance(value, dict) else {}
    failed = _optional_json(staging / "calibration/fixed_calibration.failed.json")
    if failed:
        return _mapping(failed.get("fixed_calibration"))
    return _optional_json(staging / "calibration/fixed_calibration.frame_gate_failed.json")


def _optional_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value = load_json(path)
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]


def _failure_list(value: str | None) -> list[str]:
    return [value] if value else []


def _numbers(values: list[dict[str, object]], key: str) -> list[float]:
    result: list[float] = []
    for item in values:
        value = item.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            result.append(float(value))
    return result


def _optional_distribution(values: list[dict[str, object]], key: str) -> dict[str, float] | None:
    numbers = _numbers(values, key)
    return distribution(numbers) if numbers else None
