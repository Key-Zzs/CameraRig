from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from camera_rig.calibration.fixed.depth_sanity import (
    evaluate_native_depth_sanity,
    validate_native_depth_evaluation,
)
from camera_rig.core.errors import ArtifactError, ContractError
from camera_rig.core.quality import QualityReport
from camera_rig.provision.bootstrap_qualification import (
    STRUCTURED_RESIDUAL_PRODUCTION_GATE,
    build_bootstrap_qualification,
    qualification_fingerprint,
    validate_bootstrap_qualification_data,
)
from camera_rig.provision.config import (
    BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
    BOOTSTRAP_METRIC_DEPTH_THRESHOLDS,
)
from camera_rig.targets.metrology import (
    TargetMetrologyReceipt,
    TargetScaleAcceptance,
    build_target_scale_acceptance_policy,
    evaluate_target_metrology,
)
from tests.test_fixed_camera_bundle import (
    TARGET_SHA,
    _factory,
    _fixed,
    _stream_validation,
    _target_detection,
)
from tests.test_fixed_diagnostics import _factory as _depth_factory
from tests.test_fixed_diagnostics import _frames, _front_pose, _target


def _metric_depth() -> dict[str, object]:
    thresholds = BOOTSTRAP_METRIC_DEPTH_THRESHOLDS
    return evaluate_native_depth_sanity(
        target=_target(),
        calibration=_depth_factory(),
        T_detection_from_target=_front_pose(),
        detection_stream="color",
        frames=_frames(750, count=30),
        frame_indices=tuple(range(30)),
        minimum_valid_samples=int(thresholds["minimum_valid_samples"]),
        minimum_valid_frames=int(thresholds["minimum_valid_frames"]),
        minimum_valid_sample_ratio=float(thresholds["minimum_valid_sample_ratio"]),
        minimum_region_valid_samples=int(thresholds["minimum_region_valid_samples"]),
        minimum_frame_valid_samples=int(thresholds["minimum_frame_valid_samples"]),
        minimum_passing_frames=int(thresholds["minimum_passing_frames"]),
        minimum_passing_frame_ratio=float(thresholds["minimum_passing_frame_ratio"]),
        maximum_median_error_mm=float(thresholds["maximum_median_error_mm"]),
        maximum_p95_error_mm=float(thresholds["maximum_p95_error_mm"]),
        maximum_plane_offset_mm=float(thresholds["maximum_plane_offset_mm"]),
        maximum_plane_normal_error_deg=float(thresholds["maximum_plane_normal_error_deg"]),
        maximum_scale_ratio_error=float(thresholds["maximum_scale_ratio_error"]),
        threshold_policy={
            "schema_version": BOOTSTRAP_METRIC_DEPTH_POLICY_VERSION,
            "source": "immutable_fixed_provision_contract",
        },
        fail_closed=True,
    )


def _metrology() -> TargetMetrologyReceipt:
    target = _target().with_artifact_sha256(TARGET_SHA)
    policy = build_target_scale_acceptance_policy(
        created_at="2026-09-05T07:00:00Z",
        target=target,
        acceptance=TargetScaleAcceptance(3.0, 1500.0),
        provenance={"authority": "test contract", "measurement_values_available": False},
    )
    return evaluate_target_metrology(
        created_at="2026-09-05T08:00:00Z",
        target=target,
        horizontal_square_count=5,
        vertical_square_count=4,
        horizontal_measurements_mm=(150.0, 150.0, 150.0),
        vertical_measurements_mm=(120.0, 120.0, 120.0),
        measurement_method="caliper",
        instrument="caliper",
        instrument_resolution_mm=0.01,
        measurement_uncertainty_mm=0.05,
        acceptance_policy=policy,
        provenance={},
    )


def test_structured_residual_failure_is_diagnostic_only() -> None:
    factory = _factory()
    detection = _target_detection()
    fixed = _fixed(factory, detection)
    checks = {
        "minimum_accepted_frames": True,
        "minimum_accepted_ratio": True,
        "gross_global_reprojection_rmse": True,
        "gross_global_reprojection_p95": True,
        "pose_translation_p95": True,
        "pose_rotation_p95": True,
        "split_translation_delta": True,
        "split_rotation_delta": True,
        "native_depth_sanity": True,
        "observable_frame_ratio": True,
        "ambiguous_frame_ratio": True,
        "final_pose_observability": True,
        "final_pose_full_rank": True,
        "final_pose_translation_uncertainty": True,
        "final_pose_rotation_uncertainty": True,
        "final_pose_condition_number": True,
        "final_pose_unambiguous": True,
    }
    fixed = replace(
        fixed,
        solver={**fixed.solver, "pose_policy": "uncertainty_validated"},
        aggregate={**fixed.aggregate, "native_depth_sanity": _metric_depth()},
        quality=QualityReport(
            passed=True,
            metrics={
                **fixed.quality.metrics,
                "checks": checks,
                "reprojection_decision": {
                    "policy": "uncertainty_gross_model_consistency",
                    "passed": True,
                    "checks": {
                        "rmse_within_applied_threshold": True,
                        "p95_within_applied_threshold": True,
                    },
                    "metrics": {"rmse_px": 0.1, "p95_px": 0.2},
                    "applied_thresholds": {
                        "maximum_final_rmse_px": 1.5,
                        "maximum_final_p95_px": 2.0,
                    },
                    "legacy_precision_thresholds": {
                        "maximum_frame_rmse_px": 0.5,
                        "maximum_frame_p95_px": 1.0,
                    },
                },
                "final_structured_residual": {"passed": False, "warning": "diagnostic"},
            },
            thresholds=fixed.quality.thresholds,
        ),
    )
    report = build_bootstrap_qualification(
        camera_identity_sha256="a" * 64,
        camera_bundle_fingerprint="b" * 64,
        target_identity_sha256=TARGET_SHA,
        target_metrology_sha256="c" * 64,
        metric_depth_receipt_sha256="d" * 64,
        stream_validation=_stream_validation(),
        target_detection=detection,
        target_metrology=_metrology(),
        fixed_calibration=fixed,
        provenance={},
    )
    assert report["status"] == "PASS"
    structured = report["structured_residual"]
    assert isinstance(structured, dict)
    assert structured["enforced"] is False
    assert structured["production_gate"] == STRUCTURED_RESIDUAL_PRODUCTION_GATE
    assert validate_bootstrap_qualification_data(report) == report
    catastrophic = report["catastrophic_reprojection"]
    assert isinstance(catastrophic, dict)
    catastrophic["role"] = "forged"
    report["qualification_fingerprint"] = qualification_fingerprint(report)
    with pytest.raises(ArtifactError, match="disclaimer"):
        validate_bootstrap_qualification_data(report)


def test_bootstrap_rejects_rehashed_catastrophic_threshold_forgery() -> None:
    factory = _factory()
    detection = _target_detection()
    fixed = _fixed(factory, detection)
    checks = {
        "minimum_accepted_frames": True,
        "minimum_accepted_ratio": True,
        "gross_global_reprojection_rmse": True,
        "gross_global_reprojection_p95": True,
        "pose_translation_p95": True,
        "pose_rotation_p95": True,
        "split_translation_delta": True,
        "split_rotation_delta": True,
        "native_depth_sanity": True,
        "observable_frame_ratio": True,
        "ambiguous_frame_ratio": True,
        "final_pose_observability": True,
        "final_pose_full_rank": True,
        "final_pose_translation_uncertainty": True,
        "final_pose_rotation_uncertainty": True,
        "final_pose_condition_number": True,
        "final_pose_unambiguous": True,
    }
    decision = {
        "policy": "uncertainty_gross_model_consistency",
        "passed": True,
        "checks": {
            "rmse_within_applied_threshold": True,
            "p95_within_applied_threshold": True,
        },
        "metrics": {"rmse_px": 0.1, "p95_px": 0.2},
        "applied_thresholds": {
            "maximum_final_rmse_px": 1.5,
            "maximum_final_p95_px": 2.0,
        },
        "legacy_precision_thresholds": {
            "maximum_frame_rmse_px": 0.5,
            "maximum_frame_p95_px": 1.0,
        },
    }
    fixed = replace(
        fixed,
        solver={**fixed.solver, "pose_policy": "uncertainty_validated"},
        aggregate={**fixed.aggregate, "native_depth_sanity": _metric_depth()},
        quality=QualityReport(
            passed=True,
            metrics={**fixed.quality.metrics, "checks": checks, "reprojection_decision": decision},
            thresholds=fixed.quality.thresholds,
        ),
    )
    report = build_bootstrap_qualification(
        camera_identity_sha256="a" * 64,
        camera_bundle_fingerprint="b" * 64,
        target_identity_sha256=TARGET_SHA,
        target_metrology_sha256="c" * 64,
        metric_depth_receipt_sha256="d" * 64,
        stream_validation=_stream_validation(),
        target_detection=detection,
        target_metrology=_metrology(),
        fixed_calibration=fixed,
        provenance={},
    )
    forged = deepcopy(report)
    catastrophic = forged["catastrophic_reprojection"]
    catastrophic["decision"]["applied_thresholds"] = {
        "maximum_final_rmse_px": 999.0,
        "maximum_final_p95_px": 999.0,
    }
    forged["qualification_fingerprint"] = qualification_fingerprint(forged)
    with pytest.raises(ArtifactError, match="not frozen"):
        validate_bootstrap_qualification_data(forged)


def test_bootstrap_rejects_rehashed_metric_depth_threshold_forgery() -> None:
    metric = _metric_depth()
    metric["thresholds"]["maximum_plane_offset_mm"] = 1000.0
    with pytest.raises(ContractError, match="frozen contract"):
        validate_native_depth_evaluation(
            metric, require_pass=True, require_fixed_bootstrap_policy=True
        )
