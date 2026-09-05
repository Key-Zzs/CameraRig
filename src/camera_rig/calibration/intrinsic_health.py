"""Train/holdout factory-intrinsic health diagnostics for one physical camera."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.io import atomic_write_json
from camera_rig.calibration.pose.camera_model import to_opencv_camera_model
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.calibration.pose.intrinsic_diagnostic import (
    evaluate_intrinsic_model,
    intrinsic_observation_pose_diversity,
)
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics

INTRINSIC_HEALTH_SCHEMA_VERSION: Final = "camera-rig.intrinsic-health.v1"
IntrinsicHealthStatus = Literal["PASS", "SUSPECT", "INSUFFICIENT_EVIDENCE"]


@dataclass(frozen=True)
class IntrinsicHealthThresholds:
    """Preregistered engineering decision and diversity thresholds."""

    minimum_train_poses: int = 12
    minimum_holdout_poses: int = 4
    minimum_corners_per_pose: int = 12
    minimum_image_centroid_span_fraction: float = 0.20
    minimum_distance_span_fraction: float = 0.05
    minimum_tilt_span_deg: float = 5.0
    maximum_centroid_design_condition_number: float = 100.0
    minimum_holdout_absolute_improvement_px: float = 0.005
    minimum_holdout_relative_improvement: float = 0.20
    minimum_improved_holdout_pose_fraction: float = 0.75
    minimum_engineering_focal_shift_fraction: float = 0.005
    minimum_engineering_principal_shift_fraction: float = 0.005
    minimum_engineering_distortion_shift: float = 0.01
    maximum_refit_focal_shift_fraction: float = 0.10
    maximum_refit_principal_shift_fraction: float = 0.05
    maximum_refit_distortion_shift: float = 0.20

    def __post_init__(self) -> None:
        if self.minimum_train_poses < 2 or self.minimum_holdout_poses < 1:
            raise ContractError("intrinsic-health split thresholds are invalid")
        if self.minimum_corners_per_pose < 4:
            raise ContractError("intrinsic-health corner threshold is invalid")
        for name, value in self.to_dict().items():
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0.0):
                raise ContractError(f"intrinsic-health threshold {name} must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_train_poses": self.minimum_train_poses,
            "minimum_holdout_poses": self.minimum_holdout_poses,
            "minimum_corners_per_pose": self.minimum_corners_per_pose,
            "minimum_image_centroid_span_fraction": self.minimum_image_centroid_span_fraction,
            "minimum_distance_span_fraction": self.minimum_distance_span_fraction,
            "minimum_tilt_span_deg": self.minimum_tilt_span_deg,
            "maximum_centroid_design_condition_number": (
                self.maximum_centroid_design_condition_number
            ),
            "minimum_holdout_absolute_improvement_px": (
                self.minimum_holdout_absolute_improvement_px
            ),
            "minimum_holdout_relative_improvement": self.minimum_holdout_relative_improvement,
            "minimum_improved_holdout_pose_fraction": (self.minimum_improved_holdout_pose_fraction),
            "minimum_engineering_focal_shift_fraction": (
                self.minimum_engineering_focal_shift_fraction
            ),
            "minimum_engineering_principal_shift_fraction": (
                self.minimum_engineering_principal_shift_fraction
            ),
            "minimum_engineering_distortion_shift": self.minimum_engineering_distortion_shift,
            "maximum_refit_focal_shift_fraction": self.maximum_refit_focal_shift_fraction,
            "maximum_refit_principal_shift_fraction": (self.maximum_refit_principal_shift_fraction),
            "maximum_refit_distortion_shift": self.maximum_refit_distortion_shift,
        }


@dataclass(frozen=True)
class IntrinsicHealthObservation:
    """One target pose observed by a single camera."""

    pose_id: str
    object_points_m: npt.NDArray[np.float64]
    image_points_px: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.pose_id:
            raise ContractError("intrinsic-health pose_id must be non-empty")
        object_points = np.asarray(self.object_points_m, dtype=np.float64)
        image_points = np.asarray(self.image_points_px, dtype=np.float64)
        if (
            object_points.ndim != 2
            or object_points.shape[1:] != (3,)
            or image_points.shape != (len(object_points), 2)
            or len(object_points) < 4
            or not np.isfinite(object_points).all()
            or not np.isfinite(image_points).all()
        ):
            raise ContractError("intrinsic-health observation arrays are invalid")
        object_points = object_points.copy()
        image_points = image_points.copy()
        object_points.setflags(write=False)
        image_points.setflags(write=False)
        object.__setattr__(self, "object_points_m", object_points)
        object.__setattr__(self, "image_points_px", image_points)


def evaluate_intrinsic_health(
    observations: tuple[IntrinsicHealthObservation, ...],
    factory_intrinsics: CameraIntrinsics,
    *,
    train_pose_ids: tuple[str, ...],
    holdout_pose_ids: tuple[str, ...],
    thresholds: IntrinsicHealthThresholds | None = None,
    camera_identity_sha256: str | None = None,
    target_identity_sha256: str | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compare fixed factory K/D against a constrained diagnostic refit on untouched holdout."""
    limits = thresholds or IntrinsicHealthThresholds()
    train_ids = tuple(train_pose_ids)
    holdout_ids = tuple(holdout_pose_ids)
    if not train_ids or not holdout_ids or set(train_ids) & set(holdout_ids):
        raise ContractError("intrinsic-health train and holdout IDs must be non-empty and disjoint")
    if len(set(train_ids)) != len(train_ids) or len(set(holdout_ids)) != len(holdout_ids):
        raise ContractError("intrinsic-health split IDs must be unique")
    by_pose: dict[str, IntrinsicHealthObservation] = {}
    for observation in observations:
        if observation.pose_id in by_pose:
            raise ContractError("intrinsic-health observations must contain one entry per pose")
        by_pose[observation.pose_id] = observation
    missing = (set(train_ids) | set(holdout_ids)) - set(by_pose)
    if missing:
        raise ContractError(f"intrinsic-health split references missing poses: {sorted(missing)}")
    train = tuple(by_pose[pose_id] for pose_id in train_ids)
    holdout = tuple(by_pose[pose_id] for pose_id in holdout_ids)
    diversity = _diversity(train, factory_intrinsics)
    evidence_checks = {
        "minimum_train_poses": len(train) >= limits.minimum_train_poses,
        "minimum_holdout_poses": len(holdout) >= limits.minimum_holdout_poses,
        "minimum_corners_each_pose": all(
            len(item.object_points_m) >= limits.minimum_corners_per_pose for item in train + holdout
        ),
        "training_image_x_diversity": _as_float(diversity["centroid_span_x_fraction"])
        >= limits.minimum_image_centroid_span_fraction,
        "training_image_y_diversity": _as_float(diversity["centroid_span_y_fraction"])
        >= limits.minimum_image_centroid_span_fraction,
        "training_distance_diversity": _as_float(diversity["distance_span_fraction"])
        >= limits.minimum_distance_span_fraction,
        "training_tilt_diversity": _as_float(diversity["tilt_span_deg"])
        >= limits.minimum_tilt_span_deg,
        "training_design_conditioning": _as_condition_number(
            diversity["centroid_design_condition_number"]
        )
        <= limits.maximum_centroid_design_condition_number,
    }
    base = {
        "schema_version": INTRINSIC_HEALTH_SCHEMA_VERSION,
        "camera_identity_sha256": camera_identity_sha256,
        "target_identity_sha256": target_identity_sha256,
        "train_pose_ids": list(train_ids),
        "holdout_pose_ids": list(holdout_ids),
        "per_pose_corner_counts": {
            pose_id: len(by_pose[pose_id].object_points_m) for pose_id in train_ids + holdout_ids
        },
        "thresholds": limits.to_dict(),
        "diversity": diversity,
        "evidence_checks": evidence_checks,
        "factory_intrinsics_immutable": True,
        "provenance": dict(provenance or {}),
    }
    if not all(evidence_checks.values()):
        return {
            **base,
            "status": "INSUFFICIENT_EVIDENCE",
            "failure_reasons": [name for name, passed in evidence_checks.items() if not passed],
            "factory_model": _intrinsics_payload(factory_intrinsics),
            "refit_model": None,
            "holdout": None,
        }
    try:
        refit, refit_diagnostics = _refit_intrinsics(train, factory_intrinsics)
        holdout_values = tuple(
            (item.pose_id, item.object_points_m, item.image_points_px) for item in holdout
        )
        factory_holdout = evaluate_intrinsic_model(holdout_values, factory_intrinsics)
        refit_holdout = evaluate_intrinsic_model(holdout_values, refit)
    except ContractError as error:
        return {
            **base,
            "status": "INSUFFICIENT_EVIDENCE",
            "failure_reasons": [f"REFIT_OR_HOLDOUT_FAILED:{error}"],
            "factory_model": _intrinsics_payload(factory_intrinsics),
            "refit_model": None,
            "holdout": None,
        }
    delta = _parameter_delta(factory_intrinsics, refit)
    within_bounds = {
        "focal_shift_within_bound": _as_float(delta["maximum_focal_shift_fraction"])
        <= limits.maximum_refit_focal_shift_fraction,
        "principal_shift_within_bound": _as_float(delta["maximum_principal_shift_fraction"])
        <= limits.maximum_refit_principal_shift_fraction,
        "distortion_shift_within_bound": _as_float(delta["maximum_distortion_shift"])
        <= limits.maximum_refit_distortion_shift,
    }
    factory_rmse = _as_float(factory_holdout["rmse_px"])
    refit_rmse = _as_float(refit_holdout["rmse_px"])
    absolute_improvement = factory_rmse - refit_rmse
    relative_improvement = absolute_improvement / factory_rmse if factory_rmse > 0.0 else 0.0
    factory_per_pose = cast(dict[str, float], factory_holdout["per_pose_rmse_px"])
    refit_per_pose = cast(dict[str, float], refit_holdout["per_pose_rmse_px"])
    paired = {
        pose_id: factory_per_pose[pose_id] - refit_per_pose[pose_id] for pose_id in holdout_ids
    }
    improved_fraction = sum(value > 0.0 for value in paired.values()) / len(paired)
    engineering_shift = (
        _as_float(delta["maximum_focal_shift_fraction"])
        >= limits.minimum_engineering_focal_shift_fraction
        or _as_float(delta["maximum_principal_shift_fraction"])
        >= limits.minimum_engineering_principal_shift_fraction
        or _as_float(delta["maximum_distortion_shift"])
        >= limits.minimum_engineering_distortion_shift
    )
    suspect_checks = {
        "absolute_improvement_engineering_relevant": (
            absolute_improvement >= limits.minimum_holdout_absolute_improvement_px
        ),
        "relative_improvement_engineering_relevant": (
            relative_improvement >= limits.minimum_holdout_relative_improvement
        ),
        "paired_improvement_consistent": (
            improved_fraction >= limits.minimum_improved_holdout_pose_fraction
        ),
        "parameter_shift_engineering_relevant": engineering_shift,
    }
    if not all(within_bounds.values()):
        status: IntrinsicHealthStatus = "INSUFFICIENT_EVIDENCE"
        failures = [name for name, passed in within_bounds.items() if not passed]
    elif all(suspect_checks.values()):
        status = "SUSPECT"
        failures = []
    else:
        status = "PASS"
        failures = []
    return {
        **base,
        "status": status,
        "failure_reasons": failures,
        "factory_model": _intrinsics_payload(factory_intrinsics),
        "refit_model": _intrinsics_payload(refit),
        "refit_diagnostics": refit_diagnostics,
        "parameter_delta": delta,
        "refit_bounds": within_bounds,
        "holdout": {
            "factory": factory_holdout,
            "refit": refit_holdout,
            "absolute_improvement_px": absolute_improvement,
            "relative_improvement": relative_improvement,
            "per_pose_paired_improvement_px": paired,
            "improved_pose_fraction": improved_fraction,
            "image_location_dependence": _image_location_dependence(holdout, paired),
            "paired_improvement_structure": {
                "standard_deviation_px": float(np.std(tuple(paired.values()))),
                "range_px": max(paired.values()) - min(paired.values()),
            },
        },
        "suspect_checks": suspect_checks,
        "decision_rule": (
            "SUSPECT only when absolute, relative, paired-consistency, and engineering-parameter "
            "shift checks all pass on untouched holdout"
        ),
    }


def write_intrinsic_health(path: str | Path, report: dict[str, object]) -> None:
    validate_intrinsic_health_report(report)
    atomic_write_json(path, report)


def validate_intrinsic_health_report(
    report: dict[str, object], *, require_pass: bool = False
) -> dict[str, object]:
    """Validate a one-camera diagnostic semantically, not only by its status label."""

    common = {
        "schema_version",
        "camera_identity_sha256",
        "target_identity_sha256",
        "train_pose_ids",
        "holdout_pose_ids",
        "per_pose_corner_counts",
        "thresholds",
        "diversity",
        "evidence_checks",
        "factory_intrinsics_immutable",
        "provenance",
        "status",
        "failure_reasons",
        "factory_model",
        "refit_model",
        "holdout",
    }
    passed_extra = {
        "refit_diagnostics",
        "parameter_delta",
        "refit_bounds",
        "suspect_checks",
        "decision_rule",
    }
    if report.get("status") in {"PASS", "SUSPECT"}:
        expected_fields = common | passed_extra
    else:
        expected_fields = common
    if set(report) != expected_fields:
        raise ContractError("intrinsic-health report fields are incomplete")
    if report.get("schema_version") != INTRINSIC_HEALTH_SCHEMA_VERSION:
        raise ContractError("intrinsic-health report schema is invalid")
    if report.get("factory_intrinsics_immutable") is not True:
        raise ContractError("intrinsic-health report mutated factory intrinsics")
    for name in ("camera_identity_sha256", "target_identity_sha256"):
        value = report.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ContractError(f"intrinsic-health {name} must be a digest")
    train = _unique_string_list(report.get("train_pose_ids"), "train_pose_ids")
    holdout = _unique_string_list(report.get("holdout_pose_ids"), "holdout_pose_ids")
    if set(train) & set(holdout):
        raise ContractError("intrinsic-health splits overlap")
    threshold_data = report.get("thresholds")
    if not isinstance(threshold_data, dict):
        raise ContractError("intrinsic-health thresholds must be an object")
    try:
        limits = IntrinsicHealthThresholds(**threshold_data)
    except (TypeError, ContractError) as error:
        raise ContractError(f"intrinsic-health thresholds are invalid: {error}") from error
    diversity = report.get("diversity")
    if not isinstance(diversity, dict) or set(diversity) != {
        "pose_count",
        "centroid_span_x_fraction",
        "centroid_span_y_fraction",
        "centroid_span_fraction",
        "centroid_design_condition_number",
        "distance_span_fraction",
        "tilt_span_deg",
        "conditioning",
    }:
        raise ContractError("intrinsic-health diversity payload is invalid")
    corner_counts = report.get("per_pose_corner_counts")
    all_pose_ids = train + holdout
    if (
        not isinstance(corner_counts, dict)
        or set(corner_counts) != set(all_pose_ids)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 4
            for value in corner_counts.values()
        )
    ):
        raise ContractError("intrinsic-health per-pose corner counts are invalid")
    diversity_values = {
        name: _as_float(diversity[name])
        for name in (
            "centroid_span_x_fraction",
            "centroid_span_y_fraction",
            "centroid_span_fraction",
            "distance_span_fraction",
            "tilt_span_deg",
        )
    }
    diversity_values["centroid_design_condition_number"] = _as_condition_number(
        diversity["centroid_design_condition_number"]
    )
    if (
        diversity.get("pose_count") != len(train)
        or diversity.get("conditioning")
        != (
            "FINITE"
            if math.isfinite(diversity_values["centroid_design_condition_number"])
            else "DEGENERATE"
        )
        or any(value < 0.0 for value in diversity_values.values())
        or not math.isclose(
            diversity_values["centroid_span_fraction"],
            max(
                diversity_values["centroid_span_x_fraction"],
                diversity_values["centroid_span_y_fraction"],
            ),
            abs_tol=1e-12,
        )
    ):
        raise ContractError("intrinsic-health diversity semantics are invalid")
    evidence = report.get("evidence_checks")
    expected_evidence = {
        "minimum_train_poses": len(train) >= limits.minimum_train_poses,
        "minimum_holdout_poses": len(holdout) >= limits.minimum_holdout_poses,
        "minimum_corners_each_pose": all(
            corner_counts[pose_id] >= limits.minimum_corners_per_pose for pose_id in all_pose_ids
        ),
        "training_image_x_diversity": _as_float(diversity["centroid_span_x_fraction"])
        >= limits.minimum_image_centroid_span_fraction,
        "training_image_y_diversity": _as_float(diversity["centroid_span_y_fraction"])
        >= limits.minimum_image_centroid_span_fraction,
        "training_distance_diversity": _as_float(diversity["distance_span_fraction"])
        >= limits.minimum_distance_span_fraction,
        "training_tilt_diversity": _as_float(diversity["tilt_span_deg"])
        >= limits.minimum_tilt_span_deg,
        "training_design_conditioning": _as_condition_number(
            diversity["centroid_design_condition_number"]
        )
        <= limits.maximum_centroid_design_condition_number,
    }
    if not isinstance(evidence, dict) or set(evidence) != set(expected_evidence):
        raise ContractError("intrinsic-health evidence checks are invalid")
    if any(evidence[name] is not expected for name, expected in expected_evidence.items()):
        raise ContractError("intrinsic-health evidence decision differs from metrics")
    factory_payload = report.get("factory_model")
    if not isinstance(factory_payload, dict) or set(factory_payload) != {
        "frame",
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
        "distortion_model",
        "distortion_coeffs",
    }:
        raise ContractError("intrinsic-health factory model is invalid")
    factory = CameraIntrinsics.from_dict(factory_payload)
    status = report.get("status")
    if status == "INSUFFICIENT_EVIDENCE":
        failures = report.get("failure_reasons")
        refit_failed = isinstance(failures, list) and any(
            isinstance(value, str) and value.startswith("REFIT_OR_HOLDOUT_FAILED:")
            for value in failures
        )
        if (all(evidence.values()) and not refit_failed) or require_pass:
            raise ContractError("intrinsic-health evidence is insufficient")
        return report
    if status not in {"PASS", "SUSPECT"}:
        raise ContractError("intrinsic-health status is invalid")
    refit_payload = report.get("refit_model")
    if not isinstance(refit_payload, dict) or set(refit_payload) != set(factory_payload):
        raise ContractError("intrinsic-health refit model is invalid")
    refit = CameraIntrinsics.from_dict(refit_payload)
    delta = report.get("parameter_delta")
    if not isinstance(delta, dict) or delta != _parameter_delta(factory, refit):
        raise ContractError("intrinsic-health parameter delta differs from models")
    bounds = report.get("refit_bounds")
    expected_bounds = {
        "focal_shift_within_bound": _as_float(delta["maximum_focal_shift_fraction"])
        <= limits.maximum_refit_focal_shift_fraction,
        "principal_shift_within_bound": _as_float(delta["maximum_principal_shift_fraction"])
        <= limits.maximum_refit_principal_shift_fraction,
        "distortion_shift_within_bound": _as_float(delta["maximum_distortion_shift"])
        <= limits.maximum_refit_distortion_shift,
    }
    if bounds != expected_bounds:
        raise ContractError("intrinsic-health refit bounds differ from models")
    holdout_report = report.get("holdout")
    if not isinstance(holdout_report, dict):
        raise ContractError("intrinsic-health holdout payload is invalid")
    paired = holdout_report.get("per_pose_paired_improvement_px")
    factory_eval = holdout_report.get("factory")
    refit_eval = holdout_report.get("refit")
    if (
        not isinstance(paired, dict)
        or not isinstance(factory_eval, dict)
        or not isinstance(refit_eval, dict)
    ):
        raise ContractError("intrinsic-health holdout model comparisons are invalid")
    factory_per_pose = factory_eval.get("per_pose_rmse_px")
    refit_per_pose = refit_eval.get("per_pose_rmse_px")
    if (
        not isinstance(factory_per_pose, dict)
        or not isinstance(refit_per_pose, dict)
        or set(factory_per_pose) != set(holdout)
        or set(refit_per_pose) != set(holdout)
        or set(paired) != set(holdout)
    ):
        raise ContractError("intrinsic-health holdout split differs from per-pose metrics")
    _validate_holdout_model_evaluation(
        factory_eval, holdout=holdout, corner_counts=corner_counts, label="factory"
    )
    _validate_holdout_model_evaluation(
        refit_eval, holdout=holdout, corner_counts=corner_counts, label="refit"
    )
    expected_paired = {
        pose_id: _as_float(factory_per_pose[pose_id]) - _as_float(refit_per_pose[pose_id])
        for pose_id in holdout
    }
    if any(
        not math.isclose(_as_float(paired[name]), value, abs_tol=1e-12)
        for name, value in expected_paired.items()
    ):
        raise ContractError("intrinsic-health paired holdout metrics differ")
    factory_rmse = _as_float(factory_eval.get("rmse_px"))
    refit_rmse = _as_float(refit_eval.get("rmse_px"))
    absolute = factory_rmse - refit_rmse
    relative = absolute / factory_rmse if factory_rmse > 0.0 else 0.0
    improved_fraction = sum(value > 0.0 for value in expected_paired.values()) / len(holdout)
    for name, expected in {
        "absolute_improvement_px": absolute,
        "relative_improvement": relative,
        "improved_pose_fraction": improved_fraction,
    }.items():
        if not math.isclose(_as_float(holdout_report.get(name)), expected, abs_tol=1e-12):
            raise ContractError(f"intrinsic-health holdout aggregate differs: {name}")
    engineering_shift = (
        _as_float(delta["maximum_focal_shift_fraction"])
        >= limits.minimum_engineering_focal_shift_fraction
        or _as_float(delta["maximum_principal_shift_fraction"])
        >= limits.minimum_engineering_principal_shift_fraction
        or _as_float(delta["maximum_distortion_shift"])
        >= limits.minimum_engineering_distortion_shift
    )
    expected_suspect = {
        "absolute_improvement_engineering_relevant": absolute
        >= limits.minimum_holdout_absolute_improvement_px,
        "relative_improvement_engineering_relevant": relative
        >= limits.minimum_holdout_relative_improvement,
        "paired_improvement_consistent": improved_fraction
        >= limits.minimum_improved_holdout_pose_fraction,
        "parameter_shift_engineering_relevant": engineering_shift,
    }
    if report.get("suspect_checks") != expected_suspect:
        raise ContractError("intrinsic-health suspect decision differs from metrics")
    expected_status = (
        "INSUFFICIENT_EVIDENCE"
        if not all(expected_bounds.values())
        else "SUSPECT"
        if all(expected_suspect.values())
        else "PASS"
    )
    if status != expected_status or (require_pass and status != "PASS"):
        raise ContractError("intrinsic-health status differs from recomputed decision")
    diagnostics = report.get("refit_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ContractError("intrinsic-health refit diagnostics are incomplete")
    per_view = diagnostics.get("per_view_rmse_px")
    if (
        not isinstance(per_view, list)
        or len(per_view) != len(train)
        or any(_as_float(value) < 0.0 for value in per_view)
    ):
        raise ContractError("intrinsic-health refit per-view diagnostics are invalid")
    return report


def _validate_holdout_model_evaluation(
    value: dict[str, object],
    *,
    holdout: tuple[str, ...],
    corner_counts: dict[str, object],
    label: str,
) -> None:
    if set(value) != {
        "rmse_px",
        "p95_px",
        "per_pose_rmse_px",
        "per_pose_p95_px",
        "sample_count",
    }:
        raise ContractError(f"intrinsic-health {label} holdout fields are invalid")
    per_pose_rmse = value.get("per_pose_rmse_px")
    per_pose_p95 = value.get("per_pose_p95_px")
    expected_samples = sum(cast(int, corner_counts[pose_id]) for pose_id in holdout)
    if (
        not isinstance(per_pose_rmse, dict)
        or not isinstance(per_pose_p95, dict)
        or set(per_pose_rmse) != set(holdout)
        or set(per_pose_p95) != set(holdout)
        or value.get("sample_count") != expected_samples
        or _as_float(value.get("rmse_px")) < 0.0
        or _as_float(value.get("p95_px")) < 0.0
        or any(_as_float(item) < 0.0 for item in per_pose_rmse.values())
        or any(_as_float(item) < 0.0 for item in per_pose_p95.values())
    ):
        raise ContractError(f"intrinsic-health {label} holdout semantics are invalid")


def _unique_string_list(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ContractError(f"intrinsic-health {name} must contain unique strings")
    return tuple(value)


def _refit_intrinsics(
    observations: tuple[IntrinsicHealthObservation, ...], factory: CameraIntrinsics
) -> tuple[CameraIntrinsics, dict[str, object]]:
    cv2 = cv2_module()
    camera = to_opencv_camera_model(factory)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    freeze_distortion = (
        factory.distortion_model
        in {
            "none",
            "pinhole",
            "inverse-brown-conrady",
        }
        or not factory.distortion_coeffs
    )
    if freeze_distortion:
        flags |= (
            cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        )
    object_points = [np.asarray(item.object_points_m, dtype=np.float32) for item in observations]
    image_points = [np.asarray(item.image_points_px, dtype=np.float32) for item in observations]
    try:
        calibrate_camera = getattr(cv2, "calibrate" + "CameraExtended")
        (
            reprojection_error,
            matrix,
            distortion,
            _rvecs,
            _tvecs,
            std_intrinsics,
            _std_extrinsics,
            per_view,
        ) = calibrate_camera(
            object_points,
            image_points,
            (factory.width, factory.height),
            camera.camera_matrix.copy(),
            camera.distortion_coeffs.copy(),
            flags=flags,
        )
    except cv2.error as cv_error:
        raise ContractError(f"diagnostic intrinsic refit failed: {cv_error}") from cv_error
    values = np.asarray(distortion, dtype=np.float64).reshape(-1)
    coefficient_count = len(factory.distortion_coeffs)
    model = CameraIntrinsics(
        frame=factory.frame,
        width=factory.width,
        height=factory.height,
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        distortion_model=factory.distortion_model,
        distortion_coeffs=(
            factory.distortion_coeffs
            if freeze_distortion
            else tuple(float(value) for value in values[:coefficient_count])
        ),
    )
    standard_deviations = np.asarray(std_intrinsics, dtype=np.float64).reshape(-1)
    return model, {
        "training_rmse_px": float(reprojection_error),
        "intrinsic_standard_deviations": standard_deviations.tolist(),
        "maximum_finite_intrinsic_standard_deviation": (
            float(np.max(standard_deviations[np.isfinite(standard_deviations)]))
            if np.isfinite(standard_deviations).any()
            else None
        ),
        "per_view_rmse_px": np.asarray(per_view, dtype=np.float64).reshape(-1).tolist(),
        "distortion_refit_policy": (
            "frozen_for_inverse_or_zero_distortion" if freeze_distortion else "bounded_refit"
        ),
    }


def _diversity(
    observations: tuple[IntrinsicHealthObservation, ...], intrinsics: CameraIntrinsics
) -> dict[str, object]:
    if not observations:
        return {
            "centroid_span_fraction": 0.0,
            "pose_count": 0,
            "conditioning": "INSUFFICIENT",
        }
    centroids = np.asarray([np.mean(item.image_points_px, axis=0) for item in observations])
    x_span = float(np.ptp(centroids[:, 0])) / intrinsics.width
    y_span = float(np.ptp(centroids[:, 1])) / intrinsics.height
    design = np.column_stack(
        (
            centroids[:, 0] / intrinsics.width,
            centroids[:, 1] / intrinsics.height,
            np.ones(len(centroids)),
        )
    )
    singular = np.linalg.svd(design, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else float("inf")
    distances, tilts = intrinsic_observation_pose_diversity(
        tuple((item.pose_id, item.object_points_m, item.image_points_px) for item in observations),
        intrinsics,
    )
    mean_distance = float(np.mean(distances))
    return {
        "pose_count": len(observations),
        "centroid_span_x_fraction": x_span,
        "centroid_span_y_fraction": y_span,
        "centroid_span_fraction": max(x_span, y_span),
        "centroid_design_condition_number": condition,
        "distance_span_fraction": (
            (max(distances) - min(distances)) / mean_distance if mean_distance > 0.0 else 0.0
        ),
        "tilt_span_deg": max(tilts) - min(tilts),
        "conditioning": "FINITE" if math.isfinite(condition) else "DEGENERATE",
    }


def _image_location_dependence(
    observations: tuple[IntrinsicHealthObservation, ...], paired: dict[str, float]
) -> dict[str, object]:
    centroids = np.asarray([np.mean(item.image_points_px, axis=0) for item in observations])
    values = np.asarray([paired[item.pose_id] for item in observations], dtype=np.float64)

    def correlation(axis: int) -> float | None:
        if len(values) < 3 or np.std(values) <= 1e-12 or np.std(centroids[:, axis]) <= 1e-12:
            return None
        return float(np.corrcoef(centroids[:, axis], values)[0, 1])

    return {
        "paired_improvement_vs_centroid_x_correlation": correlation(0),
        "paired_improvement_vs_centroid_y_correlation": correlation(1),
    }


def _parameter_delta(factory: CameraIntrinsics, refit: CameraIntrinsics) -> dict[str, object]:
    focal = (abs(refit.fx - factory.fx) / factory.fx, abs(refit.fy - factory.fy) / factory.fy)
    principal = (
        abs(refit.cx - factory.cx) / factory.width,
        abs(refit.cy - factory.cy) / factory.height,
    )
    length = max(len(factory.distortion_coeffs), len(refit.distortion_coeffs))
    factory_d = np.pad(
        np.asarray(factory.distortion_coeffs),
        (0, length - len(factory.distortion_coeffs)),
    )
    refit_d = np.pad(
        np.asarray(refit.distortion_coeffs),
        (0, length - len(refit.distortion_coeffs)),
    )
    distortion = np.abs(refit_d - factory_d)
    return {
        "fx": refit.fx - factory.fx,
        "fy": refit.fy - factory.fy,
        "cx": refit.cx - factory.cx,
        "cy": refit.cy - factory.cy,
        "distortion_coeffs": (refit_d - factory_d).tolist(),
        "maximum_focal_shift_fraction": max(focal),
        "maximum_principal_shift_fraction": max(principal),
        "maximum_distortion_shift": float(np.max(distortion)) if len(distortion) else 0.0,
    }


def _intrinsics_payload(value: CameraIntrinsics) -> dict[str, object]:
    return value.to_dict()


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError("intrinsic-health numeric result is invalid")
    candidate = float(value)
    if not math.isfinite(candidate):
        raise ContractError("intrinsic-health numeric result must be finite")
    return candidate


def _as_condition_number(value: object) -> float:
    """Accept positive infinity as the explicit marker for a degenerate design."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError("intrinsic-health condition number is invalid")
    candidate = float(value)
    if math.isnan(candidate) or candidate < 0.0:
        raise ContractError("intrinsic-health condition number must be non-negative")
    return candidate
