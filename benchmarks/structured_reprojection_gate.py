"""Deterministic development/holdout protocol for structured reprojection gates.

The split is frozen before any metrics are evaluated. ``development`` refuses to
evaluate holdout families. No holdout evaluator exists while development is failing
and this split's planned final holdout denominator is release-ineligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from camera_rig.artifacts.hashing import sha256_file
from camera_rig.artifacts.io import atomic_write_json, load_json
from camera_rig.calibration.fixed.aggregation import pose_delta
from camera_rig.calibration.fixed.structured_residuals import (
    StructuredResidualThresholds,
    evaluate_final_shared_structured_residuals,
    evaluate_structured_residuals,
)
from camera_rig.calibration.pose import (
    PlanarPoseEstimator,
    evaluate_pose_observability,
    project_points_px,
    refine_planar_pose_lm,
)
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

SPLIT_SCHEMA_VERSION = "camera-rig.structured-reprojection-split.v1"
DEVELOPMENT_SCHEMA_VERSION = "camera-rig.structured-reprojection-development.v1"
THRESHOLD_SCAN_SCHEMA_VERSION = "camera-rig.structured-reprojection-threshold-scan.v1"
GENERATOR_VERSION = "physical_model_counterfactuals_v1"
SPLIT_SALT = "camera-rig-structured-reprojection-2026-09-04-v1"
ENGINEERING_TRANSLATION_BAD_MM = 5.0
ENGINEERING_ROTATION_BAD_DEG = 0.5
MODEL_NAMES = ("image_physical", "board_quadratic", "minimal_image_board_union")
GROSS_RMSE_PX = 1.5
GROSS_P95_PX = 2.0


@dataclass(frozen=True)
class Family:
    family_id: str
    assignment: Literal["development", "holdout"]
    geometry: str
    intrinsics_profile: str
    distance: str
    tilt_deg: int
    placement: str
    visibility: str
    noise_px: float
    seed_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "assignment": self.assignment,
            "geometry": self.geometry,
            "intrinsics_profile": self.intrinsics_profile,
            "distance": self.distance,
            "tilt_deg": self.tilt_deg,
            "placement": self.placement,
            "visibility": self.visibility,
            "noise_px": self.noise_px,
            "seed_index": self.seed_index,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Family:
        assignment = str(value["assignment"])
        if assignment not in {"development", "holdout"}:
            raise ContractError("invalid structured split assignment")
        return cls(
            family_id=str(value["family_id"]),
            assignment=assignment,  # type: ignore[arg-type]
            geometry=str(value["geometry"]),
            intrinsics_profile=str(value["intrinsics_profile"]),
            distance=str(value["distance"]),
            tilt_deg=int(value["tilt_deg"]),
            placement=str(value["placement"]),
            visibility=str(value["visibility"]),
            noise_px=float(value["noise_px"]),
            seed_index=int(value["seed_index"]),
        )


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    amount: float = 0.0
    axis: str = ""
    sign: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "amount": self.amount,
            "axis": self.axis,
            "sign": self.sign,
        }


@dataclass(frozen=True)
class _ModelCase:
    truth_points: np.ndarray
    assumed_points: np.ndarray
    assumed_intrinsics: CameraIntrinsics
    local_bias: np.ndarray


def build_split_manifest() -> dict[str, object]:
    """Enumerate family-level assignments so descendants cannot cross the split."""
    families: list[Family] = []
    for geometry in ("a4_30mm", "large_500x700_equivalent_100mm"):
        for intrinsics_profile in ("d435i_wide", "d435i_narrow"):
            for distance in ("near", "medium", "far"):
                for tilt_deg in (0, 15, 30, 45, 60):
                    for placement in ("center", "edge", "corner"):
                        for visibility in ("full", "distributed_partial"):
                            for noise_px in (0.1, 0.25, 0.5, 0.75, 1.0):
                                for seed_index in (0, 1):
                                    spec = {
                                        "geometry": geometry,
                                        "intrinsics_profile": intrinsics_profile,
                                        "distance": distance,
                                        "tilt_deg": tilt_deg,
                                        "placement": placement,
                                        "visibility": visibility,
                                        "noise_px": noise_px,
                                        "seed_index": seed_index,
                                    }
                                    canonical = json.dumps(
                                        spec, sort_keys=True, separators=(",", ":")
                                    )
                                    digest = hashlib.sha256(
                                        f"{SPLIT_SALT}|{canonical}".encode()
                                    ).hexdigest()
                                    assignment: Literal["development", "holdout"] = (
                                        "holdout" if int(digest[:8], 16) % 5 == 0 else "development"
                                    )
                                    families.append(Family(digest, assignment, **spec))  # type: ignore[arg-type]
    counts = {
        name: sum(family.assignment == name for family in families)
        for name in ("development", "holdout")
    }
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split_salt": SPLIT_SALT,
        "split_unit": "family_id",
        "family_binding": (
            "positive and all K/D/target/warp/local/combined descendants remain together"
        ),
        "holdout_access_policy": (
            "no metrics, summaries, threshold scans, or plots before criteria SHA is frozen"
        ),
        "counts": counts,
        "families": [family.to_dict() for family in families],
    }


def run_development(manifest_path: Path) -> dict[str, object]:
    raw = load_json(manifest_path)
    if not isinstance(raw, dict) or raw.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ContractError("structured split manifest has the wrong schema version")
    values = raw.get("families")
    if not isinstance(values, list):
        raise ContractError("structured split manifest is missing families")
    families = [Family.from_dict(dict(value)) for value in values if isinstance(value, dict)]
    if len(families) != len(values):
        raise ContractError("structured split manifest contains a non-object family")
    _validate_frozen_split(raw, manifest_path)
    development = [family for family in families if family.assignment == "development"]
    rows: list[dict[str, object]] = []
    for family in development:
        rows.append(_evaluate(family, Variant("correct_model", "positive")))
        if _counterfactual_family(family):
            rows.extend(_evaluate(family, variant) for variant in _variants())
    summaries = {model: _summarize(rows, model) for model in MODEL_NAMES}
    final_rows: list[dict[str, object]] = []
    for family in development:
        if not _final_family(family):
            continue
        final_rows.append(_evaluate_final(family, Variant("correct_model", "positive")))
        final_rows.extend(_evaluate_final(family, variant) for variant in _final_variants())
    final_summaries = {model: _summarize(final_rows, model) for model in MODEL_NAMES}
    return {
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "status": "DEVELOPMENT_ONLY",
        "generator_version": GENERATOR_VERSION,
        "split_manifest_sha256": sha256_file(manifest_path),
        "holdout_family_count": sum(family.assignment == "holdout" for family in families),
        "holdout_metrics_opened": False,
        "engineering_bad_pose_definition": {
            "translation_error_mm_strictly_greater_than": ENGINEERING_TRANSLATION_BAD_MM,
            "rotation_error_deg_strictly_greater_than": ENGINEERING_ROTATION_BAD_DEG,
            "logic": "OR",
            "status": "development_candidate_not_preregistered",
        },
        "gross_scalar_candidate": {
            "maximum_rmse_px": GROSS_RMSE_PX,
            "maximum_p95_px": GROSS_P95_PX,
            "status": "CANDIDATE_HOLD",
        },
        "structured_model_candidates": {
            model: StructuredResidualThresholds(model_name=model).to_dict() for model in MODEL_NAMES
        },
        "summaries": summaries,
        "final_shared_pose_summaries": final_summaries,
        "identifiability_limit": (
            "planar monocular residual structure cannot identify all target-scale, focal, or "
            "principal-point pose bias; native depth and trusted target metrology remain required"
        ),
        "rows": rows,
        "final_shared_pose_rows": final_rows,
    }


def run_threshold_scan(development_path: Path) -> dict[str, object]:
    """Scan a small preregistered grid using development rows only."""
    raw = load_json(development_path)
    if not isinstance(raw, dict) or raw.get("schema_version") != DEVELOPMENT_SCHEMA_VERSION:
        raise ContractError("structured development evidence has the wrong schema version")
    if raw.get("holdout_metrics_opened") is not False:
        raise ContractError("threshold selection cannot consume opened holdout evidence")
    row_values = raw.get("final_shared_pose_rows")
    if not isinstance(row_values, list) or not all(isinstance(row, dict) for row in row_values):
        raise ContractError("development evidence is missing final shared-pose rows")
    rows = [dict(row) for row in row_values]
    candidates: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        for maximum_rmse in (1.0, 1.25, 1.5, 2.0, 3.0):
            for maximum_p95 in (1.5, 2.0, 2.5, 3.0, 4.0):
                for minimum_effect in (0.05, 0.10, 0.20, 0.30, 0.40):
                    for minimum_amplitude in (0.005, 0.01, 0.025, 0.05, 0.10, 0.15):
                        for alpha in (0.01, 0.05):
                            settings = {
                                "model_name": model,
                                "maximum_final_rmse_px": maximum_rmse,
                                "maximum_final_p95_px": maximum_p95,
                                "minimum_cv_explained_fraction": minimum_effect,
                                "minimum_structured_amplitude_px": minimum_amplitude,
                                "maximum_permutation_p_value": alpha,
                            }
                            positive_errors, positive_n = _family_error_count(
                                rows,
                                eligible=lambda row: (
                                    row.get("label") == "POSITIVE_ENGINEERING_GOOD"
                                ),
                                error=lambda row, settings=settings: (
                                    not _grid_decision(row, settings)
                                ),
                            )
                            negative_errors, negative_n = _family_error_count(
                                rows,
                                eligible=lambda row: row.get("label") == "NEGATIVE_POSE_BIASED",
                                error=lambda row, settings=settings: _grid_decision(row, settings),
                            )
                            challenging_errors, challenging_n = _family_error_count(
                                rows,
                                eligible=lambda row: (
                                    row.get("label") == "NEGATIVE_POSE_BIASED"
                                    and _is_challenging_negative(row)
                                ),
                                error=lambda row, settings=settings: _grid_decision(row, settings),
                            )
                            positive_upper = _wilson_upper(positive_errors, positive_n)
                            negative_upper = _wilson_upper(negative_errors, negative_n)
                            challenging_upper = _wilson_upper(challenging_errors, challenging_n)
                            assert (
                                positive_upper is not None
                                and negative_upper is not None
                                and challenging_upper is not None
                            )
                            candidates.append(
                                {
                                    **settings,
                                    "statistical_unit": "counterfactual_family_worst_case",
                                    "positive_n": positive_n,
                                    "positive_false_reject": positive_errors,
                                    "positive_false_reject_wilson_upper_95": positive_upper,
                                    "negative_n": negative_n,
                                    "negative_false_accept": negative_errors,
                                    "negative_false_accept_wilson_upper_95": negative_upper,
                                    "challenging_negative_n": challenging_n,
                                    "challenging_negative_false_accept": challenging_errors,
                                    "challenging_negative_false_accept_wilson_upper_95": (
                                        challenging_upper
                                    ),
                                    "development_target_met": positive_upper <= 0.05
                                    and negative_upper <= 0.05
                                    and challenging_upper <= 0.05,
                                    "ranking_score": max(
                                        positive_upper, negative_upper, challenging_upper
                                    ),
                                }
                            )
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["ranking_score"]),
            float(row["positive_false_reject_wilson_upper_95"])
            + float(row["negative_false_accept_wilson_upper_95"])
            + float(row["challenging_negative_false_accept_wilson_upper_95"]),
            str(row["model_name"]),
        ),
    )
    passing = [row for row in ranked if row["development_target_met"] is True]
    return {
        "schema_version": THRESHOLD_SCAN_SCHEMA_VERSION,
        "status": "DEVELOPMENT_ONLY",
        "development_evidence_sha256": sha256_file(development_path),
        "holdout_metrics_opened": False,
        "grid_definition": {
            "models": list(MODEL_NAMES),
            "maximum_final_rmse_px": [1.0, 1.25, 1.5, 2.0, 3.0],
            "maximum_final_p95_px": [1.5, 2.0, 2.5, 3.0, 4.0],
            "minimum_cv_explained_fraction": [0.05, 0.10, 0.20, 0.30, 0.40],
            "minimum_structured_amplitude_px": [0.005, 0.01, 0.025, 0.05, 0.10, 0.15],
            "maximum_permutation_p_value": [0.01, 0.05],
        },
        "candidate_count": len(ranked),
        "development_target_met_count": len(passing),
        "recommendation": "CONTINUE_TO_REAL_DEVELOPMENT" if passing else "HOLD",
        "best_development_candidates": ranked[:25],
        "note": (
            "no threshold is frozen here; real development evidence is required before "
            "preregistration and no holdout result was opened"
        ),
    }


def _grid_decision(row: dict[str, object], settings: dict[str, object]) -> bool:
    scalar = row.get("scalar_reprojection")
    models = row.get("models")
    model = models.get(settings["model_name"]) if isinstance(models, dict) else None
    metrics = model.get("metrics") if isinstance(model, dict) else None
    if not isinstance(scalar, dict) or not isinstance(metrics, dict):
        return False
    scalar_pass = float(scalar["rmse_px"]) <= float(settings["maximum_final_rmse_px"]) and float(
        scalar["p95_px"]
    ) <= float(settings["maximum_final_p95_px"])
    support = metrics.get("sufficient_support") is True
    p_value = metrics.get("permutation_p_value")
    effect = metrics.get("cv_explained_fraction")
    amplitude = metrics.get("structured_amplitude_px")
    if not all(isinstance(value, int | float) for value in (p_value, effect, amplitude)):
        return False
    structured_failure = (
        float(p_value) <= float(settings["maximum_permutation_p_value"])
        and float(effect) >= float(settings["minimum_cv_explained_fraction"])
        and float(amplitude) >= float(settings["minimum_structured_amplitude_px"])
    )
    return (
        scalar_pass
        and row.get("observability_passed") is True
        and support
        and not structured_failure
    )


def _counterfactual_family(family: Family) -> bool:
    return (
        family.noise_px in {0.1, 0.25}
        and family.tilt_deg in {15, 45}
        and family.visibility == "full"
        and family.seed_index == 0
    )


def _final_family(family: Family) -> bool:
    return (
        family.noise_px in {0.25, 0.5, 0.75}
        and family.tilt_deg in {15, 45}
        and family.placement in {"center", "edge"}
        and family.visibility == "full"
    )


def _variants() -> tuple[Variant, ...]:
    values: list[Variant] = []
    for amount in (0.0025, 0.005, 0.01, 0.02, 0.05):
        for sign in (-1, 1):
            values.append(Variant(f"focal_{sign * amount:+.4f}", "focal", amount, sign=sign))
    for amount in (1.0, 2.0, 5.0, 10.0):
        for sign in (-1, 1):
            values.append(
                Variant(f"principal_xy_{sign * amount:+.1f}px", "principal", amount, sign=sign)
            )
    distortion = {
        "k1": (0, (0.01, 0.03)),
        "k2": (1, (0.02, 0.05)),
        "p1": (2, (0.001, 0.003)),
        "p2": (3, (0.001, 0.003)),
        "k3": (4, (0.02,)),
    }
    for axis, (_index, amounts) in distortion.items():
        for amount in amounts:
            for sign in (-1, 1):
                values.append(
                    Variant(
                        f"distortion_{axis}_{sign * amount:+.4f}",
                        "distortion",
                        amount,
                        axis,
                        sign,
                    )
                )
    for amount in (0.0025, 0.005, 0.01, 0.02):
        for sign in (-1, 1):
            values.append(
                Variant(f"target_scale_{sign * amount:+.4f}", "target_scale", amount, sign=sign)
            )
    for amount in (0.00025, 0.0005, 0.001, 0.002, 0.005):
        for sign in (-1, 1):
            values.append(
                Variant(f"board_warp_{sign * amount:+.5f}m", "board_warp", amount, sign=sign)
            )
    for amount in (0.25, 0.5, 1.0, 2.0, 3.0):
        values.append(Variant(f"local_corner_bias_{amount:.2f}px", "local_bias", amount))
    values.extend(
        (
            Variant("combined_focal_plus_warp", "combined", 1.0, sign=1),
            Variant("combined_focal_minus_warp", "combined", 1.0, sign=-1),
            Variant("catastrophic_3px_structured", "catastrophic", 3.0),
        )
    )
    return tuple(values)


def _final_variants() -> tuple[Variant, ...]:
    selected = {
        "focal_+0.0100",
        "principal_xy_+5.0px",
        "distortion_k1_+0.0300",
        "distortion_p1_+0.0030",
        "target_scale_+0.0100",
        "board_warp_+0.00100m",
        "local_corner_bias_1.00px",
        "combined_focal_plus_warp",
        "catastrophic_3px_structured",
    }
    return tuple(variant for variant in _variants() if variant.name in selected)


def _evaluate(family: Family, variant: Variant) -> dict[str, object]:
    points, rows, columns = _geometry(family.geometry)
    intrinsics = _intrinsics(family.intrinsics_profile)
    pose = _pose(family, points)
    selected = np.ones(len(points), dtype=np.bool_)
    if family.visibility == "distributed_partial":
        selected = np.arange(len(points)) % 6 != 0
    model_case = _model_case(points, rows, columns, intrinsics, variant)
    truth_pixels = project_points_px(model_case.truth_points, pose, intrinsics)
    noise = np.random.default_rng(_seed(family.family_id, "noise")).normal(
        0.0, family.noise_px, truth_pixels.shape
    )
    observed = np.asarray(truth_pixels, dtype=np.float64) + noise
    observed += model_case.local_bias
    in_image = (
        (observed[:, 0] >= 0)
        & (observed[:, 0] < intrinsics.width)
        & (observed[:, 1] >= 0)
        & (observed[:, 1] < intrinsics.height)
    )
    selected &= in_image
    ids = np.flatnonzero(selected)
    base: dict[str, object] = {
        "family_id": family.family_id,
        "assignment": family.assignment,
        "family": family.to_dict(),
        "variant": variant.to_dict(),
        "corner_count": len(ids),
        "coverage_ratio": _coverage(observed[selected], intrinsics),
        "coverage_group": _coverage_group(_coverage(observed[selected], intrinsics)),
    }
    if len(ids) < 12:
        return {
            **base,
            "evaluation_status": "POSE_SOLVE_FAILED",
            "failure_reason": "insufficient visible corners",
            "label": "POSITIVE_REJECT" if variant.kind == "positive" else "NEGATIVE_UNRESOLVED",
            "models": {},
        }
    observation = TargetObservation(
        plugin_name="structured-release-synthetic",
        target_frame="target",
        point_ids=tuple(int(value) for value in ids),
        image_points_px=observed[selected],
        object_points_m=model_case.assumed_points[selected],
        image_size=(intrinsics.width, intrinsics.height),
        quality=QualityReport(True),
        metadata={"generator_version": GENERATOR_VERSION},
    )
    try:
        estimate = PlanarPoseEstimator().estimate(observation, model_case.assumed_intrinsics)
    except ContractError as error:
        return {
            **base,
            "evaluation_status": "POSE_SOLVE_FAILED",
            "failure_reason": str(error),
            "label": "POSITIVE_REJECT" if variant.kind == "positive" else "NEGATIVE_UNRESOLVED",
            "models": {},
        }
    projected = project_points_px(
        observation.object_points_m,
        estimate.T_camera_from_target,
        model_case.assumed_intrinsics,
    )
    residuals = np.asarray(observation.image_points_px) - projected
    norms = np.linalg.norm(residuals, axis=1)
    delta = pose_delta(estimate.T_camera_from_target, pose)
    pose_bad = (
        delta.translation_mm > ENGINEERING_TRANSLATION_BAD_MM
        or delta.rotation_deg > ENGINEERING_ROTATION_BAD_DEG
    )
    if variant.kind == "positive":
        label = "POSITIVE_ENGINEERING_GOOD" if not pose_bad else "POSITIVE_POSE_BAD"
    else:
        label = "NEGATIVE_POSE_BIASED" if pose_bad else "NEGATIVE_MODEL_MISMATCH_BUT_POSE_SMALL"
    scalar_pass = (
        float(np.sqrt(np.mean(np.square(norms)))) <= GROSS_RMSE_PX
        and float(np.percentile(norms, 95)) <= GROSS_P95_PX
    )
    models: dict[str, object] = {}
    for model_name in MODEL_NAMES:
        metrics = evaluate_structured_residuals(
            object_points_m=observation.object_points_m,
            projected_points_px=projected,
            residual_vectors_px=residuals,
            intrinsics=model_case.assumed_intrinsics,
            scope="frame",
            thresholds=StructuredResidualThresholds(model_name=model_name),
            board_reference_points_m=model_case.assumed_points,
        )
        models[model_name] = {
            "metrics": metrics.to_dict(),
            "candidate_combined_pass": scalar_pass
            and estimate.observability.passed
            and metrics.passed,
        }
    return {
        **base,
        "evaluation_status": "EVALUATED",
        "label": label,
        "pose_error": {
            "translation_mm": delta.translation_mm,
            "rotation_deg": delta.rotation_deg,
        },
        "observability_passed": estimate.observability.passed,
        "scalar_reprojection": {
            "rmse_px": float(np.sqrt(np.mean(np.square(norms)))),
            "p95_px": float(np.percentile(norms, 95)),
            "passed": scalar_pass,
        },
        "models": models,
    }


def _evaluate_final(family: Family, variant: Variant) -> dict[str, object]:
    points, rows, columns = _geometry(family.geometry)
    intrinsics = _intrinsics(family.intrinsics_profile)
    truth_pose = _pose(family, points)
    model = _model_case(points, rows, columns, intrinsics, variant)
    truth_pixels = project_points_px(model.truth_points, truth_pose, intrinsics)
    in_image = (
        (truth_pixels[:, 0] >= 0)
        & (truth_pixels[:, 0] < intrinsics.width)
        & (truth_pixels[:, 1] >= 0)
        & (truth_pixels[:, 1] < intrinsics.height)
    )
    ids = np.flatnonzero(in_image)
    base: dict[str, object] = {
        "family_id": family.family_id,
        "assignment": family.assignment,
        "family": family.to_dict(),
        "variant": variant.to_dict(),
        "corner_count": len(ids),
        "frame_count": 60,
        "coverage_ratio": _coverage(truth_pixels[in_image], intrinsics),
        "coverage_group": _coverage_group(_coverage(truth_pixels[in_image], intrinsics)),
    }
    observations: list[TargetObservation] = []
    for frame_index in range(60):
        noise = np.random.default_rng(_seed(family.family_id, "final", str(frame_index))).normal(
            0.0, family.noise_px, truth_pixels.shape
        )
        observed = np.asarray(truth_pixels, dtype=np.float64) + noise + model.local_bias
        observations.append(
            TargetObservation(
                plugin_name="structured-release-synthetic-final",
                target_frame="target",
                point_ids=tuple(int(value) for value in ids),
                image_points_px=observed[in_image],
                object_points_m=model.assumed_points[in_image],
                image_size=(intrinsics.width, intrinsics.height),
                quality=QualityReport(True),
                metadata={"generator_version": GENERATOR_VERSION},
            )
        )
    try:
        initial = (
            PlanarPoseEstimator()
            .estimate(observations[0], model.assumed_intrinsics)
            .T_camera_from_target
        )
        all_objects = np.vstack([item.object_points_m for item in observations])
        all_images = np.vstack([item.image_points_px for item in observations])
        refinement = refine_planar_pose_lm(
            initial, all_objects, all_images, model.assumed_intrinsics
        )
        if not refinement.validity.valid:
            raise ContractError(f"shared pose invalid: {list(refinement.validity.failure_reasons)}")
    except ContractError as error:
        return {
            **base,
            "evaluation_status": "POSE_SOLVE_FAILED",
            "failure_reason": str(error),
            "label": "POSITIVE_REJECT" if variant.kind == "positive" else "NEGATIVE_UNRESOLVED",
            "models": {},
        }
    final_pose = refinement.T_camera_from_target
    delta = pose_delta(final_pose, truth_pose)
    pose_bad = (
        delta.translation_mm > ENGINEERING_TRANSLATION_BAD_MM
        or delta.rotation_deg > ENGINEERING_ROTATION_BAD_DEG
    )
    if variant.kind == "positive":
        label = "POSITIVE_ENGINEERING_GOOD" if not pose_bad else "POSITIVE_POSE_BAD"
    else:
        label = "NEGATIVE_POSE_BIASED" if pose_bad else "NEGATIVE_MODEL_MISMATCH_BUT_POSE_SMALL"
    all_projected = project_points_px(all_objects, final_pose, model.assumed_intrinsics)
    all_residuals = all_images - all_projected
    norms = np.linalg.norm(all_residuals, axis=1)
    scalar_pass = (
        float(np.sqrt(np.mean(np.square(norms)))) <= GROSS_RMSE_PX
        and float(np.percentile(norms, 95)) <= GROSS_P95_PX
    )
    observability = evaluate_pose_observability(
        object_points_m=all_objects,
        image_points_px=all_images,
        T_camera_from_target=final_pose,
        intrinsics=model.assumed_intrinsics,
        scope="final",
    )
    models: dict[str, object] = {}
    for model_name in MODEL_NAMES:
        structured = evaluate_final_shared_structured_residuals(
            observations,
            final_pose,
            model.assumed_intrinsics,
            thresholds=StructuredResidualThresholds(model_name=model_name),
        )
        metrics = structured["structured_metrics"]
        assert isinstance(metrics, dict)
        models[model_name] = {
            "metrics": metrics,
            "candidate_combined_pass": scalar_pass
            and observability.passed
            and metrics.get("passed") is True,
        }
    return {
        **base,
        "evaluation_status": "EVALUATED",
        "label": label,
        "pose_error": {
            "translation_mm": delta.translation_mm,
            "rotation_deg": delta.rotation_deg,
        },
        "observability_passed": observability.passed,
        "scalar_reprojection": {
            "rmse_px": float(np.sqrt(np.mean(np.square(norms)))),
            "p95_px": float(np.percentile(norms, 95)),
            "passed": scalar_pass,
        },
        "models": models,
    }


def _model_case(
    points: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    intrinsics: CameraIntrinsics,
    variant: Variant,
) -> _ModelCase:
    assumed_points = points.copy()
    truth_points = points.copy()
    assumed_intrinsics = intrinsics
    local_bias = np.zeros((len(points), 2), dtype=np.float64)
    if variant.kind == "focal":
        factor = 1.0 + variant.sign * variant.amount
        assumed_intrinsics = replace(
            intrinsics, fx=intrinsics.fx * factor, fy=intrinsics.fy * factor
        )
    elif variant.kind == "principal":
        assumed_intrinsics = replace(
            intrinsics,
            cx=intrinsics.cx + variant.sign * variant.amount,
            cy=intrinsics.cy + variant.sign * variant.amount,
        )
    elif variant.kind == "distortion":
        coefficients = list(intrinsics.distortion_coeffs)
        index = {"k1": 0, "k2": 1, "p1": 2, "p2": 3, "k3": 4}[variant.axis]
        coefficients[index] += variant.sign * variant.amount
        assumed_intrinsics = replace(intrinsics, distortion_coeffs=tuple(coefficients))
    elif variant.kind == "target_scale":
        assumed_points *= 1.0 + variant.sign * variant.amount
    elif variant.kind == "board_warp":
        truth_points[:, 2] += variant.sign * variant.amount * _warp_shape(points)
    elif variant.kind == "local_bias":
        mask = (rows >= np.median(rows)) & (columns >= np.median(columns))
        local_bias[mask] = variant.amount * np.asarray([1.0, -0.6])
    elif variant.kind == "catastrophic":
        shape = _warp_shape(points)
        vectors = np.stack((shape, -0.5 * shape), axis=1)
        local_bias = vectors * (
            variant.amount / math.sqrt(float(np.mean(np.sum(np.square(vectors), axis=1))))
        )
    elif variant.kind == "combined":
        assumed_intrinsics = replace(
            intrinsics,
            fx=intrinsics.fx * (1.0 + variant.sign * 0.005),
            fy=intrinsics.fy * (1.0 + variant.sign * 0.005),
        )
        truth_points[:, 2] += variant.sign * 0.0005 * _warp_shape(points)
    elif variant.kind != "positive":
        raise ContractError(f"unknown counterfactual kind: {variant.kind}")
    return _ModelCase(truth_points, assumed_points, assumed_intrinsics, local_bias)


def _summarize(rows: list[dict[str, object]], model: str) -> dict[str, object]:
    positive_rejects, positive_n = _family_error_count(
        rows,
        eligible=lambda row: row.get("label") == "POSITIVE_ENGINEERING_GOOD",
        error=lambda row: not _decision(row, model),
    )
    negative_accepts, negative_n = _family_error_count(
        rows,
        eligible=lambda row: row.get("label") == "NEGATIVE_POSE_BIASED",
        error=lambda row: _decision(row, model),
    )
    challenging_accepts, challenging_n = _family_error_count(
        rows,
        eligible=lambda row: (
            row.get("label") == "NEGATIVE_POSE_BIASED" and _is_challenging_negative(row)
        ),
        error=lambda row: _decision(row, model),
    )
    negative = [row for row in rows if row.get("label") == "NEGATIVE_POSE_BIASED"]
    by_kind: dict[str, dict[str, int]] = {}
    for row in negative:
        variant = row.get("variant")
        kind = str(variant.get("kind")) if isinstance(variant, dict) else "unknown"
        item = by_kind.setdefault(kind, {"count": 0, "candidate_accept": 0})
        item["count"] += 1
        item["candidate_accept"] += int(_decision(row, model))
    return {
        "statistical_unit": "counterfactual_family_worst_case",
        "positive_n": positive_n,
        "positive_false_reject": positive_rejects,
        "positive_false_reject_wilson_upper_95": _wilson_upper(positive_rejects, positive_n),
        "negative_pose_biased_n": negative_n,
        "negative_false_accept": negative_accepts,
        "negative_false_accept_wilson_upper_95": _wilson_upper(negative_accepts, negative_n),
        "challenging_negative_n": challenging_n,
        "challenging_negative_false_accept": challenging_accepts,
        "challenging_negative_false_accept_wilson_upper_95": _wilson_upper(
            challenging_accepts, challenging_n
        ),
        "negative_by_kind": dict(sorted(by_kind.items())),
        "critical_subgroups": _critical_subgroups(rows, model),
        "release_status": "DEVELOPMENT_ONLY_NOT_A_RELEASE_DECISION",
    }


def _critical_subgroups(rows: list[dict[str, object]], model: str) -> dict[str, dict[str, object]]:
    predicates = {
        "far": lambda row: _family_value(row, "distance") == "far",
        "oblique_45_or_60": lambda row: int(_family_value(row, "tilt_deg")) >= 45,
        "low_coverage_below_1_percent": lambda row: row.get("coverage_group") == "below_1_percent",
        "large_500x700_equivalent": lambda row: (
            _family_value(row, "geometry") == "large_500x700_equivalent_100mm"
        ),
        "image_edge_or_corner": lambda row: _family_value(row, "placement") in {"edge", "corner"},
    }
    result: dict[str, dict[str, object]] = {}
    for name, predicate in predicates.items():
        selected = [row for row in rows if predicate(row)]
        positive_errors, positive_n = _family_error_count(
            selected,
            eligible=lambda row: row.get("label") == "POSITIVE_ENGINEERING_GOOD",
            error=lambda row: not _decision(row, model),
        )
        negative_errors, negative_n = _family_error_count(
            selected,
            eligible=lambda row: row.get("label") == "NEGATIVE_POSE_BIASED",
            error=lambda row: _decision(row, model),
        )
        result[name] = {
            "statistical_unit": "counterfactual_family_worst_case",
            "positive_n": positive_n,
            "positive_false_reject": positive_errors,
            "positive_false_reject_wilson_upper_95": _wilson_upper(positive_errors, positive_n),
            "positive_zero_error_bound_capability": (
                "CAPABLE" if positive_n >= 73 else "DENOMINATOR_TOO_SMALL_FOR_5_PERCENT"
            ),
            "negative_n": negative_n,
            "negative_false_accept": negative_errors,
            "negative_false_accept_wilson_upper_95": _wilson_upper(negative_errors, negative_n),
            "negative_zero_error_bound_capability": (
                "CAPABLE" if negative_n >= 73 else "DENOMINATOR_TOO_SMALL_FOR_5_PERCENT"
            ),
        }
    return result


def _family_value(row: dict[str, object], name: str) -> object:
    family = row.get("family")
    return family.get(name) if isinstance(family, dict) else ""


def _decision(row: dict[str, object], model: str) -> bool:
    models = row.get("models")
    if not isinstance(models, dict):
        return False
    value = models.get(model)
    return isinstance(value, dict) and value.get("candidate_combined_pass") is True


def _is_challenging_negative(row: dict[str, object]) -> bool:
    scalar = row.get("scalar_reprojection")
    return (
        row.get("observability_passed") is True
        and isinstance(scalar, dict)
        and scalar.get("passed") is True
    )


def _family_error_count(
    rows: list[dict[str, object]],
    *,
    eligible: Callable[[dict[str, object]], bool],
    error: Callable[[dict[str, object]], bool],
) -> tuple[int, int]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if not eligible(row):
            continue
        family_id = str(_family_value(row, "family_id"))
        grouped.setdefault(family_id, []).append(row)
    errors = sum(any(error(row) for row in values) for values in grouped.values())
    return errors, len(grouped)


def _validate_frozen_split(raw: dict[str, object], manifest_path: Path) -> None:
    receipt_path = (
        Path(__file__).parents[1] / "release_manifests" / ("structured_reprojection_split_v1.json")
    )
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ContractError("structured split receipt is invalid")
    expected = build_split_manifest()
    if raw != expected:
        raise ContractError("structured split manifest differs from deterministic generator")
    checks = {
        "generated_split_manifest_sha256": sha256_file(manifest_path),
        "assignment_salt": SPLIT_SALT,
        "generator_version": GENERATOR_VERSION,
        "family_count": len(expected["families"]),  # type: ignore[arg-type]
        "development_family_count": expected["counts"]["development"],  # type: ignore[index]
        "holdout_family_count": expected["counts"]["holdout"],  # type: ignore[index]
        "planned_development_final_family_count": 233,
        "planned_holdout_final_family_count": 55,
        "release_eligible": False,
    }
    for key, value in checks.items():
        if receipt.get(key) != value:
            raise ContractError(f"structured split receipt mismatch: {key}")


def _geometry(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "a4_30mm":
        row_count, column_count, spacing = 5, 7, 0.03
    elif name == "large_500x700_equivalent_100mm":
        row_count, column_count, spacing = 6, 4, 0.10
    else:
        raise ContractError(f"unknown synthetic geometry: {name}")
    rows, columns = np.indices((row_count, column_count))
    x = spacing * (columns.reshape(-1) - (column_count - 1) / 2.0)
    y = spacing * (rows.reshape(-1) - (row_count - 1) / 2.0)
    return (
        np.column_stack((x, y, np.zeros_like(x))).astype(np.float64),
        rows.reshape(-1),
        columns.reshape(-1),
    )


def _intrinsics(name: str) -> CameraIntrinsics:
    if name == "d435i_wide":
        fx, fy = 615.0, 617.0
    elif name == "d435i_narrow":
        fx, fy = 910.0, 905.0
    else:
        raise ContractError(f"unknown intrinsics profile: {name}")
    return CameraIntrinsics(
        frame="camera/color_optical",
        width=1280,
        height=720,
        fx=fx,
        fy=fy,
        cx=639.5,
        cy=359.5,
        distortion_model="brown-conrady",
        distortion_coeffs=(-0.045, 0.018, 0.0004, -0.0003, -0.003),
    )


def _pose(family: Family, points: np.ndarray) -> RigidTransform:
    if family.geometry == "a4_30mm":
        distances = {"near": 0.45, "medium": 0.85, "far": 1.8}
    else:
        distances = {"near": 1.0, "medium": 2.0, "far": 4.0}
    distance = distances[family.distance]
    angle = math.radians(family.tilt_deg)
    rotation_y = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    offsets = {
        "center": (0.0, 0.0),
        "edge": (0.35, 0.0),
        "corner": (0.28, 0.18),
    }
    offset_x, offset_y = offsets[family.placement]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_y @ np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [offset_x * distance, offset_y * distance, distance]
    del points
    return RigidTransform("target", "camera/color_optical", matrix)


def _warp_shape(points: np.ndarray) -> np.ndarray:
    board = points[:, :2]
    scale = math.sqrt(float(np.mean(np.sum(np.square(board), axis=1))))
    X, Y = (board / scale).T
    shape = X * X + 0.45 * X * Y + 0.25 * Y * Y
    return shape - np.mean(shape)


def _coverage(points: np.ndarray, intrinsics: CameraIntrinsics) -> float:
    if not len(points):
        return 0.0
    width = max(0.0, float(np.max(points[:, 0]) - np.min(points[:, 0])))
    height = max(0.0, float(np.max(points[:, 1]) - np.min(points[:, 1])))
    return width * height / (intrinsics.width * intrinsics.height)


def _coverage_group(value: float) -> str:
    if value > 0.05:
        return "above_5_percent"
    if value >= 0.01:
        return "1_to_5_percent"
    return "below_1_percent"


def _seed(*values: str) -> int:
    digest = hashlib.sha256("|".join(values).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _wilson_upper(errors: int, total: int) -> float | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    observed = errors / total
    denominator = 1.0 + z * z / total
    center = observed + z * z / (2.0 * total)
    margin = z * math.sqrt(observed * (1.0 - observed) / total + z * z / (4.0 * total * total))
    return min(1.0, (center + margin) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-split")
    prepare.add_argument("--output", type=Path, required=True)
    development = subparsers.add_parser("development")
    development.add_argument("--split-manifest", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    scan = subparsers.add_parser("scan-thresholds")
    scan.add_argument("--development-evidence", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare-split":
        atomic_write_json(arguments.output, build_split_manifest())
        print(f"wrote split manifest: {arguments.output} sha256={sha256_file(arguments.output)}")
        return 0
    if arguments.command == "scan-thresholds":
        report = run_threshold_scan(arguments.development_evidence)
        atomic_write_json(arguments.output, report)
        print(
            f"wrote structured threshold scan: {arguments.output} "
            f"sha256={sha256_file(arguments.output)} recommendation={report['recommendation']}"
        )
        return 0
    report = run_development(arguments.split_manifest)
    atomic_write_json(arguments.output, report)
    print(
        f"wrote structured development evidence: {arguments.output} "
        f"sha256={sha256_file(arguments.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
