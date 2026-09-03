"""Deterministic development stress test for gross reprojection thresholds.

This benchmark is deliberately separate from ``reprojection_policy_sweep.py``:
it exercises iid noise, cellwise bounds, and frame/final structured mismatch.
It is exploratory rather than confirmatory. Residual structure stays diagnostic-only.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.io import atomic_write_json
from camera_rig.calibration.fixed.aggregation import pose_delta
from camera_rig.calibration.fixed.config import FixedCalibrationConfig, FixedSolverThresholds
from camera_rig.calibration.fixed.viability import evaluate_fixed_pose_frame_viability
from camera_rig.calibration.pose import (
    PlanarPoseEstimator,
    UncertaintyValidatedThresholds,
    project_points_px,
    refine_planar_pose_lm,
)
from camera_rig.core.errors import CameraRigError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

FloatArray = npt.NDArray[np.float64]
CANDIDATE_RMSE_PX = 0.75
CANDIDATE_P95_PX = 1.50
POSITIVE_LEVELS_PX = (0.10, 0.25, 0.50, 0.60, 0.75)
NEGATIVE_LEVELS_PX = (1.50, 2.00, 3.00)
STRUCTURED_PATTERNS = ("radial", "tangential", "board_warp_like", "local_corruption")
DEFAULT_SEEDS = 128
DEFAULT_FINAL_TRIALS = 64
DEFAULT_NEGATIVE_FINAL_TRIALS = 8
FRAMES_PER_CAPTURE = 60
REQUIRED_ACCEPTED_FRAMES = 54


@dataclass(frozen=True)
class GeometryProfile:
    name: str
    intrinsics: CameraIntrinsics
    pose: RigidTransform


def run_holdout(
    *,
    seeds: int = DEFAULT_SEEDS,
    final_trials: int = DEFAULT_FINAL_TRIALS,
    negative_final_trials: int = DEFAULT_NEGATIVE_FINAL_TRIALS,
) -> dict[str, object]:
    if seeds < 1 or final_trials < 1 or negative_final_trials < 1:
        raise ValueError("trial counts must be positive")
    profiles = _profiles()
    layouts = _layouts()
    positive = _positive_frame_trials(profiles, layouts, seeds)
    positive_boundary = [
        row
        for row in positive
        if float(row["injected_vector_rmse_target_px"]) == max(POSITIVE_LEVELS_PX)
    ]
    final_positive = _positive_final_trials(profiles, layouts, final_trials)
    negative = _negative_frame_trials(profiles, layouts, seeds)
    negative_final = _negative_final_trials(profiles, layouts, negative_final_trials)

    positive_cells = _positive_boundary_cells(positive_boundary)
    positive_upper = max(float(cell["false_reject_wilson_upper"]) for cell in positive_cells)
    capture_failure_upper = max(
        float(cell["estimated_60_frame_capture_false_reject_upper"]) for cell in positive_cells
    )
    final_cells = _positive_final_cells(final_positive)
    final_upper = max(float(cell["false_reject_wilson_upper"]) for cell in final_cells)
    gross_negative = [
        row for row in negative if float(row["injected_vector_rmse_target_px"]) >= 3.0
    ]
    gross_negative_rejected = sum(not bool(row["scalar_passed"]) for row in gross_negative)
    gross_negative_policy_rejected = sum(not bool(row["policy_passed"]) for row in gross_negative)
    gross_negative_final_rejected = sum(not bool(row["passed"]) for row in negative_final)
    release_checks = {
        "positive_frame_false_reject_wilson_upper_at_most_0_01": positive_upper <= 0.01,
        "estimated_60_frame_capture_false_reject_upper_at_most_1e_6": (
            capture_failure_upper <= 1e-6
        ),
        "positive_final_false_reject_wilson_upper_at_most_0_01": final_upper <= 0.01,
        "gross_3px_structured_negative_rejection_at_least_0_95": (
            gross_negative_rejected / len(gross_negative) >= 0.95
        ),
        "gross_3px_structured_frame_policy_rejection_at_least_0_95": (
            gross_negative_policy_rejected / len(gross_negative) >= 0.95
        ),
        "gross_3px_structured_final_rejection_at_least_0_95": (
            gross_negative_final_rejected / len(negative_final) >= 0.95
        ),
    }
    return {
        "schema_version": "camera-rig.reprojection-policy-holdout.v1",
        "role": "development_stress_test_not_confirmatory_or_real_camera_acceptance",
        "candidate_thresholds": {
            "maximum_gross_frame_rmse_px": CANDIDATE_RMSE_PX,
            "maximum_gross_frame_p95_px": CANDIDATE_P95_PX,
            "maximum_gross_final_rmse_px": CANDIDATE_RMSE_PX,
            "maximum_gross_final_p95_px": CANDIDATE_P95_PX,
        },
        "design": {
            "seed_count_per_cell": seeds,
            "final_trial_count_per_cell": final_trials,
            "negative_final_trial_count_per_cell": negative_final_trials,
            "positive_levels_vector_rmse_px": list(POSITIVE_LEVELS_PX),
            "negative_structure_levels_vector_rmse_px": list(NEGATIVE_LEVELS_PX),
            "mixed_negative_gaussian_component_std_px": 0.25,
            "profiles": [profile.name for profile in profiles],
            "corner_counts": [len(points) for _name, points in layouts],
            "frames_per_capture": FRAMES_PER_CAPTURE,
            "required_accepted_frames": REQUIRED_ACCEPTED_FRAMES,
        },
        "development_acceptance_rule": {
            "positive_frame_false_reject_wilson_upper": 0.01,
            "estimated_60_frame_capture_false_reject_upper": 1e-6,
            "positive_final_false_reject_wilson_upper": 0.01,
            "gross_3px_structured_negative_rejection_rate": 0.95,
            "gross_3px_structured_frame_policy_rejection_rate": 0.95,
            "gross_3px_structured_final_rejection_rate": 0.95,
            "scope": (
                "gross scalar sanity only; pose-absorbable single-plane mismatch is reported "
                "as an identifiability limitation and is not relabeled as scalar-detectable"
            ),
        },
        "summary": {
            "positive_frame": _trial_summary(positive),
            "positive_frame_at_0_75px_boundary": _trial_summary(positive_boundary),
            "positive_frame_at_0_75px_boundary_by_cell": positive_cells,
            "positive_frame_false_reject_wilson_upper": positive_upper,
            "estimated_60_frame_capture_false_reject_upper": capture_failure_upper,
            "positive_final": _trial_summary(final_positive),
            "positive_final_by_cell": final_cells,
            "positive_final_false_reject_wilson_upper": final_upper,
            "mixed_structured_negative": _trial_summary(negative),
            "gross_3px_structured_negative": _trial_summary(gross_negative),
            "gross_3px_structured_frame_policy": _policy_summary(gross_negative),
            "gross_3px_structured_final": _trial_summary(negative_final),
            "release_checks": release_checks,
            "release_recommendation": "PASS" if all(release_checks.values()) else "HOLD",
            "known_limit": (
                "single-planar-pose projective modes can absorb radial, tangential, or warp "
                "signals; residual structure is diagnostic-only and multipose/depth evidence "
                "is required for independent physical identification"
            ),
        },
        "positive_frame_trials": positive,
        "positive_final_trials": final_positive,
        "mixed_structured_negative_trials": negative,
        "mixed_structured_negative_final_trials": negative_final,
    }


def _positive_frame_trials(
    profiles: tuple[GeometryProfile, ...],
    layouts: tuple[tuple[str, FloatArray], ...],
    seeds: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_index, profile in enumerate(profiles):
        for layout_index, (layout_name, points) in enumerate(layouts):
            ideal = project_points_px(points, profile.pose, profile.intrinsics)
            for level_index, level in enumerate(POSITIVE_LEVELS_PX):
                for seed_index in range(seeds):
                    seed = _seed(11, profile_index, layout_index, level_index, seed_index)
                    vectors = _iid_gaussian(ideal.shape, level, seed)
                    rows.append(
                        _solve_frame(
                            profile,
                            layout_name,
                            points,
                            ideal + vectors,
                            injected_kind="gaussian",
                            injected_level=level,
                            seed=seed,
                        )
                    )
    return rows


def _positive_final_trials(
    profiles: tuple[GeometryProfile, ...],
    layouts: tuple[tuple[str, FloatArray], ...],
    trials: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_index, profile in enumerate(profiles):
        for layout_index, (layout_name, points) in enumerate(layouts):
            ideal = project_points_px(points, profile.pose, profile.intrinsics)
            objects = np.tile(points, (FRAMES_PER_CAPTURE, 1))
            for trial in range(trials):
                images = []
                for frame in range(FRAMES_PER_CAPTURE):
                    seed = _seed(23, profile_index, layout_index, trial, frame)
                    images.append(ideal + _iid_gaussian(ideal.shape, 0.75, seed))
                try:
                    refined = refine_planar_pose_lm(
                        profile.pose, objects, np.vstack(images), profile.intrinsics
                    )
                    error = pose_delta(refined.T_camera_from_target, profile.pose)
                    rows.append(
                        {
                            "profile": profile.name,
                            "layout": layout_name,
                            "corner_count": len(points),
                            "trial": trial,
                            "frames": FRAMES_PER_CAPTURE,
                            "rmse_px": refined.reprojection.rmse_px,
                            "p95_px": refined.reprojection.p95_px,
                            "translation_error_mm": error.translation_mm,
                            "rotation_error_deg": error.rotation_deg,
                            "passed": _passed(
                                refined.reprojection.rmse_px, refined.reprojection.p95_px
                            ),
                        }
                    )
                except CameraRigError as error:
                    rows.append(
                        {
                            "profile": profile.name,
                            "layout": layout_name,
                            "corner_count": len(points),
                            "trial": trial,
                            "frames": FRAMES_PER_CAPTURE,
                            "rmse_px": None,
                            "p95_px": None,
                            "translation_error_mm": None,
                            "rotation_error_deg": None,
                            "passed": False,
                            "failure": str(error),
                        }
                    )
    return rows


def _negative_frame_trials(
    profiles: tuple[GeometryProfile, ...],
    layouts: tuple[tuple[str, FloatArray], ...],
    seeds: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_index, profile in enumerate(profiles):
        for layout_index, (layout_name, points) in enumerate(layouts):
            ideal = project_points_px(points, profile.pose, profile.intrinsics)
            for pattern_index, pattern in enumerate(STRUCTURED_PATTERNS):
                base = _structured_pattern(pattern, points, ideal, profile)
                for level_index, level in enumerate(NEGATIVE_LEVELS_PX):
                    structured = _scale_vectors(base, level)
                    for seed_index in range(seeds):
                        seed = _seed(
                            37,
                            profile_index,
                            layout_index,
                            pattern_index,
                            level_index,
                            seed_index,
                        )
                        gaussian = np.random.default_rng(seed).normal(0.0, 0.25, ideal.shape)
                        rows.append(
                            _solve_frame(
                                profile,
                                layout_name,
                                points,
                                ideal + structured + gaussian,
                                injected_kind=pattern,
                                injected_level=level,
                                seed=seed,
                            )
                        )
    return rows


def _negative_final_trials(
    profiles: tuple[GeometryProfile, ...],
    layouts: tuple[tuple[str, FloatArray], ...],
    trials: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_index, profile in enumerate(profiles):
        for layout_index, (layout_name, points) in enumerate(layouts):
            ideal = project_points_px(points, profile.pose, profile.intrinsics)
            objects = np.tile(points, (FRAMES_PER_CAPTURE, 1))
            for pattern_index, pattern in enumerate(STRUCTURED_PATTERNS):
                base = _structured_pattern(pattern, points, ideal, profile)
                structured = _scale_vectors(base, max(NEGATIVE_LEVELS_PX))
                for trial in range(trials):
                    images = []
                    for frame in range(FRAMES_PER_CAPTURE):
                        seed = _seed(
                            53,
                            profile_index,
                            layout_index,
                            pattern_index,
                            trial,
                            frame,
                        )
                        gaussian = np.random.default_rng(seed).normal(0.0, 0.25, ideal.shape)
                        images.append(ideal + structured + gaussian)
                    try:
                        refined = refine_planar_pose_lm(
                            profile.pose, objects, np.vstack(images), profile.intrinsics
                        )
                        error = pose_delta(refined.T_camera_from_target, profile.pose)
                        rows.append(
                            {
                                "profile": profile.name,
                                "layout": layout_name,
                                "corner_count": len(points),
                                "pattern": pattern,
                                "trial": trial,
                                "frames": FRAMES_PER_CAPTURE,
                                "injected_vector_rmse_target_px": max(NEGATIVE_LEVELS_PX),
                                "rmse_px": refined.reprojection.rmse_px,
                                "p95_px": refined.reprojection.p95_px,
                                "translation_error_mm": error.translation_mm,
                                "rotation_error_deg": error.rotation_deg,
                                "passed": _passed(
                                    refined.reprojection.rmse_px,
                                    refined.reprojection.p95_px,
                                ),
                            }
                        )
                    except CameraRigError as error:
                        rows.append(
                            {
                                "profile": profile.name,
                                "layout": layout_name,
                                "corner_count": len(points),
                                "pattern": pattern,
                                "trial": trial,
                                "frames": FRAMES_PER_CAPTURE,
                                "injected_vector_rmse_target_px": max(NEGATIVE_LEVELS_PX),
                                "rmse_px": None,
                                "p95_px": None,
                                "translation_error_mm": None,
                                "rotation_error_deg": None,
                                "passed": False,
                                "failure": str(error),
                            }
                        )
    return rows


def _solve_frame(
    profile: GeometryProfile,
    layout_name: str,
    points: FloatArray,
    pixels: FloatArray,
    *,
    injected_kind: str,
    injected_level: float,
    seed: int,
) -> dict[str, object]:
    try:
        estimate = PlanarPoseEstimator().estimate(_observation(points, pixels), profile.intrinsics)
        error = pose_delta(estimate.T_camera_from_target, profile.pose)
        decision = evaluate_fixed_pose_frame_viability(
            frame_index=0,
            detection_success=True,
            observation=_observation(points, pixels),
            estimate=estimate,
            config=_config(),
            pose_policy="uncertainty_validated",
            uncertainty_thresholds=_candidate_thresholds(),
        )
        scalar_passed = _passed(estimate.reprojection.rmse_px, estimate.reprojection.p95_px)
        return {
            "profile": profile.name,
            "layout": layout_name,
            "corner_count": len(points),
            "seed": seed,
            "injected_kind": injected_kind,
            "injected_vector_rmse_target_px": injected_level,
            "rmse_px": estimate.reprojection.rmse_px,
            "p95_px": estimate.reprojection.p95_px,
            "translation_error_mm": error.translation_mm,
            "rotation_error_deg": error.rotation_deg,
            "passed": scalar_passed,
            "scalar_passed": scalar_passed,
            "policy_passed": decision["accepted"],
            "policy_failure_reasons": decision["failure_reasons"],
            "policy_applied_thresholds": decision["reprojection_decision"]["applied_thresholds"],
        }
    except CameraRigError as error:
        return {
            "profile": profile.name,
            "layout": layout_name,
            "corner_count": len(points),
            "seed": seed,
            "injected_kind": injected_kind,
            "injected_vector_rmse_target_px": injected_level,
            "rmse_px": None,
            "p95_px": None,
            "translation_error_mm": None,
            "rotation_error_deg": None,
            "passed": False,
            "scalar_passed": False,
            "policy_passed": False,
            "policy_failure_reasons": [f"POSE_SOLVE_FAILED: {error}"],
            "policy_applied_thresholds": {
                "maximum_frame_rmse_px": CANDIDATE_RMSE_PX,
                "maximum_frame_p95_px": CANDIDATE_P95_PX,
            },
            "failure": str(error),
        }


def _trial_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted = sum(bool(row["passed"]) for row in rows)
    rejected = len(rows) - accepted
    solved = [row for row in rows if row.get("rmse_px") is not None]
    return {
        "total": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / len(rows),
        "rmse_px": _distribution([float(row["rmse_px"]) for row in solved]),
        "p95_px": _distribution([float(row["p95_px"]) for row in solved]),
        "translation_error_mm": _distribution(
            [float(row["translation_error_mm"]) for row in solved]
        ),
        "rotation_error_deg": _distribution([float(row["rotation_error_deg"]) for row in solved]),
    }


def _policy_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted = sum(bool(row["policy_passed"]) for row in rows)
    reason_counts: dict[str, int] = {}
    for row in rows:
        reasons = row["policy_failure_reasons"]
        assert isinstance(reasons, list)
        for reason in reasons:
            key = str(reason)
            reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "total": len(rows),
        "accepted": accepted,
        "rejected": len(rows) - accepted,
        "acceptance_rate": accepted / len(rows),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
    }


def _positive_boundary_cells(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for profile in sorted({str(row["profile"]) for row in rows}):
        for layout in sorted({str(row["layout"]) for row in rows}):
            cell = [row for row in rows if row["profile"] == profile and row["layout"] == layout]
            failures = sum(not bool(row["passed"]) for row in cell)
            upper = _wilson_upper(failures, len(cell))
            cells.append(
                {
                    "profile": profile,
                    "layout": layout,
                    "trials": len(cell),
                    "failures": failures,
                    "false_reject_wilson_upper": upper,
                    "estimated_60_frame_capture_false_reject_upper": _binomial_upper_tail(
                        FRAMES_PER_CAPTURE,
                        upper,
                        FRAMES_PER_CAPTURE - REQUIRED_ACCEPTED_FRAMES,
                    ),
                }
            )
    return cells


def _positive_final_cells(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for profile in sorted({str(row["profile"]) for row in rows}):
        for layout in sorted({str(row["layout"]) for row in rows}):
            cell = [row for row in rows if row["profile"] == profile and row["layout"] == layout]
            failures = sum(not bool(row["passed"]) for row in cell)
            cells.append(
                {
                    "profile": profile,
                    "layout": layout,
                    "trials": len(cell),
                    "failures": failures,
                    "false_reject_wilson_upper": _wilson_upper(failures, len(cell)),
                }
            )
    return cells


def _distribution(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(np.max(data)),
    }


def _passed(rmse_px: float, p95_px: float) -> bool:
    return rmse_px <= CANDIDATE_RMSE_PX and p95_px <= CANDIDATE_P95_PX


def _wilson_upper(failures: int, trials: int, z: float = 1.959963984540054) -> float:
    proportion = failures / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return (center + radius) / denominator


def _binomial_upper_tail(trials: int, probability: float, maximum_failures: int) -> float:
    return sum(
        math.comb(trials, failures)
        * probability**failures
        * (1.0 - probability) ** (trials - failures)
        for failures in range(maximum_failures + 1, trials + 1)
    )


def _iid_gaussian(shape: tuple[int, ...], vector_rmse: float, seed: int) -> FloatArray:
    component_sigma = vector_rmse / math.sqrt(2.0)
    return np.random.default_rng(seed).normal(0.0, component_sigma, shape)


def _scale_vectors(vectors: FloatArray, level: float) -> FloatArray:
    rms = float(np.sqrt(np.mean(np.sum(np.square(vectors), axis=1))))
    return vectors * (level / rms)


def _structured_pattern(
    name: str,
    points: FloatArray,
    projected: FloatArray,
    profile: GeometryProfile,
) -> FloatArray:
    intrinsics = profile.intrinsics
    centered = projected - np.asarray([intrinsics.cx, intrinsics.cy], dtype=np.float64)
    x = centered[:, 0] / intrinsics.fx
    y = centered[:, 1] / intrinsics.fy
    radius_squared = x * x + y * y
    if name == "radial":
        return np.stack(
            (intrinsics.fx * x * radius_squared, intrinsics.fy * y * radius_squared), axis=1
        )
    if name == "tangential":
        p1, p2 = 1.0, -0.7
        return np.stack(
            (
                intrinsics.fx * (2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x)),
                intrinsics.fy * (p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y),
            ),
            axis=1,
        )
    if name == "board_warp_like":
        board = points.copy()
        board_xy = board[:, :2] - np.mean(board[:, :2], axis=0)
        board_scale = np.max(np.abs(board_xy), axis=0)
        board_x = board_xy[:, 0] / board_scale[0]
        board_y = board_xy[:, 1] / board_scale[1]
        board[:, 2] = 0.005 * (board_x * board_x + 0.5 * board_x * board_y)
        board[:, 2] -= np.mean(board[:, 2])
        return project_points_px(board, profile.pose, intrinsics) - projected
    if name == "local_corruption":
        vectors = np.zeros_like(projected)
        center = np.mean(points[:, :2], axis=0)
        mask = (points[:, 0] >= center[0]) & (points[:, 1] >= center[1])
        vectors[mask] = np.asarray([2.0, -1.0])
        return vectors
    raise ValueError(f"unknown structured pattern: {name}")


def _layouts() -> tuple[tuple[str, FloatArray], ...]:
    dense = _grid_points(7, 5)
    deployment = _grid_points(6, 4)
    return (
        ("dense_35", dense),
        ("deployment_full_24", deployment),
        ("deployment_partial_20", deployment[[*range(1, 11), *range(13, 23)]]),
        ("deployment_sparse_12", deployment[[0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23]]),
    )


def _grid_points(columns: int, rows: int) -> FloatArray:
    return np.asarray(
        [
            [0.03 * (column + 1), 0.03 * (rows - row), 0.0]
            for row in range(rows)
            for column in range(columns)
        ],
        dtype=np.float64,
    )


def _profiles() -> tuple[GeometryProfile, ...]:
    return (
        _profile(
            "synthetic_near_oblique",
            fx=520.0,
            fy=525.0,
            yaw_deg=-28.0,
            pitch_deg=16.0,
            distance_m=0.60,
        ),
        _profile(
            "synthetic_mid_oblique",
            fx=640.0,
            fy=635.0,
            yaw_deg=24.0,
            pitch_deg=-12.0,
            distance_m=0.80,
        ),
        _profile(
            "synthetic_far_steep",
            fx=780.0,
            fy=790.0,
            yaw_deg=8.0,
            pitch_deg=38.0,
            distance_m=1.05,
        ),
    )


def _profile(
    name: str,
    fx: float,
    fy: float,
    yaw_deg: float,
    pitch_deg: float,
    distance_m: float,
) -> GeometryProfile:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    rotation_y = np.asarray(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    printed_face_toward_camera = np.diag([1.0, -1.0, -1.0])
    rotation = rotation_y @ rotation_x @ printed_face_toward_camera
    nominal_board_center = np.asarray([0.105, 0.075, 0.0], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (
        np.asarray([0.0, 0.0, distance_m], dtype=np.float64) - rotation @ nominal_board_center
    )
    intrinsics = CameraIntrinsics(
        frame=f"{name}/color_optical",
        width=640,
        height=480,
        fx=fx,
        fy=fy,
        cx=320.0,
        cy=240.0,
        distortion_model="none",
    )
    return GeometryProfile(
        name=name,
        intrinsics=intrinsics,
        pose=RigidTransform("target", intrinsics.frame, matrix),
    )


def _observation(points: FloatArray, pixels: FloatArray) -> TargetObservation:
    return TargetObservation(
        plugin_name="synthetic-charuco",
        target_frame="target",
        point_ids=tuple(range(len(points))),
        image_points_px=pixels,
        object_points_m=points,
        image_size=(640, 480),
        quality=QualityReport(True),
    )


def _config() -> FixedCalibrationConfig:
    return FixedCalibrationConfig(
        workspace_frame="workspace",
        target_frame="target",
        T_workspace_from_target=RigidTransform("target", "workspace", np.eye(4)),
        detection_stream="color",
        reference_stream="color",
        solver=FixedSolverThresholds(
            method="ippe",
            refinement="lm",
            minimum_corners_per_frame=12,
            minimum_accepted_frames=54,
            minimum_accepted_ratio=0.9,
            maximum_frame_rmse_px=0.5,
            maximum_frame_p95_px=1.0,
            maximum_pose_translation_p95_mm=3.0,
            maximum_pose_rotation_p95_deg=0.3,
            maximum_split_translation_delta_mm=2.0,
            maximum_split_rotation_delta_deg=0.2,
        ),
        native_depth_check=True,
    )


def _candidate_thresholds() -> UncertaintyValidatedThresholds:
    return UncertaintyValidatedThresholds(
        maximum_gross_frame_rmse_px=CANDIDATE_RMSE_PX,
        maximum_gross_frame_p95_px=CANDIDATE_P95_PX,
        maximum_gross_final_rmse_px=CANDIDATE_RMSE_PX,
        maximum_gross_final_p95_px=CANDIDATE_P95_PX,
    )


def _seed(*indices: int) -> int:
    value = 2166136261
    for index in indices:
        value = ((value ^ index) * 16777619) & 0xFFFFFFFF
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--final-trials", type=int, default=DEFAULT_FINAL_TRIALS)
    parser.add_argument("--negative-final-trials", type=int, default=DEFAULT_NEGATIVE_FINAL_TRIALS)
    arguments = parser.parse_args()
    report = run_holdout(
        seeds=arguments.seeds,
        final_trials=arguments.final_trials,
        negative_final_trials=arguments.negative_final_trials,
    )
    atomic_write_json(arguments.output, report)
    recommendation = report["summary"]["release_recommendation"]  # type: ignore[index]
    print(f"wrote reprojection-policy holdout: {arguments.output} ({recommendation})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
