"""Deterministic synthetic evidence for uncertainty reprojection-policy thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

from camera_rig.artifacts.io import atomic_write_json
from camera_rig.calibration.fixed.aggregation import pose_delta
from camera_rig.calibration.fixed.config import FixedCalibrationConfig, FixedSolverThresholds
from camera_rig.calibration.fixed.residuals import evaluate_residual_vector_field
from camera_rig.calibration.fixed.viability import evaluate_fixed_pose_frame_viability
from camera_rig.calibration.pose import PlanarPoseEstimator, project_points_px
from camera_rig.core.errors import CameraRigError
from camera_rig.core.intrinsics import CameraIntrinsics
from camera_rig.core.quality import QualityReport
from camera_rig.core.transforms import RigidTransform
from camera_rig.targets.observation import TargetObservation

FloatArray = npt.NDArray[np.float64]
LEVELS_PX = (0.1, 0.25, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0)
PATTERNS = ("gaussian", "radial", "tangential", "board_warp_like", "local_corruption")
RMSE_THRESHOLDS_PX = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
P95_THRESHOLDS_PX = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def run_sweep() -> dict[str, object]:
    intrinsics = _intrinsics()
    pose = _pose()
    points = _points()
    ideal = project_points_px(points, pose, intrinsics)
    rows: list[dict[str, object]] = []
    for pattern in PATTERNS:
        for level in LEVELS_PX:
            vectors = _scaled_pattern(pattern, points, ideal, pose, intrinsics, level)
            observation = _observation(points, ideal + vectors)
            expected_class = _expected_class(pattern, level)
            try:
                estimate = PlanarPoseEstimator().estimate(observation, intrinsics)
                decision = evaluate_fixed_pose_frame_viability(
                    frame_index=0,
                    detection_success=True,
                    observation=observation,
                    estimate=estimate,
                    config=_config(),
                    pose_policy="uncertainty_validated",
                )
                diagnostics = evaluate_residual_vector_field(
                    observation, estimate.T_camera_from_target, intrinsics
                )
                error = pose_delta(estimate.T_camera_from_target, pose)
                rows.append(
                    {
                        "pattern": pattern,
                        "injected_vector_rmse_px": level,
                        "expected_class": expected_class,
                        "evaluation_status": "EVALUATED",
                        "solved_reprojection": estimate.reprojection.to_dict(),
                        "translation_worst_std_mm": (
                            estimate.observability.translation_worst_axis_std_mm
                        ),
                        "rotation_worst_std_deg": (
                            estimate.observability.rotation_worst_axis_std_deg
                        ),
                        "pose_error_translation_mm": error.translation_mm,
                        "pose_error_rotation_deg": error.rotation_deg,
                        "scaled_condition_number": estimate.observability.scaled_condition_number,
                        "gross_reprojection_passed": _gross_passed(decision),
                        "policy_passed": decision["accepted"],
                        "failure_reasons": decision["failure_reasons"],
                        "residual_trends": diagnostics["trends"],
                        "residual_structure": {
                            "aggregate": diagnostics["aggregate"],
                            "board_coordinate_polynomial": diagnostics[
                                "board_coordinate_polynomial"
                            ],
                        },
                        "candidate_threshold_results": _candidate_threshold_results(
                            estimate.reprojection.rmse_px, estimate.reprojection.p95_px
                        ),
                    }
                )
            except CameraRigError as error:
                rows.append(
                    {
                        "pattern": pattern,
                        "injected_vector_rmse_px": level,
                        "expected_class": expected_class,
                        "evaluation_status": "POSE_SOLVE_FAILED",
                        "solved_reprojection": None,
                        "translation_worst_std_mm": None,
                        "rotation_worst_std_deg": None,
                        "pose_error_translation_mm": None,
                        "pose_error_rotation_deg": None,
                        "scaled_condition_number": None,
                        "gross_reprojection_passed": None,
                        "policy_passed": False,
                        "failure_reasons": [f"POSE_SOLVE_FAILED: {error}"],
                        "residual_trends": None,
                        "residual_structure": None,
                        "candidate_threshold_results": None,
                    }
                )
    return {
        "schema_version": "camera-rig.reprojection-policy-sweep.v1",
        "role": "synthetic_threshold_evidence_not_real_camera_acceptance",
        "levels_px": list(LEVELS_PX),
        "patterns": list(PATTERNS),
        "candidate_thresholds": {
            "maximum_gross_frame_rmse_px": 1.5,
            "maximum_gross_frame_p95_px": 2.0,
        },
        "threshold_grid": {
            "maximum_gross_frame_rmse_px": list(RMSE_THRESHOLDS_PX),
            "maximum_gross_frame_p95_px": list(P95_THRESHOLDS_PX),
        },
        "threshold_selection": {
            "positive_class": "stable_gaussian_noise_at_or_below_0.75_px",
            "negative_class": "model_mismatch_at_or_above_1.5_px_or_gaussian_at_or_above_2_px",
            "selection_rule": (
                "zero_evaluable_positive_rejections_and_zero_evaluable_negative_acceptances; "
                "pose-solve failures remain NOT_EVALUATED"
            ),
            "candidate_summaries": _candidate_threshold_summaries(rows),
        },
        "rows": rows,
    }


def _gross_passed(decision: dict[str, object]) -> bool:
    reprojection = decision.get("reprojection_decision")
    if not isinstance(reprojection, dict):
        return False
    checks = reprojection.get("checks")
    return isinstance(checks, dict) and all(value is True for value in checks.values())


def _scaled_pattern(
    name: str,
    points: FloatArray,
    projected: FloatArray,
    pose: RigidTransform,
    intrinsics: CameraIntrinsics,
    target_rmse: float,
) -> FloatArray:
    center = np.asarray([intrinsics.cx, intrinsics.cy], dtype=np.float64)
    centered = projected - center
    x = centered[:, 0] / intrinsics.fx
    y = centered[:, 1] / intrinsics.fy
    radius_squared = x * x + y * y
    if name == "gaussian":
        vectors = np.random.default_rng(9102).normal(0.0, 1.0, projected.shape)
    elif name == "radial":
        vectors = np.stack(
            (intrinsics.fx * x * radius_squared, intrinsics.fy * y * radius_squared), axis=1
        )
    elif name == "tangential":
        p1 = 1.0
        p2 = -0.7
        vectors = np.stack(
            (
                intrinsics.fx * (2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x)),
                intrinsics.fy * (p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y),
            ),
            axis=1,
        )
    elif name == "board_warp_like":
        board = points.copy()
        board_xy = board[:, :2] - np.mean(board[:, :2], axis=0)
        board_scale = np.max(np.abs(board_xy), axis=0)
        board_x = board_xy[:, 0] / board_scale[0]
        board_y = board_xy[:, 1] / board_scale[1]
        board[:, 2] = 0.005 * (board_x * board_x + 0.5 * board_x * board_y)
        board[:, 2] -= np.mean(board[:, 2])
        vectors = project_points_px(board, pose, intrinsics) - projected
    elif name == "local_corruption":
        vectors = np.zeros_like(projected)
        board_x = points[:, 0]
        board_y = points[:, 1]
        mask = (board_x >= np.quantile(board_x, 0.7)) & (board_y >= np.quantile(board_y, 0.5))
        vectors[mask] = np.asarray([2.0, -1.0])
    else:
        raise ValueError(f"unknown residual pattern: {name}")
    rms = float(np.sqrt(np.mean(np.sum(np.square(vectors), axis=1))))
    return vectors * (target_rmse / rms)


def _candidate_threshold_results(rmse_px: float, p95_px: float) -> list[dict[str, object]]:
    return [
        {
            "maximum_rmse_px": rmse,
            "maximum_p95_px": p95,
            "passed": rmse_px <= rmse and p95_px <= p95,
        }
        for rmse in RMSE_THRESHOLDS_PX
        for p95 in P95_THRESHOLDS_PX
    ]


def _expected_class(pattern: str, level_px: float) -> str:
    if pattern == "gaussian" and level_px <= 0.75:
        return "POSITIVE_STABLE_NOISE"
    if (pattern == "gaussian" and level_px >= 2.0) or (pattern != "gaussian" and level_px >= 1.5):
        return "NEGATIVE_MODEL_MISMATCH"
    return "EXPLORATORY_UNLABELED"


def _candidate_threshold_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for maximum_rmse in RMSE_THRESHOLDS_PX:
        for maximum_p95 in P95_THRESHOLDS_PX:
            counts = {
                "positive_accepted": 0,
                "positive_rejected": 0,
                "negative_rejected": 0,
                "negative_accepted": 0,
                "not_evaluated": 0,
            }
            for row in rows:
                expected = row["expected_class"]
                if expected == "EXPLORATORY_UNLABELED":
                    continue
                solved = row.get("solved_reprojection")
                if not isinstance(solved, dict):
                    counts["not_evaluated"] += 1
                    continue
                rmse = float(solved["rmse_px"])
                p95 = float(solved["p95_px"])
                passed = rmse <= maximum_rmse and p95 <= maximum_p95
                if expected == "POSITIVE_STABLE_NOISE":
                    counts["positive_accepted" if passed else "positive_rejected"] += 1
                else:
                    counts["negative_accepted" if passed else "negative_rejected"] += 1
            summaries.append(
                {
                    "maximum_rmse_px": maximum_rmse,
                    "maximum_p95_px": maximum_p95,
                    **counts,
                    "meets_selection_rule": (
                        counts["positive_rejected"] == 0 and counts["negative_accepted"] == 0
                    ),
                }
            )
    return summaries


def _points() -> FloatArray:
    return np.asarray(
        [[0.03 * column, 0.03 * row, 0.0] for row in range(5) for column in range(7)],
        dtype=np.float64,
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        frame="camera/color_optical",
        width=1280,
        height=720,
        fx=900.0,
        fy=905.0,
        cx=639.5,
        cy=359.5,
        distortion_model="none",
    )


def _pose() -> RigidTransform:
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = [-0.09, 0.06, 0.72]
    return RigidTransform("target", "camera/color_optical", matrix)


def _observation(points: FloatArray, pixels: FloatArray) -> TargetObservation:
    return TargetObservation(
        plugin_name="synthetic-grid",
        target_frame="target",
        point_ids=tuple(range(len(points))),
        image_points_px=pixels,
        object_points_m=points,
        image_size=(1280, 720),
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
            minimum_accepted_frames=50,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    atomic_write_json(arguments.output, run_sweep())
    print(f"wrote synthetic reprojection-policy sweep: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
